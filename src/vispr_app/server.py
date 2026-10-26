"""Loopback application API and a single sequential inference worker."""
import copy
import io
import json
import os
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .adapters import DirectQwen, LMStudio, ModelManager, local_url
from .pipeline import ATTRIBUTES, MAX_BYTES, REGISTRY, ROOT, parse_detections, read_upload, redact


class Settings(BaseModel):
    mode: Literal['lmstudio', 'direct', 'finetuned'] = 'lmstudio'
    base_url: str = 'http://127.0.0.1:1234/v1'
    model: str = Field(default='', max_length=1024)
    enabled: list[str] = Field(default_factory=lambda: list(REGISTRY))
    margin: int = Field(default=0, ge=0, le=20)


def create_app(storage=None, manager=None):
    storage = Path(storage or ROOT / 'data/vispr_app')
    storage.mkdir(parents=True, exist_ok=True)
    session = storage / ('session-' + uuid.uuid4().hex)
    session.mkdir()
    # Only stale app-owned session directories; never touch annotation data.
    for old in storage.glob('session-*'):
        if old.is_dir() and time.time() - old.stat().st_mtime > 86400:
            shutil.rmtree(old)
    lock = threading.RLock()
    executor = ThreadPoolExecutor(max_workers=1)
    jobs = {}
    model_manager = manager or ModelManager()
    token = secrets.token_urlsafe(32)
    limit = min(10, max(1, int(os.environ.get('VISPR_BATCH_LIMIT', '5'))))
    preference_path = storage / 'preferences.json'
    settings = Settings()
    if preference_path.exists():
        try:
            settings = Settings(**json.loads(preference_path.read_text(encoding='utf-8')))
            if set(settings.enabled) - REGISTRY.keys():
                raise ValueError('Invalid attribute preferences')
            local_url(settings.base_url)
        except (ValueError, TypeError):
            # Never silently replace corrupt saved preferences with defaults.
            raise RuntimeError('Invalid data/vispr_app/preferences.json; repair or explicitly remove it to restore defaults.')

    @asynccontextmanager
    async def lifespan(app):
        yield
        with lock:
            for job in jobs.values():
                job['cancel_requested'] = True
        executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(session, ignore_errors=True)

    app = FastAPI(title='VISPR local review', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.middleware('http')
    async def privacy_guard(request: Request, call_next):
        if request.method in {'POST', 'PUT', 'DELETE', 'PATCH'}:
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'detail': 'Cross-origin changes are forbidden.'}, status_code=403)
            if not secrets.compare_digest(request.headers.get('x-vispr-token', ''), token):
                return JSONResponse({'detail': 'Reload this local application before making changes.'}, status_code=403)
        # Content-Length plus a streaming byte counter bounds multipart bodies.
        cap = MAX_BYTES * limit + 1024 * 1024
        try:
            if int(request.headers.get('content-length', '0')) > cap:
                return JSONResponse({'detail': 'Upload is too large.'}, status_code=413)
        except ValueError:
            return JSONResponse({'detail': 'Invalid content length.'}, status_code=400)
        received = 0
        receive = request._receive
        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get('body', b''))
            if received > cap:
                raise HTTPException(413, 'Upload is too large.')
            return message
        request._receive = bounded_receive
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
        return response

    def active():
        return any(j['status'] in {'queued', 'processing'} for j in jobs.values())

    @app.get('/api/state')
    def state():
        with lock:
            return {'token': token, 'attributes': ATTRIBUTES, 'settings': settings.model_dump(),
                    'jobs': copy.deepcopy(list(jobs.values())), 'batch_limit': limit}

    @app.put('/api/settings')
    def save_settings(value: Settings):
        nonlocal settings
        if set(value.enabled) - REGISTRY.keys() or len(value.enabled) != len(set(value.enabled)):
            raise HTTPException(422, 'Unknown or duplicate attribute IDs.')
        try:
            value.base_url = local_url(value.base_url)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        with lock:
            temp = preference_path.with_suffix('.tmp')
            temp.write_text(json.dumps(value.model_dump(), indent=2), encoding='utf-8')
            temp.replace(preference_path)
            settings = value
        return {'saved': True}

    @app.post('/api/models/check')
    def model_check(value: Settings):
        try:
            if value.mode == 'lmstudio':
                adapter = LMStudio(value.model_dump())
                models = adapter.models()
                selected = next((m for m in models if m['id'] == value.model), None)
                ready = bool(selected and selected['loaded'] and selected['vision'])
                return {'reachable': True, 'ready': ready, 'models': models,
                        'message': 'Loaded vision model available; localization and JSON schema support remain subject to inference validation.' if ready else 'Select a loaded vision model. Text-only or unloaded models cannot process images.'}
            adapter = DirectQwen(value.model_dump())
            adapter.inspect()
            return {'reachable': True, 'ready': False, 'models': [], 'message': 'Local checkpoint found. Runtime compatibility and memory are checked during loading; inference is not yet verified.'}
        except ValueError as exc:
            return {'reachable': False, 'ready': False, 'models': [], 'message': str(exc)}

    def update(job_id, **values):
        with lock:
            jobs[job_id].update(values)

    def cancelled(job_id):
        with lock:
            return jobs[job_id]['cancel_requested']

    def process(job_id, image, snapshot):
        started = time.monotonic()
        try:
            if cancelled(job_id):
                return
            update(job_id, status='processing', stage='Loading / checking model')
            adapter = model_manager.get(snapshot)
            if cancelled(job_id):
                return
            update(job_id, stage='Model inference')
            raw = adapter.infer(image)
            if cancelled(job_id):
                return
            update(job_id, stage='Validating boxes')
            detections, warnings = parse_detections(raw, image.size, snapshot['enabled'], snapshot['margin'])
            update(job_id, stage='Redacting pixels')
            output = session / f'{job_id}-redacted.png'
            redact(image, detections, output)
            update(job_id, stage='Verifying saved output')
            if cancelled(job_id):
                output.unlink(missing_ok=True)
                return
            update(job_id, status='completed', stage='Review required', detections=detections,
                   warnings=warnings, incomplete=any(d['status'] != 'valid' for d in detections),
                   output=f'/api/images/{job_id}/redacted', original=f'/api/images/{job_id}/original')
        except Exception as exc:
            # Do not log exceptions/model responses: they could contain sensitive content.
            message = str(exc) if isinstance(exc, ValueError) else 'Processing failed. Check the local runtime, available memory and model configuration.'
            update(job_id, status='failed', stage='Failed', error=message)
        finally:
            if cancelled(job_id):
                (session / f'{job_id}-redacted.png').unlink(missing_ok=True)
                update(job_id, status='cancelled', stage='Cancelled', output=None)
            update(job_id, seconds=round(time.monotonic()-started, 2))

    @app.post('/api/jobs')
    def submit(files: list[UploadFile] = File(...)):
        with lock:
            if active():
                raise HTTPException(409, 'A batch is already running.')
            if not 1 <= len(files) <= limit:
                raise HTTPException(422, f'Select between 1 and {limit} images.')
            if len(jobs) + len(files) > 50:
                raise HTTPException(409, 'Session limit reached. Download your results and clear the session.')
            snapshot = settings.model_dump()
            if not snapshot['model'].strip():
                raise HTTPException(422, 'Select a loaded LM Studio vision model or configure a local checkpoint first.')
            # Validate the entire upload before enqueueing any images.
            images = []
            try:
                for f in files:
                    images.append(read_upload(f.file.read(MAX_BYTES+1)))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            finally:
                for f in files:
                    f.file.close()
            ids = []
            for f, image in zip(files, images):
                job_id = uuid.uuid4().hex
                image.save(session / f'{job_id}-original.png')
                jobs[job_id] = {'id': job_id, 'filename': Path((f.filename or 'image').replace('\\', '/')).name[:150],
                    'status': 'queued', 'stage': 'Queued', 'cancel_requested': False,
                    'width': image.width, 'height': image.height, 'settings': snapshot,
                    'detections': [], 'warnings': [], 'error': None, 'output': None, 'seconds': None}
                executor.submit(process, job_id, image, snapshot)
                ids.append(job_id)
            return {'ids': ids}

    @app.post('/api/jobs/cancel')
    def cancel():
        with lock:
            for job in jobs.values():
                if job['status'] in {'queued', 'processing'}:
                    job['cancel_requested'] = True
                    job['stage'] = 'Cancellation requested; waiting for safe boundary'
        return {'requested': True}

    @app.delete('/api/session')
    def clear():
        with lock:
            if active():
                raise HTTPException(409, 'Wait for processing or cancellation to finish before clearing.')
            for path in session.iterdir():
                if path.is_file():
                    path.unlink()
            jobs.clear()
        return {'cleared': True}

    @app.get('/api/images/{job_id}/{kind}')
    def image_file(job_id: str, kind: Literal['original', 'redacted'], download: bool = False):
        with lock:
            if job_id not in jobs or jobs[job_id]['status'] != 'completed':
                raise HTTPException(404, 'Completed image not found.')
            return FileResponse(session / f'{job_id}-{kind}.png', media_type='image/png',
                filename=f'vispr-{job_id[:8]}-{kind}.png' if download else None)

    @app.get('/api/download.zip')
    def batch_download():
        buffer = io.BytesIO()
        with lock:
            complete = [j for j in jobs.values() if j['status'] == 'completed']
            if not complete:
                raise HTTPException(404, 'No completed outputs available.')
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                for job in complete:
                    archive.write(session / f"{job['id']}-redacted.png", f"vispr-{job['id'][:8]}-redacted.png")
                archive.writestr('REVIEW_REQUIRED.txt', 'These are model predictions, not a privacy guarantee. Inspect every output.\n' +
                    '\n'.join(f"{j['id'][:8]}: {len(j['warnings'])} warnings; incomplete={j.get('incomplete', False)}" for j in complete))
        return Response(buffer.getvalue(), media_type='application/zip', headers={'Content-Disposition': 'attachment; filename="vispr-redacted.zip"'})

    app.mount('/', StaticFiles(directory=Path(__file__).parent / 'static', html=True), name='frontend')
    return app
