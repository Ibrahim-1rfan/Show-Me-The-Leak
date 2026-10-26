"""Register local image folders for an existing Label Studio project; never sync tasks.

Run with .venv-annotation/Scripts/python.exe. Defaults to read-only inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    package = ROOT / 'manual_annotation_dataset'
    data = ROOT / 'data/label_studio'
    database = data / 'label_studio.sqlite3'
    if not database.exists():
        raise SystemExit('Existing local Label Studio database required.')
    os.environ.update(LABEL_STUDIO_BASE_DATA_DIR=str(data), LABEL_STUDIO_DJANGO_DB='sqlite',
                      LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED='true',
                      LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=str(package), COLLECT_ANALYTICS='false',
                      LABEL_STUDIO_LATEST_VERSION_CHECK='false', LABEL_STUDIO_SENTRY_DSN='',
                      LABEL_STUDIO_FRONTEND_SENTRY_DSN='')
    import label_studio
    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    os.environ['DJANGO_SETTINGS_MODULE'] = 'label_studio.core.settings.label_studio'
    import django
    django.setup()
    from django.conf import settings
    from django.db import transaction
    from io_storages.localfiles.models import LocalFilesImportStorage
    from io_storages.localfiles.views import localfiles_data
    from projects.models import Project
    from tasks.models import Task, Annotation
    from rest_framework.test import APIRequestFactory, force_authenticate

    assert Path(settings.DATABASES['default']['NAME']).resolve() == database.resolve()
    project = Project.objects.get(pk=args.project)
    if project.created_by is None:
        raise SystemExit('Project owner missing; inspect the project first.')
    tasks = list(Task.objects.filter(project=project).order_by('id').values('id', 'data'))
    annotations_before = list(Annotation.objects.filter(task__project=project).order_by('id').values())
    folders = [package / 'images', package / 'annotation_views']
    urls = []
    for task in tasks:
        url = task['data'].get('image', '')
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or parsed.path != '/data/local-files/':
            raise SystemExit('Project has a non-package image URL; inspect before applying.')
        relative = parse_qs(parsed.query).get('d', [''])[0]
        path = (package / relative).resolve()
        if path.parent not in [folder.resolve() for folder in folders] or not path.is_file():
            raise SystemExit('An image is missing or outside the two supported image folders.')
        urls.append((url, path))
    if not tasks:
        raise SystemExit('Import the assignment JSON first, then configure storage.')
    existing = list(LocalFilesImportStorage.objects.filter(project=project).values('id', 'path'))
    print(json.dumps({'project_id': project.id, 'tasks': len(tasks), 'existing_storages': existing}))
    factory = APIRequestFactory()
    responses = []

    def check_images():
        statuses = {}
        matched = 0
        for url, path in urls:
            request = factory.get(url)
            force_authenticate(request, user=project.created_by)
            response = localfiles_data(request)
            responses.append(response)
            statuses[str(response.status_code)] = statuses.get(str(response.status_code), 0) + 1
            if response.status_code == 200:
                payload = b''.join(response.streaming_content)
                assert hashlib.sha256(payload).digest() == hashlib.sha256(path.read_bytes()).digest()
                matched += 1
        return {'statuses': statuses, 'byte_matches': matched}

    before = check_images()
    print(json.dumps({'before': before}))
    if not args.apply:
        for response in responses:
            response.close()
        return
    backup_dir = data / 'backups'
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_path = backup_dir / f'before_storage_{stamp}.sqlite3'
    with sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True) as source:
        with sqlite3.connect(backup_path) as target:
            source.backup(target)
    with transaction.atomic():
        created = 0
        for folder in folders:
            folder.resolve().relative_to(package.resolve())
            if LocalFilesImportStorage.objects.filter(project=project, path=str(folder)).exists():
                continue
            storage = LocalFilesImportStorage(project=project, path=str(folder),
                title=f'Annotation {folder.name} (serve only)', regex_filter=r'(?!)',
                use_blob_urls=True, recursive_scan=False)
            storage.validate_connection()
            storage.save()
            created += 1
        after = check_images()
        assert after['byte_matches'] == len(tasks), after
        assert tasks == list(Task.objects.filter(project=project).order_by('id').values('id', 'data'))
        assert annotations_before == list(Annotation.objects.filter(task__project=project).order_by('id').values())
    # Django response.close emits request_finished and closes DB connections.
    # Defer it until the storage transaction has committed.
    for response in responses:
        response.close()
    report = {'project_id': project.id, 'tasks': len(tasks), 'created_storages': created,
              'before': before, 'after': after, 'tasks_and_annotations_unchanged': True,
              'verification': 'Installed authenticated view; exact image bytes',
              'backup': str(backup_path.relative_to(ROOT)), 'synced_tasks': 0}
    (data / 'storage_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
