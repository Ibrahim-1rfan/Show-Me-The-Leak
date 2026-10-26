"""Explicit TEST-ONLY UI fixture: never imported by the production launcher.

Run python tests/vispr_ui_fixture.py, then open http://127.0.0.1:8766.
Uses a temporary directory and generated fake PII only. No model is loaded.
"""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastapi.responses import HTMLResponse
import uvicorn
from vispr_app.pipeline import REGISTRY
from vispr_app.server import Settings, create_app


class SyntheticAdapter:
    def get(self, config):
        return self

    def infer(self, image):
        return json.dumps({'coordinate_system': 'xyxy_1000', 'detections': [
            {'attribute_id':'a19_name_full','bbox':[50,250,240,340],'confidence':.95},
            {'attribute_id':'a90_email','bbox':[250,250,510,340],'confidence':.8},
            {'attribute_id':'a49_phone','bbox':None,'confidence':None}]})


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='vispr-test-only-') as storage:
        preferences = Settings(model='TEST ONLY — synthetic predictions', enabled=[a for a in REGISTRY if a != 'a90_email'])
        (Path(storage)/'preferences.json').write_text(json.dumps(preferences.model_dump()),encoding='utf-8')
        app=create_app(storage, SyntheticAdapter())
        static_mount=app.router.routes.pop()

        @app.get('/')
        def test_home():
            html=(Path(__file__).resolve().parents[1]/'src/vispr_app/static/index.html').read_text(encoding='utf-8')
            return HTMLResponse(html.replace('SEE WHAT YOU SHARE','TEST ONLY · SYNTHETIC PREDICTIONS').replace('A little less exposed.','Synthetic UI fixture.'))

        app.router.routes.append(static_mount)
        uvicorn.run(app,host='127.0.0.1',port=8766,access_log=False)
