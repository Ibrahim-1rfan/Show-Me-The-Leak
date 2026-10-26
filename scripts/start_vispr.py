"""Launch the local application without modifying research environments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import uvicorn
from vispr_app.server import create_app

if __name__ == '__main__':
    print('VISPR: http://127.0.0.1:8765 — local only. Ctrl+C to stop and clean this session.')
    uvicorn.run(create_app(), host='127.0.0.1', port=8765, access_log=False)
