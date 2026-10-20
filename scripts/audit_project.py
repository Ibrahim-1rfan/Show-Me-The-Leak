"""Read-only project audit; output hashes/metadata, never credential values."""
from pathlib import Path
import hashlib, json, statistics, re
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'audit'
OUT.mkdir(parents=True, exist_ok=True)
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
inventory = []
for p in sorted(ROOT.iterdir()):
    if p.name == '.git': continue
    inventory.append({'path': p.name, 'kind': 'directory' if p.is_dir() else 'file',
        'bytes': p.stat().st_size if p.is_file() else None,
        'sha256': digest(p) if p.is_file() and p.stat().st_size < 50_000_000 else None,
        'action': 'retain; no deletion or relocation in initial migration'})
(OUT/'inventory.json').write_text(json.dumps(inventory, indent=2))
timings = []
for p in sorted((ROOT/'Leak_Benchmark_Results').rglob('results.json')):
    rows = json.loads(p.read_text(encoding='utf-8'))
    t = [r['inference_seconds'] for r in rows if isinstance(r.get('inference_seconds'), (int,float))]
    config = json.loads((p.parent/'run_config.json').read_text(encoding='utf-8'))
    timings.append({'model': p.parent.parent.name, 'path': str(p.relative_to(ROOT)),
        'sha256': digest(p), 'records': len(rows), 'successful': sum(r.get('status')=='ok' for r in rows),
        'timed_calls': len(t), 'median_detection_seconds': statistics.median(t) if t else None,
        'maximum_detection_seconds': max(t) if t else None, 'load_seconds': config.get('load_seconds'),
        'audit_redacted': config.get('audit_redacted'), 'max_new_tokens': config.get('max_new_tokens')})
(OUT/'historical_timings.json').write_text(json.dumps(timings, indent=2))
# Report only filenames/locations of potential literal credentials, never matches.
secret_locations = []
for p in ROOT.glob('*.py'):
    for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if re.search(r'hf_[A-Za-z0-9]{15,}|AIza[A-Za-z0-9_-]{20,}',line):
            secret_locations.append({'file':p.name, 'line':i, 'action':'preserve locally; exclude from exports and Git'})
(OUT/'credential_locations.json').write_text(json.dumps(secret_locations, indent=2))
print('Audit saved:', len(inventory), 'top-level entries;',len(timings),'historical runs;',len(secret_locations),'credential locations (values omitted)')
