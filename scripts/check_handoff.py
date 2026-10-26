"""Check source handoff without GPU, network, dataset or private results.

--clean-copy builds and tests an allowlisted source copy under ignored tmp/.
No Git commits, uploads, model execution or deletion are performed.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb'hf_[A-Za-z0-9]{15,}|AIza[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def source_files(root):
    files = [*root.glob('*.md'), root / '.gitignore', root / 'requirements-dev.txt', root / 'requirements-annotation.txt',
             root / 'Model_Testing.ipynb', root / 'Start Annotation.cmd', *root.glob('benchmark_*.py')]
    files += [root / name for name in ['apply_crosswalk.py', 'check_category_distribution.py',
        'check_label_distribution.py', 'reconcile_vispr.py', 'taxonomy_crosswalk.csv',
        'manifest.csv', 'manifest_with_categories.csv', 'duplicate_groups.csv', 'issues_report.csv']]
    for folder in ['src', 'scripts', 'tests', 'tools']:
        files += [p for p in (root / folder).rglob('*.py') if '__pycache__' not in p.parts]
    files += [root / 'attributes.tsv', root / 'Images Classification.txt', *list((root / 'docs/attribute_selection').glob('*')), *list((root / 'docs/manual_annotation').glob('*'))]
    files += list((root / 'configs').glob('*')) + list((root / 'notebooks').glob('*.ipynb'))
    files += list((root / 'docs').glob('*.md')) + list((root / 'docs/audit').glob('*.json'))
    return sorted(set(p for p in files if p.is_file()))


def check(root):
    errors = []
    files = source_files(root)
    for p in files:
        if SECRET.search(p.read_bytes()):
            errors.append('Potential credential in ' + p.relative_to(root).as_posix())
    for name in ['README.md', 'GUIDE.md', 'TRACKER.md', 'BENCHMARK_REPORT.md']:
        p = root / name
        if not p.exists():
            errors.append('Missing ' + name)
            continue
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            path = (p.parent / target.split('#')[0]).resolve()
            if not path.exists():
                errors.append(name + ': broken local link ' + target)
    for p in (root / 'notebooks').glob('*.ipynb'):
        nb = json.loads(p.read_text(encoding='utf-8'))
        for c in nb['cells']:
            if c['cell_type'] != 'code':
                continue
            text = ''.join(c['source'])
            ast.parse(text)
            if c.get('outputs') or c.get('execution_count') is not None:
                errors.append(p.name + ': executed output in source notebook')
            if 'bootstrap' not in c.get('metadata', {}).get('tags', []):
                continue
            payload = next(ast.literal_eval(n.value) for n in ast.parse(text).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
                for name in z.namelist():
                    if z.read(name) != (root / name).read_bytes():
                        errors.append(p.name + ': stale snapshot ' + name)
                    if SECRET.search(z.read(name)):
                        errors.append(p.name + ': credential pattern in snapshot ' + name)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({'source_files_checked': len(files), 'documentation_links': 'pass',
        'clean_notebooks_and_exact_snapshots': 'pass', 'credential_pattern_scan': 'pass',
        'limitation': 'Pattern scan is not a guarantee; Git history, GPU, external downloads and live Drive are not verified.'}))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-copy', action='store_true')
    args = parser.parse_args()
    files = check(ROOT)
    if not args.clean_copy:
        return
    (ROOT / 'tmp').mkdir(exist_ok=True)
    clean = Path(tempfile.mkdtemp(prefix='handoff_source_', dir=ROOT / 'tmp'))
    for p in files:
        dest = clean / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    assert not (clean / 'data').exists() and not (clean / 'results').exists()
    subprocess.run([sys.executable, 'scripts/build_notebooks.py'], cwd=clean, check=True)
    assert not (clean / 'dist/vispr_eval_50.zip').exists(), 'Do not emit an empty dataset ZIP'
    check(clean)
    with (clean / 'test.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-q'],
                                cwd=clean, stdout=log, stderr=subprocess.STDOUT)
    print('Clean source copy:', clean)
    print('Offline tests:', 'PASS' if result.returncode == 0 else 'FAIL', '; details in test.log')
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
