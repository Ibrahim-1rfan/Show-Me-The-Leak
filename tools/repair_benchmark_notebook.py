"""Preserve the user's executed notebook, update setup cells, embed fixed helpers."""
from pathlib import Path
import base64
import io
import json
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'Model_Testing.ipynb'
backup = ROOT / 'outputs' / 'notebook_before_runtime_repair.ipynb'
backup.parent.mkdir(parents=True, exist_ok=True)
if not backup.exists():
    backup.write_bytes(path.read_bytes())
nb = json.loads(path.read_text(encoding='utf-8'))
manual_flash_cells = [c for c in nb['cells'] if c['cell_type'] == 'code'
                      and ''.join(c['source']).strip() in {'!pip install flash_attn', '!pip install flash-attn'}]
if manual_flash_cells:
    manual_backup = ROOT / 'outputs' / 'notebook_before_loader_fix.ipynb'
    if not manual_backup.exists():
        manual_backup.write_bytes(path.read_bytes())
    for cell in manual_flash_cells:
        cell['source'] = ['# Manual kernel FlashAttention install disabled.\n',
                          '# Florence needs no flash-attn; dots uses its isolated worker environment.\n']
        cell['outputs'] = []
        cell['execution_count'] = None
by_tag = {c.get('metadata', {}).get('tags', [''])[0]: c for c in nb['cells'] if c.get('metadata', {}).get('tags')}


def source(cell):
    return ''.join(cell['source'])


def update(cell, text, kind='code'):
    cell['source'] = textwrap.dedent(text).strip().splitlines(True)
    cell['cell_type'] = kind
    if kind == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    else:
        cell.pop('execution_count', None)
        cell.pop('outputs', None)


# Worker uses identical detection/redaction/evaluation helpers, without executing
# notebook download, installation, plotting, or model-loading cells.
pipeline = '"""Notebook pipeline helpers used by isolated model workers."""\nfrom pathlib import Path\nFORMAT_RETRIES = 1\n\n'
pipeline += '\n\n'.join(source(by_tag[t]) for t in
    ['data_helpers', 'detection_helpers', 'inference_helpers', 'pipeline_helpers']) + '\n'
(ROOT / 'benchmark_pipeline.py').write_text(pipeline, encoding='utf-8')

helper_names = sorted(p.name for p in ROOT.glob('benchmark_*.py')) + ['taxonomy_crosswalk.csv']
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for name in helper_names:
        bundle.writestr(name, (ROOT / name).read_bytes())
payload = base64.b64encode(buffer.getvalue()).decode('ascii')

bootstrap = next(c for c in nb['cells'] if 'https://github.com/Ibrahim-1rfan/Show-Me-The-Leak.git' in source(c))
bootstrap['metadata']['tags'] = ['bootstrap']
update(bootstrap, '''
# Your repository and dataset links are retained. Existing data is reused.
from pathlib import Path
import base64, io, subprocess, zipfile

REPOSITORY_URL = 'https://github.com/Ibrahim-1rfan/Show-Me-The-Leak.git'
DATASET_URL = 'https://huggingface.co/datasets/Ibrahim1rfan/VISPR/resolve/main/train2017.zip'
WORK_ROOT = Path('/content') if Path('/content').is_dir() else Path.cwd()
if WORK_ROOT == Path('/content'):
    repo_dir = WORK_ROOT / 'Show-Me-The-Leak'
    if not repo_dir.exists():
        subprocess.check_call(['git', 'clone', '--depth', '1', REPOSITORY_URL, str(repo_dir)])
    BENCHMARK_CODE_ROOT = repo_dir
else:
    BENCHMARK_CODE_ROOT = WORK_ROOT

# The fixed helper snapshot travels with this notebook. A stale GitHub checkout
# therefore cannot silently restore the broken environment setup.
EMBEDDED_HELPERS = '__PAYLOAD__'
with zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_HELPERS))) as bundle:
    for name in bundle.namelist():
        if Path(name).name != name:
            raise ValueError('Unexpected helper path in notebook bundle')
        (BENCHMARK_CODE_ROOT / name).write_bytes(bundle.read(name))

if not (WORK_ROOT / 'train2017').is_dir():
    archive = WORK_ROOT / 'train2017.zip'
    subprocess.check_call(['wget', '-c', '-O', str(archive), DATASET_URL])
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            (WORK_ROOT / entry.filename).resolve().relative_to(WORK_ROOT.resolve())
        bundle.extractall(WORK_ROOT)
    if not (WORK_ROOT / 'train2017').is_dir():
        raise FileNotFoundError('Downloaded archive did not contain train2017/')
print('Data:', WORK_ROOT / 'train2017', '| Fixed code:', BENCHMARK_CODE_ROOT)
'''.replace('__PAYLOAD__', payload))

config = source(by_tag['config'])
config = config.replace('CODE_ROOT = DATA_ROOT', "CODE_ROOT = Path(globals().get('BENCHMARK_CODE_ROOT', DATA_ROOT))")
config = config.replace("TAXONOMY_PATH = DATA_ROOT / 'taxonomy_crosswalk.csv'", "TAXONOMY_PATH = CODE_ROOT / 'taxonomy_crosswalk.csv'")
config = config.replace("MANIFEST_PATH = DATA_ROOT / 'manifest.csv'", "MANIFEST_PATH = CODE_ROOT / 'manifest.csv'")
config = config.replace('INSTALL_PACKAGES = False', 'INSTALL_PACKAGES = True')
config += '''

# New: model libraries live in subprocess environments, never the notebook kernel.
RUN_GPU_INFERENCE = True  # False: review the existing results without a GPU or installs.
ENV_ROOT = Path('/content/benchmark_envs') if Path('/content').is_dir() else DATA_ROOT / '.benchmark_envs'
ALLOW_COMPATIBLE_SAVED_RUNS = True  # Equal settings required; differing code hashes are disclosed.
RESULTS_ROOTS = [
    DATA_ROOT / 'Leak_Benchmark_Results' / 'outputs' / 'multimodel_pii',
    DATA_ROOT / 'outputs' / 'multimodel_pii',
    Path('/content/drive/MyDrive/Leak_Benchmark_Results/outputs/multimodel_pii'),
]
'''
# Keep reruns of this authoring helper idempotent.
config = config.split('# New: model libraries')[0].rstrip() + config[config.rfind('# New: model libraries') - 1:]
update(by_tag['config'], config)

update(by_tag['install'], '''
import sys, subprocess, importlib
sys.path.insert(0, str(CODE_ROOT.resolve()))
importlib.invalidate_caches()
for module_name in ('benchmark_adapters', 'benchmark_ocr_adapters', 'benchmark_workflow',
                    'benchmark_metrics', 'benchmark_environment'):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
from benchmark_environment import prepare_environment

MODEL_PYTHON = None
if RUN_GPU_INFERENCE:
    MODEL_PYTHON = prepare_environment(INSTALL_PROFILE, ENV_ROOT, install=INSTALL_PACKAGES)
    print('All model execution will use:', MODEL_PYTHON)
else:
    print('Review-only mode: skipping GPU environment setup.')
# Do not run manual pip --no-deps cells. The worker checks Python, Torch,
# Torchvision, Transformers, tokenizers, GPU access, and profile-specific imports.
''')

for cell in nb['cells']:
    if cell is by_tag['install']:
        continue
    if cell['cell_type'] == 'code' and '!pip install transformers==4.41.2 --no-deps' in source(cell):
        update(cell, 'The previous manual `--no-deps` installation has been replaced by the isolated worker setup above. The executed original notebook and its errors are preserved in `outputs/notebook_before_runtime_repair.ipynb`.', 'markdown')

load_cell = source(by_tag['load_model'])
load_cell = load_cell.split('\ndef load_selected_adapter(')[0].rstrip() + '\n'
load_cell = load_cell.replace("['benchmark_adapters.py', 'benchmark_ocr_adapters.py', 'benchmark_workflow.py']",
    "['benchmark_adapters.py', 'benchmark_ocr_adapters.py', 'benchmark_workflow.py', 'benchmark_pipeline.py', 'benchmark_environment.py', 'benchmark_worker.py']")
update(by_tag['load_model'], load_cell)

update(by_tag['run_pipeline'], '''
current_runs = {}
if RUN_GPU_INFERENCE:
    if MODEL_PYTHON is None:
        raise RuntimeError('Run the isolated setup cell first.')
    job_dir = sample_dir / 'worker_jobs'
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / (INSTALL_PROFILE + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    write_json(job_path, {'profile': INSTALL_PROFILE, 'models': MODELS_TO_RUN,
        'data_root': str(DATA_ROOT.resolve()), 'sample_dir': str(sample_dir.resolve()), 'protocol': protocol})
    # Separate interpreter, shared GPU and files. Live progress and errors appear here.
    completed = subprocess.run([str(MODEL_PYTHON), '-u', str(CODE_ROOT / 'benchmark_worker.py'),
                                '--job', str(job_path.resolve())], cwd=CODE_ROOT)
    status_path = job_path.with_suffix('.status.json')
    if status_path.exists():
        current_runs = json.loads(status_path.read_text(encoding='utf-8'))
        display(current_runs)
    if completed.returncode:
        raise RuntimeError('Model worker failed. Read the error above; saved results and earlier runs remain available.')
else:
    print('Skipped model runs; proceeding to saved-result comparison.')
''')

comparison = source(by_tag['compare'])
comparison = comparison.replace('reports, provenance = collect_runs(sample_dir, COMPARE_MODELS, protocol_id)', '''
extra_sample_dirs = [root / sample_dir.name for root in RESULTS_ROOTS if (root / sample_dir.name).is_dir()]
reports, provenance = collect_runs(sample_dir, COMPARE_MODELS, protocol_id,
    compatible_settings=protocol_settings if ALLOW_COMPATIBLE_SAVED_RUNS else None,
    additional_sample_dirs=extra_sample_dirs)
if any(p.get('protocol_match') == 'same_settings_different_code' for p in provenance):
    print('Including older code versions with identical evaluation settings; exact source runs are disclosed below.')'''.strip())
update(by_tag['compare'], comparison)

inspection = source(by_tag['inspect'])
inspection = inspection.replace("model_reports = [r for r in reports if r['model_key'] == INSPECT_MODEL]", """model_reports = [dict(r) for r in reports if r['model_key'] == INSPECT_MODEL]
current_images = {r['id']: str(r['image_path']) for r in selected}
for report in model_reports:
    report['image_path'] = current_images.get(report['image_id'], report.get('image_path'))""")
update(by_tag['inspect'], inspection)

# Correct obsolete instructions without replacing the user's custom setup/backup cells.
for cell in nb['cells']:
    if cell['cell_type'] != 'markdown':
        continue
    text = source(cell)
    text = text.replace('The earlier Qwen notebook was not executed, so there is **no measured previous baseline**.\nQwen3 is rerun here as the baseline.',
        'Your saved Qwen3, Qwen2.5, Llama, and Chandra runs are available for review.\nDots and Florence need retries after their recorded setup failures.')
    text = text.replace('Restart into a fresh runtime before changing profiles.',
        'Each profile runs in its own Python 3.11 environment; changing profiles does not replace notebook-kernel packages.')
    text = text.replace('run each profile in a fresh runtime,', 'run each profile in its isolated worker environment,')
    text = text.replace('Run the profiles in turn, retaining', 'Run the profiles in turn, retaining')
    text = text.replace('No GPU inference has been executed or claimed during notebook authoring.',
        'The saved user runs are retained. The environment repairs have offline validation; the repaired dots/Florence workers still need a GPU rerun.')
    text = text.replace('with `INSTALL_PACKAGES=False`', 'with `RUN_GPU_INFERENCE=False`')
    cell['source'] = text.splitlines(True)

note = '''## Recovery for your completed run

**October 10 loader update:** Florence runs with eager attention and does not need
`flash-attn`. Its loader now handles the legacy scanner's optional-import error.
Dots now constructs its processor with the required video component while preserving
the native image settings and tokens. Both fixes are embedded in this notebook.
In the same Colab runtime, set `INSTALL_PACKAGES=False` to reuse an already installed
worker environment; use `True` for a fresh or incomplete environment. Run bootstrap,
configuration, setup, helper and execution cells again. Dots still uses FlashAttention,
so reuse its installed environment rather than compiling it again for this fix.

Your saved run uses Python 3.13 in Colab. Dots stopped on a Transformers version mismatch;
Florence's original Transformers 4.41.2 requires tokenizers 0.19.x. Installing it with
`--no-deps` left an incompatible environment. The setup now creates a separate **Python 3.11**
worker for each profile, pins compatible Torch/Torchvision, and verifies imports before inference.
It does not downgrade the Colab kernel or substitute a different model checkpoint.

Keep `INSTALL_PROFILE='florence'`, set `INSTALL_PACKAGES=True`, and run setup and execution.
Then change to `'dots'` and repeat. Your L4 GPU meets the dots hardware requirement.
The notebook carries the fixed helper snapshot and retains your GitHub/dataset links.
For viewing existing results only, set `RUN_GPU_INFERENCE=False`. Your Drive backup is searched
automatically. Existing successful runs are not overwritten. Llama's 45 formatting failures,
Qwen2.5's five rejected outputs, and Chandra's four incomplete outputs remain visible as failures.

Compatibility sources: [Transformers 4.41.2 requirements](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/dependency_versions_table.py),
[uv managed Python environments](https://docs.astral.sh/uv/guides/install-python/).
'''
if not any(c.get('metadata', {}).get('tags') == ['recovery_note'] for c in nb['cells']):
    nb['cells'].insert(1, {'cell_type': 'markdown', 'metadata': {'tags': ['recovery_note']}, 'source': note.splitlines(True)})
for i, cell in enumerate(nb['cells']):
    cell.setdefault('id', f'recovered-{i:02d}')
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
with zipfile.ZipFile(ROOT / 'outputs' / 'multimodel_benchmark_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    files = [path, ROOT / 'BENCHMARK_README.md', *[ROOT / name for name in helper_names],
             *sorted((ROOT / 'tests').glob('test_*.py'))]
    for file in files:
        bundle.write(file, file.relative_to(ROOT).as_posix())
print('Updated notebook; original preserved at', backup)
