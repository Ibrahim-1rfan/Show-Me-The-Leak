"""Build independent Colab entry points from an allowlisted source snapshot."""
from pathlib import Path
import base64,hashlib,io,json,textwrap,zipfile
ROOT=Path(__file__).resolve().parents[1]
def md(text):return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip().splitlines(True)}
def code(text,tag):return {'cell_type':'code','execution_count':None,'outputs':[],
    'metadata':{'tags':[tag]},'source':textwrap.dedent(text).strip().splitlines(True)}
files=[*sorted((ROOT/'src/leakbench').glob('*.py')),*sorted((ROOT/'configs').glob('*'))]
files.append(ROOT/'TRACKER.md')
memory=io.BytesIO()
with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as z:
    for path in files:z.writestr(path.relative_to(ROOT).as_posix(),path.read_bytes())
payload=base64.b64encode(memory.getvalue()).decode();checksum=hashlib.sha256(memory.getvalue()).hexdigest()
bootstrap=f'''
from pathlib import Path
import base64,hashlib,io,os,sys,zipfile
PROJECT = Path('/content/Show-Me-The-Leak-v2') if Path('/content').exists() else Path.cwd()
PROJECT.mkdir(parents=True,exist_ok=True)
PAYLOAD = {payload!r}
data=base64.b64decode(PAYLOAD)
assert hashlib.sha256(data).hexdigest()=={checksum!r}, 'Embedded source checksum mismatch'
with zipfile.ZipFile(io.BytesIO(data)) as bundle:
    for member in bundle.namelist():
        target=(PROJECT/member).resolve();target.relative_to(PROJECT.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        # Preserve runtime tracker additions when rerunning the bootstrap.
        if member=='TRACKER.md' and target.exists():continue
        if not target.exists() or target.read_bytes()!=bundle.read(member):target.write_bytes(bundle.read(member))
sys.path.insert(0,str(PROJECT/'src'))
for name in list(sys.modules):
    if name=='leakbench' or name.startswith('leakbench.'):del sys.modules[name]
os.environ['PYTHONPATH']=str(PROJECT/'src')
os.environ.setdefault('HF_HOME',str(PROJECT/'.model_cache/huggingface'))
os.environ['MAX_JOBS']='2'
os.environ['NVCC_THREADS']='2'
print('Verified source snapshot ready:', PROJECT)
'''
setup='''
from leakbench.execution import run_command
run_command([sys.executable,'-m','pip','install','Pillow==11.3.0','pandas==2.2.3',
             'matplotlib==3.10.6','tqdm==4.67.1','ipywidgets>=8.1,<9'], 'Notebook dependencies', timeout=900)
import subprocess
gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20)
print(gpu.stdout)
if gpu.returncode:raise RuntimeError('Choose a GPU runtime in Colab before continuing.')
'''
cloud='''
from google.colab import drive, userdata
drive.mount('/content/drive')
from leakbench.cloud import DRIVE_ROOT, verified_copy_tree, backup
print('Existing backup root:', DRIVE_ROOT)
try:
    token=userdata.get('HF_TOKEN')
    if token:os.environ['HF_TOKEN']=token
    print('Hugging Face secret loaded.' if token else 'HF_TOKEN secret is empty; gated models cannot load.')
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    print('No HF_TOKEN secret available. Public models may run; gated Llama needs authorized access.')
'''
dataset='''
from leakbench.dataset import prepare_subset,validate_subset,records
from leakbench.common import read_json
dataset_dir=PROJECT/'data/vispr_eval_50'
cloud_dataset=DRIVE_ROOT/'benchmark_v2/data/vispr_eval_50'
uploaded_subset=Path('/content/vispr_eval_50.zip')
if not (dataset_dir/'manifest.csv').exists() and (cloud_dataset/'manifest.csv').exists():
    verified_copy_tree(cloud_dataset,dataset_dir)
if not (dataset_dir/'manifest.csv').exists() and uploaded_subset.exists():
    with zipfile.ZipFile(uploaded_subset) as archive:
        for entry in archive.infolist():
            target=(PROJECT/'data'/entry.filename).resolve();target.relative_to((PROJECT/'data').resolve())
        archive.extractall(PROJECT/'data')
dataset_dir=prepare_subset(PROJECT,PROJECT/'configs/selection.json')
display(read_json(dataset_dir/'validation.json'))
verified_copy_tree(dataset_dir,cloud_dataset)
'''
preview='''
from PIL import Image
import matplotlib.pyplot as plt
cohort=records(dataset_dir)
fig,axes=plt.subplots(2,5,figsize=(15,7))
for ax,row in zip(axes.flat,[r for r in cohort if r['category']=='documents'][:5]+[r for r in cohort if r['category']=='images'][:5]):
    with Image.open(row['image_path']) as im:ax.imshow(im.copy())
    ax.set_title(row['filename']);ax.axis('off')
plt.tight_layout();plt.show()
'''
summaries='''
import pandas as pd
from IPython.display import display,Markdown,HTML
summary=read_json(report_dir/'summary.json')
if summary.get('execution'):
    print('Execution mode/status:',summary['execution']['mode'],summary['execution']['status'])
    display(pd.DataFrame(summary['execution']['reconciliation']))
display(Markdown('## Final report\\n\\n'+summary['conclusion']+'\\n\\n'+summary['limitations']))
display(pd.DataFrame(summary['coverage']).rename(columns={'successful_images':'usable_predictions',
    'success_rate_of_attempts':'usable_prediction_rate'}))
quality=pd.read_csv(report_dir/'full_taxonomy_summaries.csv')
display(quality[(quality.stratum=='all') & (quality.scope=='common_successful')])
display(pd.read_csv(report_dir/'runtime.csv'))
for name in ['phase_timings','image_timings']:
    if (report_dir/f'{name}.csv').exists():
        display(Markdown('### '+name.replace('_',' ')))
        display(pd.read_csv(report_dir/f'{name}.csv').head(10))
        print('Complete timing table:',report_dir/f'{name}.csv')
display(pd.DataFrame(summary.get('redaction_checks',[])))
for scope in ['full_taxonomy','shared_text_scope']:
    attribute_table=pd.read_csv(report_dir/f'{scope}_per_attribute.csv')
    display(Markdown('### Most missed attributes '+scope))
    display(attribute_table[(attribute_table.scope=='all_successful') & (attribute_table.stratum=='all')]
            .sort_values('missed_count',ascending=False).head(15))
print('HTML report with image links:',report_dir/'report.html')
print('Source run:',run_dir)
'''
inspect='''
from leakbench.viewer import show_results_viewer
viewer=show_results_viewer(PROJECT,category=CATEGORY,initial_run=globals().get('run_dir'))
'''
for category,number in [('vlm','01'),('ocr','02')]:
    models=['qwen3_vl','qwen25_vl','llama32_vision'] if category=='vlm' else ['florence2','chandra','dots_ocr']
    cells=[md(f'''# {category.upper()} privacy benchmark
    Independent Colab notebook. Select a GPU runtime and run cells in order. No other
    notebook or Git clone is needed. This notebook contains its own verified source
    snapshot and fixed 25-document/25-general cohort. Results go to your existing Drive
    destination. Training is not performed. **Repaired GPU validation is pending.**
    Prior executed notebooks and results are preserved in the project audit folder.
    Start with one known-positive document; then four pilot images (two per category).

    Models: {', '.join(models)}. Each profile uses an isolated worker environment.
    Florence does not require FlashAttention. Dots retains the FlashAttention profile
    and needs an Ampere-or-newer GPU, such as L4/A100. Installation can still take time;
    the notebook displays native output and periodic elapsed-time messages.
    '''),md('## 1 Restore the bundled implementation'),code(bootstrap,'bootstrap'),
    md('## 2 Install notebook dependencies and verify GPU'),code(setup,'setup'),
    md('## 3 Connect existing Drive backup and optional Hugging Face secret'),code(cloud,'cloud'),
    md('## 4 Prepare and validate the frozen cohort'),code(dataset,'dataset'),code(preview,'preview'),
    md('''## 5 Configure the experiment
    The default preserves 1280 pixels, 4096 output tokens and one format retry.
    repair_partial_entries=True permits one model-authored invalid-ID correction
    within that retry budget; accepted boxes cannot be erased by an empty retry.
    The old hidden redacted-image audit is excluded as an explicit protocol change.
    One image per batch supports heterogeneous model APIs and isolates failures.
    Timeouts stop stalled workers; they are limits, not promised runtimes. A timeout
    remains an error and blocks continuation when it occurs in the pilot.
    Start with RUN_STAGE='single'. Inspect the saved raw output, mask and redaction.
    Then use RUN_STAGE='pilot' for two document and two general images. Invalid
    single-image results are retained as diagnostic evidence; they do not block
    the four-image pilot. A full run requires four intact predictions, correct mask
    application and explicit per-image review. Missed PII remains a measured failure,
    even when the mechanics of saving an empty mask worked correctly.
    For controlled size diagnostics only: max_input_edge=None delegates resizing
    to the model processor; 1280 retains the baseline cap. OCR can use
    ocr_preprocessing='tiles', tile_edge=1024, tile_overlap=0.15. Florence can test
    florence_precision='float32' separately. Each setting creates a distinct run.
    These are experiments, not demonstrated accuracy fixes. Do not change multiple
    settings at once or mistake the processor's pixel tensor for original geometry.
    '''),code(f'''
    from leakbench.orchestrate import prepare_execution_run,execute_category
    CATEGORY={category!r}
    RUN_STAGE='single'  # single -> pilot -> explicit review -> full
    REUSE_PREDICTIONS=True  # False deliberately repeats even intact predictions in a fresh run
    CONFIG={{'models':{models!r}, 'max_input_edge':1280,'max_new_tokens':4096,
             'format_retries':1,'load_in_4bit':True,'redaction_padding':0.01,
             'repair_partial_entries':True,
             'audit_redacted':False,'batch_size':1,'image_timeout_seconds':600,
             'run_timeout_seconds':43200,'download_timeout_seconds':3600,'load_timeout_seconds':1800,
             'ocr_preprocessing':'whole','tile_edge':1024,'tile_overlap':0.15,'max_tiles':16,
             'florence_precision':'float16'}}
    import ipywidgets as widgets
    mode_control=widgets.ToggleButtons(options=['single','pilot','full'],value=RUN_STAGE,description='Mode:')
    display(mode_control)
    print('Single = 1 document; pilot = 2 documents + 2 general images; full = all 50.')
    print('Select the mode above, then run Section 6. Each execution creates a new folder.')
    display(CONFIG)
    ''','configuration'),
    md('''## 6 Model setup then timed pilot and full benchmark
    Downloads, loading and processing are separate visible stages. Each worker checks
    actual CUDA placement and records GPU utilization and peak memory. Pilot mode
    processes two documents and two general images. Mask checks alone
    do not prove every sensitive region was found. Completed outputs are reused;
    a full worker loads once for the remaining
    images. Other models can proceed if one fails. Model initialization and native
    imports live in the bundled leakbench adapters and are included in the run source.
    '''),code('''
    RUN_STAGE=mode_control.value
    run_dir=prepare_execution_run(PROJECT,CATEGORY,CONFIG,stage=RUN_STAGE,reuse=REUSE_PREDICTIONS)
    print('Selected mode:',RUN_STAGE,'| New run:',run_dir)
    report_dir=execute_category(PROJECT,run_dir,stage=RUN_STAGE,save_cloud=True)
    execution=read_json(report_dir/'summary.json').get('execution',{})
    print('Final status:',execution.get('status'),'— this does not certify privacy removal.')
    display(execution.get('reconciliation',[]))
    ''','execution'),
    md('## 7 Metrics comparison and final summary'),code(summaries,'summary'),
    md('''## 8 Browse saved results
    Run Sections 1–4 first to restore helpers, widgets, Drive and the image mapping.
    After Section 6, run this cell. You can also open existing saved runs without
    running inference. Choose **Run → Model → Type → Image**, or use **Previous**,
    **Next** and **Jump to #**. The three panels show original, predicted regions
    and redacted output. Open the tabs beneath them for detections, masks and errors
    or raw model output. Missing images are explicitly labelled.

    `document_12` means the twelfth document in the fixed selection, not VISPR image
    12 and not a score. Each entry also shows the original VISPR ID, model, run and
    output folder. The viewer includes all 50 originals; those outside the selected
    workload are marked **Not requested**. If no runs appear, mount Drive or run
    Section 6. Dataset previews are labelled and never presented as model outputs.
    '''),code(inspect,'inspection'),
    md('''### Pilot review gate
    After inspecting all four pilot originals, overlays, masks and redactions, supply
    image-specific review notes for each model you accept for a diagnostic benchmark.
    Describe remaining misses and OCR scope limitations. No review is preapproved.
    The gate binds your notes to exact saved predictions and rejects failed checks.
    Then choose full in the mode selector and run Section 6. New runs reuse only exact-protocol intact predictions.
    '''),code('''
    from leakbench.gates import pilot_records,record_pilot_review
    print('Review these four IDs:',[r['id'] for r in pilot_records(cohort)])
    PILOT_REVIEWS = {}  # model key -> {image ID: your review note}; empty does nothing
    for model,notes in PILOT_REVIEWS.items():record_pilot_review(run_dir,model,cohort,notes)
    ''','pilot-review'),
    md('''## 9 Retry cloud backup without rerunning inference
    Every copied artifact is checked by read-back SHA-256 through the mounted Drive
    filesystem. Matching files are skipped. This verifies the mounted copy, not an
    independent Google server durability guarantee. Download the whole run folder to
    preserve relative image links in report.html. Credentials are never included.
    '''),code('backup(PROJECT,CATEGORY,run_dir.name)','backup'),
    md('## 10 Download this run (preserves your result-export workflow)'),code('''
    import shutil
    from google.colab import files
    export=shutil.make_archive(str(PROJECT/f'{CATEGORY}_{run_dir.name}'),'zip',root_dir=run_dir.parent,base_dir=run_dir.name)
    files.download(export)
    ''','download')]
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
         'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':f'{number}_{category}_benchmark.ipynb'}},'nbformat':4,'nbformat_minor':5}
    for i,c in enumerate(cells):c['id']=f'{category}-{i:02}'
    folder=ROOT/'notebooks';folder.mkdir(exist_ok=True)
    (folder/f'{number}_{category}_benchmark.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')

training=[md('# Training readiness\n\nThis notebook reviews readiness only. It never loads weights or starts training.'),
          md((ROOT/'docs/TRAINING_READINESS.md').read_text(encoding='utf-8')),
          code(bootstrap,'bootstrap'),
          code('''
          import json
          plan=json.loads((PROJECT/'configs/training_plan.json').read_text())
          assert plan['enabled'] is False
          print('Training disabled. Complete the benchmark and review the readiness conditions first.')
          display(plan)
          selection=json.loads((PROJECT/'configs/selection.json').read_text())
          excluded_ids={r['original_id'] for r in selection['images']}
          excluded_hashes={r['image_sha256'] for r in selection['images']}
          print('Exclude these 50 original IDs and their duplicate groups from training.')
          print('Reserved evaluation images:',len(excluded_ids))
          ''','readiness')]
for i,c in enumerate(training):c['id']=f'training-{i}'
(ROOT/'notebooks/03_training_readiness.ipynb').write_text(json.dumps({'cells':training,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=1),encoding='utf-8')
dist=ROOT/'dist';dist.mkdir(exist_ok=True)
from build_reporting_recovery import build_recovery
recovery=build_recovery(ROOT)
with zipfile.ZipFile(dist/'show_me_the_leak_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in [*files,*sorted((ROOT/'notebooks').glob('*.ipynb')),ROOT/'README.md',
                 ROOT/'GUIDE.md',ROOT/'BENCHMARK_REPORT.md',ROOT/'requirements-dev.txt',
                 *sorted((ROOT/'docs').glob('*.md')),ROOT/'scripts/build_notebooks.py',
                 *sorted((ROOT/'docs/audit').glob('*.json')),
                 ROOT/'tests/test_leakbench_v2.py',ROOT/'tests/test_regression_repair.py',ROOT/'tests/test_execution_recovery.py',
                 ROOT/'scripts/build_reporting_recovery.py',ROOT/'scripts/audit_saved_runs.py',
                 ROOT/'scripts/audit_handoff.py',ROOT/'scripts/check_handoff.py',recovery]:
        z.write(path,path.relative_to(ROOT).as_posix())
if (ROOT/'data/vispr_eval_50/manifest.csv').exists():
    with zipfile.ZipFile(dist/'vispr_eval_50.zip','w',zipfile.ZIP_STORED) as z:
        for path in sorted((ROOT/'data/vispr_eval_50').rglob('*')):
            if path.is_file():z.write(path,path.relative_to(ROOT/'data').as_posix())
    print('Built optional fixed dataset archive from the local subset.')
else:
    print('Dataset absent: skipped optional dataset archive; Colab prepares it from the pinned source.')
print('Built three independent notebooks and source bundle.')
