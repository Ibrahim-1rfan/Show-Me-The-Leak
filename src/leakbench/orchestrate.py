"""Run category-specific pilots, continue only passing models, preserve progress."""
import os, shutil, sys
from datetime import datetime,timezone
from pathlib import Path
from .common import read_json,write_json,fingerprint,sha,log,clean
from .dataset import records
from .execution import setup_profile,run_command
from .worker import valid_completed,output_dir
from .cloud import DRIVE_ROOT,verified_copy_tree,backup
from .reporting import report
from .gates import checked_predictions,require_review
from .progress import begin_execution,update_model,refresh_execution,selected_records
PROFILES={'qwen3_vl':'vlm','qwen25_vl':'vlm','llama32_vision':'vlm','chandra':'chandra','dots_ocr':'dots','florence2':'florence'}
CATEGORIES={'vlm':['qwen3_vl','qwen25_vl','llama32_vision'],'ocr':['florence2','chandra','dots_ocr']}

def start_run(project,category,config,resume=True):
    project=Path(project)
    models=config['models']
    if not models or set(models)-set(CATEGORIES[category]):raise ValueError('Models must belong to the selected notebook category')
    contract={'category':category,'settings':config,'selection_fingerprint':read_json(project/'data/vispr_eval_50/validation.json')['selection_fingerprint'],
              'source_hashes':{str(p.relative_to(project)):sha(p) for p in sorted((project/'src/leakbench').glob('*.py'))},
              'taxonomy_sha256':sha(project/'configs/taxonomy_crosswalk.csv')}
    contract['model_pins']=read_json(project/'configs/models.json')
    # The requested comparison panel is not a model inference setting. This lets
    # a reviewed subset continue when another model needs access or repair.
    protocol=fingerprint({**contract,'settings':{k:v for k,v in config.items() if k!='models'}})
    local=project/'results'/category
    if resume:
        roots=[local,DRIVE_ROOT/'benchmark_v2/results'/category]
        for base in roots:
            for path in sorted(base.glob('*/run_config.json'),reverse=True):
                saved=read_json(path)
                if saved.get('protocol_id')==protocol:
                    target=local/path.parent.name
                    if path.parent.resolve()!=target.resolve():verified_copy_tree(path.parent,target)
                    log('Resuming matching run',path=str(target));return target
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=local/stamp;folder.mkdir(parents=True)
    write_json(folder/'run_config.json',{**contract,'protocol_id':protocol,'run_id':stamp})
    shutil.copytree(project/'configs',folder/'configs')
    shutil.copytree(project/'src/leakbench',folder/'source/leakbench',ignore=shutil.ignore_patterns('__pycache__'))
    return folder

def execute_category(project,run_dir,full=False,save_cloud=True,stage='single'):
    project,run_dir=Path(project),Path(run_dir);run_config=read_json(run_dir/'run_config.json')
    config=run_config['settings'];cohort=records(project/'data/vispr_eval_50');environments={}
    stage='full' if full else stage
    if stage not in ('single','pilot','full'):raise ValueError('Choose single, pilot or full')
    selected=selected_records(cohort,stage)
    execution_id=begin_execution(project,run_dir,stage)
    log('Execution plan',run_id=run_dir.name,execution_id=execution_id,mode=stage,
        intended_images_per_model=len(selected),models=config['models'],output_directory=str(run_dir))
    os.environ['PYTHONPATH']=str(project/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    def save_backup():
        if save_cloud:
            try:backup(project,run_config['category'],run_dir.name)
            except Exception as exc:
                write_json(run_dir/'cloud_failure.json',{'error':clean(exc),'recovery':'Rerun backup cell; inference need not be repeated.'})
                log('Cloud backup failed; local artifacts retained',error=clean(exc))
    def save_report():
        # Reporting failures must not be labelled as inference failures or prevent
        # other models and cloud backup from running.
        failure=run_dir/'reporting_failure.json'
        try:
            out=report(project,run_dir,config['models'])
        except Exception as exc:
            write_json(failure,{'status':'failed','error':clean(exc),
                'recovery':'Repair reporting and regenerate reports from saved predictions; inference need not be repeated.'})
            log('Reporting failed; retaining predictions and continuing',error=clean(exc))
            return None
        if failure.exists():
            write_json(failure,{**read_json(failure),'status':'resolved',
                'resolved_at':datetime.now(timezone.utc).isoformat()})
        return out
    save_backup()
    for model in config['models']:
        folder=run_dir/model;folder.mkdir(exist_ok=True);(folder/'logs').mkdir(exist_ok=True)
        step='review_gate'
        try:
            # Gate BEFORE environment setup or any new inference.
            if stage=='full':require_review(run_dir,model,cohort,run_config['protocol_id'])
            # Pilot is a diagnostic workload, independent of single-image recall.
            # Low recall must not prevent measuring both document/general images.
            if all(valid_completed(output_dir(run_dir,model,r),run_config['protocol_id'],r,model) for r in selected):
                log('Selected predictions reused without loading model',model=model)
                update_model(project,run_dir,model,status='completed',stage='reuse')
                continue
            step='dependency_installation'
            update_model(project,run_dir,model,status='processing',stage=step)
            profile=PROFILES[model]
            if profile not in environments:environments[profile]=setup_profile(profile,project/'.environments',folder/'logs')
            for phase in (stage,):
                job={'project':str(project),'run_dir':str(run_dir),'model':model,'phase':phase,'config':config,'protocol_id':run_config['protocol_id']}
                if save_cloud:job['cloud_model_dir']=str(DRIVE_ROOT/'benchmark_v2/results'/run_config['category']/run_dir.name/model)
                write_json(folder/'job.json',job)
                state=folder/'state.json'
                write_json(state,{'stage':'starting','elapsed_seconds':0})
                step='worker'
                run_command([environments[profile],'-m','leakbench.worker','--job',folder/'job.json'],f'{model} {phase}',
                    timeout=config['run_timeout_seconds'],log_path=folder/'logs'/f'{phase}.log',state_path=state,
                    image_timeout=config['image_timeout_seconds'],stage_timeouts={
                        'model_download':config.get('download_timeout_seconds',3600),
                        'model_loading':config.get('load_timeout_seconds',1800), 'image_backup':300})
                update_model(project,run_dir,model,status='completed',stage='finished')
        except Exception as exc:
            write_json(folder/'orchestration_failure.json',{'error':clean(exc),'recovery':'Inspect stage log, correct the issue and rerun; completed valid images are reused.'})
            state=read_json(folder/'state.json') if (folder/'state.json').exists() else {}
            update_model(project,run_dir,model,status='blocked' if step=='review_gate' else 'failed',
                         stage=state.get('stage',step) if step=='worker' else step,error=clean(exc))
            image_id=state.get('image_id')
            r=next((r for r in cohort if r['id']==image_id),None)
            if r and not valid_completed(output_dir(run_dir,model,r),run_config['protocol_id'],r,model):
                target=output_dir(run_dir,model,r);target.mkdir(parents=True,exist_ok=True)
                if not (target/'prediction.json').exists():
                    write_json(target/'prediction.json',{'image_id':r['id'],'model_key':model,'stratum':r['stratum'],
                        'status':'error','error':clean(exc),'ground_truth_attributes':[a for a in r['labels'] if a!='a0_safe'],
                        'protocol_id':run_config['protocol_id'],'redaction_status':'not_generated_worker_failed'})
            log('Model stopped; continuing to other models',model=model,error=clean(exc))
        finally:
            try:save_report()
            finally:save_backup()
    tracker=project/'TRACKER.md'
    if tracker.exists():
        with tracker.open('a',encoding='utf-8') as f:f.write(f'\nRuntime event: stage `{stage}`, run `{run_dir.name}`. See `{run_dir}/reports/summary.json`; execution does not certify privacy removal.\n')
    execution=refresh_execution(project,run_dir,finish=True)
    log('Execution reconciled',mode=stage,status=execution['status'],counts=execution['reconciliation'])
    try:report_dir=save_report()
    finally:save_backup()
    if report_dir is None:
        raise RuntimeError(f'Report generation failed after model processing and backup attempts. '
            f'Saved predictions remain available; see {run_dir / "reporting_failure.json"}. '
            'Regenerate reports after repairing reporting; do not rerun inference for this error.')
    return report_dir

def prepare_execution_run(project,category,config,stage='single',reuse=True):
    """New artifact run for each click; copy only exact-protocol intact predictions."""
    project=Path(project)
    selected=selected_records(records(project/'data/vispr_eval_50'),stage)
    run=start_run(project,category,config,resume=False)
    contract=read_json(run/'run_config.json')
    contract.update(mode=stage,intended_images_per_model=len(selected),
                    intended_image_ids=[r['id'] for r in selected],reused_predictions={})
    if reuse:
        bases=[project/'results'/category]
        if DRIVE_ROOT.is_dir():bases.append(DRIVE_ROOT/'benchmark_v2/results'/category)
        candidates=sorted([p for base in bases for p in base.glob('*/run_config.json')
                           if p.parent.resolve()!=run.resolve()],key=lambda p:p.parent.name,reverse=True)
        for path in candidates:
            previous=read_json(path)
            if previous.get('protocol_id')!=contract['protocol_id']:continue
            for model in config['models']:
                for r in selected:
                    destination=output_dir(run,model,r);source=output_dir(path.parent,model,r)
                    if not destination.exists() and valid_completed(source,contract['protocol_id'],r,model):
                        shutil.copytree(source,destination,ignore=shutil.ignore_patterns('attempts','__pycache__'))
                        contract['reused_predictions'][f'{model}/{r["id"]}']=str(source)
                review=path.parent/model/'pilot_review.json';destination=run/model/'pilot_review.json'
                if review.exists() and not destination.exists():
                    destination.parent.mkdir(exist_ok=True);shutil.copy2(review,destination)
    write_json(run/'run_config.json',contract)
    return run
