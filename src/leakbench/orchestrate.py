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
    protocol=fingerprint(contract);local=project/'results'/category
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

def execute_category(project,run_dir,full=True,save_cloud=True):
    project,run_dir=Path(project),Path(run_dir);run_config=read_json(run_dir/'run_config.json')
    config=run_config['settings'];cohort=records(project/'data/vispr_eval_50');environments={}
    os.environ['PYTHONPATH']=str(project/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
    def save_backup():
        if save_cloud:
            try:backup(project,run_config['category'],run_dir.name)
            except Exception as exc:
                write_json(run_dir/'cloud_failure.json',{'error':clean(exc),'recovery':'Rerun backup cell; inference need not be repeated.'})
                log('Cloud backup failed; local artifacts retained',error=clean(exc))
    save_backup()
    for model in config['models']:
        folder=run_dir/model;folder.mkdir(exist_ok=True);(folder/'logs').mkdir(exist_ok=True)
        if all(valid_completed(output_dir(run_dir,model,r),run_config['protocol_id']) for r in cohort):
            log('Complete model reused without environment setup or inference',model=model);continue
        try:
            profile=PROFILES[model]
            if profile not in environments:environments[profile]=setup_profile(profile,project/'.environments',folder/'logs')
            for phase in ('pilot','full') if full else ('pilot',):
                if phase=='full':
                    pilot=[next(r for r in cohort if r['category']==c) for c in ('documents','images')]
                    if not all(valid_completed(output_dir(run_dir,model,r),run_config['protocol_id']) for r in pilot):
                        raise RuntimeError('Pilot did not produce two valid outputs. Full run blocked; inspect pilot report and failure artifacts.')
                job={'project':str(project),'run_dir':str(run_dir),'model':model,'phase':phase,'config':config,'protocol_id':run_config['protocol_id']}
                if save_cloud:job['cloud_model_dir']=str(DRIVE_ROOT/'benchmark_v2/results'/run_config['category']/run_dir.name/model)
                write_json(folder/'job.json',job)
                state=folder/'state.json'
                write_json(state,{'stage':'starting','elapsed_seconds':0})
                run_command([environments[profile],'-m','leakbench.worker','--job',folder/'job.json'],f'{model} {phase}',
                    timeout=config['run_timeout_seconds'],log_path=folder/'logs'/f'{phase}.log',state_path=state,
                    image_timeout=config['image_timeout_seconds'],stage_timeouts={
                        'model_download':config.get('download_timeout_seconds',3600),
                        'model_loading':config.get('load_timeout_seconds',1800), 'image_backup':300})
                report(project,run_dir,config['models'])
                save_backup()
        except Exception as exc:
            write_json(folder/'orchestration_failure.json',{'error':clean(exc),'recovery':'Inspect stage log, correct the issue and rerun; completed valid images are reused.'})
            state=read_json(folder/'state.json') if (folder/'state.json').exists() else {}
            image_id=state.get('image_id')
            r=next((r for r in cohort if r['id']==image_id),None)
            if r and not valid_completed(output_dir(run_dir,model,r),run_config['protocol_id']):
                target=output_dir(run_dir,model,r);target.mkdir(parents=True,exist_ok=True)
                if not (target/'prediction.json').exists():
                    write_json(target/'prediction.json',{'image_id':r['id'],'model_key':model,'stratum':r['stratum'],
                        'status':'error','error':clean(exc),'ground_truth_attributes':[a for a in r['labels'] if a!='a0_safe'],
                        'protocol_id':run_config['protocol_id'],'redaction_status':'not_generated_worker_failed'})
            log('Model stopped; continuing to other models',model=model,error=clean(exc))
        finally:
            report(project,run_dir,config['models'])
            save_backup()
    return report(project,run_dir,config['models'])
