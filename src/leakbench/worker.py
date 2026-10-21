"""One GPU model per worker, loaded once; persist every image before continuing."""
import argparse, json, shutil, subprocess, time, traceback, threading
from pathlib import Path
from .common import read_json,write_json,sha,log,stage,clean
from .dataset import records,validate_subset
from . import pipeline
from .adapters import load_adapter
from .ocr_adapters import create_ocr_adapter
from .legacy_workflow import VLM_KEYS,TEXT_RULE_ATTRIBUTES,make_detector,normalize_report
from .validation import validate_redaction

def verify_gpu_model(model):
    mapping=getattr(model,'hf_device_map',{}) or {}
    if any(str(v) in ('cpu','disk','meta') for v in mapping.values()):
        raise RuntimeError('CPU/disk offloading detected. Use a larger Colab GPU or a documented smaller/quantized model; do not run a slow hidden CPU benchmark.')
    devices={str(p.device) for p in model.parameters()}
    if not devices or any(not d.startswith('cuda') for d in devices):raise RuntimeError('Model parameters are not entirely on CUDA: '+str(devices))
    return {'device_map':{k:str(v) for k,v in mapping.items()},'parameter_devices':sorted(devices)}

def output_dir(run,key,record):return run/key/record['category']/Path(record['filename']).stem

def valid_completed(folder,protocol,record=None,model=None):
    path=folder/'prediction.json'
    if not path.exists():return False
    try:
        r=read_json(path)
        if not isinstance(r,dict):return False
        hashes=r.get('artifact_sha256',{})
        if not isinstance(hashes,dict):return False
        if model is not None and r.get('model_key')!=model:return False
        if record is not None:
            if r.get('image_id')!=record['id']:return False
            if record.get('image_sha256'):
                if r.get('input_sha256')!=record['image_sha256']:return False
                if record.get('image_path') and sha(record['image_path'])!=record['image_sha256']:return False
                if not any(name.startswith('original.') and value==record['image_sha256'] for name,value in hashes.items()):return False
            if record.get('annotation_sha256'):
                if r.get('annotation_sha256')!=record['annotation_sha256']:return False
                if record.get('annotation_path') and sha(record['annotation_path'])!=record['annotation_sha256']:return False
        required={'predicted.png','redacted.png','raw_predictions.json','redaction_mask.png','redaction_validation.json'}
        return (r.get('status')=='ok' and r.get('protocol_id')==protocol and required.issubset(hashes)
            and r.get('redaction_validation',{}).get('schema_version')==1
            and any(name.startswith('original.') for name in hashes)
            and all(Path(name).name==name and (folder/name).is_file() and sha(folder/name)==value for name,value in hashes.items()))
    except (ValueError,OSError,TypeError):return False

def archive_attempt(folder):
    """Keep all prior artifacts together; never show a stale redaction after a retry."""
    files=[p for p in folder.iterdir() if p.is_file()]
    if files:
        archive=folder/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
        for path in files:path.replace(archive/path.name)

def run(job):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU; choose a Colab GPU runtime.')
    project=Path(job['project']);run_dir=Path(job['run_dir']);key=job['model'];config=job['config']
    model_dir=run_dir/key;model_dir.mkdir(parents=True,exist_ok=True)
    state_path=model_dir/'state.json';dataset=project/'data/vispr_eval_50'
    validate_subset(dataset,read_json(project/'configs/selection.json'))
    from .progress import selected_records,refresh_execution
    selected=selected_records(records(dataset),job['phase'])
    pending=[r for r in selected if not valid_completed(output_dir(run_dir,key,r),job['protocol_id'],r,key)]
    if not pending:log('All selected images already verified; model load skipped',model=key);return
    taxonomy=pipeline.load_taxonomy(project/'configs/taxonomy_crosswalk.csv')
    pipeline.FORMAT_RETRIES=config['format_retries']
    adapter=None;started=time.perf_counter();timings=[]
    sampling_stop=threading.Event()
    def sample_gpu():
        while not sampling_stop.is_set():
            try:
                sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=5)
                with (model_dir/'gpu_samples.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'elapsed_seconds':time.perf_counter()-started,
                        'gpu_utilization_memory_csv':sample.stdout.strip(),'returncode':sample.returncode})+'\n')
            except (OSError,subprocess.TimeoutExpired) as exc:
                log('GPU utilization sampler unavailable',reason=clean(exc));return
            sampling_stop.wait(10)
    sampler=threading.Thread(target=sample_gpu,daemon=True);sampler.start()
    try:
        pinned=read_json(project/'configs/models.json')[key]
        with stage('model_download',state_path,model=key):
            from .downloads import snapshot_with_retry
            download_start=time.perf_counter()
            download_kwargs={'repo_id':pinned['model_id'],'revision':pinned['revision'],
                'allow_patterns':['*.safetensors','*.json','*.py','*.txt','*.model','*.jinja']}
            if key=='dots_ocr':download_kwargs['local_dir']=str(project/'.model_cache/DotsOCR')
            snapshot_with_retry(**download_kwargs)
            download_seconds=time.perf_counter()-download_start
        with stage('model_loading',state_path,model=key):
            load_start=time.perf_counter()
            kw={'max_new_tokens':config['max_new_tokens'],'max_input_edge':config['max_input_edge'],
                'load_in_4bit':config['load_in_4bit'] if key in VLM_KEYS or key=='chandra' else False,
                'revision':pinned['revision']}
            if key=='dots_ocr':kw['local_model_dir']=str(project/'.model_cache/DotsOCR')
            if key=='florence2':kw['precision']=config.get('florence_precision','float16')
            adapter=load_adapter(key,**kw) if key in VLM_KEYS or key=='florence2' else create_ocr_adapter(key,**kw)
            torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
            placement=verify_gpu_model(adapter.model)
            environment={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                         'cuda_compute_capability':list(torch.cuda.get_device_capability()),
                         'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,**placement}
            write_json(model_dir/f'{job["phase"]}_environment.json',environment)
            write_json(model_dir/f'{job["phase"]}_loading.json',{'download_seconds':download_seconds,'load_seconds':load_seconds,
                'model_id':pinned['model_id'],'revision':pinned['revision']})
            freeze=subprocess.run([__import__('sys').executable,'-m','pip','freeze'],text=True,capture_output=True,timeout=60)
            (model_dir/'requirements_resolved.txt').write_text(clean(freeze.stdout),encoding='utf-8')
        # Instrument the actual generate boundary; synchronize GPU for meaningful timings.
        original_generate=adapter.model.generate
        current={}
        def timed_generate(*args,**kwargs):
            torch.cuda.synchronize();before=time.perf_counter()
            current['preprocessing_seconds']+=before-current['call_started']
            current['generation_calls']+=1
            write_json(Path(current['artifact_directory'])/'attempt_state.json',
                {'status':'processing','stage':'inference','inference_attempted':True})
            log('generation',model=key,image_id=current.get('image_id'),call=current['generation_calls'],
                retry=current['generation_calls']>1)
            try:return original_generate(*args,**kwargs)
            finally:
                torch.cuda.synchronize();current['generation_seconds']+=time.perf_counter()-before
                current['generation_finished']=time.perf_counter()
        adapter.model.generate=timed_generate
        method_name='infer' if key in VLM_KEYS or key=='florence2' else 'extract'
        original_infer=getattr(adapter,method_name)
        def timed_infer(*args,**kwargs):
            current['call_started']=time.perf_counter();current['generation_finished']=None
            try:return original_infer(*args,**kwargs)
            finally:
                end=current.get('generation_finished')
                if end is not None:current['postprocessing_seconds']+=time.perf_counter()-end
        setattr(adapter,method_name,timed_infer)
        if key not in VLM_KEYS and config.get('ocr_preprocessing')=='tiles':
            from .preprocessing import tiled_extract
            native_extract=adapter.extract
            adapter.extract=lambda image:tiled_extract(native_extract,image,edge=config.get('tile_edge',1024),
                overlap=config.get('tile_overlap',.15),max_tiles=config.get('max_tiles',16))
        detector=make_detector(adapter,key,lambda image,taxonomy,generator:pipeline.detect_pii(
            image,taxonomy,generator,repair_partial_entries=config.get('repair_partial_entries',False)))
        for index,record in enumerate(selected,1):
            folder=output_dir(run_dir,key,record);folder.mkdir(parents=True,exist_ok=True)
            if valid_completed(folder,job['protocol_id'],record,key):
                log('Verified completed image reused',model=key,image_id=record['id']);continue
            archive_attempt(folder)
            write_json(folder/'attempt_state.json',{'status':'processing','stage':'image_loading','inference_attempted':False})
            refresh_execution(project,run_dir)
            torch.cuda.reset_peak_memory_stats()
            from .image_execution import process_image
            with stage('image',state_path,model=key,image_id=record['id'],index=index,total=len(selected)):
                report=process_image(record,folder,key,job['protocol_id'],config,taxonomy,detector,current,
                                     peak_memory=torch.cuda.max_memory_allocated)
                elapsed=report['timing']['total_seconds'];timings.append(elapsed)
                refresh_execution(project,run_dir)
                log('Image attempt saved',model=key,image_id=record['id'],inference_status=report['status'],
                    failure_stage=report.get('failure_stage'),redaction_status=report.get('redaction_status'),
                    mask_check_passed=report.get('redaction_validation',{}).get('technical_pass',False),
                    completed=index,total=len(selected),remaining=len(selected)-index)
            if job.get('cloud_model_dir'):
                try:
                    from .cloud import verified_copy_tree
                    with stage('image_backup',state_path,model=key,image_id=record['id']):
                        verified_copy_tree(folder,Path(job['cloud_model_dir'])/record['category']/folder.name)
                except Exception as exc:log('Image backup failed; local output retained for final backup retry',error=clean(exc))
        write_json(model_dir/f'{job["phase"]}_timing.json',{'download_seconds':download_seconds,'load_seconds':load_seconds,'wall_seconds':time.perf_counter()-started,
            'newly_processed_images':len(timings),'generation_boundary_instrumented':True})
    finally:
        sampling_stop.set();sampler.join(timeout=6)
        if adapter:adapter.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    job=read_json(args.job)
    try:run(job)
    except Exception as exc:
        folder=Path(job['run_dir'])/job['model'];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/f'{job["phase"]}_failure.json',{'error':clean(exc),'traceback':clean(traceback.format_exc())})
        raise
