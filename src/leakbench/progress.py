"""Explicit execution plans and per-image reconciliation, without GPU imports."""
from datetime import datetime, timezone
from pathlib import Path
from .common import read_json,write_json,clean,sha
from .dataset import records

def selected_records(cohort,mode):
    if mode=='single':return [r for r in cohort if r['category']=='documents'][:1]
    if mode=='pilot':
        from .gates import pilot_records
        return pilot_records(cohort)
    if mode=='full':return list(cohort)
    raise ValueError('Choose single, pilot or full')

def begin_execution(project,run,mode):
    run=Path(run);config=read_json(run/'run_config.json')
    cohort=selected_records(records(Path(project)/'data/vispr_eval_50'),mode)
    from .worker import output_dir,valid_completed
    reused=dict(config.get('reused_predictions',{}))
    for model in config['settings']['models']:
        for r in cohort:
            source=output_dir(run,model,r)
            if valid_completed(source,config['protocol_id'],r,model):
                reused.setdefault(f'{model}/{r["id"]}',str(source))
    ident=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    execution={'execution_id':ident,'artifact_run_id':run.name,'mode':mode,
        'started_at':datetime.now(timezone.utc).isoformat(),'status':'processing',
        'intended_images_per_model':len(cohort),'intended_image_ids':[r['id'] for r in cohort],
        'models':{key:{'status':'pending'} for key in config['settings']['models']},
        'settings':config['settings'],'protocol_id':config['protocol_id'],'output_directory':str(run),
        'reused_predictions':reused}
    folder=run/'executions'/ident;folder.mkdir(parents=True)
    write_json(folder/'execution.json',execution)
    write_json(run/'active_execution.json',{'execution_id':ident})
    refresh_execution(project,run)
    return ident

def active_execution(run):
    run=Path(run);path=run/'active_execution.json'
    if not path.exists():return None,None
    ident=read_json(path)['execution_id']
    if Path(ident).name!=ident:raise ValueError('Invalid execution identifier')
    folder=run/'executions'/ident
    return folder,read_json(folder/'execution.json')

def update_model(project,run,model,**updates):
    folder,execution=active_execution(run)
    if execution is None:return
    execution['models'][model].update(updates)
    write_json(folder/'execution.json',execution)
    refresh_execution(project,run)

def refresh_execution(project,run,finish=False):
    from .worker import output_dir,valid_completed
    run=Path(run);folder,execution=active_execution(run)
    if execution is None:return None
    cohort={r['id']:r for r in records(Path(project)/'data/vispr_eval_50')}
    rows=[]
    for model,model_state in execution['models'].items():
        state_path=run/model/'state.json'
        state=read_json(state_path) if state_path.exists() else {}
        for index,ident in enumerate(execution['intended_image_ids'],1):
            r=cohort[ident];output=output_dir(run,model,r);path=output/'prediction.json'
            attempt_state=read_json(output/'attempt_state.json') if (output/'attempt_state.json').exists() else {}
            reused=execution['reused_predictions'].get(f'{model}/{ident}')
            row={'image_id':ident,'filename':r['filename'],'index':index,'category':r['category'],
                'model_key':model,'run_id':run.name,'execution_id':execution['execution_id'],
                'mode':execution['mode'],'output_directory':str(output),'status':'pending',
                'inference_attempted':bool(attempt_state.get('inference_attempted')) and not bool(reused),
                'inference_completed':False,'prediction_generated':False,'valid_prediction':False,
                'redaction_generated':(output/'redacted.png').is_file(),'output_saved':path.is_file(),
                'redaction_check_passed':False,'privacy_verified':False,'duration_seconds':None,
                'failure_stage':None,'error':None,'reused_from':reused}
            prediction=None
            if path.exists():
                try:
                    prediction=read_json(path)
                    if not isinstance(prediction,dict):raise ValueError('Prediction must be an object')
                except (ValueError,OSError) as exc:row.update(status='failed',failure_stage='saving',error=clean(exc))
            if isinstance(prediction,dict):
                valid=valid_completed(output,execution['protocol_id'],r,model)
                checks=prediction.get('redaction_validation',{})
                native=prediction.get('prediction',{}).get('native_outputs',[])
                attempted=bool(prediction.get('inference_attempted') or attempt_state.get('inference_attempted') or
                               prediction.get('timing',{}).get('generation_calls',0) or native)
                row.update(inference_attempted=attempted and not bool(reused),
                    inference_completed=bool(native),
                    prediction_generated=bool(native) or prediction.get('status') in ('ok','partial'),
                    prediction_status=prediction.get('status'),valid_prediction=valid,
                    redaction_check_passed=valid and checks.get('technical_pass') is True,
                    privacy_verified=valid and checks.get('privacy_verified') is True,
                    duration_seconds=prediction.get('timing',{}).get('total_seconds'),
                    failure_stage=prediction.get('failure_stage'),
                    error=prediction.get('error') or prediction.get('prediction',{}).get('error'),
                    redaction_issues=checks.get('issues',[]),redaction_warnings=checks.get('warnings',[]),
                    prediction_sha256=sha(path))
                row['detected_pii_regions']=len(prediction.get('detections') or [])
                row['ocr_text_regions']=sum(len(n.get('text_regions',[])) for n in native)
                row['diagnostic']=prediction.get('prediction',{}).get('diagnostic')
                if not valid:row['status']='failed'
                elif not row['redaction_check_passed']:row['status']='redaction_failed'
                else:row['status']='success'
                if row['status']=='failed' and not row['error']:
                    row['error']='Prediction incomplete or required artifact hash/identity verification failed.'
                if row['status']=='redaction_failed':row['failure_stage']='sensitive_detection_or_redaction'
                if reused:row['outcome']=row['status'];row['status']='skipped'
                # Keep an immutable prediction copy with each execution record.
                saved=folder/'predictions'/model/(ident+'.json');saved.parent.mkdir(parents=True,exist_ok=True)
                if not saved.exists() or not execution.get('finished_at'):write_json(saved,prediction)
            elif not path.exists():
                if model_state['status'] in ('blocked','failed','completed'):
                    row.update(status='failed' if row['inference_attempted'] else 'skipped',failure_stage=model_state.get('stage'),
                        error=model_state.get('error') or 'Worker ended without this intended output.')
                elif model_state['status']=='processing' and (state.get('image_id')==ident or attempt_state.get('status')=='processing'):
                    row.update(status='processing',failure_stage=attempt_state.get('stage',state.get('stage')),
                               inference_attempted=bool(attempt_state.get('inference_attempted')))
            row['missing_output']=not row['output_saved']
            rows.append(row)
    reconciliation=[]
    for model in execution['models']:
        selected=[r for r in rows if r['model_key']==model]
        counts={status:sum(r['status']==status for r in selected) for status in
                ('pending','processing','success','failed','skipped','redaction_failed')}
        reconciliation.append({'model_key':model,'intended_images':len(selected),
            'attempted_images':sum(r['inference_attempted'] for r in selected),
            'successful_inferences':sum(r['inference_completed'] for r in selected),
            'valid_predictions':sum(r['valid_prediction'] for r in selected),
            'successful_redaction_checks':sum(r['redaction_check_passed'] for r in selected),
            'privacy_verified_images':sum(r['privacy_verified'] for r in selected),
            'missing_outputs':sum(r['missing_output'] for r in selected),
            'reused_images':sum(bool(r['reused_from']) for r in selected),
            'remaining_unprocessed':sum(not r['output_saved'] for r in selected),
            'accounted_for':sum(counts.values()),**counts})
    if finish:
        execution['finished_at']=datetime.now(timezone.utc).isoformat()
        execution['wall_seconds']=(datetime.fromisoformat(execution['finished_at'])-datetime.fromisoformat(execution['started_at'])).total_seconds()
        execution['status']=('complete_inference_needs_privacy_review'
            if all(r['valid_prediction'] and r['redaction_check_passed'] for r in rows) else
            'completed_with_redaction_failures' if all(r['valid_prediction'] for r in rows) else
            'blocked' if all(s['status']=='blocked' for s in execution['models'].values()) else 'incomplete')
    execution['reconciliation']=reconciliation
    write_json(folder/'processing_manifest.json',rows);write_json(folder/'execution.json',execution)
    write_json(run/'processing_manifest.json',rows)
    return execution
