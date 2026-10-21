"""Persist a single image attempt, including partial evidence after failure."""
import shutil,time
from pathlib import Path
from . import pipeline
from .common import write_json,sha,clean,log
from .validation import validate_redaction
from .legacy_workflow import normalize_report,VLM_KEYS,TEXT_RULE_ATTRIBUTES

def process_image(record,folder,model,protocol,config,taxonomy,detector,current,peak_memory=lambda:None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    original=folder/('original'+Path(record['filename']).suffix)
    started=time.perf_counter();read_seconds=0.;prediction={};image=None;failure_stage='image_loading'
    report={'image_id':record['id'],'status':'error','protocol_id':protocol,'image_path':str(original)}
    current.clear();current.update(image_id=record['id'],preprocessing_seconds=0.,generation_seconds=0.,
                                   postprocessing_seconds=0.,generation_calls=0,artifact_directory=str(folder))
    try:
        shutil.copy2(record['image_path'],original)
        image=pipeline.read_image(record['image_path']);read_seconds=time.perf_counter()-started
        report['image_size']=list(image.size)
        failure_stage='inference';prediction=detector(image,taxonomy)
        report.update(status=prediction['status'],prediction=prediction)
        # Persist native evidence BEFORE coordinate conversion or image saving.
        failure_stage='saving_native_output'
        write_json(folder/'raw_predictions.json',{'native_outputs':prediction.get('native_outputs',[]),
            'attempts':prediction.get('attempts',[]),'error':prediction.get('error')})
        if prediction['status'] in ('ok','partial'):
            failure_stage='coordinate_conversion';post_started=time.perf_counter()
            redacted,overlay,detections=pipeline.redact_and_draw(image,prediction['detections'],config['redaction_padding'])
            prediction['detections']=detections
            failure_stage='saving_redaction';overlay.save(folder/'predicted.png');redacted.save(folder/'redacted.png')
            report.update(overlay_path=str(folder/'predicted.png'),redacted_path=str(folder/'redacted.png'),
                evaluation=pipeline.evaluate_labels(record['labels'],detections,taxonomy))
            failure_stage='redaction_validation'
            validation,mask=validate_redaction(original,folder/'redacted.png',detections,record['labels'],
                prediction_status=prediction['status'],expected_sha256=record.get('image_sha256') or sha(record['image_path']),
                in_scope_labels=None if model in VLM_KEYS else TEXT_RULE_ATTRIBUTES)
            failure_stage='saving_mask';mask.save(folder/'redaction_mask.png')
            write_json(folder/'redaction_validation.json',validation)
            report.update(redaction_validation=validation,redaction_status=validation['status'])
            if prediction['status']=='partial':report.update(failure_stage='output_parsing',error=prediction.get('error'))
            elif not validation['technical_pass']:
                report['failure_stage']='sensitive_detection' if not detections else 'redaction_validation'
            current['postprocessing_seconds']+=time.perf_counter()-post_started
        else:
            report.update(error=prediction.get('error') or 'No valid prediction; inspect raw output.',
                failure_stage=prediction.get('failure_stage','output_parsing'),redaction_status='not_generated_detection_failed')
        if not (folder/'predicted.png').exists():
            from PIL import ImageDraw
            overlay=image.copy();ImageDraw.Draw(overlay).text((10,10),'FAILED - no verified redaction',fill='red')
            overlay.save(folder/'predicted.png')
    except Exception as exc:
        # One malformed image/coordinate/save must not terminate the whole model.
        report.update(status='error',prediction=prediction,error=clean(exc),failure_stage=failure_stage,
                      redaction_status='not_verified_pipeline_failure')
        log('Image failed; continuing to remaining inputs',model=model,image_id=record['id'],
            failure_stage=failure_stage,error=clean(exc))
    report['residual_audit']={'status':'not_run','reason':'Independent region truth required for privacy completeness'}
    report['inference_attempted']=bool(current.get('generation_calls') or prediction.get('native_outputs'))
    report=normalize_report(report,record,model)
    report['timing']={k:current.get(k,0) for k in ('preprocessing_seconds','generation_seconds','postprocessing_seconds','generation_calls')}
    report['timing'].update(read_seconds=read_seconds,total_seconds=time.perf_counter()-started,
                           peak_gpu_memory_bytes=peak_memory() or 0)
    report['artifact_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name in
        (original.name,'predicted.png','redacted.png','raw_predictions.json','redaction_mask.png','redaction_validation.json')}
    report['input_sha256']=record.get('image_sha256');report['annotation_sha256']=record.get('annotation_sha256')
    write_json(folder/'prediction.json',report)
    write_json(folder/'attempt_state.json',{'status':'finished','stage':report.get('failure_stage','finished'),
                                          'inference_attempted':report['inference_attempted']})
    return report
