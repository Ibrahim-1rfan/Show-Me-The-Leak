"""Read exported runs and reconstruct the six requests without changing evidence."""
import csv,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from leakbench.common import read_json,write_json,sha
from leakbench.dataset import records
from leakbench.progress import selected_records
from leakbench.validation import validate_redaction
from leakbench.legacy_workflow import VLM_KEYS,TEXT_RULE_ATTRIBUTES
from PIL import Image,ImageDraw

def audit():
    out=ROOT/'docs/recovery_20261021';out.mkdir(exist_ok=True)
    cohort=records(ROOT/'data/vispr_eval_50');runs=[];requests=[];models=[];images=[]
    for pattern in ('vlm_*/**/run_config.json','ocr_*/**/run_config.json'):
        for path in ROOT.glob(pattern):
            run=path.parent;cfg=read_json(path);tracker=run/'TRACKER.md'
            # This historical reconstruction applies only to the original two
            # exports. New execution-aware runs require audit_handoff.py.
            if run.name not in {'20261020T234601512294Z','20261021T011930055215Z'}:continue
            stages=re.findall(r'Runtime event: stage `(single|pilot|full)`',tracker.read_text(encoding='utf-8')) if tracker.exists() else []
            source_checks={name:sha(run/name.replace('src/','source/',1))==value
                for name,value in cfg.get('source_hashes',{}).items() if (run/name.replace('src/','source/',1)).exists()}
            runs.append({'run_id':run.name,'path':str(run.relative_to(ROOT)),'category':cfg['category'],
                'protocol_id':cfg['protocol_id'],'settings':cfg['settings'],'recorded_requests':stages,
                'saved_source_hash_checks':source_checks,'historical_outputs_modified':False})
            for model in cfg['settings']['models']:
                folder=run/model
                failures={p.name:read_json(p).get('error') for p in folder.glob('*failure.json')}
                timings={p.name:read_json(p) for p in folder.glob('*_timing.json')}
                model_row={'run_id':run.name,'model_key':model,'failures':failures,'timings':timings,'images':[]}
                for p in folder.glob('*/*/prediction.json'):
                    pred=read_json(p);raw=read_json(p.with_name('raw_predictions.json'));native=raw.get('native_outputs',[])
                    row={'model_key':model,'run_id':run.name,'filename':p.parent.parent.name+'/'+p.parent.name,
                        'image_id':pred['image_id'],'prediction_status':pred['status'],'predicted_attributes':pred.get('predicted_attributes'),
                        'generation_seconds':pred.get('timing',{}).get('generation_seconds'),
                        'image_seconds':pred.get('timing',{}).get('total_seconds'),
                        'artifact_hashes_intact':all((p.parent/name).is_file() and sha(p.parent/name)==h for name,h in pred.get('artifact_sha256',{}).items()),
                        'native_text_regions':sum(len(n.get('text_regions',[])) for n in native),
                        'ignored_nontext_blocks':sum(n.get('metadata',{}).get('ignored_nontext_blocks',0) for n in native),
                        'metadata':[n.get('metadata') for n in native]}
                    original=next(p.parent.glob('original.*'));redacted=p.parent/'redacted.png'
                    if redacted.exists():
                        r=next(r for r in cohort if r['id']==pred['image_id'])
                        validation,_=validate_redaction(original,redacted,pred.get('detections',[]),r['labels'],
                            prediction_status=pred['status'],expected_sha256=r['image_sha256'],
                            in_scope_labels=None if model in VLM_KEYS else TEXT_RULE_ATTRIBUTES)
                        row['rechecked_redaction']=validation
                    else:row['rechecked_redaction']={'status':'redaction_missing','technical_pass':False}
                    model_row['images'].append(row);images.append((model,p.parent,row))
                models.append(model_row)
            for stage in stages:
                selected=selected_records(cohort,stage);manifest=[]
                for model in cfg['settings']['models']:
                    for r in selected:
                        p=run/model/r['category']/Path(r['filename']).stem/'prediction.json'
                        pred=read_json(p) if p.exists() else None
                        manifest.append({'image_id':r['id'],'filename':r['filename'],'model_key':model,
                            'run_id':run.name,'requested_mode':stage,'inference_attempted_in_request':bool(pred) and stage=='single',
                            'status':('failed' if not pred or pred['status']!='ok' else 'redaction_failed') if stage=='single' else 'skipped',
                            'reason':'Earlier regression gate blocked this request' if stage!='single' else (pred or {}).get('error'),
                            'saved_prediction_from_single':bool(pred),'source_prediction_status':(pred or {}).get('status'),
                            'redaction_generated':p.with_name('redacted.png').exists(),'output_saved':bool(pred),
                            'duration_seconds':(pred or {}).get('timing',{}).get('total_seconds') if stage=='single' else None})
                filename=f'{cfg["category"]}_{stage}_processing_manifest.json';write_json(out/filename,manifest)
                requests.append({'category':cfg['category'],'mode':stage,'artifact_run_id':run.name,
                    'intended_images_per_model':len(selected),'intended_model_image_pairs':len(manifest),
                    'new_image_inferences':sum(r['inference_attempted_in_request'] for r in manifest),
                    'saved_predictions_from_single':sum(r['saved_prediction_from_single'] for r in manifest),
                    'result':'single_attempted_no_verified_privacy' if stage=='single' else 'blocked_before_inference',
                    'manifest':filename,'evidence':'embedded tracker, single-only job/timing files, final gate errors, saved source; separate pilot/full timestamps were not retained'})
    if not runs:
        raise SystemExit('Historical exports absent. Use scripts/audit_handoff.py for current execution-aware runs; existing historical audit retained.')
    write_json(out/'audit.json',{'runs':runs,'requests':requests,'models':models})
    events=[]
    for notebook in (out/'notebooks_before_repair').glob('*.ipynb'):
        for cell in read_json(notebook)['cells']:
            for output in cell.get('outputs',[]):
                for line in ''.join(output.get('text',[])).splitlines():
                    try:event=json.loads(line)
                    except ValueError:continue
                    if isinstance(event,dict) and ('backup' in event.get('message','').lower() or 'Model stopped' in event.get('message','')):
                        events.append({'notebook':notebook.name,**event})
    write_json(out/'notebook_cloud_and_gate_events.json',events)
    with (out/'six_requests.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(requests[0]));writer.writeheader();writer.writerows(requests)
    # Contact sheet uses real saved pixels, solely for inspecting the audit.
    sheet=Image.new('RGB',(990,len(images)*290),'white');draw=ImageDraw.Draw(sheet)
    for i,(model,folder,row) in enumerate(images):
        draw.text((5,i*290+4),f'{model} | {row["prediction_status"]} | mask pixels {row["rechecked_redaction"].get("mask_pixels","missing")}',fill='black')
        paths=[next(folder.glob('original.*')),folder/'predicted.png',folder/'redacted.png']
        for j,p in enumerate(paths):
            draw.text((j*330+5,i*290+22),p.name if p.exists() else 'Missing redaction',fill='black')
            if p.exists():
                with Image.open(p) as im:
                    im=im.convert('RGB');im.thumbnail((320,240));sheet.paste(im,(j*330+5,i*290+42))
    sheet.save(out/'redaction_contact.png')
    print(json.dumps(requests,indent=2))
    return out

if __name__=='__main__':audit()
