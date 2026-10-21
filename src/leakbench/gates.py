"""Explicit staged acceptance, bound to image artifacts and protocol."""
from pathlib import Path
from .common import read_json,write_json,sha

def pilot_records(cohort):
    return [r for category in ('documents','images') for r in [s for s in cohort if s['category']==category][:2]]

def checked_predictions(run,model,cohort,protocol,allow_detection_misses=False):
    from .worker import output_dir,valid_completed
    hashes={}
    for r in cohort:
        folder=output_dir(Path(run),model,r)
        if not valid_completed(folder,protocol,r,model):raise RuntimeError(f'Regression gate: {model}/{r["filename"]} lacks an intact complete prediction')
        result=read_json(folder/'prediction.json')
        validation=result.get('redaction_validation',{})
        passes=validation.get('technical_pass')
        if allow_detection_misses:passes=validation.get('mask_application_pass',passes)
        if not passes:
            raise RuntimeError(f'Regression gate: {model}/{r["filename"]} failed mask/known-positive checks; inspect validation issues')
        hashes[r['id']]=sha(folder/'prediction.json')
    return hashes

def record_pilot_review(run,model,cohort,reviews):
    run=Path(run);config=read_json(run/'run_config.json');chosen=pilot_records(cohort)
    hashes=checked_predictions(run,model,chosen,config['protocol_id'],allow_detection_misses=True)
    if set(reviews)!=set(hashes):raise ValueError('Review all four pilot image IDs explicitly')
    for note in reviews.values():
        if not isinstance(note,str) or len(note.strip())<10:raise ValueError('Provide an image-specific review note, including residual privacy limitations')
    write_json(run/model/'pilot_review.json',{'protocol_id':config['protocol_id'],'prediction_sha256':hashes,
        'review_notes':reviews,'accepted_for_diagnostic_benchmark':True,'privacy_completeness_certified':False})

def require_review(run,model,cohort,protocol):
    hashes=checked_predictions(run,model,pilot_records(cohort),protocol,allow_detection_misses=True)
    path=Path(run)/model/'pilot_review.json'
    if not path.exists():raise RuntimeError('Full benchmark blocked: review all four pilot originals, masks, redactions and remaining misses first')
    review=read_json(path)
    if (review.get('protocol_id')!=protocol or review.get('prediction_sha256')!=hashes
        or review.get('accepted_for_diagnostic_benchmark') is not True):raise RuntimeError('Pilot review is stale or not accepted for this exact run')
