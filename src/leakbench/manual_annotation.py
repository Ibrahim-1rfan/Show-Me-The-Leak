"""Canonical manual annotation validation; no model predictions are gold labels."""
import math

STATUSES = {'pending', 'in_progress', 'annotated', 'needs_review', 'approved'}

def validate_record(r, names, expected=None, require_approved=False):
    errors=[]
    if not isinstance(r,dict):return ['record must be object']
    def need(ok, msg):
        if not ok: errors.append(msg)
    required={'schema_version','annotation_id','source_id','image_file','sha256','width','height','split','annotator','coordinate_frame','status','image_type','quality','contains_selected_pii','selected_categories','instances','full_image_checked','updated_at','notes','review'}
    need(set(r)==required,'missing or unsupported record fields')
    need(type(r.get('full_image_checked')) is bool,'full_image_checked must be bool')
    need(isinstance(r.get('notes'),str),'notes must be string')
    review=r.get('review')
    if not isinstance(review,dict):return errors+['review must be object']
    need(set(review)=={'status','reviewer','reviewed_at'} and review.get('status') in {'pending','needs_review','approved'},'review fields/status')
    need(r.get('schema_version')=='vispr-manual-v1','schema_version')
    need(r.get('status') in STATUSES,'status')
    need(isinstance(r.get('annotation_id'),str),'annotation_id')
    need(r.get('split') in {'train','validation','test'},'split')
    need(r.get('coordinate_frame')=='exif_upright_normalized_xyxy','coordinate_frame')
    for k in ['width','height']:need(type(r.get(k)) is int and r[k]>0,k)
    if expected:
        for k in ['annotation_id','source_id','image_file','split','sha256','width','height']:
            need(str(r.get(k))==str(expected.get(k)),f'immutable {k} mismatch')
        need(r.get('annotator')==expected['annotator'],'primary annotator mismatch')
    need(r.get('image_type') in {None,'general','document','uncertain'},'image_type')
    need(r.get('quality') in {None,'clear','unclear','corrupt','ambiguous'},'quality')
    need(type(r.get('contains_selected_pii')) is bool or r.get('contains_selected_pii') is None,'contains_selected_pii')
    labels=r.get('selected_categories');instances=r.get('instances')
    if not isinstance(labels,list) or not isinstance(instances,list):return errors+['categories/instances must be lists']
    need(all(isinstance(x,str) and x in names for x in labels),'unsupported image category')
    need(len(labels)==len(set(labels)),'duplicate image category')
    seen=set();boxes=set();instance_labels=set()
    for i in instances:
        if not isinstance(i,dict):errors.append('instance must be object');continue
        need(set(i)=={'instance_id','attribute_id','attribute_name','bbox','localization','visibility','manually_verified','notes'},'missing/unsupported instance fields')
        key=i.get('instance_id');need(isinstance(key,str) and bool(key) and key not in seen,'duplicate/empty instance_id');seen.add(key)
        aid=i.get('attribute_id');need(aid in names,'unsupported instance attribute')
        need(i.get('attribute_name')==names.get(aid),'attribute name mismatch');instance_labels.add(aid)
        need(i.get('visibility') in {'fully_visible','partially_visible','obscured','uncertain'},'visibility')
        need(type(i.get('manually_verified')) is bool,'manually_verified must be bool')
        need(isinstance(i.get('notes'),str),'instance notes')
        b=i.get('bbox');state=i.get('localization')
        if b is None:
            need(state in {'uncertain','unlocalizable'},'null box requires explicit uncertainty')
            need(bool(i.get('notes','').strip()),'null box needs explanation')
        else:
            valid=isinstance(b,list) and len(b)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in b)
            need(valid,'box must contain four finite numbers')
            if valid:
                need(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1,'invalid normalized box')
                signature=(aid,tuple(b));need(signature not in boxes,'duplicate same-category box');boxes.add(signature)
            need(state=='localized','box requires localized state')
        need('text' not in i,'text collection disabled in v1')
    need(set(labels)==instance_labels,'image categories must equal instance categories, including null-box instances')
    if r.get('contains_selected_pii') is False:need(not labels and not instances,'negative has positive instances')
    if r.get('contains_selected_pii') is True:need(bool(labels),'positive lacks category')
    if r.get('contains_selected_pii') is None:need(r.get('status') not in {'annotated','approved'},'unknown presence cannot be completed')
    if r.get('status') in {'annotated','needs_review','approved'}:
        need(r.get('image_type') is not None and r.get('quality') is not None,'inspection fields missing')
        need(bool(r.get('annotator')) and bool(r.get('updated_at')),'annotator/time missing')
    if require_approved:need(r.get('status')=='approved','not approved')
    if r.get('status')=='approved':
        review=r.get('review',{})
        need(review.get('status')=='approved' and bool(review.get('reviewed_at')),'review approval/time required')
        need(bool(review.get('reviewer')) and review.get('reviewer')!=r.get('annotator'),'independent reviewer required')
        need(r.get('full_image_checked') is True,'full image must be checked for missed instances')
        need(all(i.get('manually_verified') is True for i in instances),'unverified instance')
        need(r.get('quality')=='clear' and r.get('image_type') in {'general','document'},'unresolved quality/type')
    return errors

def ls_box(value):
    """Label Studio percentages -> canonical normalized xyxy; reject rotation."""
    if value.get('rotation',0)!=0:raise ValueError('Rotated boxes are unsupported; redraw axis-aligned')
    x,y,w,h=[value[k]/100 for k in ('x','y','width','height')]
    if not all(math.isfinite(v) for v in (x,y,w,h)) or not (0<=x<x+w<=1 and 0<=y<y+h<=1):
        raise ValueError('Invalid Label Studio box')
    return [x,y,x+w,y+h]
