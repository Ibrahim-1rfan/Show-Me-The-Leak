"""Verify saved mask application, separately from unproven privacy completeness."""
import math
from PIL import Image,ImageDraw,ImageChops
from .pipeline import read_image
from .common import sha

def validate_redaction(original_path,redacted_path,detections,labels,prediction_status='ok',
                       expected_sha256=None,in_scope_labels=None,reference_boxes=None):
    original=read_image(original_path);issues=[];warnings=[]
    if expected_sha256 and sha(original_path)!=expected_sha256:issues.append('input_hash_mismatch')
    mask=Image.new('L',original.size,0);draw=ImageDraw.Draw(mask);unlocalized=0
    for det in detections:
        box=det.get('redaction_box_pixels')
        if box is None:unlocalized+=1;continue
        w,h=original.size
        if (len(box)!=4 or any(type(v) not in (float,int) or not math.isfinite(v) for v in box)
            or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h):
            issues.append('invalid_original_frame_box');continue
        l,t,r,b=map(int,box)
        if r<=l or b<=t:issues.append('zero_area_mask');continue
        draw.rectangle((l,t,r-1,b-1),fill=255)
    mask_pixels=mask.histogram()[255]
    if unlocalized:issues.append('detections_without_localization')
    if detections and not mask_pixels:issues.append('nonempty_detections_without_mask')
    expected=original.copy();expected.paste((0,0,0),(0,0),mask)
    changed=None
    try:
        with Image.open(redacted_path) as im:im.load();actual=im.convert('RGB')
        if actual.size!=original.size:issues.append('saved_dimensions_mismatch')
        else:
            diff=ImageChops.difference(original,actual)
            channels=diff.split();any_channel=ImageChops.lighter(ImageChops.lighter(channels[0],channels[1]),channels[2])
            changed=any_channel.point(lambda x:255 if x else 0).histogram()[255]
            # Exact comparison prevents overlays/originals being accepted as masks.
            if ImageChops.difference(expected,actual).getbbox() is not None:issues.append('saved_pixels_do_not_match_mask')
    except (OSError,ValueError):issues.append('saved_output_missing_or_unreadable')
    positive=set(labels)-{'a0_safe'}
    scope=positive if in_scope_labels is None else positive & set(in_scope_labels)
    if positive and not changed:warnings.append('known_positive_image_unchanged')
    mask_application_pass=not issues
    if scope and not mask_pixels:issues.append('in_scope_positive_without_redaction')
    if prediction_status!='ok':issues.append('prediction_incomplete_or_failed')
    missed=sorted(positive-{d['attribute_id'] for d in detections})
    if missed:warnings.append('ground_truth_labels_without_detection')
    ref_coverage=None
    if reference_boxes is not None:
        truth=Image.new('L',original.size,0);td=ImageDraw.Draw(truth)
        for box in reference_boxes:
            l,t,r,b=box
            if not 0<=l<r<=original.width or not 0<=t<b<=original.height:raise ValueError('Invalid independently supplied reference box')
            td.rectangle((math.floor(l),math.floor(t),math.ceil(r)-1,math.ceil(b)-1),fill=255)
        pixels=truth.histogram()[255]
        covered=ImageChops.darker(truth,mask).histogram()[255]
        ref_coverage=covered/pixels if pixels else None
        if pixels and covered<pixels:warnings.append('reference_sensitive_pixels_uncovered')
    # Even perfect mask application does not establish that all PII was detected.
    verified=not issues and not warnings and reference_boxes is not None and ref_coverage==1
    return {'schema_version':1,'technical_pass':not issues,'mask_application_pass':mask_application_pass,
        'prediction_complete':prediction_status=='ok','privacy_verified':verified,
        'status':'verified_reference_coverage' if verified else 'needs_review',
        'issues':sorted(set(issues)),'warnings':warnings,'mask_pixels':mask_pixels,
        'changed_pixels':changed,'unlocalized_detections':unlocalized,'missed_attributes':missed,
        'reference_region_coverage':ref_coverage,'reference_scope':'supplied boxes only' if reference_boxes is not None else 'image labels only'},mask
