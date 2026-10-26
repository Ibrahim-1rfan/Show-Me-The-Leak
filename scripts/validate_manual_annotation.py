"""Validate package coverage, copy hashes and pending/completed canonical records."""
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from leakbench.manual_annotation import validate_record
from prepare_manual_annotation import sha

def validate(package,require_complete=False):
    master=json.loads((package/'manifests/master.json').read_text());expected={r['annotation_id']:r for r in master};errors=[]
    if len(master)!=1200 or len(expected)!=1200:errors.append('master must contain 1200 unique IDs')
    master_csv=list(csv.DictReader((package/'manifests/master.csv').open(encoding='utf-8')))
    if len(master_csv)!=len(master) or any(r!={k:str(v) for k,v in original.items()} for r,original in zip(master_csv,master)):errors.append('master CSV/JSON disagree')
    names={a['attribute_id']:a['name'] for a in json.loads((package/'manifests/selected_attributes.json').read_text())['attributes']}
    owners=[]
    for i in range(1,4):
        rows=list(csv.DictReader((package/f'assignments/member_{i}.csv').open(encoding='utf-8')))
        if len(rows)!=400:errors.append(f'Member {i}: expected 400')
        if Counter(r['image_type'] for r in rows)!=Counter({'general':280,'document':120}):errors.append(f'Member {i}: type quota')
        for r in rows:
            owners.append(r['annotation_id'])
            if r['annotation_id'] not in expected or r['annotator']!=f'Team Member {i}' or r['annotator']!=expected[r['annotation_id']]['annotator']:errors.append('assignment ownership mismatch')
            elif r!={k:str(v) for k,v in expected[r['annotation_id']].items()}:errors.append('assignment metadata mismatch')
    if Counter(owners)!=Counter(expected.keys()):errors.append('primary coverage or uniqueness failed')
    frozen={r['original_id'] for r in json.loads((package/'manifests/excluded_benchmark.json').read_text())['images']}
    if frozen & {r['source_id'] for r in master}:errors.append('benchmark leakage')
    if len({r['duplicate_group'] for r in master})!=len(master):errors.append('duplicate groups overlap')
    pilot=list(csv.DictReader((package/'assignments/pilot_shared.csv').open(encoding='utf-8')))
    if len(pilot)!=27 or len({r['annotation_id'] for r in pilot})!=27 or any(r['annotation_id'] not in expected or expected[r['annotation_id']]['split']!='train' for r in pilot):errors.append('pilot must have 27 unique training IDs')
    review=list(csv.DictReader((package/'assignments/cross_review.csv').open(encoding='utf-8')))
    if len(review)!=120 or len({r['annotation_id'] for r in review})!=120:errors.append('secondary review coverage')
    for r in review:
        if r['annotation_id'] not in expected or r['primary']!=expected[r['annotation_id']]['annotator'] or r['primary']==r['reviewer']:errors.append('secondary review ownership')
    completed=0;approved=0
    for aid,r in expected.items():
        for field,h in [('image_file','sha256'),('annotation_view','view_sha256')]:
            p=(package/r[field]).resolve()
            if not p.is_relative_to(package.resolve()) or not p.is_file() or sha(p)!=r[h]:errors.append(f'{aid}: {field} missing/changed')
        original=package/'annotations/original_vispr'/Path(r['original_annotation']).name
        if not original.is_file() or sha(original)!=r['original_json_sha256']:errors.append(f'{aid}: original JSON missing/changed')
        template=package/'annotations/templates'/f'{aid}.json'
        errors.extend(f'{aid} template: {e}' for e in validate_record(json.loads(template.read_text()),names,r))
        p=package/'annotations/completed'/f'{aid}.json'
        if p.exists():
            record=json.loads(p.read_text());completed+=1;approved+=record.get('status')=='approved'
            errors.extend(f'{aid}: {e}' for e in validate_record(record,names,r,require_approved=require_complete))
        elif require_complete:errors.append(f'{aid}: missing completed annotation')
    for p in (package/'annotations/completed').glob('*.json'):
        if p.stem not in expected:errors.append(f'Unknown completed annotation {p.name}')
    return {'images':len(master),'completed_records':completed,'approved_records':approved,'missing_completed':len(master)-completed,'errors':errors,'structural_checks_passed':not errors,'human_annotation_complete':approved==len(master) and not errors}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=Path('manual_annotation_dataset'));ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    result=validate(args.package,args.require_complete);print(json.dumps(result,indent=2));raise SystemExit(bool(result['errors']))
