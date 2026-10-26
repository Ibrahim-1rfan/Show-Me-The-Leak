"""Audit local VISPR and prepare provisional classification targets; never trains.

Run from a source checkout: python scripts/prepare_training_data.py --root .
Image checks are cached by file metadata under ignored data/training_preparation.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps

SELECTED = {7, 8, 9, 10, 11, 19, 20, 21, 24, 29, 30, 31, 32, 33,
            35, 37, 38, 49, 73, 74, 75, 78, 79, 82, 85, 90, 103, 104}
DIRECT = {19,20,21,29,30,31,32,33,49,78,85,90,103}
VISUAL = {7,8,9,10,11}

def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(p)

def csv_write(p, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def describe(a):
    n=int(a['attribute_id'].split('_')[0][1:]); selected=n in SELECTED
    kind=('direct identifier' if n in DIRECT else 'biometric/identifying visual feature' if n in VISUAL
          else 'quasi-identifier' if n in {24,73,74,75,79,82,104} else 'sensitive/contextual attribute')
    method='specialized detector + VLM review' if n in VISUAL else 'VLM context + OCR' if selected else 'VLM/context review'
    if selected:
        reason=('Direct identity/contact/financial exposure; retain despite limited support.' if n in DIRECT else
                'Visual identifying feature not recoverable by OCR alone; spatial review required.' if n in VISUAL else
                'Location/time linkage or document context complements direct identifiers; context alone does not identify a text field.')
    elif n in {1,2,3,4,5,6,16,17,18,25,27}:
        reason='Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference.'
    elif n in {23,26,102}:
        reason='Useful quasi-identifier, deferred for narrower identity/contact coverage within 28 labels; handwriting is not equivalent to a signature.'
    elif n in {12,13,39,41,43,55,56,62,92,97,99}:
        reason='Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe.'
    else:
        reason='Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable.'
    return kind,method,reason

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--seed',type=int,default=421)
    args=ap.parse_args();root=args.root.resolve();out=root/'data/training_preparation';out.mkdir(parents=True,exist_ok=True)
    public=root/'docs/attribute_selection';public.mkdir(parents=True,exist_ok=True)
    attrs=list(csv.DictReader((root/'attributes.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
    text=(root/'Images Classification.txt').read_text(encoding='utf-8-sig')
    reported={m[2]:int(m[0]) for m in re.findall(r'(\d+)\s+\(\s*([\d.]+)%\)\s+(a\d+_\w+)',text)}
    ids={a['attribute_id'] for a in attrs}; annotations=[];counts=Counter();issues=[]
    for p in sorted((root/'train2017').rglob('*.json')):
        try:
            d=json.loads(p.read_text(encoding='utf-8'));labels=d['labels']
            if not isinstance(labels,list) or not set(labels)<=ids:raise ValueError('invalid labels')
            if len(set(labels))!=len(labels):issues.append({'id':d['id'],'reason':'duplicate annotation labels'})
            counts.update(set(labels));annotations.append((p,d))
        except Exception as e:issues.append({'path':str(p.relative_to(root)),'reason':str(e)})
    ranking={k:i+1 for i,(k,v) in enumerate(sorted(((a['attribute_id'],counts[a['attribute_id']]) for a in attrs if a['attribute_id']!='a0_safe'),key=lambda x:(-x[1],x[0])))}
    analysis=[]
    for a in attrs:
        aid=a['attribute_id']
        if aid=='a0_safe':continue
        kind,method,reason=describe(a);n=counts[aid]
        analysis.append(dict(attribute_id=aid,name=a['description'],description=a['description'],category=a['group'],privacy_type=kind,count=n,percent=round(100*n/len(annotations),4),rank=ranking[aid],reported_count=reported.get(aid),selected=int(aid.split('_')[0][1:]) in SELECTED,method=method,reason=reason,support_note='scarce: fewer than 50; targeted sampling and reviewed test support needed' if n<50 else 'limited: fewer than 150; inspect per-split support' if n<150 else 'larger pool; still check split support'))
    selected=[a for a in analysis if a['selected']];assert len(selected)==28
    assert len(analysis)==67 and len({a['attribute_id'] for a in analysis})==67
    write_json(root/'configs/selected_attributes.json',{'version':'vispr-28-v1','training_enabled':False,'attributes':selected})
    write_json(public/'all_attributes.json',analysis);csv_write(public/'all_attributes.csv',analysis);csv_write(public/'selected_attributes.csv',selected)
    print('Reconstructed all attribute counts; auditing image integrity with resumable cache.',flush=True)
    image_index=defaultdict(list)
    for p in (root/'train2017').rglob('*'):
        if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:image_index[p.stem].append(p)
    cachefile=out/'image_checks.json';cache=json.loads(cachefile.read_text()) if cachefile.exists() else {}
    rows=[]; selected_ids={a['attribute_id'] for a in selected};doc_ids={a['attribute_id'] for a in attrs if a['group']=='Documents'}
    frozen=json.loads((root/'configs/selection.json').read_text())['images'];excluded={x['original_id'] for x in frozen}
    from concurrent.futures import ThreadPoolExecutor
    def audit_one(item):
        p,d=item;images=image_index.get(d['id'],[])
        if len(images)!=1:return None, None
        image=images[0];stat=image.stat();key=f'{image.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}'
        if key in cache:return key,cache[key]
        try:
            h=hashlib.sha256()
            with image.open('rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            with Image.open(image) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.load()
                pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                small=list(im.convert('L').resize((9,8)).getdata());bits=''.join('1' if small[y*9+x]>small[y*9+x+1] else '0' for y in range(8) for x in range(8))
            result={'sha256':h.hexdigest(),'pixel_sha256':pixel,'dhash':f'{int(bits,2):016x}'}
        except Exception as e:result={'error':f'{type(e).__name__}: {e}'}
        return key,result
    executor=ThreadPoolExecutor(max_workers=8)
    checked=executor.map(audit_one,annotations)
    for i,((p,d),(key,result)) in enumerate(zip(annotations,checked)):
        aid=d['id'];images=image_index.get(aid,[])
        if key is None:issues.append({'id':aid,'reason':f'expected one image, found {len(images)}'});continue
        image=images[0];cache[key]=result
        check=cache[key]
        if 'error' in check:issues.append({'id':aid,'reason':check['error']});continue
        labels=sorted(set(d['labels']));rows.append(dict(id=aid,image_path=image.relative_to(root).as_posix(),annotation_path=p.relative_to(root).as_posix(),original_labels=labels,selected_labels=sorted(set(labels)&selected_ids),label_source='original_vispr_image_labels',regions=None,region_status='not_annotated',image_type='document' if set(labels)&doc_ids else 'general',image_type_status='metadata_proxy_requires_review',safe_conflict='a0_safe' in labels and len(labels)>1,**check))
        if (i+1)%250==0:write_json(cachefile,cache);print(f'Checked {i+1}/{len(annotations)} annotations',flush=True)
    write_json(cachefile,cache)
    # Connected components across exact bytes, decoded pixels and conservative dHash equality.
    parent={r['id']:r['id'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for r in rows:
        for key in ['sha256','pixel_sha256','dhash']:
            pair=(key,r[key])
            if pair in seen:parent[find(r['id'])]=find(seen[pair])
            else:seen[pair]=r['id']
    blocked={find(x) for x in excluded if x in parent};groups=defaultdict(list)
    for r in rows:groups[find(r['id'])].append(r)
    rng=random.Random(args.seed);candidates=[]
    for gid,members in sorted(groups.items()):
        if gid in blocked:continue
        r=rng.choice(members)
        if not r['safe_conflict']:r['duplicate_group']=gid;candidates.append(r)
    rng.shuffle(candidates);chosen=[];coverage=Counter()
    # Explicit safe negatives prevent the coverage heuristic selecting positives only.
    negatives=[r for r in candidates if r['original_labels']==['a0_safe'] and r['image_type']=='general'][:120]
    chosen.extend(negatives);negative_ids={r['id'] for r in negatives}
    # Deterministic inverse-frequency coverage preference; no synthetic duplicates.
    for category,quota in [('document',360),('general',840-len(negatives))]:
        pool=[r for r in candidates if r['image_type']==category and r['id'] not in negative_ids]
        for _ in range(min(quota,len(pool))):
            best=max(range(len(pool)),key=lambda j:sum(1/(1+coverage[x]) for x in pool[j]['selected_labels']))
            r=pool.pop(best);chosen.append(r);coverage.update(r['selected_labels'])
    # Keep one representative per duplicate group; split within each image type.
    for category in ['document','general']:
        part=[r for r in chosen if r['image_type']==category];rng.shuffle(part)
        cut=round(len(part)*.8)
        for i,r in enumerate(part):r['split']='train' if i<cut else 'validation'
    for r in chosen:r['split_status']='provisional_pending_near_duplicate_and_type_review'
    with (out/'classification_targets.jsonl').open('w',encoding='utf-8') as f:
        for r in sorted(chosen,key=lambda x:x['id']):f.write(json.dumps(r)+'\n')
    csv_write(out/'manifest.csv',[{k:'|'.join(v) if isinstance(v,list) else v for k,v in r.items()} for r in chosen])
    csv_write(out/'review_queue.csv',[{'id':r['id'],'image_path':r['image_path'],'proposed_type':r['image_type'],'original_selected_labels':'|'.join(r['selected_labels']),'reviewed_type':'','reviewed_selected_labels':'','review_status':'pending','reviewer':'','notes':''} for r in chosen])
    write_json(out/'all_images.json',rows);write_json(out/'issues.json',issues)
    write_json(out/'duplicate_groups.json',{k:[r['id'] for r in v] for k,v in groups.items() if len(v)>1})
    for a in analysis:
        a['selected_train_count']=sum(a['attribute_id'] in r['original_labels'] and r['split']=='train' for r in chosen)
        a['selected_validation_count']=sum(a['attribute_id'] in r['original_labels'] and r['split']=='validation' for r in chosen)
    write_json(public/'all_attributes.json',analysis);csv_write(public/'all_attributes.csv',analysis)
    overlap=[{'a':a,'b':b,'images':sum(a in r['original_labels'] and b in r['original_labels'] for r in rows)} for a in sorted(selected_ids) for b in sorted(selected_ids) if a<b]
    csv_write(public/'selected_overlap.csv',overlap)
    summary=dict(annotation_count=len(annotations),usable_images=len(rows),issues=len(issues),safe_conflicts=sum(r['safe_conflict'] for r in rows),selected_count=len(chosen),composition=dict(Counter(r['image_type'] for r in chosen)),splits=dict(Counter(r['split'] for r in chosen)),seed=args.seed,excluded_benchmark_ids=len(excluded),blocked_duplicate_groups=len(blocked),duplicate_groups=sum(len(v)>1 for v in groups.values()),near_duplicate_review_complete=False,training_ready=False,reported_total=9999,report_count_disagreements={a['attribute_id']:{'reported':a['reported_count'],'reconstructed':a['count']} for a in analysis if a['reported_count']!=a['count']},input_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['attributes.tsv','Images Classification.txt','configs/selection.json']})
    write_json(public/'summary.json',summary)
    assert excluded <= set(parent), 'A frozen benchmark image is unavailable; exclusions cannot be fully verified'
    assert not ({r['id'] for r in chosen}&excluded)
    assert len({r['duplicate_group'] for r in chosen})==len(chosen)
    assert len(chosen)==1200, f'Insufficient eligible quota pools; inspect saved summary: {summary}'
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
