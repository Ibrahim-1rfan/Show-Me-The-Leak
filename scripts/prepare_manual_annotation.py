"""Copy the frozen selected dataset into a non-overwriting manual annotation package."""
import csv, hashlib, json, random, shutil, sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manual_annotation_dataset'
SEED=421

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')

def table(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def assign(rows):
    """Exact type/split quotas, greedy balancing of rare-label positive counts."""
    rng=random.Random(SEED);result={};support=Counter(x for r in rows for x in r['selected_labels'])
    counts=[Counter() for _ in range(3)]
    for kind,split in [('document','train'),('document','validation'),('general','train'),('general','validation')]:
        pool=[r for r in rows if r['image_type']==kind and r['split']==split];rng.shuffle(pool)
        pool.sort(key=lambda r:sum(1/support[x] for x in r['selected_labels']),reverse=True)
        capacities=[len(pool)//3+(i<len(pool)%3) for i in range(3)]
        for r in pool:
            options=[i for i in range(3) if capacities[i]];rng.shuffle(options)
            i=min(options,key=lambda i:sum(counts[i][x]/support[x] for x in r['selected_labels']))
            result[r['id']]=f'Team Member {i+1}';capacities[i]-=1;counts[i].update(r['selected_labels'])
    return result

def improve_pilot(pilot,master):
    """Increase label coverage while preserving owner/type quotas and training-only status."""
    def labels(rows):return {x for r in rows for x in r['expected_selected_categories'].split('|') if x}
    while True:
        current=len(labels(pilot));best=None;score=current;ids={r['annotation_id'] for r in pilot}
        for candidate in master:
            if candidate['split']!='train' or candidate['annotation_id'] in ids:continue
            for i,old in enumerate(pilot):
                if (old['annotator'],old['image_type'])!=(candidate['annotator'],candidate['image_type']):continue
                trial=pilot[:i]+[candidate]+pilot[i+1:];n=len(labels(trial))
                if n>score:score=n;best=trial
        if best is None:return pilot
        pilot=best

def main():
    if OUT.exists():raise SystemExit('Destination exists; preserve annotation work. Choose a separately versioned package, do not overwrite.')
    rows=[json.loads(s) for s in (ROOT/'data/training_preparation/classification_targets.jsonl').read_text().splitlines()]
    csvrows=list(csv.DictReader((ROOT/'data/training_preparation/manifest.csv').open(encoding='utf-8')))
    assert len(rows)==1200 and len({r['id'] for r in rows})==1200
    assert {r['id'] for r in rows}=={r['id'] for r in csvrows}
    lookup={r['id']:r for r in csvrows}
    for r in rows:
        for k in ['image_path','annotation_path','split','sha256']:assert r[k]==lookup[r['id']][k],(r['id'],k)
    frozen={r['original_id'] for r in json.loads((ROOT/'configs/selection.json').read_text())['images']}
    assert not frozen & {r['id'] for r in rows}
    attrs=json.loads((ROOT/'configs/selected_attributes.json').read_text())['attributes'];names={a['attribute_id']:a['name'] for a in attrs}
    assert len(names)==28
    for r in rows:
        assert r['split'] in {'train','validation'} and set(r['selected_labels'])<=names.keys()
        assert (ROOT/r['image_path']).is_file() and (ROOT/r['annotation_path']).is_file()
    owners=assign(rows)
    for sub in ['images','annotations/templates','annotations/completed','annotations/original_vispr','annotations/raw_exports','annotations/reviews','assignments','guidelines','manifests','annotation_views']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    master=[];orientations=0
    for number,r in enumerate(sorted(rows,key=lambda r:r['id']),1):
        source=ROOT/r['image_path'];dest=OUT/'images'/source.name
        if dest.exists():raise ValueError('Filename collision')
        if sha(source)!=r['sha256']:raise ValueError('Source changed: '+r['id'])
        shutil.copy2(source,dest);assert sha(dest)==r['sha256']
        original=ROOT/r['annotation_path'];backup=OUT/'annotations/original_vispr'/original.name;shutil.copy2(original,backup);assert sha(original)==sha(backup)
        with Image.open(dest) as im:
            orientation=im.getexif().get(274,1);upright=ImageOps.exif_transpose(im);width,height=upright.size
            if orientation!=1:
                view=OUT/'annotation_views'/(r['id']+'.png');upright.convert('RGB').save(view);orientations+=1
            else:view=dest
        aid='vispr_'+r['id'];row=dict(annotation_id=aid,source_id=r['id'],image_file=dest.relative_to(OUT).as_posix(),annotation_view=view.relative_to(OUT).as_posix(),source_path=r['image_path'],original_annotation=r['annotation_path'],original_json_sha256=sha(original),sha256=r['sha256'],view_sha256=sha(view),width=width,height=height,split=r['split'],split_status=r['split_status'],image_type=r['image_type'],image_type_status=r['image_type_status'],original_labels='|'.join(r['original_labels']),expected_selected_categories='|'.join(r['selected_labels']),annotation_status='pending',annotator=owners[r['id']],review_status='pending',duplicate_group=r['duplicate_group'])
        master.append(row)
        template={k:row[k] for k in ['annotation_id','source_id','image_file','sha256','width','height','split','annotator']}
        template.update(schema_version='vispr-manual-v1',coordinate_frame='exif_upright_normalized_xyxy',status='pending',image_type=None,quality=None,contains_selected_pii=None,selected_categories=[],instances=[],full_image_checked=False,updated_at=None,notes='',review={'status':'pending','reviewer':None,'reviewed_at':None})
        save(OUT/'annotations/templates'/f'{aid}.json',template)
        if number%100==0:print(f'Copied and hash-verified {number}/1200',flush=True)
    table(OUT/'manifests/master.csv',master);save(OUT/'manifests/master.json',master)
    shutil.copy2(ROOT/'configs/selected_attributes.json',OUT/'manifests/selected_attributes.json')
    shutil.copy2(ROOT/'attributes.tsv',OUT/'manifests/attributes.tsv')
    shutil.copy2(ROOT/'Images Classification.txt',OUT/'manifests/Images Classification.txt')
    shutil.copy2(ROOT/'data/training_preparation/manifest.csv',OUT/'manifests/selected_source_manifest.csv')
    shutil.copy2(ROOT/'configs/selection.json',OUT/'manifests/excluded_benchmark.json')
    distributions=[];rng=random.Random(SEED)
    def tasks(batch):return [{'data':{'image':'/data/local-files/?d='+quote(r['annotation_view']),'annotation_id':r['annotation_id']}} for r in batch]
    pilot=[];reviews=[]
    for i in range(1,4):
        member=f'Team Member {i}';batch=[r for r in master if r['annotator']==member]
        table(OUT/f'assignments/member_{i}.csv',batch);save(OUT/f'assignments/member_{i}_label_studio.json',tasks(batch))
        distributions.append({'annotator':member,'images':len(batch),'types':dict(Counter(r['image_type'] for r in batch)),'splits':dict(Counter(r['split'] for r in batch)),'selected_label_counts':dict(Counter(x for r in batch for x in r['expected_selected_categories'].split('|') if x))})
        # Nine pilot images per member, all from training; calibration never consumes validation examples.
        for kind,n in [('document',4),('general',5)]:
            pool=[r for r in batch if r['image_type']==kind and r['split']=='train'];rng.shuffle(pool)
            pilot.extend(pool[:n])
        for kind,n in [('document',12),('general',28)]:
            pool=[r for r in batch if r['image_type']==kind];rng.shuffle(pool)
            for r in pool[:n]:reviews.append({'annotation_id':r['annotation_id'],'primary':member,'reviewer':f'Team Member {i%3+1}','review_status':'pending'})
    pilot=improve_pilot(pilot,master)
    table(OUT/'assignments/pilot_shared.csv',pilot);save(OUT/'assignments/pilot_label_studio.json',tasks(pilot));table(OUT/'assignments/cross_review.csv',reviews)
    for p in (ROOT/'docs/manual_annotation').iterdir():
        if p.is_file():shutil.copy2(p,OUT/'guidelines'/p.name)
    shutil.copy2(ROOT/'docs/manual_annotation/START_HERE.md',OUT/'README.md')
    summary={'seed':SEED,'images':len(master),'attributes':28,'primary_assignments':distributions,'pilot_shared_images':len(pilot),'cross_review_images':len(reviews),'upright_derivatives':orientations,'completed_annotations':0,'approved_annotations':0,'source_manifest_sha256':sha(ROOT/'data/training_preparation/manifest.csv'),'selected_attributes_sha256':sha(ROOT/'configs/selected_attributes.json'),'originals_modified':False}
    save(OUT/'manifests/package_summary.json',summary);save(ROOT/'docs/manual_annotation/package_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
