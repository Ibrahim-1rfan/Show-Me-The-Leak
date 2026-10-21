"""Frozen visually reviewed cohort; byte-preserving materialization and checks."""
import csv, hashlib, io, json, shutil, urllib.request, urllib.error, zipfile, re, time, http.client
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from .common import fingerprint, read_json, write_json, sha, log, stage

SOURCE_URL='https://huggingface.co/datasets/Ibrahim1rfan/VISPR/resolve/831e0de941ddf741fa00217c9d78d91b57c9af93/train2017.zip'
FORMATS={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp','BMP':'.bmp','GIF':'.gif','MPO':'.mpo'}

def download_archive(destination,url=SOURCE_URL,attempts=3):
    from tqdm.auto import tqdm
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if not zipfile.is_zipfile(destination):raise ValueError('Cached archive is invalid; inspect it before replacing.')
        log('Reusing dataset archive',path=str(destination));return destination
    part=destination.with_suffix('.zip.part');meta=destination.with_suffix('.download.json')
    log('Dataset source',dataset='Ibrahim1rfan/VISPR',representation='ZIP of train2017 image/JSON pairs; not a datasets table',url=url)
    for attempt in range(1,attempts+1):
        offset=part.stat().st_size if part.exists() else 0
        previous=read_json(meta) if meta.exists() else {}
        if offset and previous.get('url')!=url:raise ValueError('Unverified partial download source; retain file and choose a fresh cache path')
        headers={'Range':f'bytes={offset}-'} if offset else {}
        if offset and previous.get('etag'):headers['If-Range']=previous['etag']
        try:
            log('Dataset connect/download',attempt=attempt,resume_bytes=offset)
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                append=offset>0 and response.status==206
                if offset and not append:raise ValueError('Server did not honor resume range; partial file retained, use a fresh cache path')
                length=response.headers.get('Content-Length');total=offset+int(length) if length else None
                if append:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset:raise ValueError('Invalid download Content-Range; refusing mixed archive')
                    total=int(match[3])
                    if previous.get('total') and total!=previous['total']:raise ValueError('Source size changed during resume')
                    if previous.get('etag') and response.headers.get('ETag')!=previous['etag']:raise ValueError('Source ETag changed during resume')
                if total and shutil.disk_usage(destination.parent).free < total-offset+200*1024**2:
                    raise OSError('Insufficient disk for the full VISPR ZIP plus subset; upload the verified vispr_eval_50.zip instead')
                write_json(meta,{'url':url,'etag':response.headers.get('ETag'),'total':total,'status':'downloading'})
                with part.open('ab' if append else 'wb') as f,tqdm(total=total,initial=offset,unit='B',unit_scale=True,desc='VISPR download') as bar:
                    while block:=response.read(1024*1024):f.write(block);bar.update(len(block))
                if total and part.stat().st_size!=total:raise ConnectionError('Incomplete dataset transfer; cached bytes retained')
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (401,403):raise RuntimeError(f'Dataset access denied ({exc.code}); confirm repository access. No alternate dataset used.') from exc
            if exc.code==404:raise RuntimeError('Dataset revision or archive path not found; no alternate dataset used') from exc
            if exc.code not in (429,500,502,503,504) or attempt==attempts:raise
            log('Dataset HTTP retry',status=exc.code,attempt=attempt);time.sleep(5*attempt)
        except (urllib.error.URLError,TimeoutError,ConnectionError,http.client.IncompleteRead) as exc:
            if attempt==attempts:raise RuntimeError(f'Dataset network/transfer stage failed after {attempt} attempts; partial cache retained: {type(exc).__name__}') from exc
            log('Dataset network retry',attempt=attempt,error_type=type(exc).__name__);time.sleep(5*attempt)
    if not zipfile.is_zipfile(part):raise ValueError('Download is not a ZIP archive; incomplete file retained for inspection.')
    part.replace(destination);return destination

def image_info(data):
    with Image.open(io.BytesIO(data)) as im:
        fmt=im.format;im.load();raw_size=im.size
        upright=ImageOps.exif_transpose(im).convert('RGB')
        pixel_sha=hashlib.sha256(str(upright.size).encode()+upright.tobytes()).hexdigest()
        small=upright.convert('L').resize((9,8));px=list(small.getdata())
        dhash=sum((px[y*9+x]>px[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
        return {'format':fmt,'extension':FORMATS[fmt],'width':upright.width,'height':upright.height,
                'source_width':raw_size[0],'source_height':raw_size[1], 'pixel_sha256':pixel_sha,'dhash':f'{dhash:016x}'}

def prepare_subset(project, selection_path, archive=None):
    from tqdm.auto import tqdm
    project=Path(project);selection=read_json(selection_path);folder=project/'data'/'vispr_eval_50'
    if (folder/'manifest.csv').exists():return validate_subset(folder,selection)
    if Counter(r['category'] for r in selection['images'])!={'documents':25,'images':25}:
        raise ValueError('Selection must contain exactly 25 reviewed images per category.')
    archive=Path(archive) if archive else download_archive(project/'data'/'cache'/'train2017.zip')
    counters=Counter();rows=[]
    with zipfile.ZipFile(archive) as z:
        for item in tqdm(selection['images'],desc='Materialize reviewed images',unit='image'):
            category=item['category'];counters[category]+=1
            data=z.read(item['archive_member'])
            if hashlib.sha256(data).hexdigest()!=item['image_sha256']:raise ValueError('Source image changed: '+item['original_id'])
            info=image_info(data);stem=('document_' if category=='documents' else 'image_')+str(counters[category])
            relative=f'{category}/{stem}{info["extension"]}'
            target=folder/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=item['image_sha256']:raise ValueError('Partial subset contains changed file: '+relative)
            target.write_bytes(data)
            ann_data=z.read(item['annotation_member'])
            if hashlib.sha256(ann_data).hexdigest()!=item['annotation_sha256']:raise ValueError('Source annotation changed')
            annotation=folder/'annotations'/f'{stem}.json';annotation.parent.mkdir(exist_ok=True)
            annotation.write_bytes(ann_data)
            rows.append({'filename':relative,'category':category,'original_id':item['original_id'],
                'original_filename':Path(item['archive_member']).name,'source_split':'train2017',
                'source_dataset':selection['source_dataset'],'archive_member':item['archive_member'],
                'image_sha256':item['image_sha256'],'annotation_sha256':item['annotation_sha256'],
                'annotation_reference':f'annotations/{stem}.json','labels':json.dumps(item['labels']),
                'visual_review':item['review_note'],**info})
    with (folder/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(folder/'selection_config.json',selection)
    return validate_subset(folder,selection)

def validate_subset(folder,selection=None):
    from tqdm.auto import tqdm
    folder=Path(folder)
    with (folder/'manifest.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    frozen=read_json(folder/'selection_config.json')
    if selection is not None and fingerprint(selection)!=fingerprint(frozen):raise ValueError('Frozen selection changed; create an explicit new dataset version.')
    if len(rows)!=50 or Counter(r['category'] for r in rows)!={'documents':25,'images':25}:raise ValueError('Invalid cohort counts')
    for key in ('original_id','filename','image_sha256','pixel_sha256'):
        if len({r[key] for r in rows})!=50:raise ValueError('Duplicate cohort '+key)
    selected={r['original_id']:r for r in frozen['images']}
    for category in ('documents','images'):
        actual=[p for p in (folder/category).iterdir() if p.is_file()]
        if len(actual)!=25:raise ValueError('Unexpected files in category: '+category)
        prefix='document_' if category=='documents' else 'image_'
        if {p.stem for p in actual}!={prefix+str(i) for i in range(1,26)}:raise ValueError('Sequential filenames changed')
    near=[]
    for row in tqdm(rows,desc='Validate fixed cohort',unit='image'):
        expected=selected[row['original_id']]
        if any(row[key]!=expected[key] for key in ('category','image_sha256','annotation_sha256','archive_member')):raise ValueError('Manifest changed')
        for key in ('filename','annotation_reference'):
            (folder/row[key]).resolve().relative_to(folder.resolve())
        if sha(folder/row['filename'])!=expected['image_sha256']:raise ValueError('Image checksum mismatch')
        if sha(folder/row['annotation_reference'])!=expected['annotation_sha256']:raise ValueError('Annotation checksum mismatch')
        ann=read_json(folder/row['annotation_reference'])
        if ann['id']!=row['original_id'] or sorted(ann['labels'])!=sorted(expected['labels']):raise ValueError('Annotation mapping mismatch')
        if sorted(json.loads(row['labels']))!=sorted(expected['labels']):raise ValueError('Manifest labels changed')
        info=image_info((folder/row['filename']).read_bytes())
        if any(str(info[k])!=str(row[k]) for k in info):raise ValueError('Image metadata mismatch')
        if Path(row['filename']).suffix!=info['extension']:raise ValueError('Image format and extension mismatch')
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance<=4:near.append([a['original_id'],b['original_id'],distance])
    allowed={frozenset(p) for p in frozen.get('reviewed_distinct_near_pairs',[])}
    if any(frozenset(p[:2]) not in allowed for p in near):raise ValueError('Near duplicates require visual review: '+str(near))
    result={'status':'validated','count':50,'documents':25,'general':25,
            'selection_fingerprint':fingerprint(frozen),'near_duplicate_candidates':near,
            'annotation_scope':'image labels only; no verified reference boxes or transcripts'}
    write_json(folder/'validation.json',result);log('Dataset validated',**result)
    return folder

def records(folder):
    folder=Path(folder)
    with (folder/'manifest.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    return [dict(id=r['original_id'],filename=r['filename'],category=r['category'],
            stratum='document' if r['category']=='documents' else 'general',
            image_path=folder/r['filename'],annotation_path=folder/r['annotation_reference'],
            labels=json.loads(r['labels']),image_sha256=r['image_sha256'],annotation_sha256=r['annotation_sha256']) for r in rows]
