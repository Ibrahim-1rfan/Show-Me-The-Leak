"""Record explicit visual review decisions and freeze hashes from original ZIP."""
from pathlib import Path
import sys,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from leakbench.common import write_json
from leakbench.dataset import prepare_subset
old=json.loads(next((ROOT/'Leak_Benchmark_Results').rglob('sample.json')).read_text())
docs=[r for r in old['images'] if r['stratum']=='document']
general=[r for r in old['images'] if r['stratum']=='general']
keep=[4,6,7,9,10,11,12,13,14,15,16,18,21,22]
replacement=json.loads((ROOT/'tmp/category_review/replacements.json').read_text())
add=[1,2,4,6,7,8,9,11,12,13,14]
document_notes=['Wallet contents with identification and payment cards','Transit ticket','Event admission ticket',
 'Music event ticket','Mailed printed material and shipping label','Event tickets','Passport identity page',
 'Passport identity page','Printed receipt','Entry fee receipt','Identity cards','Boarding pass',
 'Passport and boarding pass','Transit receipt','Student identity card','Printed betting tickets',
 'Open identity document','Open passport','Visa document','Itemized receipt','Restaurant receipt',
 'Sports admission ticket','Printed ticket','Transit ticket','Handwritten correspondence']
general_notes=['People skydiving','Crowd at an outdoor event','Building and business signage','Vehicle with visible plate',
 'Speaker and audience in a room','Cyclist outdoors','Portrait photograph','Seated audience','People at a gathering',
 'Panel discussion and audience','People at a street event','Close view of fingertips','Medical scene with people','Portrait photograph']
replacement_notes=['Storefront signage','Speaker and projected screen','People at conference table','Two people posing',
 'Portrait in low light','Vehicle in a street scene','Portrait beside window','Sport participant outdoors',
 'Sport participant in snow','Vehicle with visible plate','Outdoor crowd and stage']
entries=[]
with zipfile.ZipFile(ROOT/'train2017.zip') as z:
    groups=[('documents',docs,document_notes),('images',[general[i-1] for i in keep],general_notes),
            ('images',[replacement[i-1] for i in add],replacement_notes)]
    for category,rows,notes in groups:
        for row,note in zip(rows,notes):
            member=row['image_path'];ann_member=str(Path(member).with_suffix('.json')).replace('\\','/')
            a=json.loads(z.read(ann_member))
            entries.append({'original_id':a['id'],'category':category,'archive_member':member,
                'annotation_member':ann_member,'labels':a['labels'],
                'image_sha256':hashlib.sha256(z.read(member)).hexdigest(),
                'annotation_sha256':hashlib.sha256(z.read(ann_member)).hexdigest(),
                'review_note':note,'review_method':'assistant visual inspection of rendered contact sheet plus metadata',
                'review_date':'2026-10-20'})
selection={'version':'vispr-eval-50-v2','source_dataset':'Ibrahim1rfan/VISPR','source_split':'train2017',
 'source_url':'https://huggingface.co/datasets/Ibrahim1rfan/VISPR/resolve/main/train2017.zip',
 'selection_method':'Frozen explicit IDs after visual category review; prior cohort retained where suitable',
 'candidate_seed':421,'prior_sample':'1df1d1c206ae','changed_general_images':11,
 'category_definition':'documents: documents/cards/tickets/receipts dominate image; images: ordinary scene, person, object or signage',
 'review_scope':'Image type only; privacy labels retained from VISPR without new box/transcript ground truth',
 'images':entries}
write_json(ROOT/'configs'/'selection.json',selection)
prepare_subset(ROOT,ROOT/'configs/selection.json',ROOT/'train2017.zip')
