"""Convert local Label Studio JSON exports to canonical NEEDS_REVIEW records.

Requires one submitted annotation per task. Never overwrites a canonical record.
Raw export is retained by hash. Run on one primary member's export at a time.
"""
import argparse,copy,datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from leakbench.manual_annotation import validate_record,ls_box
from prepare_manual_annotation import sha

def convert(task,template,names):
    annotations=task.get('annotations',[])
    if len(annotations)!=1 or annotations[0].get('was_cancelled'):raise ValueError('Require exactly one submitted, non-cancelled annotation per task')
    results=annotations[0]['result'];fields={};instances=[]
    for x in results:
        if x['type']=='choices' and x['from_name']!='visibility':fields[x['from_name']]=x['value']['choices']
        if x['type']=='textarea':fields[x['from_name']]=x['value']['text']
    def one(k):
        v=fields.get(k,[])
        if len(v)!=1:raise ValueError('Select exactly one '+k)
        return v[0]
    r=copy.deepcopy(template);r.update(status='needs_review',image_type=one('image_type'),quality=one('quality'),contains_selected_pii={'yes':True,'no':False,'uncertain':None}[one('contains_pii')],selected_categories=fields.get('categories',[]),notes='\n'.join(fields.get('notes',[])),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_image_checked=fields.get('full_image_checked')==['checked'])
    for x in results:
        if x['type']!='rectanglelabels':continue
        if x.get('image_rotation',0)!=0:raise ValueError('Image rotation unsupported')
        if (x.get('original_width'),x.get('original_height'))!=(r['width'],r['height']):raise ValueError('Coordinate dimensions mismatch; check EXIF upright view')
        labels=x['value']['rectanglelabels']
        if len(labels)!=1:raise ValueError('One label per region required')
        aid=labels[0]
        if aid not in names:raise ValueError('Unsupported attribute '+aid)
        visibility=[z['value']['choices'] for z in results if z.get('id')==x['id'] and z.get('from_name')=='visibility']
        if len(visibility)!=1 or len(visibility[0])!=1:raise ValueError('Region visibility missing')
        instances.append({'instance_id':x['id'],'attribute_id':aid,'attribute_name':names[aid],'bbox':ls_box(x['value']),'localization':'localized','visibility':visibility[0][0],'manually_verified':False,'notes':''})
    # A present category with no box stays explicitly unlocalized, never dropped.
    for aid in set(r['selected_categories'])-{i['attribute_id'] for i in instances}:
        if aid not in names or not r['notes'].strip():raise ValueError('Unlocalized category requires supported ID and explanatory notes')
        instances.append({'instance_id':'unlocalized_'+aid,'attribute_id':aid,'attribute_name':names[aid],'bbox':None,'localization':'unlocalizable','visibility':'uncertain','manually_verified':False,'notes':r['notes']})
    r['instances']=instances
    errors=validate_record(r,names)
    if errors:raise ValueError('; '.join(errors))
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('export',type=Path);ap.add_argument('--package',type=Path,default=Path('manual_annotation_dataset'));ap.add_argument('--member',type=int,choices=[1,2,3],required=True);args=ap.parse_args();p=args.package
    names={a['attribute_id']:a['name'] for a in json.loads((p/'manifests/selected_attributes.json').read_text())['attributes']}
    master={r['annotation_id']:r for r in json.loads((p/'manifests/master.json').read_text())};pending={}
    for task in json.loads(args.export.read_text(encoding='utf-8-sig')):
        aid=task['data']['annotation_id']
        if aid not in master or master[aid]['annotator']!=f'Team Member {args.member}':raise ValueError('Unknown task or wrong primary owner')
        dest=p/'annotations/completed'/f'{aid}.json'
        if dest.exists() or aid in pending:raise ValueError('Existing/duplicate record; preserve it and resolve manually')
        template=json.loads((p/'annotations/templates'/f'{aid}.json').read_text())
        pending[aid]=convert(task,template,names)
    if not pending:raise ValueError('Export contains no submitted tasks')
    raw=p/'annotations/raw_exports'/f'{sha(args.export)}.json'
    if not raw.exists():raw.write_bytes(args.export.read_bytes())
    for aid,r in pending.items():(p/'annotations/completed'/f'{aid}.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(f'Imported {len(pending)} NEEDS_REVIEW records. None approved; validate and cross-review before use.')

if __name__=='__main__':main()
