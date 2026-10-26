import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from leakbench.manual_annotation import validate_record,ls_box
from prepare_manual_annotation import assign
from import_manual_label_studio import convert

class ManualAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names={a['attribute_id']:a['name'] for a in json.loads((ROOT/'configs/selected_attributes.json').read_text())['attributes']}
        cls.examples=json.loads((ROOT/'docs/manual_annotation/fictional_examples.json').read_text())

    def test_four_fictional_cases(self):
        for r in self.examples:self.assertEqual(validate_record(r,self.names),[])

    def test_bad_geometry_and_duplicate_instances(self):
        for box in [[.5,.1,.2,.4],[0,0,2,1],[0,0,float('nan'),1],[0,0,0,1]]:
            r=copy.deepcopy(self.examples[1]);r['instances'][0]['bbox']=box
            self.assertTrue(validate_record(r,self.names))
        r=copy.deepcopy(self.examples[1]);r['instances']*=2
        self.assertTrue(validate_record(r,self.names))

    def test_no_false_negatives_or_implicit_approval(self):
        r=copy.deepcopy(self.examples[1]);r['contains_selected_pii']=False
        self.assertTrue(validate_record(r,self.names))
        r=copy.deepcopy(self.examples[0]);r['status']='approved'
        self.assertTrue(validate_record(r,self.names))
        self.assertTrue(validate_record(self.examples[1],self.names,require_approved=True))

    def test_unknown_labels_and_missing_localization_notes(self):
        r=copy.deepcopy(self.examples[3]);r['instances'][0]['notes']=''
        self.assertTrue(validate_record(r,self.names))
        r=copy.deepcopy(self.examples[1]);r['instances'][0]['attribute_id']='invented'
        self.assertTrue(validate_record(r,self.names))

    def test_percent_conversion(self):
        self.assertEqual(ls_box({'x':10,'y':20,'width':30,'height':40}),[.1,.2,.4,.6000000000000001])
        with self.assertRaises(ValueError):ls_box({'x':90,'y':0,'width':20,'height':10})
        with self.assertRaises(ValueError):ls_box({'x':0,'y':0,'width':20,'height':10,'rotation':90})

    def test_primary_assignment_balance_and_determinism(self):
        rows=[]
        for kind,split,n in [('general','train',672),('general','validation',168),('document','train',288),('document','validation',72)]:
            rows.extend({'id':f'{kind}_{split}_{i}','image_type':kind,'split':split,'selected_labels':['rare'] if i<6 else ['common']} for i in range(n))
        owners=assign(rows);self.assertEqual(owners,assign(rows));self.assertEqual(len(owners),1200)
        for i in range(1,4):
            subset=[r for r in rows if owners[r['id']]==f'Team Member {i}']
            self.assertEqual(len(subset),400);self.assertEqual(sum(r['image_type']=='document' for r in subset),120)
            self.assertEqual(sum(r['split']=='validation' for r in subset),80)

    def test_tool_export_is_never_gold(self):
        template=copy.deepcopy(self.examples[0]);template['status']='pending'
        results=[{'type':'choices','from_name':k,'value':{'choices':v}} for k,v in {'image_type':['general'],'quality':['clear'],'contains_pii':['no'],'full_image_checked':['checked']}.items()]
        r=convert({'annotations':[{'result':results}]},template,self.names)
        self.assertEqual(r['status'],'needs_review');self.assertEqual(r['review']['status'],'pending')
        with self.assertRaises(ValueError):convert({'annotations':[]},template,self.names)

    def test_positive_export_coordinates_and_label_consistency(self):
        template=copy.deepcopy(self.examples[0])
        results=[{'type':'choices','from_name':k,'value':{'choices':v}} for k,v in {'image_type':['document'],'quality':['clear'],'contains_pii':['yes'],'categories':['a19_name_full']}.items()]
        results += [{'type':'rectanglelabels','from_name':'regions','id':'region1','original_width':1000,'original_height':800,'value':{'x':10,'y':10,'width':30,'height':10,'rectanglelabels':['a19_name_full']}}, {'type':'choices','from_name':'visibility','id':'region1','value':{'choices':['fully_visible']}}]
        task={'annotations':[{'result':results}]};r=convert(task,template,self.names)
        self.assertEqual(r['instances'][0]['bbox'],[.1,.1,.4,.2]);self.assertFalse(r['instances'][0]['manually_verified'])
        results[-2]['original_width']=800
        with self.assertRaises(ValueError):convert(task,template,self.names)

    def test_split_and_owner_are_immutable(self):
        r=copy.deepcopy(self.examples[0]);expected={k:r[k] for k in ['annotation_id','source_id','image_file','split','sha256','width','height','annotator']}
        r['split']='validation';self.assertTrue(validate_record(r,self.names,expected))
        r['split']='train';r['annotator']='Wrong owner';self.assertTrue(validate_record(r,self.names,expected))

if __name__=='__main__':unittest.main()
