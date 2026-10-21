import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image,ImageChops
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from leakbench.pipeline import detect_pii,load_taxonomy,redact_and_draw,read_image
from leakbench.validation import validate_redaction
from leakbench.preprocessing import tile_boxes,map_tile_region,tiled_extract
from leakbench.downloads import snapshot_with_retry
from leakbench.gates import record_pilot_review,require_review,pilot_records
from leakbench.adapters import FlorenceAdapter
from leakbench.common import write_json,sha
from leakbench.dataset import download_archive

class RegressionRepair(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.tax=load_taxonomy(ROOT/'configs/taxonomy_crosswalk.csv')
        self.image=Image.new('RGB',(200,100),'white');self.original=self.folder/'original.png';self.image.save(self.original)
        self.det={'attribute_id':'a19_name_full','bbox_1000':[100,200,400,600],'location':'field','evidence_source':'explicit_text'}

    def test_bad_entry_cannot_erase_valid_detection_with_empty_retry(self):
        bad={**self.det,'attribute_id':'a10_face_complete'};calls=[]
        def generate(im,prompt):
            calls.append(prompt)
            return (json.dumps({'detections':[self.det,bad]}) if len(calls)==1 else '{"detections":[]}',False)
        result=detect_pii(self.image,self.tax,generate)
        self.assertEqual(result['status'],'partial');self.assertEqual(result['detections'],[self.det]);self.assertEqual(len(calls),1)
        self.assertEqual(len(result['rejected_entries']),1)

    def test_empty_positive_is_not_verified_redaction(self):
        out=self.folder/'redacted.png';self.image.save(out)
        check,mask=validate_redaction(self.original,out,[],['a19_name_full'])
        self.assertFalse(check['technical_pass']);self.assertFalse(check['privacy_verified'])
        self.assertIn('known_positive_image_unchanged',check['warnings'])
        safe,_=validate_redaction(self.original,out,[],['a0_safe'])
        self.assertTrue(safe['technical_pass']);self.assertNotIn('known_positive_image_unchanged',safe['warnings'])

    def test_saved_mask_is_verified_and_overlay_is_rejected(self):
        red,overlay,dets=redact_and_draw(self.image,[self.det],.01);out=self.folder/'redacted.png';red.save(out)
        check,mask=validate_redaction(self.original,out,dets,['a19_name_full'],expected_sha256=sha(self.original))
        self.assertTrue(check['technical_pass']);self.assertGreater(check['mask_pixels'],0);self.assertGreater(check['changed_pixels'],0)
        self.assertFalse(check['privacy_verified'])
        overlay.save(out);check,_=validate_redaction(self.original,out,dets,['a19_name_full'])
        self.assertIn('saved_pixels_do_not_match_mask',check['issues'])

    def test_invalid_boxes_are_not_silently_clipped(self):
        for box in [[-1,0,100,100],[0,0,0,2],[0,0,float('nan'),10]]:
            with self.assertRaises(ValueError):redact_and_draw(self.image,[{**self.det,'ocr_bbox_pixels':box}])

    def test_original_and_saved_integrity_and_reference_coverage(self):
        red,_,dets=redact_and_draw(self.image,[self.det],0);out=self.folder/'redacted.png';red.save(out)
        check,_=validate_redaction(self.original,out,dets,['a19_name_full'],reference_boxes=[[20,20,80,60]])
        self.assertEqual(check['reference_region_coverage'],1);self.assertTrue(check['privacy_verified'])
        check,_=validate_redaction(self.original,out,dets,['a19_name_full'],expected_sha256='wrong')
        self.assertIn('input_hash_mismatch',check['issues'])
        out.write_bytes(b'corrupt');check,_=validate_redaction(self.original,out,dets,['a19_name_full'])
        self.assertIn('saved_output_missing_or_unreadable',check['issues'])

    def test_tiles_cover_entire_image_and_map_coordinates(self):
        size=(2048,1536);boxes=tile_boxes(size);mask=Image.new('L',size)
        from PIL import ImageDraw
        draw=ImageDraw.Draw(mask)
        for l,t,r,b in boxes:draw.rectangle((l,t,r-1,b-1),fill=255)
        self.assertEqual(mask.histogram()[255],size[0]*size[1])
        r=map_tile_region({'text':'test','bbox_pixels':[10,20,30,40]},(500,200,1524,1224))
        self.assertEqual(r['bbox_pixels'],[510,220,530,240])
        with self.assertRaises(ValueError):tile_boxes((10000,10000),max_tiles=4)

    def test_one_failed_tile_keeps_whole_image_incomplete(self):
        def extract(im):return {'text_regions':[],'parse_success':False,'metadata':{'truncated':True}}
        with contextlib.redirect_stdout(io.StringIO()):result=tiled_extract(extract,Image.new('RGB',(1500,1024)))
        self.assertFalse(result['parse_success']);self.assertEqual(result['metadata']['tile_count'],2)

    def test_download_retries_only_transient_failure(self):
        calls=[]
        def flaky(**kw):
            calls.append(kw)
            if len(calls)<2:raise ConnectionError('HTTP 429 Too Many Requests')
            return 'cached'
        with contextlib.redirect_stdout(io.StringIO()):result=snapshot_with_retry(flaky,sleep=lambda _:None,repo_id='example')
        self.assertEqual(result,'cached');self.assertEqual(calls[0]['max_workers'],2)
        with self.assertRaisesRegex(RuntimeError,'401'):
            snapshot_with_retry(lambda **kw:(_ for _ in ()).throw(RuntimeError('401 Unauthorized')),sleep=lambda _:None)

    def test_florence_invalid_region_preserves_raw_output(self):
        import numpy as np
        class Inputs(dict):
            def to(self,*args):return self
        class Processor:
            image_processor=SimpleNamespace(size={'width':768,'height':768})
            def __call__(self,**kw):return Inputs(input_ids=np.array([[1]]),pixel_values=np.zeros((1,3,768,768)))
            def batch_decode(self,*args,**kw):return ['raw malformed region evidence']
            def post_process_generation(self,*args,**kw):return {'<OCR_WITH_REGION>':{'quad_boxes':[[0]*8],'labels':['text']}}
        adapter=FlorenceAdapter.__new__(FlorenceAdapter);adapter.max_input_edge=1280;adapter.max_new_tokens=4096
        adapter.dtype='float16';adapter.torch=SimpleNamespace(inference_mode=contextlib.nullcontext)
        adapter.processor=Processor();adapter.model=SimpleNamespace(device='cuda',config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=1024)),generate=lambda **kw:np.array([[2,7,2]]))
        adapter._metadata=lambda *args:{'truncated':False}
        result=adapter.infer(self.image)
        self.assertEqual(result['raw_text'],'raw malformed region evidence');self.assertFalse(result['parse_success'])
        self.assertIn('zero or negative area',str(result['parse_warnings']))

    def test_dataset_access_denial_is_explicit_no_fallback(self):
        import urllib.error
        with patch('urllib.request.urlopen',side_effect=urllib.error.HTTPError('https://example',403,'Forbidden',{},None)):
            with self.assertRaisesRegex(RuntimeError,'access denied'):
                download_archive(self.folder/'train2017.zip',attempts=1)
        self.assertFalse((self.folder/'train2017.zip').exists())

    def test_resume_rejects_wrong_content_range(self):
        destination=self.folder/'train2017.zip';destination.with_suffix('.zip.part').write_bytes(b'partial')
        write_json(destination.with_suffix('.download.json'),{'url':'https://example','etag':'tag','total':100})
        response=SimpleNamespace(status=206,headers={'Content-Range':'bytes 0-99/100','Content-Length':'100','ETag':'tag'})
        with patch('urllib.request.urlopen',return_value=contextlib.nullcontext(response)):
            with self.assertRaisesRegex(ValueError,'Content-Range'):download_archive(destination,url='https://example')
        self.assertEqual(destination.with_suffix('.zip.part').read_bytes(),b'partial')

    def test_full_gate_requires_real_review_and_rejects_stale_results(self):
        from test_leakbench_v2 import completed
        from leakbench.worker import output_dir
        cohort=[{'id':f'id{i}','category':'documents' if i<2 else 'images','filename':f'item{i}.png',
            'stratum':'document' if i<2 else 'general','labels':['a19_name_full']} for i in range(4)]
        write_json(self.folder/'run_config.json',{'protocol_id':'protocol'})
        for r in cohort:completed(output_dir(self.folder,'qwen3_vl',r),r,'protocol')
        with self.assertRaisesRegex(RuntimeError,'review'):require_review(self.folder,'qwen3_vl',cohort,'protocol')
        notes={r['id']:'Synthetic acceptance fixture, not a human review of model outputs.' for r in cohort}
        record_pilot_review(self.folder,'qwen3_vl',cohort,notes)
        require_review(self.folder,'qwen3_vl',cohort,'protocol')
        path=output_dir(self.folder,'qwen3_vl',cohort[0])/'prediction.json'
        d=json.loads(path.read_text());d['new_field']='changed';write_json(path,d)
        with self.assertRaisesRegex(RuntimeError,'stale'):require_review(self.folder,'qwen3_vl',cohort,'protocol')

    def test_positive_empty_mask_cannot_be_reviewed_through_gate(self):
        from test_leakbench_v2 import completed
        from leakbench.worker import output_dir
        r={'id':'id','category':'documents','filename':'test.png','stratum':'document','labels':['a19_name_full']}
        write_json(self.folder/'run_config.json',{'protocol_id':'protocol'})
        folder=output_dir(self.folder,'qwen3_vl',r);completed(folder,r,'protocol')
        d=json.loads((folder/'prediction.json').read_text());d['redaction_validation']['technical_pass']=False
        write_json(folder/'prediction.json',d)
        with self.assertRaisesRegex(RuntimeError,'failed mask'):
            record_pilot_review(self.folder,'qwen3_vl',[r],{'id':'Cannot bless a failed redaction.'})

if __name__=='__main__':unittest.main()
