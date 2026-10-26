"""Synthetic-only tests. Fake adapters are injected here, never exposed in the app."""
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
from fastapi.testclient import TestClient
import httpx
from vispr_app.adapters import DirectQwen, LMStudio, local_url
from vispr_app.pipeline import ATTRIBUTES, REGISTRY, box_pixels, parse_detections, read_upload, redact
from vispr_app.server import create_app

A, B = list(REGISTRY)[:2]
def raw(items):
    return json.dumps({'coordinate_system':'xyxy_1000','detections':items})
def detection(attr=A, box=None):
    return {'attribute_id':attr,'bbox':[100,100,600,600] if box is None else box,'confidence':.9}
def image_bytes():
    buffer=io.BytesIO();Image.new('RGB',(100,100),'white').save(buffer,format='PNG');return buffer.getvalue()

class FakeManager:
    def __init__(self, response=None, fail=False, gate=None):
        self.response=response or raw([detection()]);self.fail=fail;self.gate=gate;self.started=threading.Event()
    def get(self, config):return self
    def infer(self, image):
        self.started.set()
        if self.gate:self.gate.wait(5)
        if self.fail:raise ValueError('Synthetic test failure')
        return self.response

class PipelineTests(unittest.TestCase):
    def test_registry_exactly_matches_source(self):
        self.assertEqual(len(ATTRIBUTES),28)
        self.assertEqual(REGISTRY['a29_ausweis']['name'],'National Identification')
    def test_geometry(self):
        self.assertEqual(box_pixels([100,200,600,800],(100,100))[0],[10,20,60,80])
        self.assertTrue(box_pixels([-5,0,1005,1000],(100,100))[1])
        for box in (None, [], [0,0,0,1],[1,1,0,0],[0,0,float('nan'),1],[0,0,float('inf'),1],[-11,0,20,20],[0,0,'x',20],[True,0,20,20],[1001,0,1005,20]):
            with self.subTest(box=box),self.assertRaises(ValueError):box_pixels(box,(100,100))
    def test_bad_json_and_coordinate_contract(self):
        for value in ('text', '[]','{}',json.dumps({'detections':[]})):
            with self.assertRaises(ValueError):parse_detections(value,(100,100),[A])
    def test_invalid_entries_reported(self):
        ds,warnings=parse_detections(raw([{'attribute_id':A},detection('unknown'),None]),(100,100),[A])
        self.assertEqual(len(warnings),3);self.assertTrue(all(not d['redacted'] for d in ds))
    def test_small_overflow_margin_and_uncertainty(self):
        d=detection(box=[-5,0,500,500]);d['confidence']=.2
        ds,warnings=parse_detections(raw([d]),(100,100),[A],margin=2)
        self.assertEqual(ds[0]['box'],[0,0,52,52]);self.assertEqual(len(warnings),2)
    def test_preferences_overlap_and_exact_saved_pixels(self):
        ds,_=parse_detections(raw([detection(),detection(B,[400,400,900,900])]),(100,100),[A])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'out.png';source=Image.new('RGB',(100,100),'white')
            self.assertEqual(redact(source,ds,path),1)
            with Image.open(path) as im:
                self.assertEqual(im.getpixel((50,50)),(0,0,0))
                self.assertEqual(im.getpixel((70,70)),(255,255,255))
                self.assertEqual(im.getpixel((9,10)),(255,255,255))
                self.assertEqual(im.getpixel((60,59)),(255,255,255))
            self.assertEqual(len(ds),2);self.assertFalse(ds[1]['redacted'])
            ds,_=parse_detections(raw([detection(),detection(B,[400,400,900,900])]),(100,100),[A,B])
            self.assertEqual(redact(source,ds,path),2)
    def test_invalid_upload(self):
        with self.assertRaises(ValueError):read_upload(b'not an image')
        b=io.BytesIO();Image.new('RGB',(10,10)).save(b,format='GIF')
        with self.assertRaises(ValueError):read_upload(b.getvalue())
    def test_metadata_stripped(self):
        im=Image.new('RGB',(20,10),'white');exif=im.getexif();exif[274]=6;exif[315]='Fake Person'
        b=io.BytesIO();im.save(b,format='JPEG',exif=exif)
        clean=read_upload(b.getvalue());self.assertEqual(clean.size,(10,20));self.assertFalse(clean.getexif())
    def test_transparent_hidden_pixels_not_revealed(self):
        b=io.BytesIO();Image.new('RGBA',(10,10),(0,0,0,0)).save(b,format='PNG')
        self.assertEqual(read_upload(b.getvalue()).getpixel((0,0)),(255,255,255))

class LocalAdapterTests(unittest.TestCase):
    def test_remote_urls_refused(self):
        for url in ('https://example.com/v1','http://127.0.0.1.evil/v1','http://user@localhost/v1','http://192.168.1.1/v1','http://localhost/v1?x=1','file:///tmp/v1'):
            with self.assertRaises(ValueError):local_url(url)
        self.assertEqual(local_url('http://localhost:1234/v1'),'http://127.0.0.1:1234/v1')
    def test_direct_missing_or_incompatible(self):
        with self.assertRaises(ValueError):DirectQwen({'model':'//server/checkpoint'})
        with tempfile.TemporaryDirectory() as tmp:
            adapter=DirectQwen({'model':tmp})
            with self.assertRaises(ValueError):adapter.load()
            (Path(tmp)/'config.json').write_text('{"model_type":"text-only"}')
            with self.assertRaises(ValueError):adapter.load()
    def test_lm_vision_checks_and_structured_request(self):
        requests=[]
        def handle(request):
            requests.append(request)
            if request.url.path=='/api/v1/models':return httpx.Response(200,json={'models':[{'capabilities':{'vision':True},'loaded_instances':[{'id':'test-vision'}]}]})
            body=json.loads(request.content);self.assertEqual(body['response_format']['type'],'json_schema')
            self.assertTrue(body['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':raw([])}}]})
        client=httpx.Client(transport=httpx.MockTransport(handle))
        with patch('vispr_app.adapters.httpx.Client',return_value=client):
            adapter=LMStudio({'base_url':'http://localhost:1234/v1','model':'test-vision'});adapter.load()
        client=httpx.Client(transport=httpx.MockTransport(handle))
        with patch('vispr_app.adapters.httpx.Client',return_value=client):self.assertEqual(adapter.infer(Image.new('RGB',(10,10))),raw([]))
        self.assertTrue(all(r.url.host=='127.0.0.1' for r in requests))
    def test_text_only_unloaded_and_missing_rejected(self):
        adapter=LMStudio({'base_url':'http://localhost:1234/v1','model':'test'})
        for models in ([],[{'id':'test','vision':False,'loaded':True}],[{'id':'test','vision':True,'loaded':False}]):
            with patch.object(adapter,'models',return_value=models),self.assertRaises(ValueError):adapter.load()
    def test_unavailable_no_fallback_or_redirect(self):
        def handle(request):raise httpx.ConnectError('offline')
        client=httpx.Client(transport=httpx.MockTransport(handle))
        with patch('vispr_app.adapters.httpx.Client',return_value=client),self.assertRaises(ValueError):
            LMStudio({'base_url':'http://localhost:1234/v1','model':'test'}).load()
    def test_redirect_is_not_followed(self):
        with patch('vispr_app.adapters.httpx.Client') as factory:
            client=factory.return_value.__enter__.return_value
            client.get.return_value=httpx.Response(302,headers={'Location':'https://example.com'},request=httpx.Request('GET','http://127.0.0.1/api/v1/models'))
            with self.assertRaises(ValueError):LMStudio({'base_url':'http://localhost:1234/v1','model':'test'}).load()
            self.assertFalse(factory.call_args.kwargs['follow_redirects'])
            self.assertFalse(factory.call_args.kwargs['trust_env'])
            self.assertEqual(client.get.call_count,1)
    def test_truncated_response_rejected(self):
        def handle(request):return httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':raw([])}}]})
        with patch('vispr_app.adapters.httpx.Client',return_value=httpx.Client(transport=httpx.MockTransport(handle))),self.assertRaises(ValueError):
            LMStudio({'base_url':'http://localhost:1234/v1','model':'test'}).infer(Image.new('RGB',(10,10)))

class APITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.manager=FakeManager()
        self.client=TestClient(create_app(self.temp.name,self.manager));self.client.__enter__()
        self.state=self.client.get('/api/state').json();self.headers={'X-VISPR-Token':self.state['token']}
        self.settings=self.state['settings'];self.settings['model']='TEST-ONLY-INJECTED'
        self.client.put('/api/settings',json=self.settings,headers=self.headers)
    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()
    def submit(self,n=1):
        return self.client.post('/api/jobs',files=[('files',(f'fake-{i}.png',image_bytes(),'image/png')) for i in range(n)],headers=self.headers)
    def wait_jobs(self):
        for _ in range(150):
            jobs=self.client.get('/api/state').json()['jobs']
            if jobs and all(j['status'] not in ('queued','processing') for j in jobs):return jobs
            time.sleep(.01)
        self.fail('Jobs did not finish')
    def test_batch_and_downloads(self):
        self.assertEqual(self.submit(2).status_code,200);jobs=self.wait_jobs()
        self.assertEqual([j['status'] for j in jobs],['completed','completed'])
        result=self.client.get(jobs[0]['output']);self.assertEqual(result.status_code,200)
        with Image.open(io.BytesIO(result.content)) as im:self.assertEqual(im.getpixel((20,20)),(0,0,0))
        with zipfile.ZipFile(io.BytesIO(self.client.get('/api/download.zip').content)) as z:self.assertEqual(len(z.namelist()),3)
    def test_single_preferences_change_and_reset(self):
        self.settings['enabled']=[];self.client.put('/api/settings',json=self.settings,headers=self.headers)
        self.submit();jobs=self.wait_jobs();self.assertFalse(jobs[0]['detections'][0]['redacted'])
        self.settings['enabled']=list(REGISTRY);self.client.put('/api/settings',json=self.settings,headers=self.headers)
        self.submit();jobs=self.wait_jobs();self.assertTrue(jobs[1]['detections'][0]['redacted'])
        self.assertEqual(json.loads((Path(self.temp.name)/'preferences.json').read_text())['enabled'],list(REGISTRY))
    def test_upload_errors_and_limits(self):
        self.assertEqual(self.submit(6).status_code,422)
        result=self.client.post('/api/jobs',files={'files':('bad.png',b'invalid','image/png')},headers=self.headers)
        self.assertEqual(result.status_code,422);self.assertFalse(self.client.get('/api/state').json()['jobs'])
    def test_failed_job(self):
        self.manager.fail=True;self.submit();jobs=self.wait_jobs()
        self.assertEqual(jobs[0]['status'],'failed');self.assertIsNone(jobs[0]['output'])
    def test_one_failure_does_not_discard_success(self):
        with patch.object(self.manager,'infer',side_effect=[ValueError('Synthetic failure'),raw([detection()])]):
            self.submit(2);jobs=self.wait_jobs()
        self.assertEqual([j['status'] for j in jobs],['failed','completed'])
    def test_preferences_snapshot_while_processing(self):
        gate=threading.Event();self.manager.gate=gate
        try:
            self.submit();self.assertTrue(self.manager.started.wait(2))
            self.settings['enabled']=[];self.client.put('/api/settings',json=self.settings,headers=self.headers)
        finally:gate.set()
        self.assertTrue(self.wait_jobs()[0]['detections'][0]['redacted'])
    def test_cancellation_and_duplicate_submission(self):
        gate=threading.Event();self.manager.gate=gate
        try:
            self.submit(2);self.assertTrue(self.manager.started.wait(2))
            jobs=self.client.get('/api/state').json()['jobs'];self.assertEqual(jobs[0]['stage'],'Model inference');self.assertEqual(jobs[1]['status'],'queued')
            self.assertEqual(self.submit().status_code,409)
            self.assertEqual(self.client.delete('/api/session',headers=self.headers).status_code,409)
            self.client.post('/api/jobs/cancel',headers=self.headers)
        finally:gate.set()
        self.assertTrue(all(j['status']=='cancelled' for j in self.wait_jobs()))
        self.assertEqual(self.client.get('/api/download.zip').status_code,404)
    def test_incomplete_detections_visible(self):
        self.manager.response=raw([{'attribute_id':A}]);self.submit();job=self.wait_jobs()[0]
        self.assertTrue(job['incomplete']);self.assertTrue(job['warnings']);self.assertFalse(job['detections'][0]['redacted'])
    def test_security_and_cleanup(self):
        self.assertEqual(self.client.put('/api/settings',json=self.settings).status_code,403)
        self.assertEqual(self.client.put('/api/settings',json=self.settings,headers={**self.headers,'Origin':'https://evil.example'}).status_code,403)
        self.assertEqual(self.client.get('/api/state',headers={'Host':'evil.example'}).status_code,400)
        self.settings['base_url']='http://example.com/v1';self.assertEqual(self.client.put('/api/settings',json=self.settings,headers=self.headers).status_code,422)
        self.submit();self.wait_jobs();self.assertEqual(self.client.delete('/api/session',headers=self.headers).status_code,200)
        self.assertFalse(self.client.get('/api/state').json()['jobs'])
        self.assertTrue((Path(self.temp.name)/'preferences.json').exists())

if __name__=='__main__':unittest.main()
