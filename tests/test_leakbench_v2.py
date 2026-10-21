"""Offline contract tests. Synthetic fixtures are never benchmark measurements."""
import ast
import base64
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
TEMP_ROOT=ROOT/'tmp/v2_tests'
TEMP_ROOT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from leakbench.common import read_json,write_json,sha,log
from leakbench.dataset import prepare_subset,validate_subset,records,image_info
from leakbench.worker import valid_completed,archive_attempt,verify_gpu_model,output_dir
from leakbench.cloud import verified_copy_tree
from leakbench.execution import run_command
from leakbench.reporting import report
from leakbench.orchestrate import start_run,execute_category

def completed(folder,record,protocol,model='qwen3_vl'):
    folder.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in ('original.png','predicted.png','redacted.png','raw_predictions.json','redaction_mask.png','redaction_validation.json'):
        if name=='original.png' and record.get('image_path'):shutil.copy2(record['image_path'],folder/name)
        else:(folder/name).write_text('synthetic test fixture',encoding='utf-8')
        hashes[name]=sha(folder/name)
    write_json(folder/'prediction.json',{'image_id':record['id'],'model_key':model,'stratum':record['stratum'],
        'status':'ok','protocol_id':protocol,'ground_truth_attributes':record['labels'],
        'predicted_attributes':record['labels'],'artifact_sha256':hashes,
        'redaction_validation':{'schema_version':1,'technical_pass':True,'privacy_verified':False},
        'input_sha256':record.get('image_sha256'),'annotation_sha256':record.get('annotation_sha256')})

class V2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.fixture=tempfile.TemporaryDirectory(dir=TEMP_ROOT);cls.base=Path(cls.fixture.name)
        (cls.base/'configs').mkdir()
        shutil.copy2(ROOT/'configs/taxonomy_crosswalk.csv',cls.base/'configs/taxonomy_crosswalk.csv')
        selection={'source_dataset':'synthetic unit test, not VISPR measurements','images':[]}
        rng=random.Random(137)
        archive=cls.base/'fixture.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for n in range(50):
                category='documents' if n<25 else 'images';ident=f'test_{n}'
                image=Image.frombytes('RGB',(32,32),rng.randbytes(32*32*3))
                buf=io.BytesIO();image.save(buf,format='PNG');data=buf.getvalue()
                ann=json.dumps({'id':ident,'labels':['a19_name_full']}).encode()
                item={'category':category,'original_id':ident,'archive_member':f'{ident}.png',
                    'annotation_member':f'{ident}.json','image_sha256':hashlib.sha256(data).hexdigest(),
                    'annotation_sha256':hashlib.sha256(ann).hexdigest(),'labels':['a19_name_full'],
                    'review_note':'Synthetic test fixture'}
                selection['images'].append(item);z.writestr(item['archive_member'],data);z.writestr(item['annotation_member'],ann)
        write_json(cls.base/'configs/selection.json',selection)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            prepare_subset(cls.base,cls.base/'configs/selection.json',archive)

    @classmethod
    def tearDownClass(cls):cls.fixture.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=TEMP_ROOT);self.addCleanup(self.temp.cleanup)
        self.project=Path(self.temp.name)/'project'
        shutil.copytree(self.base/'data',self.project/'data')
        shutil.copytree(self.base/'configs',self.project/'configs')
        self.cohort=records(self.project/'data/vispr_eval_50')
        self.run=self.project/'results/vlm/test';self.run.mkdir(parents=True)
        write_json(self.run/'run_config.json',{'protocol_id':'test','category':'vlm','settings':{
            'models':['qwen3_vl'],'run_timeout_seconds':5,'image_timeout_seconds':1}})

    def quiet(self,fn,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):return fn(*args,**kwargs)

    def test_fixed_cohort_materializes_and_reuses_without_download(self):
        folder=self.project/'data/vispr_eval_50'
        before=sha(folder/'documents/document_1.png')
        with patch('leakbench.dataset.download_archive',side_effect=AssertionError('No download expected')):
            self.assertEqual(self.quiet(prepare_subset,self.project,self.project/'configs/selection.json'),folder)
        self.assertEqual(before,sha(folder/'documents/document_1.png'))
        self.assertEqual(read_json(folder/'validation.json')['count'],50)

    def test_changed_image_is_rejected(self):
        folder=self.project/'data/vispr_eval_50';(folder/'documents/document_1.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):self.quiet(validate_subset,folder)

    def test_changed_manifest_labels_are_rejected(self):
        path=self.project/'data/vispr_eval_50/manifest.csv'
        with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        rows[0]['labels']='[]'
        with path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'labels'):self.quiet(validate_subset,path.parent)

    def test_changed_selection_is_rejected(self):
        selection=read_json(self.project/'configs/selection.json');selection['extra']='changed'
        with self.assertRaisesRegex(ValueError,'selection changed'):
            self.quiet(validate_subset,self.project/'data/vispr_eval_50',selection)

    def test_resume_requires_complete_matching_intact_artifacts(self):
        folder=self.run/'image';completed(folder,self.cohort[0],'test')
        self.assertTrue(valid_completed(folder,'test'))
        self.assertFalse(valid_completed(folder,'other'))
        (folder/'redacted.png').write_text('tampered')
        self.assertFalse(valid_completed(folder,'test'))
        write_json(folder/'prediction.json',{'status':'ok','protocol_id':'test','artifact_sha256':{'original.png':sha(folder/'original.png')}})
        self.assertFalse(valid_completed(folder,'test'))

    def test_corrupt_prediction_is_retryable(self):
        folder=self.run/'image';folder.mkdir();(folder/'prediction.json').write_text('{')
        self.assertFalse(valid_completed(folder,'test'))
        (folder/'prediction.json').write_text('[]');self.assertFalse(valid_completed(folder,'test'))

    def test_resume_rejects_wrong_input_or_model_association(self):
        folder=self.run/'image';r=self.cohort[0];completed(folder,r,'test')
        self.assertTrue(valid_completed(folder,'test',r,'qwen3_vl'))
        self.assertFalse(valid_completed(folder,'test',self.cohort[1],'qwen3_vl'))
        self.assertFalse(valid_completed(folder,'test',r,'qwen25_vl'))
        Path(r['image_path']).write_bytes(b'changed after dataset validation')
        self.assertFalse(valid_completed(folder,'test',r,'qwen3_vl'))

    def test_corrupt_prediction_does_not_prevent_error_report(self):
        folder=output_dir(self.run,'qwen3_vl',self.cohort[0]);folder.mkdir(parents=True)
        (folder/'prediction.json').write_text('{')
        out=self.quiet(report,self.project,self.run,['qwen3_vl'])
        self.assertEqual(read_json(out/'summary.json')['coverage'][0]['failed_images'],1)

    def test_retry_archives_every_prior_artifact(self):
        folder=self.run/'image';completed(folder,self.cohort[0],'test');archive_attempt(folder)
        self.assertFalse((folder/'redacted.png').exists())
        self.assertEqual(len(list((folder/'attempts').glob('*/redacted.png'))),1)
        self.assertFalse(valid_completed(folder,'test'))

    def test_gpu_placement_rejects_offloading(self):
        from types import SimpleNamespace
        model=SimpleNamespace(hf_device_map={'':'cuda:0'},parameters=lambda:[SimpleNamespace(device='cuda:0')])
        self.assertEqual(verify_gpu_model(model)['parameter_devices'],['cuda:0'])
        model.hf_device_map={'':'cpu'}
        with self.assertRaisesRegex(RuntimeError,'offloading'):verify_gpu_model(model)
        model.hf_device_map={};model.parameters=lambda:[SimpleNamespace(device='cpu')]
        with self.assertRaisesRegex(RuntimeError,'CUDA'):verify_gpu_model(model)

    def test_backup_reuses_and_repairs_with_readback(self):
        source=self.project/'source';source.mkdir();(source/'a.txt').write_text('valuable')
        dest=Path(self.temp.name)/'backup'
        self.quiet(verified_copy_tree,source,dest)
        check=self.quiet(verified_copy_tree,source,dest)
        self.assertEqual(check[0]['status'],'already_verified')
        (dest/'a.txt').write_text('corrupt');self.quiet(verified_copy_tree,source,dest)
        self.assertEqual(sha(source/'a.txt'),sha(dest/'a.txt'))
        with self.assertRaises(ValueError):verified_copy_tree(source,source/'child')
        with self.assertRaises(ValueError):verified_copy_tree(source,source.parent)
        with self.assertRaises(FileNotFoundError):verified_copy_tree(source/'missing',dest)

    def test_timeout_terminates_worker(self):
        started=time.monotonic()
        with self.assertRaises(TimeoutError):
            self.quiet(run_command,[sys.executable,'-c','import time; time.sleep(20)'],'test',timeout=.1)
        self.assertLess(time.monotonic()-started,5)

    def test_process_error_and_last_output_are_retained(self):
        logfile=self.project/'last.log'
        with self.assertRaises(RuntimeError):self.quiet(run_command,[sys.executable,'-c',
            "print('last useful error'); raise SystemExit(2)"],'test',log_path=logfile)
        self.assertIn('last useful error',logfile.read_text())

    def test_credentials_in_structured_logs_are_redacted(self):
        value='hf_'+'A'*30;out=io.StringIO()
        with contextlib.redirect_stdout(out):log('test',error={'token':value})
        self.assertNotIn(value,out.getvalue())

    def test_reports_do_not_rank_failures_or_missing_images(self):
        folder=output_dir(self.run,'qwen3_vl',self.cohort[0]);completed(folder,self.cohort[0],'test')
        out=self.quiet(report,self.project,self.run,['qwen3_vl','llama32_vision'])
        summary=read_json(out/'summary.json');self.assertIsNone(summary['winner'])
        self.assertEqual(summary['status'],'partial_or_pending')
        coverage={r['model_key']:r for r in summary['coverage']}
        self.assertEqual(coverage['qwen3_vl']['successful_images'],1)
        self.assertEqual(coverage['llama32_vision']['not_attempted_images'],50)
        (folder/'redacted.png').write_text('corrupt')
        out=self.quiet(report,self.project,self.run,['qwen3_vl'])
        self.assertEqual(read_json(out/'summary.json')['coverage'][0]['failed_images'],1)

    def test_failed_pilot_blocks_full_phase_and_reports_error(self):
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable),patch('leakbench.orchestrate.run_command') as call:
            self.quiet(execute_category,self.project,self.run,full=True,save_cloud=False)
        self.assertEqual(call.call_count,0)
        self.assertIn('Regression gate',read_json(self.run/'qwen3_vl/orchestration_failure.json')['error'])
        self.assertIsNone(read_json(self.run/'reports/summary.json')['winner'])

    def save_partial(self,model,record):
        from leakbench.pipeline import detect_pii,load_taxonomy
        detection={'attribute_id':'a19_name_full','bbox_1000':[100,100,500,500],
                   'location':'field','evidence_source':'explicit_text'}
        result=detect_pii(None,load_taxonomy(self.project/'configs/taxonomy_crosswalk.csv'),
            lambda *_:(json.dumps({'detections':[detection,{**detection,'attribute_id':'invalid_label'}]}),False))
        self.assertEqual(result['status'],'partial')
        folder=output_dir(self.run,model,record);folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'prediction.json',{**result,'image_id':record['id'],'model_key':model,
            'stratum':record['stratum'],'protocol_id':'test','ground_truth_attributes':record['labels'],
            'predicted_attributes':['a19_name_full'],
            'redaction_validation':{'schema_version':1,'technical_pass':False,'privacy_verified':False}})
        return folder/'prediction.json'

    def test_partial_parser_output_reports_in_both_scopes_without_quality_credit(self):
        partial_path=self.save_partial('qwen3_vl',self.cohort[0]);before=sha(partial_path)
        completed(output_dir(self.run,'qwen3_vl',self.cohort[1]),self.cohort[1],'test')
        r=self.cohort[25];folder=output_dir(self.run,'qwen3_vl',r);folder.mkdir(parents=True)
        write_json(folder/'prediction.json',{'image_id':r['id'],'model_key':'qwen3_vl','stratum':r['stratum'],
            'status':'error','protocol_id':'test','ground_truth_attributes':r['labels'],'error':'test failure'})
        out=self.quiet(report,self.project,self.run,['qwen3_vl','qwen25_vl'])
        for scope in ('full_taxonomy','shared_text_scope'):
            result=read_json(out/f'{scope}.json')
            cov=next(r for r in result['coverage'] if r['model_key']=='qwen3_vl' and r['stratum']=='all')
            self.assertEqual([cov[k] for k in ('successful_images','failed_images','partial_images','error_images','not_attempted_images')],[1,2,1,1,47])
            row=next(r for r in result['per_image'] if r['status']=='partial')
            self.assertEqual(row['predicted_attributes'],['a19_name_full'])
            self.assertIsNone(row['f1']);self.assertIsNone(row['missed_attributes'])
            self.assertFalse(row['localization_available']);self.assertFalse(row['text_available'])
            successful=next(r for r in result['summaries'] if r['model_key']=='qwen3_vl' and r['stratum']=='all' and r['scope']=='all_successful')
            self.assertEqual(successful['n_images'],1);self.assertEqual(result['common_image_ids'],[])
        summary=read_json(out/'summary.json')
        self.assertIsNone(summary['winner']);self.assertFalse(summary['all_saved_mask_checks_passed'])
        self.assertEqual(sha(partial_path),before)
        self.assertEqual(read_json(out/'all_results.json')[0]['status'],'partial')
        self.assertIn('1 partial', (out/'report.md').read_text())

    def test_partial_worker_result_does_not_stop_next_model_or_backup(self):
        config=read_json(self.run/'run_config.json');config['settings']['models']=['qwen3_vl','qwen25_vl']
        write_json(self.run/'run_config.json',config)
        def fake_worker(args,*_,**kwargs):
            job=read_json(args[-1]);self.save_partial(job['model'],self.cohort[0])
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable), \
             patch('leakbench.orchestrate.run_command',side_effect=fake_worker) as worker, \
             patch('leakbench.orchestrate.backup') as backup:
            out=self.quiet(execute_category,self.project,self.run,save_cloud=True)
        self.assertEqual(worker.call_count,2);self.assertGreaterEqual(backup.call_count,3)
        self.assertFalse(list(self.run.glob('*/orchestration_failure.json')))
        self.assertFalse((self.run/'reporting_failure.json').exists())
        self.assertTrue(all(r['partial_images']==1 for r in read_json(out/'summary.json')['coverage']))

    def test_reporting_exception_does_not_interrupt_models_or_backup(self):
        config=read_json(self.run/'run_config.json');config['settings']['models']=['qwen3_vl','qwen25_vl']
        write_json(self.run/'run_config.json',config)
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable), \
             patch('leakbench.orchestrate.run_command') as worker, \
             patch('leakbench.orchestrate.report',side_effect=ValueError('test report failure')), \
             patch('leakbench.orchestrate.backup') as backup:
            with self.assertRaisesRegex(RuntimeError,'do not rerun inference'):
                self.quiet(execute_category,self.project,self.run,save_cloud=True)
        self.assertEqual(worker.call_count,2);self.assertGreaterEqual(backup.call_count,3)
        self.assertFalse(list(self.run.glob('*/orchestration_failure.json')))
        failure=read_json(self.run/'reporting_failure.json')
        self.assertEqual(failure['status'],'failed');self.assertEqual(failure['error'],'test report failure')

    def test_transient_reporting_failure_is_resolved_without_repeating_worker(self):
        def flaky_report(*args):
            if not (self.run/'reporting_failure.json').exists():raise OSError('temporary reporting failure')
            return report(*args)
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable), \
             patch('leakbench.orchestrate.run_command') as worker, \
             patch('leakbench.orchestrate.report',side_effect=flaky_report):
            out=self.quiet(execute_category,self.project,self.run,save_cloud=False)
        self.assertEqual(worker.call_count,1);self.assertTrue((out/'summary.json').exists())
        self.assertEqual(read_json(self.run/'reporting_failure.json')['status'],'resolved')

    def test_colab_repair_rebuilds_saved_partial_without_inference_or_changing_evidence(self):
        prediction=self.save_partial('qwen3_vl',self.cohort[0])
        shutil.copytree(ROOT/'src/leakbench',self.project/'src/leakbench',ignore=shutil.ignore_patterns('__pycache__'))
        metrics=self.project/'src/leakbench/metrics.py'
        metrics.write_text(metrics.read_text().replace('not in {"ok", "partial", "error"}',
                                                      'not in {"ok", "error"}'))
        snapshot=self.run/'source/leakbench/metrics.py';snapshot.parent.mkdir(parents=True)
        shutil.copy2(metrics,snapshot)
        failure=self.run/'qwen3_vl/orchestration_failure.json'
        write_json(failure,{'error':'status must be ok or error'})
        retained={p:sha(p) for p in (prediction,self.run/'run_config.json',snapshot,failure)}
        script='''
import runpy,sys
from pathlib import Path
from unittest.mock import patch
project,run,recovery=map(Path,sys.argv[1:])
sys.path.insert(0,str(project/'src'))
from leakbench.reporting import report
try:report(project,run,['qwen3_vl'])
except ValueError:pass
else:raise AssertionError('Original partial-status failure was not reproduced')
with patch('leakbench.execution.run_command',side_effect=AssertionError('No subprocess or model allowed')),patch('leakbench.cloud.backup') as backup:
    values=runpy.run_path(str(recovery),init_globals={'PROJECT':project,'run_dir':run})
    assert backup.call_count==1
assert (values['report_dir']/'summary.json').is_file()
'''
        proc=subprocess.run([sys.executable,'-c',script,str(self.project),str(self.run),
                             str(ROOT/'dist/repair_colab_reporting.py')],capture_output=True,text=True,timeout=30)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        self.assertEqual({p:sha(p) for p in retained},retained)
        repair=read_json(next(self.run.glob('reporting_repairs/*/repair.json')))
        self.assertEqual(repair['status'],'reports_regenerated')
        self.assertEqual(repair['historical_reporting_failures'],['qwen3_vl/orchestration_failure.json'])
        cov=read_json(self.run/'reports/summary.json')['coverage'][0]
        self.assertEqual(cov['partial_images'],1);self.assertEqual(cov['successful_images'],0)

    def test_worker_crash_is_an_explicit_image_failure(self):
        def fail_worker(*args,**kwargs):
            write_json(self.run/'qwen3_vl/state.json',{'stage':'image','image_id':self.cohort[0]['id']})
            raise TimeoutError('synthetic worker timeout')
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable),patch('leakbench.orchestrate.run_command',side_effect=fail_worker):
            self.quiet(execute_category,self.project,self.run,stage='single',save_cloud=False)
        coverage=read_json(self.run/'reports/summary.json')['coverage'][0]
        self.assertEqual(coverage['failed_images'],1);self.assertEqual(coverage['not_attempted_images'],0)
        self.assertEqual(coverage['expected_images'],1)  # single mode reports its planned workload

    def test_cloud_failure_does_not_abort_valid_pilot(self):
        def fake_worker(*args,**kwargs):
            job=read_json(self.run/'qwen3_vl/job.json')
            chosen=[next(r for r in self.cohort if r['category']==c) for c in ('documents','images')]
            for r in chosen:completed(output_dir(self.run,'qwen3_vl',r),r,job['protocol_id'])
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable),patch('leakbench.orchestrate.run_command',side_effect=fake_worker) as call,patch('leakbench.orchestrate.backup',side_effect=OSError('test Drive unavailable')):
            self.quiet(execute_category,self.project,self.run,stage='single',save_cloud=True)
        self.assertEqual(call.call_count,1)
        self.assertFalse((self.run/'qwen3_vl/orchestration_failure.json').exists())
        self.assertTrue((self.run/'cloud_failure.json').exists())

    def test_protocol_resume_and_changed_source_create_distinct_runs(self):
        (self.project/'src/leakbench').mkdir(parents=True)
        source=self.project/'src/leakbench/test.py';source.write_text('VERSION=1')
        shutil.copy2(ROOT/'configs/models.json',self.project/'configs/models.json')
        config={'models':['qwen3_vl']}
        with patch('leakbench.orchestrate.DRIVE_ROOT',Path(self.temp.name)/'drive'):
            first=self.quiet(start_run,self.project,'vlm',config)
            self.assertEqual(first,self.quiet(start_run,self.project,'vlm',config))
            source.write_text('VERSION=2')
            self.assertNotEqual(first,self.quiet(start_run,self.project,'vlm',config))

class NotebookPackaging(unittest.TestCase):
    def test_notebooks_parse_and_contain_exact_independent_source(self):
        for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
            notebook=read_json(path);ids=[c['id'] for c in notebook['cells']]
            self.assertEqual(len(ids),len(set(ids)))
            bootstrap=None
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    ast.parse(''.join(cell['source']));self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])
                    if 'bootstrap' in cell['metadata'].get('tags',[]):bootstrap=''.join(cell['source'])
            self.assertIsNotNone(bootstrap)
            payload=next(ast.literal_eval(n.value) for n in ast.parse(bootstrap).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
                for name in z.namelist():
                    self.assertTrue(name.startswith(('src/leakbench/','configs/')) or name=='TRACKER.md')
                    self.assertEqual(z.read(name),(ROOT/name).read_bytes(),name)
                self.assertIn('src/leakbench/worker.py',z.namelist())
            with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
                script=Path(folder)/'bootstrap_test.py';script.write_text(bootstrap,encoding='utf-8')
                result=subprocess.run([sys.executable,str(script)],cwd=folder,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertTrue((Path(folder)/'src/leakbench/worker.py').is_file())

if __name__=='__main__':unittest.main()
