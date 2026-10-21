"""Offline execution/viewer regressions. Synthetic outputs are not GPU evidence."""
import contextlib,io,json,shutil,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_leakbench_v2 as fixtures
ROOT=fixtures.ROOT;completed=fixtures.completed
from leakbench.common import read_json,write_json,sha
from leakbench.worker import output_dir,valid_completed
from leakbench.orchestrate import execute_category,prepare_execution_run
from leakbench.progress import selected_records,begin_execution,refresh_execution,update_model,active_execution
from leakbench.image_execution import process_image
from leakbench.pipeline import load_taxonomy
from leakbench.viewer import discover_runs,image_details,show_results_viewer
from leakbench.gates import record_pilot_review,require_review
from leakbench.hardware import native_dtype

class ExecutionRecovery(unittest.TestCase):
    setUpClass=classmethod(fixtures.V2Contracts.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.V2Contracts.tearDownClass.__func__)
    setUp=fixtures.V2Contracts.setUp
    quiet=fixtures.V2Contracts.quiet

    def test_modes_are_exact_and_full_block_has_fifty_accounted_images(self):
        self.assertEqual([len(selected_records(self.cohort,s)) for s in ('single','pilot','full')],[1,4,50])
        with patch('leakbench.orchestrate.setup_profile') as setup:
            out=self.quiet(execute_category,self.project,self.run,stage='full',save_cloud=False)
        self.assertFalse(setup.called)
        execution=read_json(out/'summary.json')['execution'];self.assertEqual(execution['status'],'blocked')
        self.assertGreaterEqual(execution['wall_seconds'],0)
        self.assertIn('including reused',read_json(out/'summary.json')['runtime'][0]['timing_scope'])
        count=execution['reconciliation'][0]
        self.assertEqual(count['accounted_for'],50);self.assertEqual(count['skipped'],50)
        self.assertEqual(count['attempted_images'],0);self.assertEqual(count['missing_outputs'],50)

    def test_pilot_runs_four_images_without_single_quality_gate(self):
        def fake_worker(args,*_,**kw):
            job=read_json(args[-1])
            for r in selected_records(self.cohort,job['phase']):completed(output_dir(self.run,job['model'],r),r,'test')
        with patch('leakbench.orchestrate.setup_profile',return_value=sys.executable),patch('leakbench.orchestrate.run_command',side_effect=fake_worker) as worker:
            out=self.quiet(execute_category,self.project,self.run,stage='pilot',save_cloud=False)
        self.assertEqual(worker.call_count,1)
        summary=read_json(out/'summary.json');self.assertEqual(summary['coverage'][0]['expected_images'],4)
        self.assertEqual(summary['execution']['reconciliation'][0]['valid_predictions'],4)
        self.assertIsNone(summary['winner'])

    def test_each_execution_has_immutable_history_even_in_legacy_run(self):
        first=begin_execution(self.project,self.run,'single');refresh_execution(self.project,self.run,finish=True)
        path=self.run/'executions'/first/'execution.json';before=sha(path)
        second=begin_execution(self.project,self.run,'pilot')
        self.assertNotEqual(first,second);self.assertEqual(sha(path),before)
        self.assertEqual(len(read_json(self.run/'processing_manifest.json')),4)
        bad=output_dir(self.run,'qwen3_vl',self.cohort[0]);bad.mkdir(parents=True)
        write_json(bad/'prediction.json',[]);refresh_execution(self.project,self.run)
        self.assertEqual(read_json(self.run/'processing_manifest.json')[0]['status'],'failed')

    def test_interrupted_generation_is_recorded_as_attempted_without_prediction(self):
        begin_execution(self.project,self.run,'single')
        r=self.cohort[0];folder=output_dir(self.run,'qwen3_vl',r);folder.mkdir(parents=True)
        write_json(folder/'attempt_state.json',{'status':'processing','stage':'inference','inference_attempted':True})
        update_model(self.project,self.run,'qwen3_vl',status='processing')
        row=read_json(self.run/'processing_manifest.json')[0]
        self.assertEqual(row['status'],'processing');self.assertTrue(row['inference_attempted'])
        self.assertFalse(row['prediction_generated']);self.assertTrue(row['missing_output'])
        update_model(self.project,self.run,'qwen3_vl',status='failed',stage='inference',error='interrupted')
        row=read_json(self.run/'processing_manifest.json')[0]
        self.assertEqual(row['status'],'failed');self.assertTrue(row['inference_attempted'])

    def test_fresh_run_reuses_only_matching_protocol_and_intact_input(self):
        (self.project/'src/leakbench').mkdir(parents=True);(self.project/'src/leakbench/test.py').write_text('version=1')
        shutil.copy2(ROOT/'configs/models.json',self.project/'configs/models.json')
        config={'models':['qwen3_vl']}
        with patch('leakbench.orchestrate.DRIVE_ROOT',self.project/'no_drive'):
            first=self.quiet(prepare_execution_run,self.project,'vlm',config,stage='single')
            protocol=read_json(first/'run_config.json')['protocol_id']
            r=self.cohort[0];completed(output_dir(first,'qwen3_vl',r),r,protocol)
            second=self.quiet(prepare_execution_run,self.project,'vlm',config,stage='pilot')
            self.assertNotEqual(first,second);self.assertTrue(valid_completed(output_dir(second,'qwen3_vl',r),protocol,r,'qwen3_vl'))
            self.assertEqual(len(read_json(second/'run_config.json')['reused_predictions']),1)
            third=self.quiet(prepare_execution_run,self.project,'vlm',{**config,'max_input_edge':999},stage='single')
            self.assertEqual(read_json(third/'run_config.json')['reused_predictions'],{})

    def test_valid_model_subset_can_reuse_the_reviewed_protocol(self):
        (self.project/'src/leakbench').mkdir(parents=True);(self.project/'src/leakbench/test.py').write_text('version=1')
        shutil.copy2(ROOT/'configs/models.json',self.project/'configs/models.json')
        with patch('leakbench.orchestrate.DRIVE_ROOT',self.project/'no_drive'):
            first=self.quiet(prepare_execution_run,self.project,'vlm',{'models':['qwen3_vl','qwen25_vl']},stage='pilot')
            protocol=read_json(first/'run_config.json')['protocol_id']
            for r in selected_records(self.cohort,'pilot'):completed(output_dir(first,'qwen3_vl',r),r,protocol)
            notes={r['id']:'Synthetic review fixture; not real privacy verification.' for r in selected_records(self.cohort,'pilot')}
            record_pilot_review(first,'qwen3_vl',self.cohort,notes)
            second=self.quiet(prepare_execution_run,self.project,'vlm',{'models':['qwen3_vl']},stage='full')
        self.assertEqual(read_json(second/'run_config.json')['protocol_id'],protocol)
        require_review(second,'qwen3_vl',self.cohort,protocol)

    def test_bad_coordinate_preserves_raw_and_next_image_still_processes(self):
        tax=load_taxonomy(self.project/'configs/taxonomy_crosswalk.csv');config={'redaction_padding':0}
        good={'attribute_id':'a19_name_full','bbox_1000':[100,100,400,400],'location':'name','evidence_source':'explicit_text'}
        bad={**good,'ocr_bbox_pixels':[-10,0,10,20]}
        current={}
        def detector(det):
            def call(*_):
                current['generation_calls']=1
                return {'status':'ok','detections':[det],'native_outputs':[{'raw_text':'retained evidence'}]}
            return call
        r=self.cohort[0];folder=output_dir(self.run,'qwen3_vl',r)
        failed=self.quiet(process_image,r,folder,'qwen3_vl','test',config,tax,detector(bad),current)
        self.assertEqual(failed['failure_stage'],'coordinate_conversion');self.assertEqual(failed['status'],'error')
        self.assertIn('retained evidence',(folder/'raw_predictions.json').read_text())
        r=self.cohort[1];folder=output_dir(self.run,'qwen3_vl',r)
        good_result=self.quiet(process_image,r,folder,'qwen3_vl','test',config,tax,detector(good),current)
        self.assertTrue(good_result['redaction_validation']['technical_pass']);self.assertTrue(valid_completed(folder,'test',r,'qwen3_vl'))

    def test_review_can_document_detection_misses_without_certifying_privacy(self):
        tax=load_taxonomy(self.project/'configs/taxonomy_crosswalk.csv');notes={}
        for r in selected_records(self.cohort,'pilot'):
            self.quiet(process_image,r,output_dir(self.run,'qwen3_vl',r),'qwen3_vl','test',{'redaction_padding':0},tax,
                lambda *_:{'status':'ok','detections':[],'native_outputs':[{'raw_text':'empty'}]}, {})
            notes[r['id']]='Synthetic review: no PII detected; redaction is ineffective and must be scored as a miss.'
        with self.assertRaisesRegex(RuntimeError,'review'):require_review(self.run,'qwen3_vl',self.cohort,'test')
        record_pilot_review(self.run,'qwen3_vl',self.cohort,notes)
        require_review(self.run,'qwen3_vl',self.cohort,'test')
        self.assertFalse(read_json(self.run/'qwen3_vl/pilot_review.json')['privacy_completeness_certified'])

    def test_viewer_reads_selected_run_not_previous_summary(self):
        r=self.cohort[0];completed(output_dir(self.run,'qwen3_vl',r),r,'test')
        fresh=self.run.parent/'fresh';fresh.mkdir();shutil.copy2(self.run/'run_config.json',fresh/'run_config.json')
        self.assertTrue(image_details(self.project,self.run,'qwen3_vl',r)['prediction'])
        missing=image_details(self.project,fresh,'qwen3_vl',r)
        self.assertEqual(missing['prediction'],{});self.assertIn('dataset preview',missing['original_source'])
        self.assertEqual(len(discover_runs(self.project,'vlm')),2)

    def test_viewer_controls_navigate_model_category_image_and_run(self):
        fresh=self.run.parent/'fresh';fresh.mkdir();shutil.copy2(self.run/'run_config.json',fresh/'run_config.json')
        with patch('IPython.display.display'),patch('IPython.display.clear_output'):
            view=show_results_viewer(self.project,category='vlm',initial_run=self.run)
            first=view['image'].value;view['next'].click();self.assertNotEqual(first,view['image'].value)
            view['previous'].click();self.assertEqual(first,view['image'].value)
            view['category'].value='images';self.assertEqual(len(view['image'].options),25)
            view['jump'].value=12;self.assertEqual(view['image'].value,self.cohort[36]['id'])
            view['run'].value=str(fresh.resolve());self.assertEqual(view['jump'].value,1)

    def test_t4_uses_native_float16_and_ampere_uses_bfloat16(self):
        torch=SimpleNamespace(float16='fp16',bfloat16='bf16',cuda=SimpleNamespace(get_device_capability=lambda:(7,5)))
        self.assertEqual(native_dtype(torch),'fp16')
        torch.cuda.get_device_capability=lambda:(8,9);self.assertEqual(native_dtype(torch),'bf16')

    def test_florence_fp32_is_loaded_directly_not_cast_from_fp16(self):
        from leakbench.adapters import FlorenceAdapter,_Adapter
        from unittest.mock import Mock
        loader=Mock();model=Mock();loader.return_value=model;model.to.return_value=model
        def base(adapter,*args,**kw):
            adapter.torch=SimpleNamespace(float16='fp16',float32='fp32')
            adapter.transformers=SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=Mock()),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader))
            adapter._hub_kwargs={};adapter.model_id='microsoft/Florence-2-large-ft'
        with patch.object(_Adapter,'__init__',base),patch('leakbench.adapters.florence_eager_loader',contextlib.nullcontext):
            FlorenceAdapter(precision='float32')
        self.assertEqual(loader.call_args.kwargs['torch_dtype'],'fp32')
        model.to.assert_called_once_with('cuda')

    def test_invalid_id_correction_keeps_boxes_and_never_uses_annotation(self):
        from leakbench.pipeline import detect_pii
        tax=load_taxonomy(self.project/'configs/taxonomy_crosswalk.csv')
        good={'attribute_id':'a19_name_full','bbox_1000':[100,100,400,400],'location':'field','evidence_source':'explicit_text'}
        bad={**good,'attribute_id':'a10_face_complete','evidence_source':'visible_region'}
        fixed={**bad,'attribute_id':'a9_face_complete'}
        replies=iter([{'detections':[good,bad]},{'detections':[fixed]}]);prompts=[]
        def generator(image,prompt):prompts.append(prompt);return json.dumps(next(replies)),False
        result=detect_pii(None,tax,generator,repair_partial_entries=True)
        self.assertEqual(result['status'],'ok');self.assertEqual(result['detections'],[good,fixed])
        self.assertEqual(len(prompts),2);self.assertTrue(result['attribute_ids_corrected_by_model'])
        self.assertEqual(result['original_rejected_entries'][0]['entry'],bad)
        def failure(*_):raise RuntimeError('synthetic generation failure')
        failed=detect_pii(None,tax,failure)
        self.assertEqual(failed['failure_stage'],'inference')
        self.assertIn('synthetic generation failure',failed['error'])

    def test_empty_or_moved_correction_cannot_erase_partial_evidence(self):
        from leakbench.pipeline import detect_pii
        tax=load_taxonomy(self.project/'configs/taxonomy_crosswalk.csv')
        good={'attribute_id':'a19_name_full','bbox_1000':[100,100,400,400],'location':'field','evidence_source':'explicit_text'}
        bad={**good,'attribute_id':'a10_face_complete'}
        for correction in ([],[{**bad,'attribute_id':'a9_face_complete','bbox_1000':[0,0,10,10]}]):
            replies=iter([{'detections':[good,bad]},{'detections':correction}])
            result=detect_pii(None,tax,lambda *_:(json.dumps(next(replies)),False),repair_partial_entries=True)
            self.assertEqual(result['status'],'partial');self.assertEqual(result['detections'],[good])

if __name__=='__main__':unittest.main()
