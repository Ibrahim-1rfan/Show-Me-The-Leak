"""Export measured results, coverage and per-image evidence, including failures."""
import csv, html, statistics
from pathlib import Path
from .common import read_json,write_json,log,stage,sha
from .dataset import records
from .pipeline import load_taxonomy
from .metrics import build_comparison
from .legacy_workflow import TEXT_RULE_ATTRIBUTES,scope_records
from .worker import valid_completed
from .progress import active_execution

def export_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def table(rows,keys):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'

def report(project,run_dir,models):
    project,run_dir=Path(project),Path(run_dir);out=run_dir/'reports';out.mkdir(exist_ok=True)
    cohort=records(project/'data/vispr_eval_50');taxonomy=load_taxonomy(project/'configs/taxonomy_crosswalk.csv')
    config=read_json(run_dir/'run_config.json');rows=[];runtime=[];gallery=[]
    execution_folder,execution=active_execution(run_dir)
    if execution:
        intended=set(execution['intended_image_ids']);cohort=[r for r in cohort if r['id'] in intended]
    expected_count=len(cohort)
    history_path=project/'configs/historical_baseline.json'
    history=read_json(history_path) if history_path.exists() else {'interpretation':'No historical baseline supplied','runs':[]}
    with stage('Calculate metrics and assemble reports'):
        for model in models:
            for r in cohort:
                folder=run_dir/model/r['category']/Path(r['filename']).stem
                path=folder/'prediction.json'
                if not path.exists():continue
                try:
                    row=read_json(path)
                    if not isinstance(row,dict):raise ValueError('Prediction must be an object')
                except (ValueError,OSError) as exc:
                    row={'image_id':r['id'],'model_key':model,'stratum':r['stratum'],
                        'ground_truth_attributes':[a for a in r['labels'] if a!='a0_safe'],
                        'status':'error','error':'Unreadable prediction record: '+str(exc),'protocol_id':config['protocol_id']}
                if row.get('protocol_id')!=config['protocol_id']:raise ValueError('Mismatched result protocol')
                if row.get('status')=='ok' and not valid_completed(folder,config['protocol_id'],r,model):
                    row={**row,'status':'error','error':'Artifact integrity verification failed; rerun this image.'}
                rows.append(row)
                gallery.append((model,r,folder,row))
            measured=[r for r in rows if r['model_key']==model]
            times=[r['timing']['total_seconds'] for r in measured if r.get('timing')]
            detection_times=[r['inference_seconds'] for r in measured if r.get('inference_seconds') is not None]
            loads={p.stem.removesuffix('_loading'):read_json(p) for p in (run_dir/model).glob('*_loading.json')}
            runtime.append({'model':model,'timed_images':len(times),
                'timing_scope':'Saved prediction timings, including reused predictions; not current execution wall time.',
                'reused_predictions':sum(k.startswith(model+'/') for k in (execution or {}).get('reused_predictions',{})),
                'median_detection_seconds':statistics.median(detection_times) if detection_times else None,
                'historical_median_detection_seconds':[r['median_detection_seconds'] for r in history['runs'] if r['model']==model and r['median_detection_seconds'] is not None],
                'median_image_seconds':statistics.median(times) if times else None,
                'p95_image_seconds':sorted(times)[max(0,__import__('math').ceil(.95*len(times))-1)] if times else None,
                'images_per_second_excluding_load':len(times)/sum(times) if times and sum(times)>0 else None,
                'loading_and_download_by_phase':loads,'peak_gpu_memory_bytes':max((r.get('timing',{}).get('peak_gpu_memory_bytes',0) for r in measured),default=0) or None})
        kwargs=dict(requested_models=models,sample_image_ids=[r['id'] for r in cohort],
            sample_strata={r['id']:r['stratum'] for r in cohort},
            attribute_to_category={k:v['taxonomy_category'] for k,v in taxonomy.items()})
        comparison=build_comparison(rows,taxonomy,**kwargs)
        text_comparison=build_comparison(scope_records(rows,TEXT_RULE_ATTRIBUTES),TEXT_RULE_ATTRIBUTES,**kwargs)
        for name,result in [('full_taxonomy',comparison),('shared_text_scope',text_comparison)]:
            write_json(out/f'{name}.json',result)
            for key in ('summaries','coverage','per_image','per_attribute','per_category'):export_csv(out/f'{name}_{key}.csv',result[key])
        export_csv(out/'runtime.csv',runtime);write_json(out/'all_results.json',rows)
        redaction_checks=[{'model':r['model_key'],'image_id':r['image_id'],'prediction_status':r['status'],
            **r.get('redaction_validation',{'status':'not_verified','technical_pass':False,'privacy_verified':False})} for r in rows]
        export_csv(out/'redaction_checks.csv',redaction_checks)
        image_timings=[{'model':r['model_key'],'image_id':r['image_id'],'stratum':r['stratum'],
            'status':r['status'],**r['timing']} for r in rows if r.get('timing')]
        export_csv(out/'image_timings.csv',image_timings)
        phase_timings=[{'model':model,'phase':p.stem.removesuffix('_timing'),**read_json(p)}
            for model in models for p in (run_dir/model).glob('*_timing.json')]
        export_csv(out/'phase_timings.csv',phase_timings)
        coverage=[r for r in comparison['coverage'] if r['stratum']=='all']
        summary=[r for r in comparison['summaries'] if r['stratum']=='all' and r['scope']=='common_successful']
        candidates=[r for r in summary if r['micro_f1'] is not None]
        complete=bool(coverage) and len(coverage)==len(models) and all(r['successful_images']==expected_count for r in coverage)
        mask_checks_passed=bool(redaction_checks) and all(r.get('technical_pass') for r in redaction_checks)
        winner=max(candidates,key=lambda r:(r['micro_f1'],r['micro_recall']))['model_key'] if expected_count==50 and complete and mask_checks_passed and candidates else None
        conclusion=f'Best model by paired full-taxonomy micro F1, then recall: {winner}.' if winner else (
            'Best model is pending: complete valid coverage and passing mask checks are required for every requested model. '
            'The common-successful tables compare identical images; other tables are diagnostic, not a fair ranking.')
        limitations=('Inference success is not verified privacy removal. Inspect redaction_checks.csv and saved masks; without region truth, privacy completeness requires review. '
            'VISPR supplies image-level labels only. Missing labels are image-attribute occurrences, not physical regions. '
            'Extra labels are unannotated predictions, not confirmed false positives. IoU, reference redaction coverage, '
            'CER, WER and ROUGE-L are unavailable until independently verified region/transcript annotations are supplied. '
            'OCR scores measure native OCR plus the fixed English text rules. Review images for unlocalized predictions and excess masking. '
            'This is a curated training-split diagnostic cohort, not a held-out generalization test. The cohort differs from the old sample by 11 images.')
        errors=[]
        for model in models:
            for path in (run_dir/model).glob('*failure.json'):errors.append({'model':model,**read_json(path)})
        environment={model:{p.stem:read_json(p) for p in (run_dir/model).glob('*_environment.json')} for model in models}
        recommendations=['Inspect missed attributes and unlocalized detections before selecting a model.',
            'Label a separate set of reference boxes and transcripts before assessing localization, redaction coverage or OCR text accuracy.',
            'Review pilot GPU placement, generation calls and stage timings before changing precision, resolution or token limits.']
        write_json(out/'summary.json',{'winner':winner,'selection_criterion':'paired full-taxonomy micro F1 then recall; requires all 50 valid for all requested models',
            'conclusion':conclusion,'limitations':limitations,'runtime':runtime,'coverage':coverage,'failures':errors,
            'phase_timings':phase_timings,
            'redaction_checks':redaction_checks,'all_saved_mask_checks_passed':mask_checks_passed,
            'configuration':config,'execution':execution,'dataset_validation':read_json(project/'data/vispr_eval_50/validation.json'),
            'reporting_source_sha256':{name:sha(Path(__file__).with_name(name)) for name in ('metrics.py','reporting.py')},
            'environment':environment,'recommended_next_steps':recommendations,
            'historical_timing_context':history,
            'status':('complete_inference_needs_privacy_review' if mask_checks_passed else 'redaction_checks_failed') if complete else 'partial_or_pending','protocol_id':config['protocol_id']})
        md=f'# {config["category"].upper()} benchmark report\n\n{conclusion}\n\n{limitations}\n\n'
        if execution:md+=f'Mode: {execution["mode"]}; execution: {execution["execution_id"]}; status: {execution["status"]}.\n\n'
        md+='## Coverage\n\n'+ '\n'.join(f'- {r["model_key"]}: {r["successful_images"]}/{expected_count} valid, {r["partial_images"]} partial, {r["error_images"]} errors, {r["not_attempted_images"]} not attempted.' for r in coverage)
        md+='\n\nPartial detections are retained for inspection and excluded from quality scores and successful coverage. In CSV/JSON, failed_images includes partial attempts and errors.\n'
        md+='\n\nSee CSV/JSON tables for full and text-only scopes, per-category errors and runtime. See report.html for image artifacts.\n'
        md+='\nHistorical timing comparison: '+history['interpretation']+' See runtime.csv for current and previous detection-call medians.\n'
        md+='\n## Next steps\n\n'+'\n'.join('- '+r for r in recommendations)+'\n'
        (out/'report.md').write_text(md,encoding='utf-8')
        page='<html><head><meta charset="utf-8"><style>body{font:15px sans-serif;margin:32px;max-width:1400px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}img{max-width:30%;max-height:360px}details{margin:20px 0}</style></head><body>'
        page+='<h1>'+html.escape(config['category'].upper())+' benchmark</h1><p>'+html.escape(conclusion)+'</p><p>'+html.escape(limitations)+'</p>'
        if execution:
            page+='<h2>Execution reconciliation</h2><p>'+html.escape(f'{execution["mode"]} / {execution["execution_id"]} / {execution["status"]}')+'</p>'
            page+=table(execution['reconciliation'],['model_key','intended_images','attempted_images','valid_predictions',
                'successful_redaction_checks','failed','redaction_failed','skipped','missing_outputs','remaining_unprocessed'])
            export_csv(out/'processing_manifest.csv',read_json(run_dir/'processing_manifest.json'))
        page+='<h2>Coverage</h2>'+table(coverage,['model_key','successful_images','partial_images','error_images','not_attempted_images'])
        page+='<p>Partial detections are retained for inspection and excluded from quality scores and successful coverage. In CSV/JSON, failed_images includes partial attempts and errors.</p>'
        page+='<h2>Redaction verification (separate from inference)</h2>'+table(redaction_checks,
            ['model','image_id','prediction_status','status','technical_pass','privacy_verified','mask_pixels','changed_pixels','issues','warnings'])
        page+='<h2>Paired quality</h2>'+table(summary,['model_key','n_images','micro_precision','micro_recall','micro_f1','missed_count','unannotated_count'])
        page+='<h2>Runtime</h2>'+table(runtime,['model','timed_images','median_image_seconds','p95_image_seconds','peak_gpu_memory_bytes',
            'median_detection_seconds','historical_median_detection_seconds'])+'<p>'+html.escape(history['interpretation'])+'</p>'
        page+='<h2>Download, loading and total phase runtime</h2>'+table(phase_timings,
            ['model','phase','download_seconds','load_seconds','wall_seconds','newly_processed_images'])
        page+='<details><summary>Per-image processing times</summary>'+table(image_timings,
            ['model','image_id','status','preprocessing_seconds','generation_seconds','postprocessing_seconds','generation_calls','total_seconds','peak_gpu_memory_bytes'])+'</details>'
        page+='<h2>Execution failures</h2><pre>'+html.escape(str(errors))+'</pre>'
        page+='<details><summary>Configuration, dataset and GPU environment</summary><pre>'+html.escape(str({'configuration':config,'environment':environment}))+'</pre></details>'
        for model,r,folder,row in gallery:
            ev=row.get('evaluation',{})
            page+='<details><summary>'+html.escape(f'{model} / {r["filename"]} / {row["status"]}')+'</summary>'
            page+='<p>'+html.escape(str({'matched':ev.get('matched_attributes'),'missed':ev.get('missed_attributes'), 'unannotated':ev.get('unannotated_predictions'),
                'detected_but_unlocalized':ev.get('detected_but_unlocalized_attributes'),'error':row.get('error')}))+'</p>'
            artifacts=[*folder.glob('original.*'),folder/'predicted.png']
            if row['status'] in ('ok','partial'):artifacts.extend([folder/'redacted.png',folder/'redaction_mask.png'])
            for path in artifacts:
                if path.exists():page+=f'<img alt="{html.escape(path.name)}" src="../{html.escape(path.relative_to(run_dir).as_posix(),quote=True)}">'
            link=folder.relative_to(run_dir).as_posix()
            page+=f'<p><a href="../{link}/prediction.json">Structured prediction</a> | <a href="../{link}/raw_predictions.json">Native output and attempts</a></p></details>'
        (out/'report.html').write_text(page+'</body></html>',encoding='utf-8')
        log('Reports exported',path=str(out),status='complete' if complete else 'partial_or_pending')
        return out
