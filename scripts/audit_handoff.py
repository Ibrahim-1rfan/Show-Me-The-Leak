"""Audit imported v2 runs without changing them or executing any model.

Private per-image evidence stays in ignored docs/handoff_20261021. The public
summary contains counts, metrics, identifiers and hashes, never OCR text/images.
"""
import ast
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from leakbench.common import read_json, sha, write_json
from leakbench.dataset import records
from leakbench.legacy_workflow import TEXT_RULE_ATTRIBUTES, scope_records
from leakbench.metrics import build_comparison
from leakbench.pipeline import load_taxonomy
from leakbench.validation import validate_redaction


def equivalent_metrics(left, right):
    """Per-image row order and floating reduction roundoff are not discrepancies."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent_metrics(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        if left and isinstance(left[0], dict) and 'image_id' in left[0]:
            key = lambda row: (row.get('model_key', ''), row['image_id'])
            left, right = sorted(left, key=key), sorted(right, key=key)
        return len(left) == len(right) and all(equivalent_metrics(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def main():
    private = ROOT / 'docs/handoff_20261021'
    private.mkdir(parents=True, exist_ok=True)
    cohort = {r['id']: r for r in records(ROOT / 'data/vispr_eval_50')}
    taxonomy = load_taxonomy(ROOT / 'configs/taxonomy_crosswalk.csv')
    output = {'audit_date': '2026-10-21', 'runs': [], 'notebook_evidence': []}
    details = []
    preservation = []
    full_panels = {}
    for config_path in sorted([*ROOT.glob('vlm_*/**/run_config.json'), *ROOT.glob('ocr_*/**/run_config.json'),
                               *ROOT.glob('20*T*Z/run_config.json')]):
        run = config_path.parent
        config = read_json(config_path)
        # Old exports are kept but this audit only evaluates execution-aware runs.
        if 'mode' not in config:
            continue
        for path in run.rglob('*'):
            if path.is_file():
                preservation.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        executions = [read_json(p) for p in (run / 'executions').glob('*/execution.json')]
        execution = executions[-1]
        source_checks = {}
        for name, expected in config['source_hashes'].items():
            p = run / name.replace('src/', 'source/', 1)
            source_checks[name] = p.exists() and sha(p) == expected
        rows = []
        models = []
        for model in config['settings']['models']:
            predictions = []
            checks = []
            raw_counts = Counter()
            errors = Counter()
            for p in sorted((run / model).glob('*/*/prediction.json')):
                pred = read_json(p)
                rows.append(pred)
                predictions.append(pred)
                hashes = pred.get('artifact_sha256', {})
                intact = bool(hashes) and all((p.parent / n).exists() and sha(p.parent / n) == h for n, h in hashes.items())
                record = cohort[pred['image_id']]
                identity = pred.get('input_sha256') == record['image_sha256'] and pred.get('annotation_sha256') == record['annotation_sha256']
                validation = None
                original = next(p.parent.glob('original.*'), None)
                if original and (p.parent / 'redacted.png').exists():
                    validation, _ = validate_redaction(original, p.parent / 'redacted.png', pred.get('detections', []), record['labels'],
                        prediction_status=pred['status'], expected_sha256=record['image_sha256'],
                        in_scope_labels=TEXT_RULE_ATTRIBUTES if config['category'] == 'ocr' else None)
                raw_path = p.parent / 'raw_predictions.json'
                if raw_path.exists():
                    raw = read_json(raw_path)
                    for native in raw.get('native_outputs', []):
                        raw_counts['text_regions'] += len(native.get('text_regions', []))
                        meta = native.get('metadata', {})
                        raw_counts['truncated_outputs'] += bool(meta.get('truncated'))
                if pred.get('failure_stage') or pred.get('error'):
                    # Error strings can contain rejected OCR/JSON text: publish
                    # stage counts only; preserve detailed errors in private runs.
                    errors[str(pred.get('failure_stage') or 'unspecified')] += 1
                row = {'run_id': run.name, 'model': model, 'image_id': pred['image_id'], 'status': pred['status'],
                       'artifacts_intact': intact, 'input_identity_matches': identity,
                       'redaction': validation, 'saved_validation_matches_recheck': validation == pred.get('redaction_validation') if validation else None,
                       'detections': len(pred.get('detections', [])), 'folder': p.parent.relative_to(ROOT).as_posix()}
                checks.append(row)
                details.append(row)
            review = run / model / 'pilot_review.json'
            review_checks = {}
            if review.exists():
                review_data = read_json(review)
                for image_id, expected in review_data['prediction_sha256'].items():
                    record = cohort[image_id]
                    path = run / model / record['category'] / Path(record['filename']).stem / 'prediction.json'
                    review_checks[image_id] = path.exists() and sha(path) == expected
            runtime = next((x for x in read_json(run / 'reports/summary.json')['runtime'] if x['model'] == model), {})
            models.append({'model': model, 'prediction_statuses': dict(Counter(p['status'] for p in predictions)),
                'all_artifacts_intact': all(x['artifacts_intact'] for x in checks) if checks else None,
                'input_identity_matches': all(x['input_identity_matches'] for x in checks) if checks else None,
                'saved_validation_matches': all(x['saved_validation_matches_recheck'] for x in checks if x['redaction'] is not None),
                'redacted_files': sum(x['redaction'] is not None for x in checks),
                'nonempty_masks': sum(bool((x['redaction'] or {}).get('mask_pixels')) for x in checks),
                'mask_application_pass': sum(bool((x['redaction'] or {}).get('mask_application_pass')) for x in checks),
                'technical_pass': sum(bool((x['redaction'] or {}).get('technical_pass')) for x in checks),
                'privacy_verified': sum(bool((x['redaction'] or {}).get('privacy_verified')) for x in checks),
                'raw_counts': dict(raw_counts), 'failure_stages': dict(errors), 'review_hash_checks': review_checks,
                'runtime': {k: runtime.get(k) for k in ['median_detection_seconds', 'median_image_seconds', 'p95_image_seconds', 'peak_gpu_memory_bytes']}})
        selected = [cohort[i] for i in execution['intended_image_ids']]
        kwargs = dict(requested_models=config['settings']['models'], sample_image_ids=[r['id'] for r in selected],
                      sample_strata={r['id']: r['stratum'] for r in selected},
                      attribute_to_category={k: v['taxonomy_category'] for k, v in taxonomy.items()})
        comparisons = {}
        matches = {}
        for name, data, labels in [('full_taxonomy', rows, taxonomy), ('shared_text_scope', scope_records(rows, TEXT_RULE_ATTRIBUTES), TEXT_RULE_ATTRIBUTES)]:
            comp = build_comparison(data, labels, **kwargs)
            saved = read_json(run / 'reports' / (name + '.json'))
            matches[name] = equivalent_metrics(comp, saved)
            comparisons[name] = comp['summaries']
        if config['mode'] == 'full' and len(rows) > len(full_panels.get(config['category'], [])):
            full_panels[config['category']] = rows
        # A useful paired OCR comparison excludes Florence's two-image gate residue.
        if config['mode'] == 'full':
            pair_models = ['chandra', 'dots_ocr'] if config['category'] == 'ocr' else ['qwen3_vl', 'qwen25_vl']
            paired_kwargs = {**kwargs, 'requested_models': pair_models}
            pair = [x for x in rows if x['model_key'] in paired_kwargs['requested_models']]
            comparisons['paired_' + config['category']] = build_comparison(pair, taxonomy, **paired_kwargs)['summaries']
        output['runs'].append({'run_id': run.name, 'relative_path': run.relative_to(ROOT).as_posix(), 'category': config['category'],
            'mode': config['mode'], 'protocol_id': config['protocol_id'], 'selection_fingerprint': config['selection_fingerprint'],
            'source_hash_checks': source_checks, 'metrics_match_saved_report': matches, 'execution': execution,
            'models': models, 'summaries': comparisons})
    if set(full_panels) == {'vlm', 'ocr'}:
        panel_models = ['qwen3_vl', 'qwen25_vl', 'chandra', 'dots_ocr']
        panel = [p for group in full_panels.values() for p in group if p['model_key'] in panel_models]
        panel_kwargs = dict(requested_models=panel_models, sample_image_ids=list(cohort),
            sample_strata={k: r['stratum'] for k, r in cohort.items()},
            attribute_to_category={k: v['taxonomy_category'] for k, v in taxonomy.items()})
        output['paired_four_models'] = {
            'full_taxonomy': build_comparison(panel, taxonomy, **panel_kwargs)['summaries'],
            'shared_text_scope': build_comparison(scope_records(panel, TEXT_RULE_ATTRIBUTES), TEXT_RULE_ATTRIBUTES, **panel_kwargs)['summaries']}
    for p in sorted((private / 'notebooks_received').glob('0[12]*.ipynb')):
        notebook = read_json(p)
        events = []
        reconciliation = []
        for cell in notebook['cells']:
            for item in cell.get('outputs', []):
                for line in ''.join(item.get('text', [])).splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and ('backup' in event.get('message', '').lower() or event.get('message') == 'Execution plan'):
                        events.append(event)
                plain = ''.join(item.get('data', {}).get('text/plain', []))
                if plain.startswith('[{') and 'intended_images' in plain:
                    reconciliation = ast.literal_eval(plain)
        plans = [x for x in events if x.get('message') == 'Execution plan']
        backups = [x for x in events if '/results/' in x.get('destination', '') and len(x['destination'].split('/results/')[1].split('/')) == 2]
        output['notebook_evidence'].append({'notebook': p.name, 'sha256': sha(p), 'plans': plans,
            'reconciliation': reconciliation, 'last_run_backup': backups[-1] if backups else None})
        write_json(private / (p.stem + '_events.json'), events)
    write_json(private / 'prediction_audit.json', details)
    write_json(private / 'preservation.json', preservation)
    # The public summary deliberately omits raw text, errors, images, review notes
    # and saved absolute paths/settings already available in source configuration.
    for run in output['runs']:
        execution = run['execution']
        run['execution'] = {k: execution[k] for k in ['execution_id', 'status', 'reconciliation', 'wall_seconds']}
    write_json(ROOT / 'docs/audit/latest_benchmark.json', output)
    print(json.dumps({'runs_audited': len(output['runs']), 'predictions_audited': len(details),
        'preserved_files_hashed': len(preservation), 'hash_failures': sum(not x['artifacts_intact'] for x in details),
        'mask_recheck_disagreements': sum(x['saved_validation_matches_recheck'] is False for x in details),
        'metrics_match': {x['run_id']: x['metrics_match_saved_report'] for x in output['runs']}}, indent=2))


if __name__ == '__main__':
    main()
