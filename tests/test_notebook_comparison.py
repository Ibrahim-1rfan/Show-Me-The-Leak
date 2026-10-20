"""Execute the real notebook comparison cell on empty and synthetic saved runs.

This validates notebook wiring only; fixture results are never model scores.
"""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmark_metrics import build_comparison
from benchmark_workflow import (MODEL_KEYS, TEXT_RULE_ATTRIBUTES, attach_references,
    collect_runs, load_references, scope_records, write_json)

ROOT = Path(__file__).resolve().parents[1]


class NotebookComparisonTests(unittest.TestCase):
    def test_comparison_cell_handles_missing_models_then_all_saved_models(self):
        notebook = json.loads((ROOT / 'Model_Testing.ipynb').read_text(encoding='utf-8'))
        comparison_code = next(''.join(c['source']) for c in notebook['cells']
            if c.get('metadata', {}).get('tags') == ['compare'])
        with (ROOT / 'taxonomy_crosswalk.csv').open(encoding='utf-8-sig', newline='') as f:
            taxonomy = {r['attribute_id']: r for r in csv.DictReader(f)}
        images = [{'image_id': f'image-{i:02d}', 'stratum': 'document' if i < 25 else 'general',
                   'image_size': [10, 10]} for i in range(50)]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            ns = dict(Path=Path, pd=pd, json=json, hashlib=hashlib, display=lambda *args: None,
                sample_dir=folder, COMPARE_MODELS=MODEL_KEYS, protocol_id='test-protocol',
                RESULTS_ROOTS=[], ALLOW_COMPATIBLE_SAVED_RUNS=False, protocol_settings={},
                REFERENCE_PATH=None, sample={'sample_fingerprint': 'fixture', 'images': images},
                taxonomy=taxonomy, IOU_THRESHOLD=0.5, TEXT_RULE_ATTRIBUTES=TEXT_RULE_ATTRIBUTES,
                build_comparison=build_comparison, collect_runs=collect_runs,
                load_references=load_references, attach_references=attach_references,
                scope_records=scope_records, write_json=write_json)
            exec(compile(comparison_code, 'notebook:compare', 'exec'), ns)
            empty = ns['comparisons']['full_taxonomy']
            self.assertEqual(empty['common_image_ids'], [])
            self.assertTrue(all(row['micro_f1'] is None for row in empty['summaries']))
            self.assertEqual(next(r['expected_images'] for r in empty['coverage']
                if r['model_key'] == 'qwen3_vl' and r['stratum'] == 'document'), 25)
            for model in MODEL_KEYS:
                run_dir = folder / model / '20260101T000000Z'
                run_dir.mkdir(parents=True)
                write_json(run_dir / 'run_config.json', {'protocol_id': 'test-protocol'})
                write_json(run_dir / 'results.json', [dict(image_id=r['image_id'], model_key=model,
                    stratum=r['stratum'], status='ok', ground_truth_attributes=['a90_email'],
                    predicted_attributes=['a90_email'], detections=[]) for r in images])
            exec(compile(comparison_code, 'notebook:compare', 'exec'), ns)
            for scope in ('full_taxonomy', 'shared_text_labels'):
                comparison = ns['comparisons'][scope]
                self.assertEqual(len(comparison['common_image_ids']), 50)
                scores = [r for r in comparison['summaries']
                    if r['scope'] == 'common_successful' and r['stratum'] == 'all']
                self.assertEqual(len(scores), 6)
                self.assertTrue(all(r['micro_f1'] == 1.0 and r['n_images'] == 50 for r in scores))
                self.assertTrue(all(r['n_text_images'] == 0 and r['n_localization_images'] == 0 for r in scores))
                self.assertTrue((folder / 'comparisons' / 'test-protoco' / scope / 'summaries.csv').exists())


if __name__ == '__main__':
    unittest.main()
