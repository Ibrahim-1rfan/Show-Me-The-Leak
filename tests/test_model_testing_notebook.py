"""Offline checks for notebook logic; these do not run or simulate Qwen quality.

Run: python -m unittest discover -s tests -v
Requires Pillow and pandas, also used by the notebook.
"""
import ast
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'Model_Testing.ipynb').read_text(encoding='utf-8'))
NS = {'__name__': 'notebook_tests'}
for cell in NOTEBOOK['cells']:
    tags = cell.get('metadata', {}).get('tags', [])
    if tags and tags[0] in {'config', 'data_helpers', 'detection_helpers',
                            'inference_helpers', 'pipeline_helpers'}:
        exec(compile(''.join(cell['source']), f"notebook:{tags[0]}", 'exec'), NS)


def detection(attr='a9_face_complete', box=None, source='visible_region'):
    return {'attribute_id': attr, 'bbox_1000': box, 'location': 'test region', 'evidence_source': source}


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = NS['load_taxonomy'](ROOT / 'taxonomy_crosswalk.csv')

    def test_notebook_structure_and_python_syntax(self):
        self.assertEqual(NOTEBOOK['nbformat'], 4)
        ids = set()
        for cell in NOTEBOOK['cells']:
            self.assertNotIn(cell['id'], ids)
            ids.add(cell['id'])
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if cell.get('metadata', {}).get('tags') != ['install']:
                    ast.parse(source)
                # Executed user outputs are retained; edited cells are cleared.
                self.assertTrue(cell['execution_count'] is None or isinstance(cell['execution_count'], int))
                self.assertIsInstance(cell['outputs'], list)

    def test_taxonomy_and_prompt_cover_every_attribute(self):
        self.assertEqual(len(self.taxonomy), 68)
        prompt = NS['build_prompt'](self.taxonomy)
        for attr in self.taxonomy:
            self.assertIn(attr + ':', prompt)
        self.assertNotIn('ground_truth', prompt)

    def test_parse_empty_fenced_and_multiple_instances(self):
        parse = NS['parse_response']
        self.assertEqual(parse('{"detections":[]}', self.taxonomy), [])
        data = [detection(box=[0, 0, 1000, 1000]), detection(box=[20, 40, 100, 200])]
        raw = '```json\n' + json.dumps({'detections': data}) + '\n```'
        self.assertEqual(parse(raw, self.taxonomy), data)

    def test_invalid_boxes_and_unknown_labels_are_not_empty_predictions(self):
        invalid_boxes = [[5, 1, 5, 10], [20, 0, 1, 10], [-1, 0, 1, 10],
                         [0, 0, 1001, 1000], [True, 0, 10, 10], [0, 0, 1.5, 10], [0, 0, 10]]
        for box in invalid_boxes:
            with self.subTest(box=box), self.assertRaises(ValueError):
                NS['parse_response'](json.dumps({'detections': [detection(box=box)]}), self.taxonomy)
        for attr in ['a0_safe', 'invented_label']:
            with self.subTest(attr=attr), self.assertRaises(ValueError):
                NS['parse_response'](json.dumps({'detections': [detection(attr)]}), self.taxonomy)

    def test_sensitive_labels_require_explicit_text(self):
        with self.assertRaises(ValueError):
            NS['parse_response'](json.dumps({'detections': [detection('a16_race')]}), self.taxonomy)
        explicit = detection('a16_race', [10, 20, 80, 90], 'explicit_text')
        self.assertEqual(NS['parse_response'](json.dumps({'detections': [explicit]}), self.taxonomy), [explicit])

    def test_pixel_scaling_padding_and_lossless_redaction(self):
        image = Image.new('RGB', (200, 100), 'white')
        det = detection(box=[100, 200, 400, 600])
        redacted, overlay, enriched = NS['redact_and_draw'](image, [det], 0.01)
        self.assertEqual(enriched[0]['bbox_pixels'], [20, 20, 80, 60])
        self.assertEqual(enriched[0]['redaction_box_pixels'], [18, 19, 82, 61])
        self.assertEqual(redacted.getpixel((20, 20)), (0, 0, 0))
        self.assertEqual(redacted.getpixel((81, 60)), (0, 0, 0))
        self.assertEqual(redacted.getpixel((82, 61)), (255, 255, 255))
        self.assertEqual(image.getpixel((20, 20)), (255, 255, 255))
        self.assertEqual(NS['pixel_box']([0, 0, 1000, 1000], (200, 100), 0.01), [0, 0, 200, 100])
        self.assertIsNone(NS['pixel_box'](None, image.size))

    def test_upright_coordinates_and_metadata_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rotated.jpg'
            exif = Image.Exif()
            exif[274] = 6  # Rotate 90 degrees clockwise for display.
            exif[270] = 'Private metadata fixture'
            Image.new('RGB', (20, 10), 'white').save(path, exif=exif)
            image = NS['read_image'](path)
            self.assertEqual(image.size, (10, 20))
            self.assertEqual(image.info, {})
            redacted, _, _ = NS['redact_and_draw'](image, [detection(box=[0, 0, 1000, 1000])])
            output = Path(tmp) / 'redacted.png'
            redacted.save(output)
            with Image.open(output) as saved:
                self.assertFalse(saved.getexif())

    def test_label_misses_are_distinct_from_unlocalized_and_unannotated(self):
        detections = [detection(box=[0, 0, 10, 10]), detection('a49_phone'),
                      detection('a90_email', [0, 0, 20, 20])]
        result = NS['evaluate_labels'](['a9_face_complete', 'a49_phone', 'a19_name_full'], detections, self.taxonomy)
        self.assertEqual(result['missed_attributes'], ['a19_name_full'])
        self.assertEqual(result['missed_categories'], ['Legal Identifiers'])
        self.assertEqual(result['detected_but_unlocalized_attributes'], ['a49_phone'])
        self.assertEqual(result['ground_truth_without_predicted_box'], ['a19_name_full', 'a49_phone'])
        self.assertEqual(result['unannotated_predictions'], ['a90_email'])
        self.assertAlmostEqual(result['label_recall'], 2 / 3)
        self.assertIsNone(NS['evaluate_labels'](['a0_safe'], [], self.taxonomy)['label_recall'])

    def test_retries_and_truncation_preserve_failure(self):
        responses = iter([('not json', False), ('{"detections":[]}', False)])
        result = NS['detect_pii'](Image.new('RGB', (10, 10)), self.taxonomy,
                                  generator=lambda image, prompt: next(responses))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(result['attempts']), 2)
        result = NS['detect_pii'](Image.new('RGB', (10, 10)), self.taxonomy,
                                  generator=lambda image, prompt: ('{"detections":[]}', True))
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['detections'])

    def test_discovery_and_reproducible_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for image_id in ['one', 'two', 'three']:
                Image.new('RGB', (20, 10)).save(folder / f'{image_id}.jpg')
                (folder / f'{image_id}.json').write_text(json.dumps({'id': image_id, 'labels': ['a0_safe']}))
            (folder / 'missing.json').write_text(json.dumps({'id': 'missing', 'labels': ['a0_safe']}))
            records, issues = NS['discover_records'](folder, self.taxonomy)
            self.assertEqual(len(records), 3)
            self.assertEqual(len(issues), 1)
            first = NS['select_subset'](records, 2, 42)
            self.assertEqual(first, NS['select_subset'](list(reversed(records)), 2, 42))
            self.assertEqual(NS['select_subset'](records, image_ids=['one.jpg'])[0]['id'], 'one')
            with self.assertRaises(ValueError):
                NS['select_subset'](records, image_ids=['missing'])
            manifest = folder / 'manifest.csv'
            manifest.write_text('id\none\n')
            filtered, _ = NS['discover_records'](folder, self.taxonomy, manifest)
            self.assertEqual([r['id'] for r in filtered], ['one'])

    def test_pipeline_outputs_and_failure_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'fixture.png'
            Image.new('RGB', (200, 100), 'white').save(source)
            record = {'id': 'fixture', 'image_path': source, 'annotation_path': folder / 'fixture.json',
                      'labels': ['a9_face_complete', 'a19_name_full']}
            calls = []

            def detector(image, taxonomy):
                calls.append(image.copy())
                return {'status': 'ok', 'detections': [detection(box=[0, 0, 500, 500])] if len(calls) == 1 else [],
                        'attempts': []}

            report = NS['process_record'](record, self.taxonomy, folder / 'result', detector=detector)
            self.assertEqual(report['status'], 'ok')
            self.assertEqual(report['evaluation']['missed_attributes'], ['a19_name_full'])
            self.assertEqual(report['residual_attributes'], [])
            self.assertEqual(calls[1].getpixel((0, 0)), (0, 0, 0))
            self.assertTrue(Path(report['redacted_path']).exists())
            self.assertTrue((folder / 'result' / 'report.json').exists())
            failure = NS['process_record'](record, self.taxonomy, folder / 'failed',
                detector=lambda *args: {'status': 'error', 'detections': None, 'attempts': []})
            self.assertNotIn('evaluation', failure)
            _, attrs, metrics = NS['summarize_reports']([report, failure], self.taxonomy)
            self.assertEqual(metrics['images_failed'], 1)
            self.assertEqual(metrics['ground_truth_label_occurrences'], 2)
            self.assertEqual(metrics['micro_label_recall'], 0.5)
            self.assertEqual(len(attrs), 67)
            self.assertEqual(Image.open(source).getpixel((0, 0)), (255, 255, 255))

    def test_audit_error_does_not_erase_initial_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / 'fixture.png'
            Image.new('RGB', (10, 10)).save(source)
            record = {'id': 'fixture', 'image_path': source, 'annotation_path': folder / 'fixture.json',
                      'labels': ['a0_safe']}
            count = 0

            def detector(*args):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError('Test audit failure')
                return {'status': 'ok', 'detections': [], 'attempts': []}

            report = NS['process_record'](record, self.taxonomy, folder / 'result', detector=detector)
            self.assertEqual(report['status'], 'ok')
            self.assertEqual(report['residual_audit']['status'], 'error')
            self.assertNotIn('residual_attributes', report)


if __name__ == '__main__':
    unittest.main()
