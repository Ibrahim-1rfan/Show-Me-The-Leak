"""Regression checks for the actual Python/profile failures in the user's run."""
import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_environment import install_commands, probe_environment
from benchmark_workflow import COMPATIBILITY_FIELDS, collect_runs, settings_match, write_json
from benchmark_worker import load_job_records


class EnvironmentTests(unittest.TestCase):
    def test_profiles_use_distinct_python311_workers_and_matching_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            florence, _, fc = install_commands('florence', tmp, ['uv'])
            dots, _, dc = install_commands('dots', tmp, ['uv'])
            self.assertNotEqual(florence, dots)
            self.assertEqual(fc[0], ['uv', 'python', 'install', '3.11'])
            self.assertIn('tokenizers==0.19.1', fc[3])
            self.assertIn('--only-binary=tokenizers', fc[3])
            self.assertIn('transformers==4.56.1', dc[3])
            self.assertIn('torch==2.7.0', dc[2])
            self.assertIn('torchvision==0.22.0', dc[2])
            self.assertTrue(all(cmd[0] == str(florence) for cmd in fc[2:]))
            self.assertTrue(all(cmd[0] == str(dots) for cmd in dc[2:]))
            self.assertNotIn('--no-deps', [arg for cmd in fc + dc for arg in cmd])
            self.assertIn('--no-build-isolation', dc[-1])
            self.assertNotIn('flash-attn==2.8.0.post2', [arg for cmd in fc for arg in cmd])

    def test_probe_preserves_import_error_instead_of_generic_install_message(self):
        completed = subprocess.CompletedProcess([], 1, '', 'ImportError: tokenizers>=0.19,<0.20 is required')
        with patch('benchmark_environment.subprocess.run', return_value=completed):
            with self.assertRaisesRegex(RuntimeError, 'tokenizers>=0.19,<0.20'):
                probe_environment('/example/florence/bin/python', 'florence')


class RecoverySelectionTests(unittest.TestCase):
    def test_compatible_backup_requires_explicit_opt_in_and_equal_settings(self):
        settings = {k: 'fixture' for k in COMPATIBILITY_FIELDS}
        saved = {**settings, 'protocol_id': 'old', 'helper_sha256': {'x': 'old-code'}}
        self.assertTrue(settings_match(saved, settings))
        self.assertFalse(settings_match(saved, {**settings, 'max_new_tokens': 9000}))
        self.assertFalse(settings_match({'sample_fingerprint': 'fixture'}, settings))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backup = root / 'backup' / 'sample'
            run = backup / 'qwen3_vl' / '20261009T000000Z'
            run.mkdir(parents=True)
            write_json(run / 'run_config.json', saved)
            write_json(run / 'results.json', [{'image_id': 'image', 'status': 'ok'}])
            rows, _ = collect_runs(root / 'current', ['qwen3_vl'], 'new', additional_sample_dirs=[backup])
            self.assertEqual(rows, [])
            rows, provenance = collect_runs(root / 'current', ['qwen3_vl'], 'new',
                compatible_settings=settings, additional_sample_dirs=[backup])
            self.assertEqual(len(rows), 1)
            self.assertEqual(provenance[0]['protocol_match'], 'same_settings_different_code')
            self.assertEqual(provenance[0]['protocol_id'], 'old')

    def test_latest_failure_remains_visible_across_roots(self):
        settings = {k: 'fixture' for k in COMPATIBILITY_FIELDS}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, stamp, status in [('backup', '20261008T000000Z', 'ok'), ('current', '20261009T000000Z', 'error')]:
                run = root / folder / 'florence2' / stamp
                run.mkdir(parents=True)
                write_json(run / 'run_config.json', {**settings, 'protocol_id': 'old'})
                write_json(run / 'results.json', [{'image_id': 'image', 'status': status}])
            rows, _ = collect_runs(root / 'current', ['florence2'], 'new', compatible_settings=settings,
                                   additional_sample_dirs=[root / 'backup'])
            self.assertEqual(rows[0]['status'], 'error')


class WorkerRecordTests(unittest.TestCase):
    def test_worker_rebases_and_validates_sample_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'image.png').write_bytes(b'fixture-image-content')
            write_json(root / 'image.json', {'id': 'image', 'labels': ['a90_email']})
            item = {'image_id': 'image', 'stratum': 'document', 'image_path': 'image.png',
                    'annotation_path': 'image.json', 'ground_truth_attributes': ['a90_email'],
                    'image_sha256': hashlib.sha256(b'fixture-image-content').hexdigest()}
            write_json(root / 'sample.json', {'sample_fingerprint': 'fixture', 'images': [item]})
            job = {'sample_dir': str(root), 'data_root': str(root),
                   'protocol': {'sample_fingerprint': 'fixture'}}
            _, records = load_job_records(job)
            self.assertEqual(records[0]['image_path'], root / 'image.png')
            (root / 'image.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Image changed'):
                load_job_records(job)

    def test_worker_rejects_paths_outside_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / 'sample.json', {'sample_fingerprint': 'fixture', 'images': [
                {'image_id': 'image', 'image_path': '../outside.png', 'annotation_path': 'x.json'}]})
            with self.assertRaises(ValueError):
                load_job_records({'sample_dir': str(root), 'data_root': str(root),
                                  'protocol': {'sample_fingerprint': 'fixture'}})


if __name__ == '__main__':
    unittest.main()
