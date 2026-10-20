"""Run notebook benchmark jobs under an isolated model interpreter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_job_records(job):
    """Resolve portable sample paths and verify labels before reusing the cohort."""
    sample_dir = Path(job['sample_dir'])
    sample = json.loads((sample_dir / 'sample.json').read_text(encoding='utf-8'))
    if sample['sample_fingerprint'] != job['protocol']['sample_fingerprint']:
        raise ValueError('Worker job and saved sample fingerprint differ.')
    root = Path(job['data_root']).resolve()
    selected = []
    for item in sample['images']:
        image_path = (root / item['image_path']).resolve()
        annotation_path = (root / item['annotation_path']).resolve()
        for path in (image_path, annotation_path):
            path.relative_to(root)
            if not path.is_file():
                raise FileNotFoundError(path)
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != item['image_sha256']:
            raise ValueError(f"Image changed since sample selection: {item['image_id']}")
        if annotation['id'] != item['image_id'] or sorted(annotation['labels']) != item['ground_truth_attributes']:
            raise ValueError(f"Annotation changed since sample selection: {item['image_id']}")
        selected.append({'id': item['image_id'], 'stratum': item['stratum'], 'labels': annotation['labels'],
                         'image_path': image_path, 'annotation_path': annotation_path})
    return sample, selected


def main(job_path):
    import benchmark_pipeline as pipeline
    from benchmark_adapters import MODEL_SPECS, load_adapter
    from benchmark_ocr_adapters import OCR_MODEL_SPECS, create_ocr_adapter
    from benchmark_workflow import VLM_KEYS, run_one_model

    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    sample, selected = load_job_records(job)
    pipeline.FORMAT_RETRIES = job['protocol']['format_retries']
    specs = {**MODEL_SPECS, **OCR_MODEL_SPECS}
    profile = job['profile']
    protocol = job['protocol']

    def loader(key):
        if specs[key]['profile'] != profile:
            raise ValueError(f'{key} does not belong to the {profile} worker environment.')
        kwargs = {'max_new_tokens': protocol['max_new_tokens'], 'max_input_edge': protocol['max_input_edge'],
                  'load_in_4bit': protocol['quantize_vlm_chandra'] if key in VLM_KEYS or key == 'chandra' else False}
        return create_ocr_adapter(key, **kwargs) if key in OCR_MODEL_SPECS else load_adapter(key, **kwargs)

    status = {}
    for key in job['models']:
        folder, reports = run_one_model(key, loader, selected, sample['taxonomy'], job['sample_dir'], protocol,
            pipeline.process_record, pipeline.detect_pii, audit=protocol['audit_redacted'],
            padding=protocol['redaction_padding'])
        config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
        status[key] = {'run_directory': str(folder), 'successful_images': sum(r['status'] == 'ok' for r in reports),
                       'requested_images': len(reports), 'run_error': config.get('run_error')}
        print(json.dumps({key: status[key]}), flush=True)
    Path(job_path).with_suffix('.status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    return 1 if any(s['run_error'] for s in status.values()) else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.job))
