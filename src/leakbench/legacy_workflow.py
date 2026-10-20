"""Dataset, OCR-to-PII mapping, and run bookkeeping for Model_Testing.ipynb.

No model weights are imported by this module. The notebook runs inference on a GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL_KEYS = ['qwen3_vl', 'qwen25_vl', 'llama32_vision', 'chandra', 'dots_ocr', 'florence2']
VLM_KEYS = {'qwen3_vl', 'qwen25_vl', 'llama32_vision'}
RULES_VERSION = 'visible-text-fields-v1'

# A deliberately fixed, conservative mapper shared by all OCR pipelines. These
# rules indicate visible information types, never a person's inferred traits.
TEXT_RULES = {
    'a90_email': r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
    'a49_phone': r'(?:\b(?:phone|telephone|tel|mobile|fax)\s*[:=.]?\s*\+?[\d ()-]{7,}\d|\+\d[\d ()-]{8,}\d)',
    'a85_username': r'(?:\b(?:username|user name|handle)\s*[:=]\s*\S+|(?<![\w.])@[A-Z0-9_]{2,})',
    'a19_name_full': r'\b(?:full name|name)\s*[:=]\s*[A-Z][^\n:]{1,80}',
    'a20_name_first': r'\b(?:first name|given name)\s*[:=]\s*[A-Z][^\n:]{1,60}',
    'a21_name_last': r'\b(?:last name|surname|family name)\s*[:=]\s*[A-Z][^\n:]{1,60}',
    'a23_birth_city': r'\b(?:place of birth|birthplace|birth city)\s*[:=]\s*\S+',
    'a24_birth_date': r'\b(?:date of birth|dob|birth date)\s*[:=]?\s*\d',
    'a25_nationality': r'\bnationality\s*[:=]\s*\S+',
    'a4_gender': r'\b(?:gender|sex)\s*[:=]\s*\S+',
    'a27_marital_status': r'\bmarital status\s*[:=]\s*\S+',
    'a29_ausweis': r'\b(?:national identification|national identity|identity card|national id)\b',
    'a30_credit_card': r'\b(?:credit card|cardholder|card number|card no\.)\b',
    'a31_passport': r'\bpassport\b',
    'a32_drivers_license': r"\bdriver(?:'s|s)? licen[sc]e\b",
    'a33_student_id': r'\bstudent (?:id|identification|number)\b',
    'a35_mail': r'\b(?:postage paid|return address|postal mail)\b',
    'a37_receipt': r'\b(?:receipt|subtotal|sales tax)\b',
    'a38_ticket': r'\b(?:boarding pass|admit one|e-ticket|ticket number|seat no\.)\b',
    'a46_occupation': r'\b(?:occupation|job title|profession)\s*[:=]\s*\S+',
    'a55_religion': r'\breligion\s*[:=]\s*\S+',
    'a56_sexual_orientation': r'\bsexual orientation\s*[:=]\s*\S+',
    'a70_education_history': r'\b(?:education history|academic qualifications)\s*[:=]\s*\S+',
    # A generic address field cannot establish completeness; use partial labels.
    'a79_address_home_partial': r'\b(?:home address|residential address)\s*[:=]\s*\S+',
    'a75_address_current_partial': r'\b(?:current address|visited location)\s*[:=]\s*\S+',
    'a82_date_time': r'\b(?:date|time)\s*[:=]\s*\d[\d/:. -]{2,}',
    'a103_license_plate_complete': r'\b(?:license plate|licence plate|registration plate)\s*[:=]\s*[A-Z0-9][A-Z0-9 -]{2,}',
}
TEXT_RULE_ATTRIBUTES = sorted(TEXT_RULES)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def document_attributes(taxonomy):
    return {key for key, row in taxonomy.items() if row.get('vispr_native_group') == 'Documents'}


def stratified_sample(records, taxonomy, size=50, seed=42, image_ids=None):
    """Exactly balanced label-derived document/general groups; never silently shrink."""
    if type(size) is not int or size < 50:
        raise ValueError('The multi-model benchmark requires at least 50 images.')
    doc_attrs = document_attributes(taxonomy)
    if not doc_attrs:
        raise ValueError('No Documents group found in the taxonomy.')
    annotated = [dict(r, stratum='document' if set(r['labels']) & doc_attrs else 'general') for r in records]
    if image_ids:
        ids = list(dict.fromkeys(Path(x).stem for x in image_ids))
        by_id = {r['id']: r for r in annotated}
        if len(ids) < 50 or set(ids) - by_id.keys():
            raise ValueError('Explicit sample needs at least 50 distinct available image IDs.')
        selected = [by_id[i] for i in ids]
        if {r['stratum'] for r in selected} != {'document', 'general'}:
            raise ValueError('Explicit sample must contain both document and general images.')
        return selected
    rng = random.Random(seed)
    selected = []
    for stratum, count in [('document', size // 2), ('general', size - size // 2)]:
        pool = sorted((r for r in annotated if r['stratum'] == stratum), key=lambda r: r['id'])
        if len(pool) < count:
            raise ValueError(f'Need {count} {stratum} images; found only {len(pool)}.')
        selected.extend(rng.sample(pool, count))
    return sorted(selected, key=lambda r: r['id'])


def prepare_sample(records, taxonomy, data_root, output_root, size=50, seed=42, image_ids=None):
    from PIL import Image, ImageOps

    selected = stratified_sample(records, taxonomy, size, seed, image_ids)
    data_root = Path(data_root).resolve()
    manifest = []
    for record in selected:
        path = Path(record['image_path']).resolve()
        with Image.open(path) as original:
            original.load()  # Catch corrupt images before committing the comparison cohort.
            upright = ImageOps.exif_transpose(original)
            width, height = upright.size
        manifest.append({
            'image_id': record['id'], 'stratum': record['stratum'],
            'image_path': path.relative_to(data_root).as_posix(),
            'annotation_path': Path(record['annotation_path']).resolve().relative_to(data_root).as_posix(),
            'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'image_size': [width, height], 'ground_truth_attributes': sorted(record['labels']),
        })
    fingerprint = digest({'images': manifest, 'taxonomy': taxonomy})
    folder = Path(output_root) / f'sample_{len(selected)}_{fingerprint[:12]}'
    folder.mkdir(parents=True, exist_ok=True)
    sample = {'sample_fingerprint': fingerprint, 'seed': seed, 'images': manifest,
              'stratum_definition': 'document = any VISPR Documents-group label; general = all other images',
              'selection': 'balanced' if not image_ids else 'explicit', 'taxonomy': taxonomy}
    write_json(folder / 'sample.json', sample)
    template_path = folder / 'reference_template.json'
    if not template_path.exists():
        write_json(template_path, {
            'sample_fingerprint': fingerprint,
            'instructions': 'Copy to references.json. Add manually verified boxes and/or exact visible-text transcripts. Leave unavailable values null. Coordinates use upright original image pixels, [left,top,right,bottom]. boxes_complete=true means ALL instances of the declared box_attribute_scope are annotated, including confirmed negatives.',
            'images': {r['image_id']: {'image_size': r['image_size'], 'boxes_complete': False,
                'box_attribute_scope': [], 'ground_truth_boxes': None, 'reference_text': None} for r in manifest},
        })
    return selected, sample, folder


def load_references(path, sample, taxonomy):
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('sample_fingerprint') != sample['sample_fingerprint']:
        raise ValueError('Reference annotations belong to a different sample.')
    result = payload.get('images')
    if not isinstance(result, dict):
        raise ValueError('Reference images must be an ID-keyed object.')
    by_id = {r['image_id']: r for r in sample['images']}
    for image_id, ref in result.items():
        if image_id not in by_id or ref.get('image_size') != by_id[image_id]['image_size']:
            raise ValueError(f'Reference image ID/dimensions mismatch: {image_id}')
        scope = ref.get('box_attribute_scope', [])
        if not isinstance(scope, list) or set(scope) - (taxonomy.keys() - {'a0_safe'}):
            raise ValueError(f'Invalid box attribute scope for {image_id}')
        if ref.get('boxes_complete') and (not scope or not isinstance(ref.get('ground_truth_boxes'), list)):
            raise ValueError(f'Complete boxes need an explicit scope and list: {image_id}')
        w, h = ref['image_size']
        for box in ref.get('ground_truth_boxes') or []:
            coords = box.get('bbox_pixels')
            if box.get('attribute_id') not in scope or not valid_pixel_box(coords, (w, h)):
                raise ValueError(f'Invalid reference box for {image_id}: {box}')
        if ref.get('reference_text') is not None and not isinstance(ref['reference_text'], str):
            raise ValueError(f'reference_text must be a transcript string or null: {image_id}')
    return result


def valid_pixel_box(box, size):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])


def ocr_to_pii(native, image_size, taxonomy):
    """Classify only extracted text, not layout markup or numeric box coordinates."""
    detections = []
    regions = native.get('text_regions')
    if not isinstance(regions, list):
        raise ValueError('OCR adapter did not return a text_regions list.')
    w, h = image_size
    for index, region in enumerate(regions):
        text = region.get('text', '')
        if not isinstance(text, str):
            raise ValueError('OCR region text must be a string.')
        pixel = region.get('bbox_pixels')
        if pixel is not None and not valid_pixel_box(pixel, image_size):
            raise ValueError('OCR adapter returned out-of-frame or invalid coordinates.')
        normalized = None if pixel is None else [
            max(0, math.floor(pixel[0] / w * 1000)), max(0, math.floor(pixel[1] / h * 1000)),
            min(1000, math.ceil(pixel[2] / w * 1000)), min(1000, math.ceil(pixel[3] / h * 1000)),
        ]
        for attr, pattern in TEXT_RULES.items():
            if attr not in taxonomy or not re.search(pattern, text, re.I):
                continue
            # Avoid treating a first/last-name field as a full-name field as well.
            if attr == 'a19_name_full' and re.search(r'\b(first|given|last|family) name\s*[:=]', text, re.I) and not re.search(r'\bfull name\s*[:=]', text, re.I):
                continue
            if attr == 'a82_date_time' and re.search(r'\b(date of birth|birth date|dob)\b', text, re.I):
                continue
            detections.append({'attribute_id': attr, 'bbox_1000': normalized,
                'location': f'OCR region {index + 1}; entire region is redacted',
                'evidence_source': 'explicit_text', 'source_region_id': region.get('region_id', index),
                'ocr_bbox_pixels': pixel})
    return detections


def make_detector(adapter, model_key, vlm_detect):
    """Bridge heterogeneous native outputs into the previous notebook's detector."""
    def detector(image, taxonomy):
        started = time.perf_counter()
        native_calls = []
        try:
            if model_key in VLM_KEYS:
                def generate(image, prompt):
                    native = adapter.infer(image, prompt)
                    native_calls.append(native)
                    return native['raw_text'], native.get('metadata', {}).get('truncated', native.get('truncated', False))
                result = vlm_detect(image, taxonomy, generator=generate)
            else:
                native = adapter.extract(image) if hasattr(adapter, 'extract') else adapter.infer(image, '')
                native_calls.append(native)
                if native.get('parse_success') is False:
                    raise ValueError('Native OCR parsing failed; see native_outputs and parse_warnings.')
                if native.get('metadata', {}).get('truncated', native.get('truncated', False)):
                    raise ValueError('OCR output reached token limit; incomplete output is not a valid empty result.')
                result = {'status': 'ok', 'detections': ocr_to_pii(native, image.size, taxonomy), 'attempts': []}
                result['predicted_text'] = native.get('predicted_text', '\n'.join(r['text'] for r in native['text_regions']))
            result['native_outputs'] = native_calls
        except Exception as exc:
            result = {'status': 'error', 'detections': None, 'attempts': [], 'native_outputs': native_calls,
                      'error': f'{type(exc).__name__}: {exc}'}
        result['inference_seconds'] = time.perf_counter() - started
        return result
    return detector


def normalize_report(report, record, model_key):
    prediction = report.get('prediction', {})
    ev = report.get('evaluation', {})
    return {**report, 'model_key': model_key, 'stratum': record['stratum'],
            'ground_truth_attributes': sorted(set(record['labels']) - {'a0_safe'}),
            'predicted_attributes': ev.get('predicted_attributes', []),
            'detections': prediction.get('detections') or [],
            'predicted_text': prediction.get('predicted_text'),
            'inference_seconds': prediction.get('inference_seconds'),
            'pipeline': 'direct_visual_pii' if model_key in VLM_KEYS else 'native_ocr_plus_fixed_text_rules',
            'rules_version': None if model_key in VLM_KEYS else RULES_VERSION}


def run_one_model(model_key, loader, selected, taxonomy, sample_dir, protocol, process_record, vlm_detect,
                  audit=True, padding=0.01):
    """Sequential GPU lifecycle; startup and image failures remain in the results."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = Path(sample_dir) / model_key / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'model_key': model_key, 'started_at_utc': stamp, **protocol}
    write_json(run_dir / 'run_config.json', manifest)
    reports, adapter = [], None
    try:
        start = time.perf_counter()
        adapter = loader(model_key)
        manifest['load_seconds'] = time.perf_counter() - start
        manifest['adapter_metadata'] = getattr(adapter, 'metadata', {})
        detector = make_detector(adapter, model_key, vlm_detect)
        for index, record in enumerate(selected, 1):
            print(f"{model_key} [{index}/{len(selected)}] {record['id']}", flush=True)
            raw = process_record(record, taxonomy, run_dir / record['id'], detector=detector,
                                 audit=audit, padding=padding)
            report = normalize_report(raw, record, model_key)
            reports.append(report)
            write_json(run_dir / record['id'] / 'report.json', report)
            write_json(run_dir / 'results.json', reports)
    except Exception as exc:
        manifest['run_error'] = f'{type(exc).__name__}: {exc}'
        done = {r['image_id'] for r in reports}
        for record in selected:
            if record['id'] not in done:
                reports.append(normalize_report({'image_id': record['id'], 'status': 'error',
                    'error': manifest['run_error']}, record, model_key))
        print(f"{model_key} failed: {manifest['run_error']}", flush=True)
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                manifest['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        manifest['completed_images'] = len(reports)
        write_json(run_dir / 'run_config.json', manifest)
        write_json(run_dir / 'results.json', reports)
    return run_dir, reports


COMPATIBILITY_FIELDS = ('protocol_version', 'sample_fingerprint', 'prompt', 'rules_version',
    'max_input_edge', 'max_new_tokens', 'format_retries', 'redaction_padding',
    'audit_redacted', 'quantize_vlm_chandra')


def settings_match(saved, current):
    """Explicit cross-code comparison; all substantive evaluation settings must match."""
    return all(key in saved and key in current and saved[key] == current[key] for key in COMPATIBILITY_FIELDS)


def collect_runs(sample_dir, requested_models, protocol_id, *, compatible_settings=None, additional_sample_dirs=()):
    """Latest matching run per model, optionally including clearly marked old code versions.

    Strict protocol-ID selection remains the default. Notebook recovery can explicitly
    include equal-setting runs from backup roots; provenance records the difference.
    A later failure is never silently replaced by an older success.
    """
    records, provenance = [], []
    folders = list(dict.fromkeys(Path(p).resolve() for p in [sample_dir, *additional_sample_dirs]))
    for model_key in requested_models:
        candidates = [p for folder in folders for p in (folder / model_key).glob('*/run_config.json')]
        candidates.sort(key=lambda p: (p.parent.name, str(p)), reverse=True)
        for config_path in candidates:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            results_path = config_path.with_name('results.json')
            exact = config.get('protocol_id') == protocol_id
            compatible = compatible_settings is not None and settings_match(config, compatible_settings)
            if not (exact or compatible) or not results_path.exists():
                continue
            results = json.loads(results_path.read_text(encoding='utf-8'))
            # Backups retain original /content/... paths. Relocate output previews
            # in memory without altering the archived reports.
            for row in results:
                for field, filename in [('redacted_path', 'redacted.png'), ('overlay_path', 'predicted_boxes.png')]:
                    local = config_path.parent / row['image_id'] / filename
                    if field in row and local.is_file():
                        row[field] = str(local)
            records.extend(results)
            provenance.append({'model_key': model_key, 'run_directory': str(config_path.parent),
                               'rows': len(results), 'protocol_id': config.get('protocol_id'),
                               'protocol_match': 'exact' if exact else 'same_settings_different_code',
                               'helper_sha256': config.get('helper_sha256', {})})
            break
        else:
            provenance.append({'model_key': model_key, 'run_directory': None, 'rows': 0, 'protocol_id': protocol_id})
    return records, provenance


def attach_references(records, references):
    """Human references only enter evaluation, after inference has finished."""
    result = []
    for record in records:
        ref = references.get(record['image_id'], {})
        row = {**record, **{k: ref[k] for k in ['reference_text', 'boxes_complete', 'ground_truth_boxes', 'box_attribute_scope'] if k in ref}}
        # PII box references may cover only declared labels; do not penalize other types.
        scope = set(ref.get('box_attribute_scope', []))
        if ref.get('boxes_complete'):
            row['detections'] = [d for d in record.get('detections', []) if d['attribute_id'] in scope]
        result.append(row)
    return result


def scope_records(records, attributes):
    attrs = set(attributes)
    scoped = []
    for record in records:
        row = {**record, 'ground_truth_attributes': sorted(set(record['ground_truth_attributes']) & attrs),
               'predicted_attributes': sorted(set(record.get('predicted_attributes') or []) & attrs)}
        for key in ('detections', 'ground_truth_boxes'):
            if record.get(key) is not None:
                row[key] = [d for d in record[key] if d['attribute_id'] in attrs]
        if 'box_attribute_scope' in record:
            row['box_attribute_scope'] = sorted(set(record['box_attribute_scope']) & attrs)
            if not row['box_attribute_scope']:
                row['boxes_complete'] = False
        scoped.append(row)
    return scoped
