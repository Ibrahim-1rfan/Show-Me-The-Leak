"""Canonical attributes, strict geometry, and verified pixel redaction."""
import io
import json
import math
from pathlib import Path
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTES = json.loads((ROOT / 'configs/selected_attributes.json').read_text(encoding='utf-8'))['attributes']
REGISTRY = {a['attribute_id']: a for a in ATTRIBUTES}
assert len(REGISTRY) == 28
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000


def read_upload(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Image exceeds the 20 MB limit.')
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('Use a PNG, JPEG, or WebP image.')
            if source.width * source.height > MAX_PIXELS or getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Use a still image of at most 20 megapixels.')
            source.load()
            upright = ImageOps.exif_transpose(source).convert('RGBA')
            # Composite transparency instead of exposing invisible RGB payloads.
            background = Image.new('RGBA', upright.size, (255, 255, 255, 255))
            upright = Image.alpha_composite(background, upright).convert('RGB')
            # Construct a clean image: no original metadata survives.
            return Image.frombytes('RGB', upright.size, upright.tobytes())
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('The file is not a readable supported image.') from exc


def box_pixels(box, size, margin=0):
    if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError('Missing or malformed bounding box')
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Zero or negative area bounding box')
    if min(box) < -10 or max(box) > 1010:
        raise ValueError('Bounding box outside allowed image boundary')
    w, h = size
    clipped = [min(1000, max(0, v)) for v in box]
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Bounding box has no area inside the image')
    pixels = [max(0, math.floor(x1*w/1000)-margin), max(0, math.floor(y1*h/1000)-margin),
              min(w, math.ceil(x2*w/1000)+margin), min(h, math.ceil(y2*h/1000)+margin)]
    return pixels, clipped != box


def parse_detections(raw, size, enabled, margin=0):
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError('Model did not return valid JSON. No output was produced.') from exc
    if not isinstance(payload, dict) or payload.get('coordinate_system') != 'xyxy_1000' or not isinstance(payload.get('detections'), list):
        raise ValueError('Model response is missing detections or the explicit xyxy_1000 coordinate convention.')
    if len(payload['detections']) > 500:
        raise ValueError('Model returned too many detections.')
    detections, warnings = [], []
    for i, item in enumerate(payload['detections']):
        attr = item.get('attribute_id') if isinstance(item, dict) else None
        if not isinstance(attr, str):
            attr = None
        d = {'id': f'd{i+1}', 'attribute_id': attr if attr in REGISTRY else None,
             'name': REGISTRY[attr]['name'] if attr in REGISTRY else 'Unsupported attribute',
             'box': None, 'confidence': None, 'redacted': False, 'status': 'invalid'}
        try:
            if attr not in REGISTRY:
                raise ValueError('Unknown attribute ID')
            d['box'], clipped = box_pixels(item.get('bbox'), size, margin)
            confidence = item.get('confidence')
            if confidence is not None:
                if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    warnings.append(f"{d['id']}: invalid confidence omitted")
                else:
                    d['confidence'] = confidence
                    if confidence < .5:
                        warnings.append(f"{d['id']}: low-confidence prediction; inspect localization")
            if clipped:
                warnings.append(f"{d['id']}: small boundary overflow clipped")
            d['status'] = 'valid'
            d['redacted'] = attr in enabled
        except ValueError as exc:
            warnings.append(f"{d['id']}: {exc}; not redacted")
        detections.append(d)
    return detections, warnings


def redact(image, detections, destination):
    expected = image.copy()
    for d in detections:
        if d['redacted']:
            expected.paste((0, 0, 0), tuple(d['box']))
    expected.save(destination, format='PNG')
    with Image.open(destination) as saved:
        saved.load()
        if saved.size != image.size or ImageChops.difference(expected, saved.convert('RGB')).getbbox():
            raise ValueError('Saved output failed pixel verification.')
    return sum(d['redacted'] for d in detections)
