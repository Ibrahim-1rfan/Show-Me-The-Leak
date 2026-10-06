"""Offline metrics for the multi-model VISPR benchmark (standard library only).

Every image/model pair is one record. Label metrics compare predicted attribute
IDs with image-level JSON annotations; they do not certify privacy or establish
why a model omitted a label. ``a0_safe`` is never a positive privacy label.
Localization requires explicitly complete manual boxes in original-image pixel
coordinates. Text scores require independent reference transcriptions.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Any


SAFE_ATTRIBUTE = "a0_safe"
STRATA = ("all", "document", "general")


def _divide(numerator: float, denominator: float, default=0.0):
    return numerator / denominator if denominator else default


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
    }


def _labels(values: Iterable[str], allowed: set[str], field: str) -> set[str]:
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError(f"{field} must be a sequence of attribute IDs")
    values = list(values)
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field} contains a non-string attribute ID")
    labels = set(values) - {SAFE_ATTRIBUTE}
    unknown = labels - allowed
    if unknown:
        raise ValueError(f"Unknown {field}: {sorted(unknown)}")
    return labels


def label_metrics(ground_truth: Iterable[str], predicted: Iterable[str],
                  attribute_ids: Iterable[str]) -> dict:
    """Unique image-level labels; undefined precision/recall/F1 use zero.

    ``unannotated_attributes`` are conventional false positives relative to the
    supplied annotations, and may represent real but unannotated information.
    A correctly empty image has exact_match=True, jaccard=1, and F1=0.
    """
    allowed = set(attribute_ids) - {SAFE_ATTRIBUTE}
    truth = _labels(ground_truth, allowed, "ground_truth_attributes")
    prediction = _labels(predicted, allowed, "predicted_attributes")
    hits, missed, unannotated = truth & prediction, truth - prediction, prediction - truth
    return {
        "ground_truth_attributes": sorted(truth),
        "predicted_attributes": sorted(prediction),
        "matched_attributes": sorted(hits),
        "missed_attributes": sorted(missed),
        "unannotated_attributes": sorted(unannotated),
        "true_positives": len(hits),
        "false_negatives": len(missed),
        "unannotated_count": len(unannotated),
        **_prf(len(hits), len(unannotated), len(missed)),
        "exact_match": truth == prediction,
        "jaccard": _divide(len(hits), len(truth | prediction), 1.0),
    }


def _box(value: Sequence[float]) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_pixels must contain [x1, y1, x2, y2]")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
           for x in value):
        raise ValueError("bbox_pixels must contain finite numbers")
    x1, y1, x2, y2 = map(float, value)
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        raise ValueError("bbox_pixels must be nonnegative with positive area")
    return x1, y1, x2, y2


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Area IoU for continuous/exclusive xyxy pixel boxes (no '+1')."""
    ax1, ay1, ax2, ay2 = _box(first)
    bx1, by1, bx2, by2 = _box(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union


def localization_metrics(predictions: Sequence[Mapping], ground_truth: Sequence[Mapping] | None,
                         iou_threshold: float = 0.5,
                         ground_truth_complete: bool = False) -> dict:
    """Same-label maximum-cardinality one-to-one matching at an IoU threshold.

    Adjacency is visited in descending IoU order for deterministic matching;
    the algorithm maximizes the number of matches, not summed IoU. Duplicated
    boxes and predictions without a box remain unmatched (false positives).
    Manual boxes must exhaustively cover the evaluated label scope. Partial
    manual annotations are deliberately unavailable rather than treated empty.
    """
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, Real) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    unavailable = {
        "localization_available": False, "localization_tp": None,
        "localization_fp": None, "localization_fn": None,
        "localization_precision": None, "localization_recall": None,
        "localization_f1": None, "mean_matched_iou": None,
    }
    if ground_truth is None:
        return {**unavailable, "localization_reason": "No manual ground-truth boxes"}
    if ground_truth_complete is not True:
        return {**unavailable, "localization_reason": "Manual box annotations are not marked complete"}

    for item in [*predictions, *ground_truth]:
        if not isinstance(item, Mapping) or not isinstance(item.get("attribute_id"), str):
            raise ValueError("Every box item needs an attribute_id")
        if item["attribute_id"] == SAFE_ATTRIBUTE:
            raise ValueError("a0_safe is not a localization target")
    true_boxes = [_box(item.get("bbox_pixels")) for item in ground_truth]
    predicted_boxes = [None if item.get("bbox_pixels") is None else _box(item["bbox_pixels"])
                       for item in predictions]
    overlaps = {}
    adjacency = {}
    for pi, pbox in enumerate(predicted_boxes):
        edges = []
        if pbox is not None:
            for gi, gbox in enumerate(true_boxes):
                if predictions[pi]["attribute_id"] != ground_truth[gi]["attribute_id"]:
                    continue
                overlap = box_iou(pbox, gbox)
                if overlap >= iou_threshold:
                    overlaps[pi, gi] = overlap
                    edges.append(gi)
        adjacency[pi] = sorted(edges, key=lambda gi: (-overlaps[pi, gi], gi))

    assigned_truth = {}

    def augment(pi, seen):
        for gi in adjacency[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in assigned_truth or augment(assigned_truth[gi], seen):
                assigned_truth[gi] = pi
                return True
        return False

    for pi in adjacency:
        augment(pi, set())
    matches = [{"prediction_index": pi, "ground_truth_index": gi,
                "attribute_id": predictions[pi]["attribute_id"], "iou": overlaps[pi, gi]}
               for gi, pi in sorted(assigned_truth.items())]
    matched_predictions = set(assigned_truth.values())
    tp, fp, fn = len(matches), len(predictions) - len(matches), len(ground_truth) - len(matches)
    return {
        "localization_available": True,
        "localization_reason": None,
        "iou_threshold": float(iou_threshold),
        "localization_tp": tp, "localization_fp": fp, "localization_fn": fn,
        **{f"localization_{key}": value for key, value in _prf(tp, fp, fn).items()},
        "mean_matched_iou": _divide(sum(item["iou"] for item in matches), tp, None),
        "unlocalized_prediction_count": sum(box is None for box in predicted_boxes),
        "matches": matches,
        "unmatched_prediction_indices": [pi for pi in range(len(predictions)) if pi not in matched_predictions],
        "unmatched_ground_truth_indices": [gi for gi in range(len(ground_truth)) if gi not in assigned_truth],
    }


def _edit_distance(first: Sequence, second: Sequence) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for i, item in enumerate(first, 1):
        current = [i]
        for j, other in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (item != other)))
        previous = current
    return previous[-1]


def _lcs_length(first: Sequence, second: Sequence) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for item in first:
        current = [0]
        for j, other in enumerate(second, 1):
            current.append(previous[j - 1] + 1 if item == other else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def text_metrics(predicted_text: str | None, reference_text: str | None,
                 casefold: bool = False) -> dict:
    """Transcription ROUGE-L, character error rate (CER), word error rate (WER).

    Normalize Unicode NFKC and collapse whitespace, retaining punctuation and
    case by default. Tokens are whitespace-delimited. No stemming. ROUGE-L is
    the F1 of sequence LCS precision/recall, not unordered token agreement.
    Empty references yield undefined per-image rates (None); insertion edits
    still contribute to corpus CER/WER when other references are nonempty.
    Missing text is unavailable; an explicit empty prediction is evaluated.
    """
    if predicted_text is None or reference_text is None:
        return {"text_available": False, "text_reason": "Both predicted text and an independent reference transcription are required",
                "rouge_l_precision": None, "rouge_l_recall": None, "rouge_l_f1": None,
                "cer": None, "wer": None}
    if not isinstance(predicted_text, str) or not isinstance(reference_text, str):
        raise ValueError("Transcriptions must be strings, not attribute lists")

    def normalize(value):
        value = " ".join(unicodedata.normalize("NFKC", value).split())
        return value.casefold() if casefold else value

    prediction, reference = normalize(predicted_text), normalize(reference_text)
    predicted_tokens, reference_tokens = prediction.split(), reference.split()
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    precision = _divide(lcs, len(predicted_tokens)) if reference_tokens else None
    recall = _divide(lcs, len(reference_tokens), None)
    rouge_f1 = _divide(2 * lcs, len(predicted_tokens) + len(reference_tokens)) if reference_tokens else None
    character_edits = _edit_distance(prediction, reference)
    word_edits = _edit_distance(predicted_tokens, reference_tokens)
    return {
        "text_available": True, "text_reason": None,
        "rouge_l_precision": precision, "rouge_l_recall": recall, "rouge_l_f1": rouge_f1,
        "lcs_tokens": lcs, "predicted_words": len(predicted_tokens), "reference_words": len(reference_tokens),
        "reference_characters": len(reference), "character_edits": character_edits, "word_edits": word_edits,
        "cer": _divide(character_edits, len(reference), None),
        "wer": _divide(word_edits, len(reference_tokens), None),
    }


def _aggregate_labels(rows: list[dict], labels: list[str]) -> tuple[dict, list[dict]]:
    truth_count, prediction_count, hit_count = Counter(), Counter(), Counter()
    for row in rows:
        truth_count.update(row["ground_truth_attributes"])
        prediction_count.update(row["predicted_attributes"])
        hit_count.update(row["matched_attributes"])
    attributes = []
    for label in labels:
        tp, support, predicted = hit_count[label], truth_count[label], prediction_count[label]
        attributes.append({
            "attribute_id": label, "support": support, "predicted_count": predicted,
            "true_positives": tp, "missed_count": support - tp, "unannotated_count": predicted - tp,
            **(_prf(tp, predicted - tp, support - tp) if rows else {"precision": None, "recall": None, "f1": None}),
        })
    tp = sum(item["true_positives"] for item in attributes)
    fp = sum(item["unannotated_count"] for item in attributes)
    fn = sum(item["missed_count"] for item in attributes)
    supported = [item for item in attributes if item["support"]]
    summary = {
        "n_images": len(rows), "n_evaluated_labels": len(labels), "n_supported_labels": len(supported),
        "true_positives": tp, "unannotated_count": fp, "missed_count": fn,
        "ground_truth_positive_count": tp + fn, "predicted_positive_count": tp + fp,
        **{f"micro_{key}": value for key, value in
           (_prf(tp, fp, fn) if rows else {"precision": None, "recall": None, "f1": None}).items()},
        "macro_f1_supported": _divide(sum(item["f1"] for item in supported), len(supported), None),
        "macro_f1_all_labels": _divide(sum(item["f1"] for item in attributes), len(labels), None) if rows else None,
        "mean_image_f1": _divide(sum(row["f1"] for row in rows), len(rows), None),
        "exact_match_rate": _divide(sum(row["exact_match"] for row in rows), len(rows), None),
    }
    return summary, attributes


def _aggregate_optional(rows: list[dict]) -> dict:
    localized = [row for row in rows if row["localization_available"]]
    texts = [row for row in rows if row["text_available"]]
    tp = sum(row["localization_tp"] for row in localized)
    fp = sum(row["localization_fp"] for row in localized)
    fn = sum(row["localization_fn"] for row in localized)
    scores = _prf(tp, fp, fn) if localized else {"precision": None, "recall": None, "f1": None}
    rouge = [row["rouge_l_f1"] for row in texts if row["rouge_l_f1"] is not None]
    return {
        "n_localization_images": len(localized),
        "localization_tp": tp if localized else None,
        "localization_fp": fp if localized else None,
        "localization_fn": fn if localized else None,
        **{f"localization_{key}": value for key, value in scores.items()},
        "mean_matched_iou": _divide(sum((row["mean_matched_iou"] or 0) * row["localization_tp"]
                                           for row in localized), tp, None),
        "n_text_images": len(texts), "mean_rouge_l_f1": _divide(sum(rouge), len(rouge), None),
        "corpus_cer": _divide(sum(row["character_edits"] for row in texts),
                              sum(row["reference_characters"] for row in texts), None),
        "corpus_wer": _divide(sum(row["word_edits"] for row in texts),
                              sum(row["reference_words"] for row in texts), None),
    }


def build_comparison(records: Iterable[Mapping[str, Any]], attribute_ids: Iterable[str],
                     requested_models: Iterable[str] | None = None,
                     sample_image_ids: Iterable[str] | None = None,
                     iou_threshold: float = 0.5,
                     attribute_to_category: Mapping[str, str] | None = None,
                     sample_strata: Mapping[str, str] | None = None) -> dict:
    """Return JSON-safe tables: summaries, per_image, per_attribute, coverage.

    Required record fields: image_id, model_key, stratum (document/general),
    status (ok/error), ground_truth_attributes and, on success,
    predicted_attributes. Optional fields: detections, ground_truth_boxes,
    boxes_complete (or box_ground_truth_complete), predicted_text, reference_text.

    ``all_successful`` evaluates each model's successful outputs. Pairwise
    ranking should use ``common_successful``: the same images successfully
    processed by EVERY requested model, including models with no records.
    Failures have no quality score and remain visible in coverage. A missing
    model therefore gives an empty common subset, never a vacuous comparison.
    Supply the planned sample_image_ids to count entirely unattempted images.
    Optional sample_strata maps planned IDs to document/general, allowing
    stratum coverage before any model has run. Without it, sample strata can
    only be counted where record metadata exists.
    """
    if isinstance(attribute_ids, (str, bytes)):
        raise ValueError("attribute_ids must be a sequence of IDs")
    attribute_ids = list(attribute_ids)
    if any(not isinstance(label, str) or not label for label in attribute_ids):
        raise ValueError("attribute_ids must contain nonempty strings")
    labels = sorted(set(attribute_ids) - {SAFE_ATTRIBUTE})
    if not labels:
        raise ValueError("At least one valid privacy attribute ID is required")
    if isinstance(requested_models, (str, bytes)):
        raise ValueError("requested_models must be a sequence of model keys")
    if isinstance(sample_image_ids, (str, bytes)):
        raise ValueError("sample_image_ids must be a sequence of image IDs")
    allowed = set(labels)
    records = [dict(record) for record in records]
    models = list(dict.fromkeys(requested_models if requested_models is not None
                                else [record["model_key"] for record in records]))
    if not models or any(not isinstance(model, str) or not model for model in models):
        raise ValueError("At least one requested model key is required")
    records = [record for record in records if record.get("model_key") in models]
    known_strata = dict(sample_strata or {})
    planned = set(sample_image_ids) if sample_image_ids is not None else (
        {record["image_id"] for record in records} | known_strata.keys())
    if any(not isinstance(image_id, str) or not image_id for image_id in planned):
        raise ValueError("sample_image_ids must contain nonempty strings")
    if known_strata.keys() - planned or any(stratum not in STRATA[1:] for stratum in known_strata.values()):
        raise ValueError("sample_strata must map planned image IDs to document or general")
    image_metadata = {}
    seen = set()
    per_image = []
    for record in records:
        image_id, model = record["image_id"], record["model_key"]
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a nonempty string")
        if image_id not in planned:
            raise ValueError(f"Record {image_id} is not in sample_image_ids")
        pair = (model, image_id)
        if pair in seen:
            raise ValueError(f"Duplicate image/model result: {pair}")
        seen.add(pair)
        if record.get("stratum") not in STRATA[1:]:
            raise ValueError("stratum must be document or general")
        if image_id in known_strata and known_strata[image_id] != record["stratum"]:
            raise ValueError(f"Inconsistent planned stratum for {image_id}")
        known_strata[image_id] = record["stratum"]
        if record.get("status") not in {"ok", "error"}:
            raise ValueError("status must be ok or error")
        truth = _labels(record["ground_truth_attributes"], allowed, "ground_truth_attributes")
        metadata = (record["stratum"], frozenset(truth))
        if image_id in image_metadata and image_metadata[image_id] != metadata:
            raise ValueError(f"Inconsistent ground truth or stratum across models for {image_id}")
        image_metadata[image_id] = metadata
        row = {"image_id": image_id, "model_key": model, "stratum": record["stratum"],
               "status": record["status"], "ground_truth_attributes": sorted(truth)}
        if record["status"] == "error":
            row.update({"error": record.get("error"), "precision": None, "recall": None, "f1": None,
                        "predicted_attributes": None, "missed_attributes": None, "unannotated_attributes": None,
                        "localization_available": False, "text_available": False})
        else:
            row.update(label_metrics(truth, record["predicted_attributes"], labels))
            for field in ("detections", "ground_truth_boxes"):
                if record.get(field) is not None:
                    _labels([item["attribute_id"] for item in record[field]], allowed, field)
            row.update(localization_metrics(
                record.get("detections", []), record.get("ground_truth_boxes"), iou_threshold,
                record.get("boxes_complete", record.get("box_ground_truth_complete", False))))
            row.update(text_metrics(record.get("predicted_text"), record.get("reference_text")))
        per_image.append(row)

    successes = {model: {row["image_id"] for row in per_image
                         if row["model_key"] == model and row["status"] == "ok"} for model in models}
    common = set.intersection(*(successes[model] for model in models))
    summaries, per_attribute, per_category, coverage = [], [], [], []
    category_map = dict(attribute_to_category or {})
    categories = sorted({category_map[label] for label in labels if label in category_map})
    for model in models:
        model_rows = [row for row in per_image if row["model_key"] == model]
        for stratum in STRATA:
            expected = planned if stratum == "all" else {image_id for image_id in planned
                        if known_strata.get(image_id) == stratum}
            attempted = [row for row in model_rows if row["image_id"] in expected]
            successful = [row for row in attempted if row["status"] == "ok"]
            coverage.append({
                "model_key": model, "stratum": stratum, "expected_images": len(expected),
                "attempted_images": len(attempted), "successful_images": len(successful),
                "failed_images": len(attempted) - len(successful), "not_attempted_images": len(expected) - len(attempted),
                "success_rate_of_attempts": _divide(len(successful), len(attempted), None),
                "coverage_of_sample": _divide(len(successful), len(expected), None),
            })
            for scope in ("all_successful", "common_successful"):
                rows = successful if scope == "all_successful" else [row for row in successful if row["image_id"] in common]
                keys = {"model_key": model, "stratum": stratum, "scope": scope}
                summary, attributes = _aggregate_labels(rows, labels)
                summaries.append({**keys, **summary, **_aggregate_optional(rows)})
                per_attribute.extend({**keys, **item, "category": category_map.get(item["attribute_id"])} for item in attributes)
                for category in categories:
                    category_labels = [label for label in labels if category_map.get(label) == category]
                    category_summary, _ = _aggregate_labels(rows, category_labels)
                    # Image exact-match and sample-F1 were computed over all labels;
                    # omit them from category rows to avoid misleading semantics.
                    category_summary.pop("mean_image_f1")
                    category_summary.pop("exact_match_rate")
                    per_category.append({**keys, "category": category, **category_summary})

    return {
        "summaries": summaries, "per_image": per_image, "per_attribute": per_attribute,
        "per_category": per_category, "coverage": coverage, "requested_models": models,
        "common_image_ids": sorted(common), "sample_image_ids": sorted(planned),
        "unobserved_stratum_image_ids": sorted(planned - known_strata.keys()),
        "notes": [
            "Precision and unannotated predictions measure agreement with available image-level JSON annotations; annotations may be incomplete.",
            "Missing labels are measured omissions, not evidence that a model consciously decided they are not private.",
            "a0_safe is excluded. Labels are deduplicated per image. Undefined nonempty-sample label precision/recall/F1 use 0.",
            "macro_f1_supported averages only labels with ground-truth support; macro_f1_all_labels includes every configured privacy label, assigning 0 to unsupported labels.",
            "Errors and unattempted images are not converted to empty predictions. No-success quality scores are null. Read coverage alongside quality.",
            "common_successful contains only images successful for every requested model; missing requested models make it empty.",
            "IoU requires complete independent manual boxes in original-image xyxy pixels; matches are same-label, one-to-one maximum cardinality at the threshold.",
            "ROUGE-L/CER/WER need independent reference transcriptions. ROUGE-L uses whitespace-token sequence LCS; Unicode NFKC and collapsed whitespace, case and punctuation retained.",
            "Optional metric image counts can differ across models; compare only matching reference and successful-output subsets.",
        ],
    }
