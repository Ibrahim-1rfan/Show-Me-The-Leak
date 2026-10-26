# What remains before Qwen3-VL training

Original VISPR JSON contains image-level labels and image links/IDs. It contains no instance IDs, spatial boxes, segmentation masks, OCR strings or OCR boxes. Preserve those original files unchanged. This package adds pending canonical records; it does not add real human annotations by generating templates.

Three representations stay separate:
1. Source evidence: original VISPR JSONs and raw annotation-tool exports.
2. Canonical annotations: vispr-manual-v1 records associated with original IDs, verified original-byte hashes, EXIF-upright dimensions and original train/validation split.
3. Model-training examples: an explicit, separately tested framework-specific conversion. No such trained model or completed training export is produced in this task.

Classification conversion should use only independently approved, resolved records and their selected_categories. Localization conversion must additionally require all intended instances localized, clear quality and complete image review. Exclude unknown/uncertain boxes; do not silently remove a missed region and call the remaining target complete. Keep reason-coded exclusion reports. A reviewed category-only record can be useful for classification but not full-redaction evaluation.

For Qwen-style supervised examples, the user turn contains the image plus the task/taxonomy instruction; the assistant turn contains structured category/instance output. Keep image paths resolvable, preserve train/validation membership and use the pinned model processor/chat template. Convert normalized boxes to the exact convention declared in that experiment (for example integer 0–1000 only if the prompt and evaluator use it). Validate image-token alignment, assistant-only loss masking, padding, gradient flow and adapter checkpoint reload on a small GPU smoke test. A canonical JSON with boxes is not itself a validated Qwen training batch.

No transcription is collected in v1. OCR word alignment would need separately generated OCR word IDs/coordinates transformed into the same upright frame, then human-checked overlap/field linkage. OCR proposals are not ground truth. Do not claim transcript exact match, CER or ROUGE without reviewed text references.

The 28 labels include document-object context, not every field subtype. Passport numbers and other unsupported identifiers must not be mislabeled as names/usernames. Passport/card context can support a whole-object policy, but this is not evidence of accurate number-level detection. Expand the versioned taxonomy only through a deliberate decision and new review rules.

Approved records support per-category precision/recall, matched-box IoU and region coverage when references are sufficiently complete. Sampled cross-review is a quality signal, not certification of the entire package. Inspect sparse classes and negative examples. Current document/general assignments are proxies, and related-capture/near-duplicate review remains pending. Keep the frozen 50-image cohort excluded from training; obtain a genuinely unseen test cohort separately. Pilot calibration uses training images only, never validation examples as demonstrations. No training is started by package preparation or validation.
