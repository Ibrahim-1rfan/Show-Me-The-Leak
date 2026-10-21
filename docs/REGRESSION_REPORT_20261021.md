> Historical snapshot. Current results: [BENCHMARK_REPORT](../BENCHMARK_REPORT.md); current usage: [GUIDE](../GUIDE.md). Later runs supersede model/coverage statements below.

# Regression investigation and repair report — 21 October 2026

The saved-output data-loss bug is repaired and reproduced on the real Qwen3
document response. The dots processor now passes a real CPU image-token test.
**Fresh GPU inference, Florence generation recovery, a successful pilot and a full
benchmark have not been demonstrated.** Training remains disabled. No results or
dataset images were deleted or replaced.

## Recovered state and preserved evidence

The earlier session stopped after creating `TRACKER.md` and identifying two VLM
problems. It had not applied repairs. The project is
`D:/Github/Show-Me-The-Leak`; the embedded Colab project is `Show-Me-The-Leak-v2`.
The actual active notebooks are `notebooks/01_vlm_benchmark.ipynb` and
`notebooks/02_ocr_benchmark.ipynb`. There is no additional `ocr.ipynb` or
`ocr_benchmark.ipynb` in this checkout. Both active notebooks had 11 executed cells
and a user-added result-download cell. Their executed versions were preserved
before updating the same notebook paths. ZIP export functionality is retained.

Current evidence directories:

- VLM: `results/content/Show-Me-The-Leak-v2/results/vlm/20261019T230132317596Z/`.
- OCR: `results_ocr/content/Show-Me-The-Leak-v2/results/ocr/20261019T235754632234Z/`.
- Previous runs: `Leak_Benchmark_Results/outputs/multimodel_pii/sample_50_1df1d1c206ae/`.
- Audit, original notebook snapshots, image contact sheets, replay artifacts and
  hashes: `docs/regression_20261021/`.

`preserved_inventory.json` records 156 imported files/notebooks. The result ZIPs
remain intact. The supplied OCR result is a two-image Florence pilot plus two
startup failures, not a completed OCR benchmark.

## What actually happened, by model

| Model | Actual current evidence | Diagnosis and status |
|---|---|---|
| Qwen3 VL | Both final predictions empty; both redactions identical to originals | Document first response had three valid detections plus two invalid labels. Whole-response rejection discarded valid detections; empty retry was accepted. General image was empty from initial generation, cause unresolved. |
| Qwen2.5 VL | Both responses empty from initial generation; unchanged redactions | No valid detections were available for postprocessing to recover. Not truncated; CUDA-only parameters on A100. Prompt, precision, visual-input and model-behavior diagnosis remains pending. |
| Llama 3.2 Vision | No image inference | HTTP 401 while accessing gated checkpoint. Requires authorized account/secret. Not a redaction or accuracy score. |
| Florence 2 | Two image attempts, both errors; two failure overlays, no redacted outputs | Document reaches effective decoder output limit with repetitive text and no regions. General image raises zero-area-region parsing error; old adapter lost raw evidence before returning. |
| Chandra | No image inference | HTTP 429 during model-weight download from HF Xet CDN. This is not a VISPR dataset import failure. |
| dots.ocr | No image inference | Import of nonexistent `Qwen2_5_VLVideoProcessor` under Transformers 4.56.1. Fixed to the supported `Qwen2VLVideoProcessor`; actual CPU processor test passed, GPU model test pending. |

Original/predicted/redacted contact sheets were visually inspected. All four Qwen
redacted outputs are exactly equal to their upright original RGB pixels. The
Florence overlays only carry the failure notice. They are not privacy redactions.
Original dimensions and IDs agree with the frozen manifest for these examples.

### Comparison with earlier evidence

The same document (`2017_12973862`) and general image (`2017_24408793`) also had
zero final Qwen detections in the earlier October 8 runs. Historical Qwen3 runs
have 21 nonempty images elsewhere, but that does not establish previous success
on these two examples. The parser defect is demonstrated; a new regression caused
specifically by the project reorganization is **not** established.

The matching older Chandra examples also had zero mapped PII detections. The
English OCR-to-PII rules recognize only their documented fields/patterns. Plain
names, visual faces, contextual identity cards and sensitive content outside those
patterns are not automatically protected just because OCR produced text. No new
Chandra output exists to demonstrate recognition recovery or mapping regression.

## Florence: exact failure and bounded diagnostic changes

The document call used `<OCR_WITH_REGION>`, compatible pinned Transformers 4.41.2,
torch 2.6.0, FP16, eager attention, an L4 GPU, and a 1280×960 preprocessed image.
It generated exactly 1,023 tokens, ended with forced EOS at the bound, and produced
repetitive non-spatial text. Its native result had zero quadrilaterals and labels.
There is no recorded out-of-memory or input-context exception in this attempt.

The pinned model has 1,024 decoder position embeddings. The existing 1,023 new-token
cap leaves space for the decoder start token. Increasing the notebook's requested
4,096-token budget would not remove that bound. The cause of repetitive generation
is still unknown; truncation is a confirmed symptom, not a complete explanation.
[Pinned Florence configuration](https://huggingface.co/microsoft/Florence-2-large-ft/blob/4a12a2b54b7016a48a22037fbd62da90cd566f2a/config.json).

The adapter now preserves raw text and native output even if individual regions
fail parsing. Rejected regions and truncation remain explicit failures. It does
not reinterpret malformed regions as clean empty OCR. Precision and crop modes
are separate, recorded diagnostic experiments, not claimed fixes to generation.
FlashAttention remains unnecessary for Florence's eager profile.

For dots, the official versioned AutoVideoProcessor mapping associates
`qwen2_5_vl` with `Qwen2VLVideoProcessor`. The repaired component was tested using
the actual pinned tokenizer/template, confirming the custom image token and grid
agree for a 56×56 input. This used CPU torch 2.4.1/Python 3.10; production CUDA
torch 2.7/Python 3.11 generation still requires Colab.
[Transformers 4.56.1 video processor mapping](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/auto/video_processing_auto.py).

## Redaction repair and acceptance

VLM entries are now independently validated. A malformed sibling no longer erases
valid detections. Rejected entries are preserved and the result is marked partial;
unknown IDs are not guessed or silently converted. Partial outputs cannot pass
the full-run gate. No ground truth enters the model prompt or prediction repair.

The real Qwen3 document replay retained three boxes: credit card, student ID and
ticket. It changed **1,066,354 pixels** at the original 2048×1536 resolution. The
saved replay image was inspected: masked regions are present, while the top-left
face and other unhandled content remain visible. This is a verified mask operation
on partial predictions, **not complete privacy removal or new model inference**.
The other three Qwen responses remain empty and fail the known-positive check.

`validation.py` now checks:

- Original-file identity, output readability/dimensions and original-coordinate boxes.
- Nonempty mask area for localized detections, exact saved black-mask pixels and
  changed-pixel count; saving an overlay or original instead of the redaction fails.
- Unlocalized detections, partial predictions, unchanged known-positive images and
  in-scope positive labels without redaction.
- Optional independently supplied reference-box coverage, without fabricating boxes.

Prediction validity, mask application and privacy completeness are separate fields.
Valid empty predictions remain in label-quality scoring as misses; they are not
discarded to inflate accuracy. Mask checks and privacy warnings are exported in
`redaction_checks.csv`. Image-level VISPR labels cannot certify spatial privacy
coverage, so outputs remain reviewable even when mask application is correct.

The staged workflow is now single known-positive document → four-image pilot
(two documents, two general images) → explicit per-image review → full run.
Missing, partial, mismatched or failed artifacts quarantine the affected model.
Review is bound to the protocol and hashes of exact predictions; changing a result
invalidates its review. OCR scope limitations must be recorded during review.

## Image dimensions and preprocessing decision

The existing 50-image cohort is retained unchanged. Nine of 25 documents and 16
of 25 general images exceed the baseline 1280-pixel edge cap.

| Category | Width range | Height range | Aspect ratio range | Formats |
|---|---:|---:|---:|---|
| Documents, 25 | 480–3648 | 328–2448 | 0.75–1.78 | JPEG |
| General, 25 | 402–5616 | 468–5184 | 0.61–1.54 | JPEG, MPO |

Policy: preserve source bytes and use the upright, original-resolution RGB frame
for redaction and scoring. Keep the 1280 aspect-ratio cap as the unchanged baseline;
record the actual intermediate size, scale, native processor tensor/grid and any
tile offsets. Do not stretch every source image into a common arbitrary square.

Florence's pinned processor itself resizes to 768×768 and disables center cropping.
Its quadrilaterals are postprocessed to the supplied image size, then scaled to the
original dimensions. That differs from the shared pre-cap and is now explicitly
recorded. Removing the pre-cap alone does not make Florence see a native-resolution
document. [Pinned processor settings](https://huggingface.co/microsoft/Florence-2-large-ft/blob/4a12a2b54b7016a48a22037fbd62da90cd566f2a/preprocessor_config.json).

Controlled experiment options in the existing notebooks:

1. Baseline whole image, `max_input_edge=1280`.
2. Processor-native path, `max_input_edge=None`; the processor may still resize.
3. OCR original-image crops, 1024-pixel edge, 15% overlap, explicit maximum of 16
   tiles. No tile is silently dropped; any failed/truncated tile marks the whole
   image incomplete. Crops are offset back to the original frame, including quads.
4. Florence FP32 as a separate precision diagnostic after the unchanged FP16 run.

Pure geometry tests verify full crop-plan coverage and coordinate offsets. No new
recognition/F1/runtime comparison between these policies has been measured. Keep
the same image IDs and change one factor at a time. Crop overlap can duplicate OCR
regions; this is retained for audit, while label scoring uses sets and masks union
overlapping boxes. It is not used to inflate instance counts.

## Hugging Face and cloud status

The public metadata endpoint confirms `Ibrahim1rfan/VISPR` is ungated and contains
`train2017.zip`, not a dataset table with configurations and columns. The archive
is about 22 GB. The source URL is now pinned to the inspected repository revision.
No unrelated dataset or empty fallback was introduced.
[Repository metadata](https://huggingface.co/api/datasets/Ibrahim1rfan/VISPR).

Network metadata retrieval succeeded after an initial sandbox DNS failure. The
reported original dataset-import cause remains unknown because its failing log
was not supplied. The full archive was not downloaded again. New diagnostics cover
access denial, missing revision, transient network errors, disk shortage, partial
transfer length, resumed byte range and source ETag. Cached subset validation still
checks all image/annotation hashes, dimensions, source IDs and category counts.

Model downloads now use bounded transient-error retries and lower concurrency;
401/403 are not retried as if transient. HF cache is retained. This improves
recovery from Chandra's 429 but cannot guarantee provider availability.

Drive authentication and destination are unchanged. Repaired runs copy the current
tracker into their unique run before backup. Prior logs show mounted-Drive copy
verification, but this session has not independently inspected live Drive storage.

## Timing evidence and benchmark status

| Model/image | Earlier detection-call seconds | Current saved measurement |
|---|---:|---|
| Qwen3 document | 26.52–28.03 | 23.86 detection, two generation calls; empty final output |
| Qwen3 general | 1.70 | 0.788 detection; empty |
| Qwen2.5 document | 2.17–2.18 | 1.648 detection; empty |
| Qwen2.5 general | 1.76–1.78 | 0.793 detection; empty |
| Florence document | No comparable successful earlier call | 13.61 generation, 14.94 image wall seconds; failed |
| Florence general | No comparable successful earlier call | 0.217 generation, 2.40 image wall seconds; failed |

Hardware, versions and stages differ; these are not causal speedup measurements.
Replay CPU timing is recorded only as replay work, not model latency. No model has
completed a verified new 50-image privacy benchmark. Llama access, Chandra network
availability, new dots GPU inference, Florence generation and empty Qwen responses
remain blockers. No winner or training candidate is declared from failed pilots.

## Changed implementation and tests

| Files | Purpose |
|---|---|
| `src/leakbench/pipeline.py` | Preserve independently valid entries; reject invalid mask geometry |
| `adapters.py`, `ocr_adapters.py`, `environment.py` | Florence raw evidence, strict coordinates/transform metadata, actual dots import and early profile check |
| `legacy_workflow.py` in package | Keep incomplete OCR status and usable partial regions separate |
| `validation.py`, `gates.py` | Saved-mask checks and explicit staged acceptance/review |
| `preprocessing.py` | Bounded, original-coordinate crop experiment |
| `downloads.py`, `dataset.py` | Transient retries, dataset range/integrity diagnostics and source hashes |
| `worker.py`, `orchestrate.py` | Single/pilot/full staging, per-image mask evidence and source/model association |
| `reporting.py`, `cloud.py` | Separate redaction checks from inference metrics; tracker backup |
| `scripts/build_notebooks.py`, existing three notebooks | Embed repairs, retain export workflow, stage/review and preprocessing controls |
| `tests/test_regression_repair.py`, `tests/test_leakbench_v2.py` | Parser/mask/geometry/download/gate/association regressions |
| `tests/test_model_testing_notebook.py` | Parse user-added IPython commands correctly without executing them |
| `TRACKER.md`, `README.md`, `docs/VALIDATION.md`, this report | Live status, changed instructions and historical-check caveats |
| `.gitignore` | Keep imported OCR exports and local regression image/notebook evidence out of accidental Git additions |

All **112 offline tests passed**. Tests use synthetic fixtures where needed and actual saved-output replay separately.
The legacy root notebook/source remain preserved; use the repaired category
notebooks for new work. Final test count and pending steps are in `TRACKER.md`.

## Cleanup and next steps

`cleanup_inventory.csv` classifies 45 root entries with exact paths. No item meets
all deletion criteria yet, and nothing has been deleted. Future candidates are
`D:/Github/Show-Me-The-Leak/tmp/`, `D:/Github/Show-Me-The-Leak/__pycache__/` and
`D:/Github/Show-Me-The-Leak/dist/`, because they can be regenerated. The temporary
processor probe and released bundles are still useful evidence. Preserve them
until benchmark completion, verified backup, dependency review and user approval.

Next: run the repaired single-image tests in Colab; resolve access/download failures;
inspect native Florence generation before trying one precision/crop change at a
time; investigate empty Qwen outputs with recorded vision inputs; then run the
four-image pilot. Only reviewed passing models may proceed to 50 images. Confirm
Drive copies before cleanup. Training remains blocked until a valid baseline and
suitable training/reference annotations exist.
