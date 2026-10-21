# Setup, usage and reproducibility

Read [README](README.md) for the overview, [BENCHMARK_REPORT](BENCHMARK_REPORT.md)
for current evidence and [TRACKER](TRACKER.md) for history. This is the maintained
v2 workflow; `BENCHMARK_README.md` describes the older mixed notebook only.

## Environment and access

**Recommended execution:** import either benchmark notebook into Google Colab,
select a GPU, then run from the top. The notebook carries a checksum-verified
snapshot of `src/leakbench` and `configs`; it does not fetch an unpinned Git branch.
Colab Pro does not guarantee a particular GPU. The current dots profile requires
compute capability 8 or newer (for example L4/A100); it rejects T4.

Section 2 installs notebook utilities: Pillow 11.3.0, pandas 2.2.3, matplotlib
3.10.6, tqdm 4.67.1 and ipywidgets 8.1–8.x. Model setup runs in isolated Python
3.11 workers, not in the notebook's Torch environment. The exact command plans
and probes live in [environment.py](src/leakbench/environment.py):

| Profile | Torch / torchvision | Transformers | Additional requirements |
|---|---|---|---|
| VLM | 2.8.0 / 0.23.0, cu128 | 4.57.1 | accelerate, bitsandbytes |
| Chandra | 2.8.0 / 0.23.0, cu128 | 4.57.1 | chandra-ocr 0.1.7, accelerate, bitsandbytes |
| dots | 2.7.0 / 0.22.0, cu128 | 4.56.1 | flash-attn 2.8.0.post2, CUDA build toolchain |
| Florence | 2.6.0 / 0.21.0, cu124 | 4.41.2 | tokenizers 0.19.1, timm 1.0.7, numpy <2; no FlashAttention |

Some supporting dependencies have bounded ranges rather than exact pins. Keep
saved environment manifests for reproducibility; this is not a complete lockfile.
Section 6 automatically prepares and probes the required worker profiles.
`INSTALL_PACKAGES` belongs to the legacy notebook, not these v2 entry points.
Installing into the notebook kernel does not repair an isolated worker.

Public models can download without a token. For
[Meta Llama 3.2 Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct),
obtain model access with your own Hugging Face account and add a Colab secret named
`HF_TOKEN`, enabling notebook access. Section 3 loads it into the process without
printing it. Never put credentials in cells, configuration files or commits.
A 401 requires authorization investigation, not another package installation.

For offline development on Windows or Linux, use Python 3.10 or newer (offline checks use 3.10.11; GPU workers use 3.11):

```console
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux, then:

```console
python -m pip install -r requirements-dev.txt
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
python scripts/check_handoff.py
```

These tests use synthetic fixtures and do not download model weights or establish
model quality. GPU environment creation is Linux/Colab-only. After editing source,
configuration or the training plan, preserve any executed notebooks, then run:

```console
python scripts/build_notebooks.py
```

This regenerates clean notebook snapshots, resetting outputs, mode to `single`
and pilot reviews to empty. It builds the source ZIP in `dist/`; the optional
dataset ZIP is built only when the local subset exists. Do not regenerate a
notebook to resume a run unless the resulting protocol change is intended.

## Workflows and configuration

| Notebook | Purpose |
|---|---|
| [01 VLM](notebooks/01_vlm_benchmark.ipynb) | Qwen3-VL 8B, Qwen2.5-VL 7B, Llama 3.2 Vision 11B: ask directly for taxonomy IDs and boxes |
| [02 OCR](notebooks/02_ocr_benchmark.ipynb) | Florence, Chandra, dots: decode native OCR/layout, then apply fixed English PII rules |
| [03 readiness](notebooks/03_training_readiness.ipynb) | Review the training plan and excluded evaluation IDs; never trains |

All six models receive both document and general images. OCR scores measure the
complete OCR-plus-rules system, not recognition alone. Picture descriptions are
not literal recognized text. Florence uses `<OCR_WITH_REGION>` and its native
postprocessor; JSON is an internal normalized format, not a model requirement.
VLM prompts exclude ground truth. Sensitive traits require explicit text evidence;
annotation agreement must not be improved by guessing traits from appearance.

Model IDs, revisions and recorded licenses: [models.json](configs/models.json).
Section 5 controls the following baseline:

| Variable | Default / role |
|---|---|
| `CONFIG['models']` | Explicit model keys from models.json; no silent substitution |
| `RUN_STAGE` / live selector | `single`, `pilot`, `full`; Section 6 reads the live value |
| `REUSE_PREDICTIONS` | True: copy intact, compatible evidence into a new run; False: deliberate rerun |
| `max_input_edge` | 1280; aspect-ratio cap without upscaling, then native processor resizing |
| `max_new_tokens` | 4096 requested; Florence effective maximum 1023 due to decoder bound |
| `format_retries`, `repair_partial_entries` | 1, True; bounded correction, rejected/accepted evidence retained |
| `load_in_4bit` | True where supported; Florence and dots use their native profiles |
| `redaction_padding` | 0.01 of original dimensions; lossless black mask in original coordinates |
| `batch_size`, `audit_redacted` | 1, False; no hidden second inference on redacted images |
| `ocr_preprocessing` | `whole`; `tiles` is a separately recorded experiment |
| `tile_edge`, `tile_overlap`, `max_tiles` | 1024, 0.15, 16; coverage checked |
| `florence_precision` | `float16`; `float32` is an unverified quality diagnostic |
| Timeouts, seconds | image 600; run 43200; download 3600; load 1800 |

Rerun Section 5 after editing configuration. Changing only the live selector does
not require rerunning that cell. Source/model/input/preprocessing changes invalidate
reuse; using the same filenames does not establish compatible results.

## Dataset and modes

[selection.json](configs/selection.json) fixes 25 document-category and 25 general
images from the VISPR **training split**. It records original IDs, image and annotation
hashes, source ZIP members and review decisions. The pinned source ZIP is
[Ibrahim1rfan/VISPR](https://huggingface.co/datasets/Ibrahim1rfan/VISPR/tree/831e0de941ddf741fa00217c9d78d91b57c9af93).
This curated, repeatedly inspected diagnostic cohort is not an untouched test set.
Eleven images differ from the original October 8 cohort; do not merge their scores.

Section 4 first reuses a verified mounted-Drive subset or an uploaded
`/content/vispr_eval_50.zip`. Otherwise it downloads the pinned full source ZIP and
extracts only the selected pairs. Allow disk space for that archive; the downloader
checks available space and retains partial downloads. `data/vispr_eval_50/` contains
`documents/`, `images/`, `annotations/`, `manifest.csv` and validation metadata.
Original bytes and actual formats are retained, including MPO. Hashes, readability,
50 unique IDs, 25/25 balance and duplicate checks are enforced.

| Mode | Images per model | Use |
|---|---:|---|
| single | 1 (`2017_12973862`) | Diagnose loading, generation and masking |
| pilot | 4 (first two in each category) | Examine both categories; no single-image recall gate |
| full | 50 | Benchmark after four complete predictions and explicit review |

Run Sections 1–4, choose mode/models in Section 5, execute Section 6, read Section 7
and inspect Section 8. For a full run, review all four pilot originals, raw outputs,
boxes and redactions, fill `PILOT_REVIEWS` with your own image-specific notes, and
run that review cell before selecting `full`. Reviews are bound to prediction hashes.
Valid detection misses can be accepted for diagnostic measurement; malformed
predictions or incorrectly applied masks cannot. Never copy another run's notes
as automatic approval. A blocked model remains in coverage as unprocessed.

## Output locations and interpretation

Colab project root: `/content/Show-Me-The-Leak-v2`. Local source checkout uses its
working directory. Runs use UTC timestamp IDs, with a separate protocol fingerprint:

```text
results/<vlm|ocr>/<run_id>/
  run_config.json, active_execution.json, processing_manifest.json
  executions/<execution_id>/execution.json, processing_manifest.json
  source/, configs/, TRACKER.md
  <model>/<documents|images>/<image_stem>/
    original.<format>, predicted.png, redacted.png, redaction_mask.png
    prediction.json, raw_predictions.json, redaction_validation.json
  <model>/pilot_review.json, *_failure.json, *_timing.json
  reports/summary.json, report.html, report.md, *.csv, *.json
```

Files depend on stage success: failed parsing can retain raw output without a
redacted image. Stage logs are saved under each model's `logs/` directory. Check the
manifest rather than counting PNGs. `document_12` is a position in the frozen
selection, not the original VISPR ID; the viewer and manifest show both identifiers.

Section 8 reads actual selected-run files. Choose **Run → Model → Type → Image**,
use Previous/Next or Jump to #, then inspect three image panels and raw/error/mask
tabs. An image outside that run's workload says **Not requested**. Missing artifacts
are explicit. To inspect saved runs without inference, run Sections 1–4 and 8;
restore a downloaded export under `results/<category>/<run_id>` or mount Drive.

Interpret these separately:

- `status=ok`: structurally usable prediction, possibly an empty list.
- `partial` / `error`: excluded from complete-output quality; retain for diagnosis.
- `mask_application_pass`: saved pixels match the predicted mask, including an empty mask.
- `technical_pass`: additionally checks complete predictions and in-scope empty-mask misses.
- `privacy_verified`: needs independent region truth; none of the audited images has it.

Reports distinguish new inference attempts from copied predictions. Saved timing
medians include reused evidence; execution wall time is separate. `skipped` can mean
reused evidence or blocked work: check reuse origin and remaining-unprocessed counts.
Coverage-table `attempted_images` counts available records, including reused ones;
the execution reconciliation counts **new** inference attempts.

Label precision/recall/F1 use image-level annotations, excluding `a0_safe` from the
positive labels. Missing labels count image/attribute occurrences, not physical
regions. Extra labels are unannotated predictions, not proven false positives.
`all_successful` uses each model's usable subset; `common_successful` uses identical
images across all requested models and may be empty or tiny when a model fails.
Use document/general and shared-text-scope tables alongside full-taxonomy scores.
IoU, true redaction coverage, CER/WER and ROUGE-L need independent boxes/transcripts.

A completed workload requires all intended IDs accounted for, no pending/processing
or missing/unprocessed rows, and explicit failures reported. A complete **valid**
benchmark further needs valid predictions for every requested model/image. A model
worker finishing is not proof of this, nor of privacy removal.

## Drive backup and reproduction

The existing cloud integration is **mounted Google Drive**, not a GCS bucket.
Section 3 calls `drive.mount('/content/drive')`; keep
`/content/drive/MyDrive/Leak_Benchmark_Results/benchmark_v2` as the destination.
Subfolders are `data/vispr_eval_50` and `results/<category>/<run_id>`. No service-account
JSON or GCS configuration is needed. Section 9 retries backup without inference;
Section 10 downloads the entire run to preserve relative links. Verify the final
backup message and read-back file hashes before ending Colab. This verifies the
mounted filesystem, not independent server durability. Do not publish originals,
OCR text or identifying outputs to GitHub as part of a source handoff.

To reproduce: retain the exact selection, model revisions, saved source snapshot,
settings, environment versions, GPU metadata, review records and execution IDs.
For a rerun explicitly disable reuse; for recovery keep reuse enabled. Changed
protocols are new experiments. To audit downloaded runs locally, restore the seven
exports listed in the report, prepare the subset, preserve executed notebooks in
`docs/handoff_20261021/notebooks_received/`, then run
`python scripts/audit_handoff.py`. This reads files only; its public output is
`docs/audit/latest_benchmark.json`, with private details under the ignored handoff
folder. Missing exports must remain labelled missing, not reconstructed as facts.

## Troubleshooting and model status

| Model / symptom | Verified state and next action |
|---|---|
| Qwen3 | 48/50 valid full outputs; strongest paired F1 among the four evaluated models, but low recall and broad masks. Four retained labels on first image; invalid-ID correction succeeded on the saved first image, not a general quality guarantee. |
| Qwen2.5 | Empty pilot predictions; verified 40/50 usable full outputs. Full raw outputs identify JSON/schema failures as well as detection misses. Native T4 precision selection is tested, not proof of recall improvement. |
| Llama | Single/pilot HTTP 401 at gated download; no current inference. Authorize access first, then test one image. No current evidence of an OOM or dependency failure. |
| Florence | Eager loader works without FlashAttention. First image repeats to 1023 tokens with no regions; two copied pilot outputs parse text but detect no PII. Full blocked. Try isolated single-image FP32/native-size/tile diagnostics; quality fix unresolved. |
| Chandra | Fifty recorded outcomes; 46 valid, three errors and one partial. Four native outputs truncated. OCR plus rules has low recall; loading works. |
| dots | Fifty recorded outcomes; 46 valid, four truncated-output errors. Earlier video-processor failure is resolved in these GPU exports. OCR/layout and PII misses remain. |

Do not increase Florence beyond its decoder position bound or install FlashAttention
to cure repetition. Do not change pins indiscriminately in the shared kernel. For
`status must be ok or error`, the current reporter accepts partial records; see the
[report-only recovery procedure](docs/REPORTING_RECOVERY_20261021.md) for older runs.
Failed backups retain local evidence. Inspect the precise stage error before retrying.

Training is planning-only. See [training readiness](docs/TRAINING_READINESS.md).

## Local files and GitHub handoff

Push the maintained source, notebooks, configuration, tests and documentation.
`.gitignore` excludes datasets, imported run folders, private handoff evidence,
generated reports/bundles, checkpoints and local credential scripts. Retain these
locally when needed; do not force-add them. Legacy root helpers remain because
compatibility tests use them. Disposable verification environments and caches
were deleted after testing; recreate your environment from `requirements-dev.txt`.
See [cleanup decisions](docs/AUDIT_AND_MIGRATION.md) for the exact scope.
