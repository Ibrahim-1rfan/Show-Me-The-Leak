# Show Me the Leak

Use the **v2 notebooks** for new work. Each contains a verified source snapshot,
configuration and the fixed image selection; no Git clone or other notebook is
required. Model dependencies run in separate Python 3.11 workers. GPU inference
and Drive authentication happen in your Colab session.

| Entry point | Models |
|---|---|
| `notebooks/01_vlm_benchmark.ipynb` | Qwen3 VL 8B, Qwen2.5 VL 7B, Llama 3.2 Vision 11B |
| `notebooks/02_ocr_benchmark.ipynb` | Florence 2 large ft, Chandra 9B, dots.ocr 3B |
| `notebooks/03_training_readiness.ipynb` | Readiness checks and plan only; never starts training |

## Run in Colab

1. Import either benchmark notebook, select a GPU runtime, and run from the top.
   For dots use L4/A100 or another supported Ampere-or-newer GPU; a T4 cannot run
   the current dots profile. A Pro subscription does not replace the GPU check.
2. Mount Drive when prompted. The destination remains
   `MyDrive/Leak_Benchmark_Results`, with new artifacts under `benchmark_v2/`.
3. For Llama, accept its Hugging Face terms and add a Colab secret named `HF_TOKEN`
   with notebook access. Do not paste a token into a cell. Public models can run
   without that secret; a denied gated model remains an explicit failure.
4. Dataset preparation reuses the verified Drive subset if present. Otherwise it
   downloads the original Hugging Face ZIP once and extracts only the 50 fixed
   image/annotation pairs. Optionally upload `dist/vispr_eval_50.zip` to `/content`
   to avoid the first full-source download; the hashes are still verified.
5. Every model first runs a two-image pilot (one per category). Review its timings,
   predictions and redactions, then set `RUN_FULL_BENCHMARK=True` and rerun sections
   5 onward for all 50 images. A full run proceeds only if both pilot outputs are
   valid. Completed pilot images are reused in the 50-image benchmark.
6. Review the displayed coverage, quality and runtime tables and the image viewer.
   Reports are exported as HTML, Markdown, CSV and JSON. The backup cell can be
   rerun independently if Drive copying fails. A partially failed run is never
   reported as a complete benchmark or assigned a fabricated winning model.

Matching run configurations resume automatically from local or Drive results.
Only completed outputs whose artifact hashes still match are skipped. Failures
are retried on an explicit rerun, with all previous image artifacts archived.
Completed image artifacts are backed up individually; full reports, logs and
configuration are backed up before and after each model phase. Backup failures
retain local results and can be retried independently.
New source, model revisions, selection or inference settings create a new protocol.
Cloud copies are verified by read-back hashes through the mounted filesystem.

## What has and has not been validated

The fixed cohort is locally validated: exactly 25 visually reviewed document
images and 25 general images, source/annotation hashes, readability, unique original
IDs, decoded-pixel duplicate checks and perceptual duplicate screening. All inputs
retain their original bytes and actual format. Eleven entries differ from the old
cohort, so old model scores are historical evidence only.

Offline checks exercise dataset integrity, failure accounting, safe resume, copy
verification, notebook syntax and packaging. GPU timing, full-model correctness,
cloud persistence and final model selection await your Colab runs. No training is
enabled. IoU and ground-truth redaction coverage are unavailable with image labels
alone. OCR plus fixed English rules is reported separately from direct visual PII.

## Organization and maintenance

- Maintained implementation: `src/leakbench/`; fixed inputs: `configs/`.
- New data: `data/vispr_eval_50/`; generated runs: `results/<category>/<run_id>/`.
- Each model/image folder contains the original, prediction visualization,
  redacted PNG on successful detection, raw predictions and structured metadata.
  Failed images explicitly lack a verified redaction.
- Legacy notebook, root helpers, proposals and saved runs are retained in place.
  See `docs/AUDIT_AND_MIGRATION.md` before any cleanup.
- `docs/TRAINING_READINESS.md` covers candidate selection and leakage prevention;
  `docs/BRANCHING_STRATEGY.md` explains Git options without executing them.
- Regenerate notebooks after package edits with `python scripts/build_notebooks.py`.
  Run `python -m unittest discover -s tests -v` before rebuilding distribution.

The default baseline retains 1,280-pixel input edge and 4,096 output-token limits,
one format retry, and single-image batches. It removes the hidden second audit
pass, recording that protocol change. Batch size 1 is deliberate for heterogeneous
variable-resolution models and per-image failure isolation; batching is a future
measured experiment, not an unverified speed claim. Environment setup, downloads,
GPU load, image preprocessing, generation and postprocessing are timed separately.
