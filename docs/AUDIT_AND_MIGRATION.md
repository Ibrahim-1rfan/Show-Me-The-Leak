# Current cleanup and GitHub handoff — 21 October 2026

Current usage: [GUIDE](../GUIDE.md). Current measured results:
[BENCHMARK_REPORT](../BENCHMARK_REPORT.md). Development history: [TRACKER](../TRACKER.md).
The October 20 migration notes below are preserved as historical evidence.

## Inventory and decisions

The recursive metadata inventory is `docs/handoff_20261021/inventory_final.json`
(private/local); its [public summary](audit/handoff_inventory_summary.json) records
counts and scope. Run-source and image artifact hashes are verified by the benchmark
audit; an inventory listing does not imply every byte of the source dataset was hashed.

| Classification | Exact path or path family | Decision / dependency |
|---|---|---|
| Keep | `src/`, `configs/`, `notebooks/`, `tests/`, `scripts/`, four root handoff documents | Maintained pipeline, pinned selection, clean entry points and reproduction |
| Keep | `docs/audit/latest_benchmark.json`, `docs/TRAINING_READINESS.md` | Public aggregate evidence and planning-only training |
| Keep privately | `vlm_*/`, `ocr_*/`, `20261021T*Z/`, `data/vispr_eval_50/` | Current downloaded evidence and frozen data; intentionally ignored |
| Archive in place | `results/`, `results_ocr/`, `Leak_Benchmark_Results/`, `outputs/`, `output/` | Regression/proposal evidence; old tools and documentation still reference these paths |
| Keep compatibility | `Model_Testing.ipynb`, root `benchmark_*.py`, `taxonomy_crosswalk.csv`, root manifests and reconciliation utilities | Legacy tests import the notebook/helpers; not safe to relocate blindly |
| Keep privately | `docs/handoff_20261021/notebooks_received/` | Exact received notebooks/reviews/outputs before cleaning |
| Investigate | `train2017/`, `train2017.zip`, model caches, `tmp/category_review/`, `tmp/regression_sources/`, `tmp/dots_processor_probe/`, `tmp/proposal_render/` | Expensive source or potentially useful diagnostic/visual evidence; not deletion candidates |

No complete run is a duplicate merely because it reuses prediction files: run IDs,
reviews, logs and execution accounting differ. Preserve every such run.

## Completed source cleanup

- Replaced 20,000 redundant per-dataset-file ignore entries with the existing broad
  directory rule; retained other exclusions and added private imports/checkpoints.
- Preserved received executed notebooks, then rebuilt active notebooks with empty
  outputs/reviews and single mode. Legacy notebook outputs were preserved separately
  and cleared from its source copy; its code remains for compatibility tests.
- Removed `output/` and `outputs/` from the Git index only. All local files remain.
  Two previously absent tracked archives, `results.zip` and `results_ocr.zip`, were
  also untracked; their absence predates this task. No filesystem deletion occurred.
- Historical Git objects are unchanged. A later commit records the source cleanup;
  this task does not commit, push or rewrite history. A clone of the old remote
  revision will still have its old contents until the handoff changes are published.
- Local ignored upload scripts contain literal tokens and are excluded from the
  source allowlist. Their values are not copied into documentation or packages.
  This audit's pattern scan covers the source handoff, not credential revocation
  or every historical Git object. Do not publish those local scripts.

## Authorized deletion completed

On 21 October 2026 the user authorized removal of old unimportant files.
The [exact deletion record](audit/cleanup_candidates.json) records 25,184 files
and 294,389,475 bytes removed: Python caches, synthetic scratch files, temporary
widget/development environments, the disposable source verification copy and two
duplicate installation logs. Targets were resolved inside the workspace and
checked for reparse points before deletion. Original test/install logs were
preserved under `docs/handoff_20261021/`; duplicate log hashes matched.

The earlier inventory is a **pre-cleanup snapshot**, not current disk usage.
Datasets, benchmark results, proposal files, executed notebook backups, diagnostic
experiments and legacy source were retained in place. There is no remaining
approval requirement for the deletion already completed. Further removals need
a new dependency/evidence review. Install `requirements-dev.txt` in a new local
environment to run the full development suite again; temporary environments
are not part of the GitHub project.

## Backups and final structure

Received notebooks and earlier documentation were copied locally before editing.
The seven imported runs are hash-inventoried; tracker appendices preserve their
original versions separately. Saved Colab logs verify mounted-Drive copies as
reported in BENCHMARK_REPORT. No new live remote durability check is claimed.

The source structure in README is the final maintained structure. Generated data,
outputs and private evidence remain local at their original paths so links continue
to work. Further physical decluttering can move a reviewed group into ignored
`archive/` after dependencies and backup integrity are rechecked. Training artifacts
will use the separate Drive training destination; no training has started.

---

# Audit and phased migration

Audit date: 20 October 2026. The existing files are preserved in place. No branch,
commit, push, deletion, or Git history modification is part of this migration.

## Findings

- The original entry point is a 31-cell mixed notebook with embedded utility
  snapshots and an additional manual installation history. Tested pure parsing,
  image redaction, taxonomy mapping and metric functions are reusable.
- Backups use **Google Drive**, not a Google Cloud Storage bucket. The existing
  authentication is `google.colab.drive.mount('/content/drive')`; the root is
  `/content/drive/MyDrive/Leak_Benchmark_Results`. New runs use its `benchmark_v2`
  child directory and never replace the older `outputs/multimodel_pii` results.
- The local image directory has been reorganized into `part_0` and `part_1`.
  Historical sample paths no longer all resolve there. The original ZIP retains
  the source entries and is used to materialize the reviewed subset, without
  extracting or overwriting the entire dataset.
- Image-level VISPR labels are available. Verified boxes and transcripts are
  absent. IoU and reference redaction coverage cannot yet be claimed.
- Two local upload scripts contain literal credentials. Their values were not
  printed, copied into the new package, or changed. The local scripts are excluded
  from future Git adds and distribution. This does not remove a credential from
  existing history if it was committed previously; no history was rewritten.
- `.gitignore` contains 20,000 individual dataset entries. New broad directory
  exclusions supplement them without deleting the existing list.

## Performance evidence

`audit/historical_timings.json` records source result hashes and timings for 17
saved runs. These measure the detection call, not full end-to-end notebook time.
Examples across saved runs: Qwen3 median 1.95–3.42 seconds, Qwen2.5 about 1.92
seconds, Llama about 31.36 seconds, Chandra about 33 seconds. Some calls exceed
300 seconds. Chandra loading took about 650 seconds; other successful loads
ranged from roughly 70 to 124 seconds.

These records do not substantiate an hour of inference per image. The exact
cause of the user's observed wait is unresolved. Confirmed sources of extra work:

1. A second inference on every redacted image, enabled by default.
2. Format retries and a 4,096-token generation ceiling; OCR sometimes truncates.
3. Package setup and FlashAttention compilation, distinct from model inference.
4. Weak stage visibility: long calls show no progress until they finish.

Models were already loaded outside the image loop; repeated loading per image is
not a demonstrated cause. GPU availability was checked, but `device_map='auto'`
could allow CPU/disk placement and parameter placement was not comprehensively
verified. The new runner rejects that placement and measures download, load,
preprocessing, generation, postprocessing, image wall time and peak GPU memory.
GPU utilization is sampled during the run. The pilot must establish new timings.

## New structure

```text
notebooks/       independent VLM, OCR, and training-readiness entry points
src/leakbench/   reusable adapters, dataset, execution, metrics, reporting, backup
configs/         frozen cohort, taxonomy, checkpoint revisions, run parameters
data/vispr_eval_50/{documents,images,annotations}/
results/{vlm,ocr}/<run_id>/<model>/{documents,images}/<image_stem>/
results/{vlm,ocr}/<run_id>/reports/
docs/            audit, cleanup inventory, training and branching guidance
scripts/         reproducible notebook/package builders and validation utilities
dist/            generated notebook distribution and optional 50-image archive
```

The new package is the maintained implementation for v2. The root `benchmark_*`
modules remain the legacy notebook's frozen implementation. They are deliberately
not imported by v2. This transitional duplication preserves old reproducibility;
an eventual legacy archive can remove the root clutter after cloud verification.

## Dataset decision

All 50 historical images and replacement candidates were visually inspected as
rendered contact sheets. The 25 document images are retained. Eleven general
entries are replaced: nine safe-labeled images, including a document-like entry,
and two other document-style images. The replacement set retains privacy-labeled
general scenes. Category review does not invent privacy labels or region truth.
`configs/selection.json` records exact decisions, source members, annotation and
image hashes. The new cohort must not be merged into old results as if identical.
Byte and decoded-pixel duplicates are rejected; close perceptual-hash matches
require explicit visual review. A source JPEG-named file is actually MPO and is
preserved as `.mpo`, without re-encoding.

## Phase status

1. Audit: completed; exact hour-long slowdown remains unconfirmed.
2. Reorganization: implemented; new entry points and package, legacy retained.
3. Dataset: materialized and locally validated at exactly 25 plus 25.
4. Optimize and pilot: instrumentation and pilot gates implemented; GPU run pending.
5. Full benchmark: workflow ready; measured results and model selection pending.
6. Cloud persistence: checksum-verifying backup implemented; real Colab backup and
   subsequent cleanup pending. A local copy test is not cloud verification.
7. Training readiness: plan and disabled configuration prepared; no training run.
8. Branch strategy: documented only; no branches or history changed.

## Controlled cleanup proposal

The hash inventory is `audit/inventory.json`. Retain all historical result folders,
dataset archives, upload scripts and document proposals. After a real Drive backup
passes read-back verification, group the original notebook, root helpers, old
inspection scripts and original manifests under `archive/legacy_2026_10/` in a
separate reviewed migration. Update any relative paths before moving them. Leave
dataset relocation until all old source references are recorded. Generated
`__pycache__` and rendering intermediates are cleanup candidates only; they have
not been deleted. Do not remove an input simply because its outputs exist.
