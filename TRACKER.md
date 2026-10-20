# Project tracker

## A. Project overview

Objective: detect visible privacy information, localize it, redact images and assess
quality without confusing valid inference with verified privacy removal.

Confirmed root: `D:/Github/Show-Me-The-Leak`. Imported Colab results are inside
`results/content/Show-Me-The-Leak-v2/results/vlm/20261019T230132317596Z/`.
`results.zip` preserves the supplied export. Maintained code is `src/leakbench/`;
`scripts/build_notebooks.py` embeds it in the existing category notebooks under
`notebooks/`. The root `Model_Testing.ipynb` and `benchmark_*` helpers are legacy.

Dataset: VISPR, `Ibrahim1rfan/VISPR`, original train2017 ZIP and JSON image labels;
frozen `configs/selection.json` and `data/vispr_eval_50/` contain 25 document and 25
general images. There are no verified reference region boxes or transcripts.
VLMs: Qwen3 VL 8B, Qwen2.5 VL 7B, Llama 3.2 Vision 11B. OCR: Florence 2 large ft,
Chandra 9B, dots.ocr 3B. Model profiles use isolated Python 3.11 environments on
Colab. Imported Qwen runs used an A100 80GB and CUDA-only parameters. Local GPU
inference is unavailable; the user runs prepared notebooks in Colab.

Storage: existing `drive.mount('/content/drive')` authentication and
`MyDrive/Leak_Benchmark_Results/benchmark_v2/`. No GCS bucket is involved. Preserve
all previous outputs, credentials and source snapshots. Never record secret values.

## B. Work completed

| Date | Files/action | Purpose and outcome |
|---|---|---|
| 2026-10-20 | Read new request; inspected root, imported results, saved source and raw predictions | Found four Qwen pilot predictions, all empty final results; Llama access failure. No new full benchmark or training launched. |
| 2026-10-20 | Compared imported `source/leakbench/*.py` with maintained source | All 14 files match before repairs; establishes exact implementation baseline. |
| 2026-10-20 | Created `TRACKER.md` | Persistent chronology, evidence, issues and handover started before repairs. |

## C. Errors and troubleshooting

| Issue ID | Date | Symptoms | Root cause | Files affected | Fix applied | Validation | Status |
|---|---|---|---|---|---|---|---|
| REG-001 | 2026-10-20 | Qwen3 document_1 loses three valid detections | Confirmed: all-or-nothing parsing rejects `a10_face_complete` in two entries; empty retry replaces entire first response | `pipeline.py`, imported Qwen3 document_1 raw/prediction JSON | Pending | Saved first response contains valid credit-card, student-ID and ticket boxes | investigating |
| REG-002 | 2026-10-20 | Unchanged known-positive images counted as successful pilot outputs | Confirmed: status checks parsing/artifact hashes, not redaction completeness or changed pixels | `worker.py`, `orchestrate.py`, `reporting.py` | Pending | All four saved final detections are empty with positive VISPR labels | investigating |
| REG-003 | 2026-10-20 | Qwen2.5 both images and Qwen3 general image return empty lists | Unknown; raw inference is already empty, so parser cannot recover detections | Imported Qwen predictions; model prompt/configuration | Pending controlled GPU diagnosis | Outputs are not truncated; GPU placement is CUDA on A100 | open |
| ACCESS-001 | 2026-10-20 | Llama never reaches inference | Confirmed 401 gated repository denial at model download | Imported Llama `pilot_failure.json` | Requires authorized HF account and notebook secret; no credential changes | Exact saved error inspected | open |
| DATA-001 | 2026-10-20 | User reports Hugging Face dataset import failure | Unknown; no dataset download failure log yet identified | `dataset.py`, notebook dataset cell | Inspect and harden diagnostics without substituting data | Pending | investigating |

## D. Model performance regressions

Latest run: `20261019T230132317596Z`, protocol
`eb723527bb7bdc14e13cc2c37b89c5e4307456aba5f911065226516fe07f0572`.
Examples: document_1 `2017_12973862`; image_1 `2017_24408793`.
Qwen3 first document attempt contains five regions, three with valid taxonomy IDs;
the two invalid IDs must not be silently guessed or remapped. No confirmed saving
or coordinate-scaling defect established yet. OCR has no new imported results.
Previous matching-image comparisons and actual pixel inspection are pending.

## E. Dataset and Hugging Face status

Selection fingerprint:
`0beae8a380049243b74c8c9e2605ef3d01ecbdfcdfaa14f99dc74e8a3cb50b72`.
The locally validated cohort exists; this does not prove live Hugging Face import.
Original IDs, byte hashes and annotations remain in the frozen manifest. Current
implementation downloads a ZIP using HTTP, not `datasets.load_dataset`; inspect
that actual path before attributing an issue to the `datasets` API.

## F. Benchmark results

Imported detection-call seconds: Qwen3 document 23.8635 (two generation calls),
general 0.7879; Qwen2.5 document 1.6475, general 0.7933. These are measurements,
not post-fix performance claims. Four parse-success outputs do not establish four
successful privacy redactions. No new full benchmark is authorized before repair
and pilot verification. Existing historical runs remain in `Leak_Benchmark_Results/`.

## G. Pending tasks and blockers

- [ ] Preserve hash inventory and inspect original/predicted/redacted image pixels.
- [ ] Compare matching earlier results and saved implementation.
- [ ] Repair parser data loss and add explicit redaction correctness validation.
- [ ] Replay known-positive saved outputs; validate document and general cases.
- [ ] Harden dataset access, integrity, resume and diagnostic stages.
- [ ] Repair existing notebook gates; quarantine failed models before full run.
- [ ] Record cleanup inventory with exact paths; delete nothing.
- [ ] New GPU single-image and small pilot: requires user's Colab runtime.
- [ ] Llama authorization: requires user's authorized Hugging Face account.
- [ ] Full benchmark, verified Drive persistence and training selection: blocked
      until correctness checks and human pilot review pass.

## H. Cleanup inventory

No deletion approved or performed. Inventory is being prepared. Original datasets,
annotations, exported `results.zip`, imported results, prior successful outputs,
source snapshots and dependency records must be retained for regression analysis.

## I. Next-session handover

Session in progress. Open this tracker, imported run `run_config.json`, Qwen3
document_1 `prediction.json`, and `src/leakbench/pipeline.py` first. Confirmed bug:
one invalid label invalidates valid detections and a later empty response hides
the loss. Do not claim model recovery from replay alone, launch a full benchmark,
or infer reference boxes from image-level labels.
