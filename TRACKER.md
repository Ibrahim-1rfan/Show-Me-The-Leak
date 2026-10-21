# Project tracker

## Current handoff status — 21 October 2026

Read [README](README.md) → [GUIDE](GUIDE.md) → [BENCHMARK_REPORT](BENCHMARK_REPORT.md).
Sections A–K below are dated historical records; **Section L supersedes their
coverage, model-status and archive-presence statements**. Section M records the
subsequently authorized filesystem cleanup. The handoff is local;
no commit/push or model training has been performed.

- Seven exports cover all six requested modes plus an extra blocked OCR full
  request. 237 saved prediction records and 2,694 run files were audited; all listed
  artifact/source/input hashes and saved-mask rechecks pass. Both metric scopes
  recompute consistently for every export.
- Full valid output counts: Qwen3 48/50, Qwen2.5 40/50, Chandra 46/50, dots 46/50.
  Florence reuses only 2/50 and is blocked; Llama has 0/50 (authorization failure).
  A complete valid six-model benchmark has not been achieved.
- Qwen3 is the recommended first training candidate, supported by paired metrics;
  training is disabled pending annotations, disjoint splits and a GPU training test.
- Authorized cleanup removed 25,184 disposable files (294,389,475 bytes): caches,
  temporary dependencies/source copy and duplicate logs. Important source, datasets
  and benchmark evidence remain. See [deletion record](docs/audit/cleanup_candidates.json).
- Source-only reproduction passed 131 tests in both the existing environment and
  a fresh development environment. GPU/model-quality fixes are not inferred from
  documentation changes. Full evidence: docs/audit/handoff_validation.json.


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

Recovery on 2026-10-21: prior session stopped after audit/tracker creation. No
repair or repair-validation claim from that session is accepted. New OCR results
and executed notebooks are now present. Exact notebook names are
`notebooks/01_vlm_benchmark.ipynb` and `notebooks/02_ocr_benchmark.ipynb`; there is
no separate `ocr.ipynb`. Both contain 11 executed cells and a user-added ZIP export
cell. Executed copies are preserved under
`docs/regression_20261021/notebooks_before_repair/`; 156 imported files/notebooks
are recorded in `preserved_inventory.json`. User export functionality will be kept.

| Date | Files/action | Purpose and outcome |
|---|---|---|
| 2026-10-20 | Read new request; inspected root, imported results, saved source and raw predictions | Found four Qwen pilot predictions, all empty final results; Llama access failure. No new full benchmark or training launched. |
| 2026-10-20 | Compared imported `source/leakbench/*.py` with maintained source | All 14 files match before repairs; establishes exact implementation baseline. |
| 2026-10-20 | Created `TRACKER.md` | Persistent chronology, evidence, issues and handover started before repairs. |
| 2026-10-21 | Preserved executed notebooks and 156 imported files in the hash inventory | Both user ZIP-download cells identified; replacement export cells retained in repaired notebooks. No prior result modified. |
| 2026-10-21 | Audited both imported runs and compared matching historical images | Four Qwen redactions exactly equal original pixels. Florence has two failure overlays and no redacted images. Earlier matching Qwen/Chandra examples also had zero detections; no evidence those exact examples recently regressed. |
| 2026-10-21 | Repaired per-entry validation, saved-mask validation and staged gates | Valid detections survive malformed siblings; empty positive outputs cannot pass a redaction gate. Four-image pilot plus artifact-bound human review required before full run. |
| 2026-10-21 | Replayed four saved Qwen responses without model inference | Qwen3 document retains 3 regions and masks 1,066,354 pixels, but stays partial; other 3 empty outputs remain unresolved. Inspected replay image: remaining face/other content still visible. |
| 2026-10-21 | Fixed dots import; installed isolated CPU probe libraries under `tmp/dots_processor_probe` | Transformers 4.56.1 imports correct classes; real pinned dots tokenizer/processor passes the 56x56 image-token/grid check. Python 3.10 and CPU torch 2.4.1 differ from production Python 3.11/CUDA torch 2.7; full GPU model still unverified. Main Python packages untouched. |
| 2026-10-21 | Inspected official Florence config and processor; repaired diagnostics | Effective decoder cap retained; raw generation survives invalid-region parsing. Added explicit whole/native/crop and float32 diagnostic settings without claiming generation recovery. |
| 2026-10-21 | Dataset source and downloads audited | Public HF API confirms ZIP-only repository, revision `831e0de941ddf741fa00217c9d78d91b57c9af93`, 22,400,164,976 stored bytes. Initial sandbox DNS attempt failed; approved public-metadata request succeeded. Full 22GB transfer not repeated. Added range/ETag, disk and stage diagnostics; no alternate dataset. |
| 2026-10-21 | Ran initial 11 repair tests | Passed parser retention, masks, integrity, coordinate offsets, crop coverage, raw-output preservation and network error checks. |
| 2026-10-21 | Ran full 109-test suite | 108 passed; legacy syntax test rejected user's valid `!pip install flash_attn` cell. Adjusted test to use IPython input translation without executing the cell; notebook preserved. Final rerun pending. |
| 2026-10-21 | Final full regression suite | All 112 tests passed, including model/input association, failed positive mask gates, stale reviews and notebook source restoration. Earlier 109-test syntax failure is resolved. |
| 2026-10-21 | Final cohort/preservation checks | All 50 original image/annotation mappings and hashes validated; all 156 preserved imported artifacts/notebook copies match the audit inventory. No full benchmark, training, live cloud mutation or deletion performed. |
| 2026-10-21 | Updated existing notebooks, README, report, distribution and ignore rules | Same active notebook paths, user export workflow retained, tracker embedded/backed up. Regression evidence folders excluded from accidental Git adds. Final handoff in `docs/REGRESSION_REPORT_20261021.md`. |

## C. Errors and troubleshooting

| Issue ID | Date | Symptoms | Root cause | Files affected | Fix applied | Validation | Status |
|---|---|---|---|---|---|---|---|
| REG-001 | 2026-10-20 | Qwen3 document_1 loses three valid detections | Confirmed: all-or-nothing parsing rejects `a10_face_complete` in two entries; empty retry replaces entire first response | `pipeline.py`, imported Qwen3 document_1 raw/prediction JSON | Keep separately validated entries; record rejected entries; no guessed label remap or empty overwrite | Real saved-response replay retains and masks 3 regions; partial output remains quarantined | verified for saved-output replay |
| REG-002 | 2026-10-20 | Unchanged known-positive images counted as successful pilot outputs | Confirmed: status checked parsing/artifact hashes, not redaction completeness or changed pixels | `validation.py`, `gates.py`, worker/reporting/orchestration | Independent mask validation and explicit artifact-bound pilot review | Saved empty responses flagged; synthetic overlay, corrupt output, zero-mask, wrong-input and stale-review checks | verified offline; GPU acceptance pending |
| REG-003 | 2026-10-20 | Qwen2.5 both images and Qwen3 general image return empty lists | Unknown; raw inference is already empty, so parser cannot recover detections | Imported Qwen predictions; model prompt/configuration | Pending controlled GPU diagnosis | Outputs are not truncated; GPU placement is CUDA on A100 | open |
| ACCESS-001 | 2026-10-20 | Llama never reaches inference | Confirmed 401 gated repository denial at model download | Imported Llama `pilot_failure.json` | Requires authorized HF account and notebook secret; no credential changes | Exact saved error inspected | open |
| DATA-001 | 2026-10-20 | User reports Hugging Face dataset import failure | Original cause unknown; repo is public ZIP-only, not a datasets table | `dataset.py`, notebook dataset cell | Pin source revision; stage diagnostics, bounded retries, disk checks, range/ETag validation | Live metadata read and mocked access/range tests; local cohort revalidation; full network transfer not repeated | original failure unresolved; diagnostics tested |
| OCR-001 | 2026-10-21 | Chandra never reaches inference | Confirmed HTTP 429 from HF Xet CDN during weight download, not dataset import | `downloads.py`, worker | Bounded transient retries, two download workers, retained cache | Simulated 429 recovery and nonretryable 401 tests; live model retry pending | recovery logic verified; provider retry pending |
| OCR-002 | 2026-10-21 | dots fails during processor construction | Confirmed nonexistent `Qwen2_5_VLVideoProcessor` import in pinned Transformers 4.56.1 | `ocr_adapters.py` | Use `Qwen2VLVideoProcessor`, keep original tokenizer/template/image token | Real pinned-processor CPU image-token/grid check passed | verified CPU processor; GPU pending |
| OCR-003 | 2026-10-21 | Florence document has no usable OCR regions | Confirmed output reaches effective 1023-token budget, forced EOS, repetitive text, zero parsed regions; cause of repetitive generation unresolved | `adapters.py`, imported Florence document_1 | Keep decoder bound; diagnostic crops/precision experiment pending | Raw output and original checkpoint config inspected; not an OOM error | investigating |
| OCR-004 | 2026-10-21 | Florence general image parser raises and loses raw evidence | Confirmed zero-area-region exception occurs before native output returns to caller | `adapters.py` | Preserve raw/native output and per-region rejection details; never accept malformed empty output | Regression test exercises adapter with malformed region; old missing raw output cannot be recovered | evidence-retention fix verified offline; fresh generation pending |

## D. Model performance regressions

Latest run: `20261019T230132317596Z`, protocol
`eb723527bb7bdc14e13cc2c37b89c5e4307456aba5f911065226516fe07f0572`.
Examples: document_1 `2017_12973862`; image_1 `2017_24408793`.
Qwen3 first document attempt contains five regions, three with valid taxonomy IDs;
the two invalid IDs must not be silently guessed or remapped. No confirmed saving
or coordinate-scaling defect established. OCR audit is now available in
`results_ocr/content/Show-Me-The-Leak-v2/results/ocr/20261019T235754632234Z/`.
The two matching image IDs also had zero final detections in earlier Qwen runs.
Other historical Qwen3 runs contain 21 nonempty images; these cannot establish
success on the two currently failed images. No paired before/after GPU improvement
has been measured. Current masks are independently checked against saved pixels.

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

- [x] Preserve hash inventory and inspect original/predicted/redacted image pixels.
- [x] Compare matching earlier results and saved implementation.
- [x] Repair parser data loss and add explicit redaction correctness validation.
- [x] Replay known-positive saved outputs; document partial/failed general cases without inventing detections.
- [x] Harden dataset access, integrity, resume and diagnostic stages (actual original network failure unresolved).
- [x] Repair existing notebook gates; quarantine failed models before full run.
- [x] Record cleanup inventory with exact paths; delete nothing.
- [ ] New GPU single-image and small pilot: requires user's Colab runtime.
- [ ] Llama authorization: requires user's authorized Hugging Face account.
- [ ] Full benchmark, verified Drive persistence and training selection: blocked
      until correctness checks and human pilot review pass.

## H. Cleanup inventory

No deletion approved or performed. Inventory is being prepared. Original datasets,
annotations, exported `results.zip`, imported results, prior successful outputs,
source snapshots and dependency records must be retained for regression analysis.
Exact-path inventory: `docs/regression_20261021/cleanup_inventory.csv` (45 root
entries, all retained). Future candidates only: `D:/Github/Show-Me-The-Leak/tmp/`,
`D:/Github/Show-Me-The-Leak/__pycache__/`, `D:/Github/Show-Me-The-Leak/dist/`.
They are regenerable but **not eligible for deletion now**: backup and dependency
criteria are not all satisfied. The processor probe and released bundles are
currently useful evidence. No deletion is proposed for immediate approval.

## I. Next-session handover

Open this tracker and `docs/REGRESSION_REPORT_20261021.md`, then the existing
category notebook and its corresponding imported failure/prediction JSON.
Parser data loss, mask validation, per-image identity, raw Florence evidence,
correct dots import, network diagnostics and explicit gates have been implemented.
The actual dots processor CPU check passed. Qwen3 saved-response replay masks three
valid regions but remains partial; the original empty Qwen responses are unresolved.
Florence's repetitive generation and zero-area response still need GPU diagnosis.
Chandra needs a live retry after rate limiting; Llama needs authorized gated access.

Start Colab with `RUN_STAGE='single'`, not a full benchmark. Keep the same 50-image
selection. If Florence fails unchanged, compare one factor at a time: FP32, then
native/crop policy, preserving every diagnostic run. Run `RUN_STAGE='pilot'` only
after the positive-document mask check passes. Inspect all four pilot outputs,
record honest per-image notes, then permit a diagnostic full benchmark. No training
or cleanup until valid results and verified Drive backup exist. No fresh GPU
quality or runtime improvement has been claimed in this session.
### Image-size decision (2026-10-21)

All 25 documents are JPEG, 480–3648 pixels wide and 328–2448 high. General images
are JPEG/MPO, 402–5616 wide and 468–5184 high. The baseline 1280 cap downsizes 9
documents and 16 general images. Original bytes and original-resolution redaction
remain authoritative. Florence's pinned processor resizes to 768×768; the native
processor behavior is recorded rather than hidden by a nominal common cap.

Default baseline is retained. Processor-native (`max_input_edge=None`) and bounded
1024-pixel overlapping OCR tiles are explicit separate experiments, not proven
recognition improvements. Coordinate offset and complete tile-plan coverage tests
passed. No GPU size/precision quality comparison has been run. Independent region
and transcript labels are still needed for IoU/redaction coverage/CER/WER claims.

## J. Partial-result reporting regression (2026-10-21)

New user-supplied log: run `20261020T230625225013Z`, Qwen3 single document
`2017_12973862`. Worker completed; image processing averaged 26.75 seconds. Saved
status was `partial`, redaction `needs_review`, mask check false. Seven image
artifact copies were verified through mounted Drive according to the supplied log.
This is new GPU execution evidence, **not** a passing privacy/pilot result. Other
models did not proceed because report generation rejected `partial` and the same
unguarded call raised again in `finally`. The previous 112-test suite missed the
parser-to-report boundary. The defect belongs to our reporting integration.

- Metrics now accept partial records, retain their inspectable labels, exclude them
  from quality and successful coverage, and report separate partial/error counts.
  `failed_images` remains the sum of partial and error attempts for compatibility.
- Reporting errors are recorded separately; they no longer block subsequent models
  or backup attempts. Persistent report failures surface after processing/backups.
- Added a reporting-only Colab recovery script, with source backups/hashes and
  live-module refresh. Predictions, original run configuration/source snapshots and
  historical failures remain unchanged. No model rerun, install or download.
- Added five regression cases: parser partial to both report scopes with mixed
  statuses, partial worker plus next model, persistent/transient report failures,
  and a live-module Colab recovery simulation preserving saved evidence.
- Validation: all 117 local tests passed (40.1 seconds), including the real
  parser-to-report integration and simulated live Colab repair. An initial recovery
  test caught Windows path separators in the repair manifest; paths now use portable
  forward slashes. No fresh local GPU inference or live Drive check was performed.
- Next user action: apply `dist/repair_colab_reporting.py` in the existing runtime,
  inspect saved partial detections/rejected entries and mask issues. Do not advance
  to pilot/full just because reports now render. See
  `docs/REPORTING_RECOVERY_20261021.md` for exact recovery steps.

## K. Six-mode execution and viewer recovery (2026-10-21, in progress)

Preservation inventory: `docs/recovery_20261021/preserved_inventory.json` (335
entries); current notebooks copied into `notebooks_before_repair/` there. Latest
imported runs are VLM `20261020T234601512294Z` and OCR `20261021T011930055215Z`.
Their embedded trackers record single, pilot and full invocations. Only single
worker jobs/timings/predictions exist: later calls hit the regression gates before
inference and reused the same report/output directory. OCR notebook execution 21
is the full request; execution 20 resumes that same OCR run. The VLM execution
counters are cleared but its saved outputs confirm resuming the full request at
00:27:54 UTC. Its outputs were checked independently of the cleared counters.

- RUN-01 — Same displayed results. Confirmed cause: pilot/full blocked by initial
  quality checks, stable protocol run ID, reports over old single outputs. Fix in
  progress: unique execution/artifact run, explicit plan/manifest/reconciliation,
  diagnostic pilot independent of single-image recall. Validation pending.
- VIEW-01 — Section 8 always resets INSPECT_INDEX=0 and reads only saved records.
  No interactive navigation or missing-image inventory. Viewer replacement pending.
- MODEL-01 — Qwen3 produces three retained regions and two invalid attribute IDs
  (`a10_face_complete`). Partial, not complete redaction. Prompt/quality issue
  remains under investigation; no silent label remapping.
- MODEL-02 — Qwen2.5 emits a genuine empty JSON response (11 tokens, not truncated).
  No mask changes. Detection failure remains unresolved. T4 native precision check
  is being corrected: logged BF16 can include emulation; no speed/quality claim yet.
- MODEL-03 — Llama still fails model download with gated-repository HTTP 401.
  Requires the user's authorized HF account; not fixed by package installation.
- MODEL-04 — Florence reaches its 1,023-token decoder budget with repetitive text
  and zero spatial regions on L4. Parsing correctly rejects this generation.
  Underlying generation quality unresolved; FP32/crop diagnostics require GPU.
- MODEL-05 — Chandra now downloads/loads/runs, returning six Image descriptions,
  zero OCR text regions; they are deliberately not treated as literal text evidence.
  Empty redaction is a detection limitation, not a fresh download failure.
- MODEL-06 — dots now downloads/loads/runs, returning six Picture blocks with no
  text. The previous processor import repair is GPU-confirmed by this export.
  Text recognition/redaction on this photograph still fails.
- SAVE-01 — Per-image coordinate/save exceptions can terminate the model loop.
  Per-image recovery and early raw-output persistence are being added; tests pending.

No GPU reruns, cloud uploads, cleanup or model substitutions have been performed
locally. Existing mounted-Drive authentication/destination remain unchanged.

### K validation progress

- RUN-01 / VIEW-01 / SAVE-01: first 128-test suite passed. It includes real widget
  selection/Previous/Next/category/jump/run switching, unique runs, compatible
  model-subset reuse, full gating, plan reconciliation and per-image failure recovery.
  The widget test initially found a missing default selection; it is now fixed.
- MODEL-01: bounded invalid-ID correction implemented. Valid detections and
  rejected-entry geometry cannot be erased/replaced by an invalid retry. Original
  and corrective responses remain auditable; no label mapping or ground truth is
  used. Added tests; new GPU output remains pending.
- MODEL-02: native T4 FP16 selection is tested with hardware stubs; detection
  quality/speed remains unverified. Not a claim to have fixed empty Qwen2.5 output.
- MODEL-04: FP32 now enters the original weight loader directly. A loader-spy test
  confirms it does not load FP16 then cast. Repetitive GPU generation unresolved.
- Added attempted-inference state before generation, so interrupted calls are
  distinguished from never-started images. Additional regression validation pending.
- All 18 saved source hashes in each latest export match. All five predictions'
  listed artifact hashes match. Original/mask/redacted pixels were independently
  rechecked: Qwen3 masks 1,065,906 pixels but leaves PII; three empty predictions
  change zero pixels; Florence has no redaction. See `redaction_contact.png` and
  `audit.json` in `docs/recovery_20261021/`.
- Cloud: saved OCR notebook verifies 93 result files and 103 dataset files through
  mounted Drive (last shown 2026-10-21 01:36:34 UTC). Connected Drive searches for
  Leak_Benchmark_Results and benchmark_v2 returned no accessible results here.
  No new upload/server-side verification claimed; the Colab destination is unchanged.

### K final issue status and handover

| Issue | Confirmed cause / fix | Validation and status |
|---|---|---|
| RUN-01 | Later stages were blocked while the same artifact folder/report stayed visible. Fresh runs, explicit plans, immutable execution history, live mode selector, scoped reports and reconciliation added. Pilot no longer requires good single-image recall. | Offline mode/reuse/gate tests passed; full GPU benchmark NOT completed. |
| VIEW-01 | Fixed INSPECT_INDEX=0 had no navigation and ignored missing images. Replaced with run/model/category/image selectors, Previous/Next/jump, three image panels, raw/error and mask tabs. | Actual ipywidgets control tests passed, including default selection and switching runs. New Colab UI run still pending. |
| SAVE-01 | Conversion/save failures could terminate the loop and lose raw output. Raw is now saved first; individual errors are stage-labelled; inference-attempt state survives interruption. | Invalid-coordinate → next valid image and interrupted-attempt tests passed. |
| MODEL-01 | Qwen3 used two invalid taxonomy IDs. Optional bounded model-authored correction preserves accepted detections and rejected geometry. | Correct/empty/changed-box correction tests passed. Fresh GPU correction quality UNVERIFIED. |
| MODEL-02 | Empty Qwen2.5 JSON is a detection miss, not truncation. T4 BF16 support check could select emulation; native FP16 now selected. | Hardware-stub test passed. Empty response cause/quality remains UNDER INVESTIGATION. |
| MODEL-03 | Llama gated repository HTTP 401. | BLOCKED on user's HF account authorization; no model substitution or quality score. |
| MODEL-04 | Florence repetitive output at 1,023 tokens with no regions. FP32 diagnostic now loads original FP32 weights directly. | Loader-spy test passed; generation problem UNRESOLVED pending GPU diagnostic. |
| MODEL-05 | Chandra generated six Image descriptions, no literal text regions. | Download/load/inference confirmed by latest export; redaction on this photo FAILED. |
| MODEL-06 | dots generated six Picture blocks, no text. | Previous processor fix confirmed by GPU export; redaction on this photo FAILED. |
| CLOUD-01 | Existing mounted-Drive destination retained; notebook records verified backup. Connected search cannot see it here. | No new live upload/server durability verification claimed. Colab backup still required for new runs. |

Validation: **131 tests passed**, including 14 execution/viewer/correction cases,
in 67.9 seconds. A transient Windows notebook-write error cleared on the immediate
retry; the resulting notebooks are being checked against their embedded source.
Viewer test dependencies were installed only in `tmp/viewer_test_deps`, not the
main Python installation. Full test command used that folder on PYTHONPATH.

Files changed for this recovery: maintained `orchestrate`, `reporting`, `worker`,
`pipeline`, `legacy_workflow`, `adapters`, `ocr_adapters`, `gates`, `validation`;
new `progress`, `image_execution`, `hardware`, `viewer`; notebook/recovery builders,
new audit script and execution tests; existing notebooks, README, tracker and
recovery documentation. Historical model outputs and source snapshots remain intact.

Next: open the existing category notebook; Sections 1–4 then 8 can inspect saved
Drive runs without inference. Use the live pilot selector for the four-image
diagnostic. Review all four outputs before full; do not mistake mechanics passing
for complete PII removal. Isolate Florence FP32 and then crop experiments, resolve
Llama authorization, and measure Qwen correction/native T4 precision on GPU.
No full benchmark, new quality improvement, new cloud upload, training, cleanup or
model ranking is claimed. Detailed findings: `docs/EXECUTION_RECOVERY_20261021.md`.

Preservation reconciliation verified **333 distinct files** from the initial
335-entry inventory (two active training-notebook entries resolve to its saved
original). Historical outputs/source files are unchanged. The two imported run
trackers received only appended audit notes; both originals are saved separately.
Both exported ZIP archives remain present and are hashed in the final preservation
record. The VLM notebook has saved outputs despite cleared execution counters:
its final backup records 84 run files at 00:28:05 UTC. OCR records 93 files at
01:36:34 UTC; each dataset backup records 103 files. No live cloud claim is inferred.


## L. Final export audit and GitHub handoff (2026-10-21)

HANDOFF-01 — Planned/in progress: audit newly supplied single/pilot/full evidence,
verify metrics and reviews, investigate Florence/Llama, and update README, GUIDE,
BENCHMARK_REPORT and this tracker with distinct responsibilities. Preserve received
executed notebooks before producing clean entry points. Inventory cleanup candidates
and references; no permanent deletion is authorized. No training or GPU rerun.
New imports: OCR 055306013536Z and 073101588696Z; VLM 055302824843Z and
060533341923Z (all 20261021). Earlier K conclusions are historical and must not be
used as current coverage. Two tracked ZIP deletions already existed on arrival
(results.zip, results_ocr.zip); do not attribute them to this audit.

HANDOFF-02 — In progress: consolidate .gitignore, preserve received notebooks and
replace active entry points with clean generated copies (empty reviews, single
mode). Remove generated output/ and outputs/ from the Git index only; keep their
local files and historical Git objects. This is repository hygiene, not permanent
deletion. Existing root legacy modules/notebook stay because regression tests and
historical tooling reference them. Complete deletion candidates will be listed
in docs/AUDIT_AND_MIGRATION.md for separate user review.

### L audit findings and decisions

| Issue / date | Symptoms and evidence | Cause / action | Verification and status |
|---|---|---|---|
| AUDIT-01 / 2026-10-21 | New files supersede the earlier single-only exports | Audit configuration, execution IDs, raw predictions, source/artifact hashes and reviews | Seven local exports, 237 predictions; all masks/metrics rechecked. VERIFIED. Details in BENCHMARK_REPORT and docs/audit/latest_benchmark.json. |
| RUN-02 / 2026-10-21 | OCR folder 072049931932Z was thought to be pilot | Its config/manifest says full; all models blocked before new inference. Reviews were added later. Actual pilot is 060848155665Z | All six modes plus extra blocked full accounted for. Original paths supplied by user and verified. |
| MODEL-01 / 2026-10-21 update | Qwen3 invalid a10_face_complete IDs | Saved corrective model response supplies a9_face_complete, preserving geometry | GPU success on first image confirmed; full export 48 valid, 1 partial, 1 token-limit error. Not a universal quality fix. |
| MODEL-02 / 2026-10-21 update | Qwen2.5 empty pilot, ten invalid full predictions | Seven non-schema JSON errors, two all-entry schema failures, one partial; many valid outputs still miss PII | Parsing evidence VERIFIED; detection/schema reliability unresolved. No speculative new GPU fix. |
| MODEL-03 / 2026-10-21 update | Llama single/pilot fail before inference | HTTP 401 at pinned gated chat_template.json download | Confirmed authorization blocker; full gated. Account access and subsequent model/GPU quality NOT TESTED. Excluded from training candidate selection. |
| MODEL-04 / 2026-10-21 update | Florence only 2/4 usable pilot outputs | Card image repeats to 1023 tokens; skydiver image emits zero-width quad. Other two decode text but no PII matches | Native parsing correctly rejects two failures; eager loader works. FP32/crop quality remains NOT TESTED. Retain for single-image diagnostics, not training. |
| MODEL-05/06 / 2026-10-21 update | Chandra/dots each process 50 but only 46 valid | Four truncated native outputs each; separate OCR/rule detection misses | Loading/inference VERIFIED on L4; nine nonempty masks each. Very low label recall, no privacy completeness proof. |
| REVIEW-01 / 2026-10-21 | Review notes mention OCR faces/demographics and unverified privacy expectations | Reviews are diagnostic acceptances, not ground truth or permission to infer sensitive traits | Copied reviews for both Qwens and Chandra/dots match four hashes each. Scope caveats documented; no notes silently rewritten. |
| HANDOFF-02 / 2026-10-21 | Generated artifacts were already tracked despite ignore rules | Untracked output/, outputs/ and two previously absent result ZIPs; preserved local files. Consolidated ignore rules and cleaned notebooks after backup | No permanent filesystem deletion/history rewrite. Exact candidates and inventory in AUDIT_AND_MIGRATION. |
| TRAIN-01 / 2026-10-21 | Need one evidence-based candidate | Qwen3 leads on same 34 valid images; full F1 0.1969 vs 0.0264/0.0517/0.0517; low recall remains | Recommend Qwen3 for first planned adaptation. Training config stays disabled; data/gradient/license gates pending. |

The received VLM/OCR notebooks and legacy notebook are copied in ignored
`docs/handoff_20261021/notebooks_received/`; their saved logs report Drive read-back
of 943 full VLM and 970 full OCR files, plus the 103-file dataset. This is saved
cloud evidence, not a new live upload/server check. All imported run trackers gain
only this audit's appendices after original copies; predictions and source snapshots
remain unchanged. Earlier `results.zip`/`results_ocr.zip` were already absent.

Documentation responsibilities are separated: README overview; GUIDE practical
setup/reproduction; BENCHMARK_REPORT measured results; TRACKER history. The old
BENCHMARK_README and recovery reports are explicitly historical. Training details
remain in the existing TRAINING_READINESS document. No competing guide/report added.

Next verification: clean source-copy build without private data, complete offline
suite, notebook snapshot/credential-pattern/link checks, final preservation hashes.
Known limits: live Colab setup, external downloads, CUDA diagnostics, checkpoint
training/reload, complete privacy coverage and independent cloud durability cannot
be established by offline tests.

### L final verification

- `python scripts/check_handoff.py --clean-copy`: source-only copy with no data or
  saved results built all notebooks; optional absent dataset ZIP was correctly
  skipped. 131 offline tests passed in 74.510 seconds using the existing Python
  and temporary widget dependencies. Source links, clean outputs, exact embedded
  snapshots and credential-pattern checks passed.
- Created a fresh project-local Python 3.10.11 venv and installed
  `requirements-dev.txt`. The first sandbox installation failed DNS; a permitted
  network retry completed. A premature test while pip was still installing failed
  imports; this was not a dependency incompatibility. After completion,
  `python -m pip check` passed and all **131 tests passed in 77.488 seconds** in the
  source-only copy using that fresh environment. Logs and exact package freeze are
  preserved under docs/handoff_20261021; no global packages were changed.
- `python scripts/audit_handoff.py`: all seven exports, 237 records including reuse,
  22/22 source hashes per export, input/artifact hashes and pixel validation checked.
  Saved full-taxonomy and shared-text metrics match recomputation. Per-image ordering
  and approximately 1e-17 floating summation differences were normalized for comparison;
  no meaningful metric discrepancy found. Aggregate public audit omits OCR text/images.
- Preservation verification: all 2,694 imported files matched their initial hashes,
  with seven intentionally appended trackers checked against preserved originals
  and their new hashes. The received executed notebook hashes also match. No
  model output/source snapshot was changed and no permanent filesystem deletion occurred.
- Fixed a source-only builder defect found during handoff preparation: absence of
  local data previously produced an empty optional dataset ZIP. Now the builder
  explicitly skips it. Added the four handoff documents, development requirements,
  audit/check tools and public evidence to distribution; private artifacts remain excluded.
- A final Windows notebook rebuild raised transient OSError 22; the immediate
  retry succeeded. Final source snapshot checks are repeated after that rebuild.
- Narrowed the older audit_saved_runs script to its two historical run IDs so it
  cannot apply the obsolete single-only reconstruction to new execution-aware exports.
- Required GitHub files are prepared locally. Generated output removals are staged
  index-only changes; source/document edits remain working-tree changes. No commit,
  push or history rewrite was performed. Existing history is not scrubbed by ignore rules.

Remaining: approve any permanent cleanup separately; obtain independent region/text
annotations and duplicate-disjoint splits; test Qwen3 training gradients/checkpoint
reload before training. Diagnose Florence on one image and resolve Llama access only
if retaining them in future comparisons. Current outcomes do not justify rerunning
all completed models or claiming complete privacy removal.


