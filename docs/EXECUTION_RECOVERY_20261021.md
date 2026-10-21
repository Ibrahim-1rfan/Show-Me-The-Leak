> Historical snapshot. Current results: [BENCHMARK_REPORT](../BENCHMARK_REPORT.md); current usage: [GUIDE](../GUIDE.md). Later runs supersede model/coverage statements below.

# What happened, what changed, and how to use the repaired workflow

## What the six requests actually did

The full benchmark has **not completed**. There are two exported artifact runs,
each containing three recorded requests, rather than six independently completed
benchmarks. All observed image inference was on the first document,
`documents/document_1.jpg`, original VISPR ID `2017_12973862`.

| Notebook / request | Intended images per model | New image inference attempts | What happened |
|---|---:|---:|---|
| VLM single | 1 | 2 of 3 model/image pairs | Both Qwens ran; Llama could not download |
| VLM pilot | 4 | 0 of 12 | Single-image regression gates blocked every model |
| VLM full | 50 | 0 of 150 | Pilot/review gates blocked every model |
| OCR single | 1 | 3 of 3 | All loaded and generated output, but none passed redaction checks |
| OCR pilot | 4 | 0 of 12 | Single-image regression gates blocked every model |
| OCR full | 50 | 0 of 150 | Pilot/review gates blocked every model |

VLM artifact run: `vlm_20261020T234601512294/20261020T234601512294Z`.
OCR artifact run: `ocr_20261021T011930055215Z/20261021T011930055215Z`.
The fixed 50-image cohort is unchanged: 25 document-category and 25 general images.
“Document” describes the VISPR content category; it does not necessarily mean a
clean scanned page. The first document is a photograph of several cards and tickets.

The mode was passed to execution correctly. Worker selection already distinguished
1, 4 and 50 images. The blocking happened **before the worker started**. The previous
configuration cell resumed the same protocol folder for each request; reports then
read the same saved single-image predictions. Gate exceptions were caught so other
models could proceed, but every model was blocked, and the notebook did not explain
the zero new workload clearly. Final per-model failure files overwrote earlier gate
errors; separate pilot/full start timestamps were not preserved. The embedded
trackers, final gate errors, saved source, single-only jobs/timings and predictions
support the reconstruction. The saved OCR notebook confirms the full call at
execution 21 and its configuration at execution 20. The VLM notebook's execution
counters are cleared, but its saved outputs confirm the full request resuming the
same VLM run at 00:27:54 UTC. Cleared counters do not mean no execution occurred.

All 18 recorded source hashes in each export match the saved source snapshot.
Every artifact hash listed in the five saved predictions matches its local file.
Existing older `results` / `results_ocr` exports remain available and represent
different October 19 runs; they must not be mistaken for the latest execution.

## Model findings, with stage and evidence

| Model | Latest stage/outcome | Generation time on the saved image | Remaining limitation |
|---|---|---:|---|
| Qwen3-VL | Partial parsing: three valid detections; two invalid `a10_face_complete` IDs rejected | 125.03 s, T4 | Three regions masked correctly, visible PII remains; no complete prediction |
| Qwen2.5-VL | Inference completed; genuine empty JSON, 11 tokens, normal EOS | 23.70 s, T4 | Zero detections, unchanged output; cause of missed content unresolved |
| Llama 3.2 Vision | Model download: gated repository HTTP 401 | Not measured | Authorized HF account/access needed; never reached image inference |
| Florence-2 | Generation/decoding: repetitive output at 1,023-token limit, zero spatial regions | 15.21 s, L4 | No usable OCR or redaction; not a new FlashAttention installation failure |
| Chandra | Layout inference completed: six Image descriptions, zero OCR text regions | 27.41 s, L4 | Descriptions are not literal OCR; no text enters PII rules, no masking |
| dots.ocr | Layout inference completed: six Picture blocks without text | 9.02 s, L4 | No text enters PII rules, no masking; earlier processor import failure is resolved in this export |

Chandra's older HTTP 429 and dots' older processor import failure are not the
current failures. The latest logs show both models running on CUDA. The OCR parsers
deliberately exclude picture descriptions and layout coordinates from literal text
rules; treating those descriptions as verified text would misstate this benchmark.
No replacement model was silently substituted. No observed latest error establishes
GPU memory exhaustion: the five loaded models fit the supplied GPUs. Llama access
can be addressed by accepting its license and providing an authorized `HF_TOKEN`;
otherwise select the public models explicitly and report the smaller comparison.

Qwen3 now has an explicit, bounded invalid-ID correction option,
`repair_partial_entries=True`, within the existing format-retry budget. Only the
rejected IDs are sent back to the model for correction. Accepted regions are
retained, and a correction must preserve every rejected box and evidence field.
Empty, truncated or changed-geometry corrections leave the result partial. The
original response, rejected entries and correction response are retained; no
ground-truth labels or automatic ID remapping are used. Offline tests verify this
contract, but whether Qwen supplies correct IDs on the next GPU run is unverified.

The T4 VLM exports used BF16. PyTorch's support check includes emulation by default,
so the loader now selects native FP16 on T4-class devices and BF16 on Ampere or
newer. This is a hardware-selection fix, **not proof of better recall or speed**.
[PyTorch API](https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html).

Florence's failure remains under investigation. Its native task and coordinate
decoder are being used; it does not need JSON input/output. Raising the token budget
beyond the decoder position bound is not a valid repair. The optional FP32 diagnostic
now loads FP32 weights directly, instead of first loading FP16 and casting them back.
The default remains the documented FP16 profile. A new GPU single-image FP32 run,
followed separately by native-size/tile diagnostics, is still needed.
[Microsoft model card](https://huggingface.co/microsoft/Florence-2-large-ft).

For dots and Chandra, use the four-image pilot to test both categories before deciding
they cannot recognize any dataset text. The existing `ocr_preprocessing='tiles'`
option is a separate, recorded crop experiment, not an automatic quality fix.
The original dots layout prompt explicitly omits text from Picture blocks.
[Original pinned model](https://huggingface.co/rednote-hilab/dots.ocr/tree/c0111ce6bc07803dbc267932ffef0ae3a51dc951).

## Redaction verification and fair comparison

The saved images were visually inspected together and their masks reapplied to the
original pixel dimensions. Qwen3's three masks cover 1,065,906 pixels, with 1,065,876
changed pixels (some original pixels were already black). They cover the predicted
card/ticket regions but leave the upper-left photo ID visible. Qwen2.5, Chandra and
dots each have zero mask pixels and zero changed pixels. Florence has no redacted
image. Llama has no image output to inspect. No model has verified complete privacy
removal on this saved example.

Valid empty predictions still receive label precision/recall/F1 of zero on this
positive image; they are not successful redactions. Qwen3's retained three labels
are diagnostic evidence, excluded from complete-output quality aggregates because
the response is partial. Failed and unattempted predictions are not converted to
empty predictions. Rankings require all 50 valid outputs on the same image cohort
and passing redaction checks. T4 and L4 timings above are descriptive, not a fair
hardware-controlled speed ranking.

VISPR JSON supplies image-level attributes, not complete reference boxes or exact
text transcripts. “Missed labels” is measurable; a count of missed physical regions,
unnecessary masking area, true privacy coverage, IoU, CER/WER or ROUGE-L is not
established without independent annotations. Full-taxonomy and shared text-rule
scope tables remain separate. Sensitive characteristics are not inferred from
appearance to improve agreement with annotation labels.

## Workflow repairs

- Every execution from Section 6 creates a fresh run directory and an execution
  record containing the chosen mode, settings, intended IDs, model states, timing
  and final reconciliation. Repeated use of the legacy execution helper also keeps
  separate execution records.
  Folder names are UTC timestamps, not image IDs. The protocol fingerprint is a
  separate value used to check compatible settings. Saved per-image timings include
  reused evidence; execution wall time and reused-image counts are reported separately.
- `processing_manifest.json` contains one row per intended model/image pair,
  including pending, processing, success, failed, skipped and redaction_failed
  states. It separates current inference attempts, generated outputs, valid
  predictions, saved redactions, mask checks, missing files and reused evidence.
- Single reports cover one intended image per model; pilot reports cover four;
  full reports cover 50. No result from a different workload is silently counted.
- The four-image pilot is now independent of single-image detection recall.
  Full execution still needs four intact predictions, correct application of the
  predicted masks, and explicit review bound to prediction hashes. A reviewed
  detection miss can be benchmarked as a miss; it does not become a successful
  privacy redaction. Malformed outputs or broken mask application still block full.
- Exact-protocol, input-hash and artifact-verified completed predictions may be
  copied into a new run, with their origin recorded. Model selection can be reduced
  without invalidating unchanged reviewed model settings. Changing preprocessing,
  model pins or implementation invalidates reuse. Historical exports are preserved.
- Raw output is saved before coordinate conversion/redaction. Per-image failures
  retain their stage and evidence and allow the next image to proceed. Global
  initialization failures and timeouts remain explicit model errors with unprocessed
  images accounted for. A reporting failure cannot prevent backup attempts.
- A new browser replaces Section 8's fixed `INSPECT_INDEX=0`. It reads selected run
  files directly, including missing outputs, rather than a hardcoded first report row.

## Using the corrected notebooks

Open the existing `notebooks/01_vlm_benchmark.ipynb` or
`notebooks/02_ocr_benchmark.ipynb` in Colab. They are updated in place.

1. Run Sections 1–4 for the bundled helpers, notebook dependencies, existing Drive
   mount and validated image mapping. L4/A100 is needed for the current dots profile.
2. Section 5 displays the configuration and a live **single / pilot / full** selector.
   Choose the models explicitly in `CONFIG['models']`. Changing a setting requires
   rerunning that configuration cell; changing the live mode selector does not.
3. Run Section 6. It prints the mode and new folder before processing, then shows
   actual counts and final status. `REUSE_PREDICTIONS=True` avoids repeated intact
   work; `False` deliberately reruns selected images in another new folder.
4. Run Section 8. Choose **Run → Model → Type → Image**. Use **Previous**, **Next**
   or **Jump to #**. Original, predicted-region and redacted panels appear together
   or wrap on smaller screens. Tabs show structured detections/errors, raw output
   and the saved mask. Missing artifacts and previews are labelled explicitly.
5. `document_12` is an index in the fixed document selection; it is not the original
   dataset ID. Every selector entry shows its filename and original VISPR ID.
   All 50 source images remain browsable; images outside the selected execution
   say **Not requested**. A dataset preview is never labelled a processed original.
6. Inspect all four pilot images before adding image-specific notes in the review
   cell. Then select **full** and execute Section 6. Models still failing checks
   stay blocked and are reported as such. Do not fill in reviews blindly.
7. Backup still uses the same Drive root. The final download cell exports the whole
   run, preserving relative image/report links.

To inspect old results without any model inference, run Sections 1–4 and Section 8.
The viewer also searches the mounted Drive result folders. If no run appears, check
the Drive mount or restore the saved run into `results/vlm/<id>` or `results/ocr/<id>`.
No model has to be loaded to use the viewer.

## Preservation, cloud status and remaining work

`docs/recovery_20261021/` contains the initial preservation inventory, the copied
executed notebooks, six reconstructed manifests, source/artifact checks, pixel
validation, a contact sheet and saved notebook cloud/gate events. Existing archives,
results and source snapshots were not deleted. Tracker updates are appended, with
their original versions separately preserved.

The existing cloud destination is **Google Drive mounted by Colab**, not a newly
created Google Cloud Storage bucket:
`/content/drive/MyDrive/Leak_Benchmark_Results/benchmark_v2`.
The saved notebooks record read-back verification of 84 VLM files (00:28:05 UTC),
93 OCR files (01:36:34 UTC), and 103 dataset files on October 21. The connected Drive search
in this environment returned no matching accessible folder; this does not prove
the folder is absent. No new cloud upload or server-side verification is claimed.

Local regression tests cover the changed selection, manifests, reuse, gates,
per-image recovery, reporting and actual widget controls. The exact test results
and any remaining failures are recorded in `TRACKER.md`. No local CUDA runtime is
available, so new model-quality tests and a full GPU benchmark remain pending.
Next: inspect the existing outputs in the viewer, run the new four-image pilot,
isolate Florence's FP32 diagnostic, resolve Llama access, and review the pilot before
any full benchmark. Training remains disabled.
