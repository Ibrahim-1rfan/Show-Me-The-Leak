# Benchmark results and model comparison

Audited 21 October 2026. This is the current consolidated results report.
[GUIDE](GUIDE.md) covers reproduction; [TRACKER](TRACKER.md) records development.
Earlier reports in `docs/` and `outputs/benchmark_review/` describe older exports
and must not be read as the current benchmark.

## Evidence and methodology

Fixed VISPR training-split diagnostic cohort: 50 images, 25 document-category and
25 general images. Document category includes photographs of IDs/cards/tickets,
not just scans. [Selection](configs/selection.json) fingerprint:
`0beae8a380049243b74c8c9e2605ef3d01ecbdfcdfaa14f99dc74e8a3cb50b72`.
All 68 taxonomy entries are supplied; `a0_safe` is excluded from positive metrics,
leaving 67 positive labels. This reviewed cohort is not a held-out generalization test.

Both approaches use original bytes, upright coordinates, 1280-pixel edge cap and
native processor resizing. Baseline OCR preprocessing is `whole`, not tiling.
Requested generation budget is 4096; Florence is capped at 1023. Batch size is one;
one format retry is allowed, including bounded invalid-ID correction. No hidden
redacted-image inference. Model revisions are in [models.json](configs/models.json).
The seven downloaded exports have matching 22/22 saved source hashes each.

VLMs predict labels and regions directly. OCR systems recognize native text/layout
and then apply fixed English PII rules. They cannot be judged as standalone face
or demographic detectors. Sensitive traits require explicit text evidence, so
missed image annotations can also reflect the project's intentionally narrower policy.

**Verified locally:** seven artifact folders, 237 prediction records (including reused copies); all listed
artifact hashes and input identities match. Every available saved redaction was
reapplied and pixel-checked, matching its recorded validation. Saved metrics were
recomputed; row ordering and floating-point roundoff are not metric discrepancies.
Machine-readable evidence: [latest audit](docs/audit/latest_benchmark.json), generated
by [audit_handoff.py](scripts/audit_handoff.py). Private images/reviews remain outside Git.

**Evidence coverage:** all six requested modes now have local exports, plus an
additional blocked OCR full attempt. The two initially missing paths were supplied
during this audit; their actual configurations, rather than folder naming assumptions,
determine the mode. Historical intermediate OCR attempts are tracker-only evidence.

## Six requested modes

All times/IDs below are UTC on 2026-10-21. Intended counts are per model.

| Workflow / mode | Artifact run ID | Intended | Evidence and actual workload |
|---|---|---:|---|
| VLM single | `20261021T055302824843Z` | 1 | Local export: both Qwens inferred; Llama download failed |
| VLM pilot | `20261021T060533341923Z` | 4 | Local export: each Qwen added 3 inferences and reused 1; Llama download failed |
| VLM full | `20261021T063515607538Z` | 50 | Local export: each Qwen added 46 and reused 4; Llama blocked |
| OCR single | `20261021T055306013536Z` | 1 | Local export: all 3 inferred; Florence output invalid; other two empty |
| OCR pilot | `20261021T060848155665Z` | 4 | Local export: Florence attempted 4 (2 valid); Chandra/dots each added 3 and reused 1 (4 valid) |
| OCR full | `20261021T073101588696Z` | 50 | Local export: Chandra/dots each added 46 and reused 4; Florence added 0, reused 2, left 48 unprocessed |

Additional downloaded OCR run `20261021T072049931932Z` is **full**, not pilot.
It reused 2/4/4 predictions for Florence/Chandra/dots but added zero inference;
Florence lacked valid pilot outputs and the other two lacked recorded review at
execution time. Reviews were added afterward; the next full run could proceed.
Earlier full requests `064241946057Z`, `064335916709Z`, `064538216954Z` on the same
date are mentioned in imported trackers, but their folders are not supplied here.

These are genuinely different workloads. Identical pilot examples in full results
are intentional, recorded reuse. Unlike the earlier exports audited in Section K
of the tracker, full inference now occurred for four models. Neither category is
a complete valid comparison across all three requested models; no winner is certified.

## Model outcomes

A valid prediction can be empty. Technical checks do not certify all PII is hidden.

| Model | Most extensive evidence | Valid predictions | Invalid / missing | Technical redaction checks passed | Median detection seconds |
|---|---|---:|---|---:|---:|
| Qwen3-VL 8B | Full export | 48/50 | 1 partial + 1 error | 25/50 | 4.88 |
| Qwen2.5-VL 7B | Full export | 40/50 | 1 partial + 9 errors | 5/50 | 1.91 |
| Llama 3.2 Vision 11B | Single/pilot download failures; full blocked | 0/50 | All full images unprocessed | 0/50 | Unavailable |
| Florence-2-large-ft | Full export, copied pilot evidence only | 2/50 | 48 unprocessed; first-image single also failed | 0/50 | 1.18 on only 2 copied images |
| Chandra 9B | Full export | 46/50 | 1 partial + 3 errors (8% invalid) | 12/50 | 24.12 |
| dots.ocr 3B | Full export | 46/50 | 4 errors (8% invalid) | 13/50 | 14.75 |

Qwen3 has 49 redacted files (26 nonempty masks), Qwen2.5 has 41 (6 nonempty),
including partial outputs. Chandra has 47 redacted files including one partial, dots 46; both have **nine
nonempty masks**, and all saved files match the masks they intended to apply.
Technical-pass counts can exceed nonempty masks because OCR's in-scope labels
exclude many visual attributes. Neither count measures complete privacy removal.
All audited `privacy_verified` counts are zero because reference region truth is absent.

Saved medians include copied pilot timings and exclude model load/setup. Chandra
median whole-image time is 29.70 s (p95 275.37); dots is 22.02 s (p95 163.40).
Reported peak allocated GPU memory is approximately 6.72 GiB / 6.04 GiB respectively,
not total GPU capacity or a training requirement. Current inspected metadata uses
L4; do not compare these with older T4/A100 runs as a controlled speedup experiment.

## Detection scores and slices

The following full-taxonomy scores are **conditional on valid outputs**. Failed
and partial predictions are excluded, not turned into empty correct predictions.
Each model's own valid subset differs; these rows are descriptive, not a ranking.

| Model / mode | Valid n | Micro precision | Micro recall | Micro F1 | Matched / missed / extra labels |
|---|---:|---:|---:|---:|---|
| Qwen3 single | 1 | 1.0000 | 0.2667 | 0.4211 | 4 / 11 / 0 |
| Qwen3 pilot | 4 | 1.0000 | 0.1000 | 0.1818 | 4 / 36 / 0 |
| Qwen2.5 single / pilot | 1 / 4 | 0 | 0 | 0 | 0 / 15 or 40 / 0 |
| Chandra single | 1 | 0 | 0 | 0 | 0 / 15 / 0 |
| dots single | 1 | 0 | 0 | 0 | 0 / 15 / 0 |
| Florence copied pilot subset in full | 2 | 0 | 0 | 0 | 0 / 17 / 0 |
| Qwen3 full valid subset | 48 | 0.6719 | 0.1348 | 0.2245 | 43 / 276 / 21 |
| Qwen2.5 full valid subset | 40 | 0.8000 | 0.0160 | 0.0314 | 4 / 246 / 1 |
| Chandra/dots pilot (each) | 4 | 0 | 0 | 0 | 0 / 40 / 0 |
| Chandra full valid subset | 46 | 0.6667 | 0.0262 | 0.0505 | 8 / 297 / 4 |
| dots full valid subset | 46 | 0.4375 | 0.0222 | 0.0423 | 7 / 308 / 9 |

All displayed local-run scores were independently recomputed. Florence single has
no valid score; Florence pilot has 2 valid/2 failed outputs and F1 zero on the valid
pair. Llama has no current quality score. Missing-label counts are
image/attribute occurrences, not physical regions; extra labels are unannotated
predictions, not confirmed false positives.

A **paired four-model comparison** excludes failed/partial outputs and the blocked
Llama/Florence models explicitly. The intersection contains 34 images: 14 documents
and 20 general. This avoids comparing scores on different successful subsets, but
still has survivorship bias and is not a complete six-model benchmark.

| Model | Full-taxonomy F1 | Document F1 | General F1 | Shared-text F1 |
|---|---:|---:|---:|---:|
| Qwen3 | 0.1969 | 0.4304 | 0.0914 | 0.3797 |
| Qwen2.5 | 0.0264 | 0.0645 | 0.0121 | 0.0625 |
| Chandra | 0.0517 | 0.1791 | 0 | 0.1714 |
| dots | 0.0517 | 0.1791 | 0 | 0.1714 |

Paired Qwen-only comparison: 39 valid common images, F1 0.1915 vs 0.0317.
An additional **paired OCR diagnostic** uses the same 44 valid images for Chandra
and dots (20 documents, 24 general), explicitly excluding Florence from that panel.
Full-taxonomy micro F1 is 0.0516 vs 0.0447; document F1 is 0.1270 vs 0.1085; general
F1 is zero for both. This is not a six-model winner and excludes failure cases.
The original three-model common-successful table contains only two images because
Florence has only two usable copied records; zero scores there do not summarize 50 images.

On each OCR model's own 46-image valid subset, shared-text-scope F1 is 0.1468 for
Chandra and 0.1186 for dots. Document-only shared-text F1 is 0.1975 and 0.1556;
general is zero for both. OCR plus rules is still missing most annotated text labels.
Examples of missed in-scope text labels on each model's own valid subset:
Chandra misses full names in 8/9 annotated images, date/time in 14/16, tickets in
11/12 and partial addresses in 8/8; dots misses 7/8, 15/17, 11/12 and 9/9 respectively.
These are label misses, not proof of which physical text spans were missed. Many
VLM full-taxonomy misses are appearance-based sensitive annotations; policy excludes
such inference without explicit text, so they are not automatically training targets.

Region IoU, missed-region counts, unnecessary masking area, true redaction coverage,
CER/WER and ROUGE-L remain unavailable without independent boxes/transcripts.

## Model reviews and failure analysis

**Qwen3:** the saved first-image response used invalid `a10_face_complete` twice.
The bounded correction returned `a9_face_complete`, preserving boxes and evidence;
five detections across four unique labels survived. This is GPU evidence that the
correction worked on this image. Its photo-ID and card/ticket masks now appear in
the actual redaction. The remaining three pilot images returned empty predictions.
The full export has one token-limit error and one partial validation result.
Its 48/50 valid coverage and paired scores support first-candidate status, while
low recall and annotation limitations prevent a production privacy claim.

**Qwen2.5:** all four local pilot predictions are empty, with unchanged images.
The full export has seven non-schema JSON failures, two records where every entry
violates the four-field schema, and one partial record. This is output compliance
as well as detection failure; artifacts distinguish these stages. Faster generation is not evidence of useful privacy detection.

**Llama:** exact checkpoint `meta-llama/Llama-3.2-11B-Vision-Instruct`, pinned revision
in models.json. Single and pilot fail with HTTP 401 fetching `chat_template.json`
at the model-download stage. No initialization/inference happened. Full is then
blocked by absent valid pilot predictions. This is an access failure; memory,
processor compatibility and current output quality were not tested. Retain as a
blocked comparison model, exclude from training selection, authorize access then
try one image. No model substitution or package reinstall is a verified fix.

**Florence:** loads on L4 using eager attention and native `<OCR_WITH_REGION>`.
The first image generates repetitive text for all 1023 tokens, forces EOS at the
bound, and returns zero regions. The parser correctly rejects incomplete output;
no final redacted image is saved. This precedes coordinate conversion/redaction.
The first general pilot image emits a zero-width quadrilateral (all four x values
coincide), correctly rejected despite normal EOS after 13 tokens. Two other pilot
images yield 12 and 1 text regions with normal EOS (132 and 26 tokens),
proving the loader/decoder can work on some inputs. PII rules match neither image.
Full is blocked by incomplete pilot coverage. FP32 loading is unit-tested, but no
FP32/crop quality experiment is supplied. Retain for a single-image diagnostic,
exclude from training; the root cause of repetition remains under investigation.
The [native model interface](https://huggingface.co/microsoft/Florence-2-large-ft)
does not require the model to emit the benchmark's JSON schema.

**Chandra / dots:** current GPU loading works, unlike older download/processor
failures. Four outputs each reach truncation/incomplete parsing, which explains
the invalid records. Many successful OCR outputs still yield no PII rule matches.
Improving text recognition, text-to-PII rules and visual PII detection are separate
experiments; fine-tuning the OCR model is not yet justified by one combined score.

Reviews are present in the received notebooks for both Qwens and both OCR models;
Chandra/dots copied review files match all four saved pilot prediction hashes.
No passing review file exists for Llama or Florence. The full VLM export also contains two review files whose four prediction hashes
match for each Qwen. Notes about OCR missing faces or
models missing demographics exceed OCR scope or conflict with the explicit-text
policy; treat those as reviewer expectations, not established adapter defects.

Representative inspected examples: VISPR `2017_12973862` (Qwen card/photo-ID scene),
`2017_36855676` (receipt) and `2017_40830297` (ticket). OCR masks cover some text but
leave other identifying fields visible; some boxes cover broad irrelevant content.
These are qualitative observations, not reference-box scores. Private comparison
images are in `docs/handoff_20261021/representative_contact.png`; original/predicted/
redacted files remain in each export's model/category/image folder. Restore the
export into the GUIDE's results layout to browse them in Section 8.

## Recommendation and next experiment

**Recommended first candidate: Qwen3-VL-8B-Instruct. Training remains disabled**
until annotation, split, license and GPU-training checks pass. This choice is based
on verified 48/50 valid full outputs (versus Qwen2.5 40/50), stronger paired F1 on
34 common images, and useful localized detections in both image categories.
It is slower than Qwen2.5 but detects substantially more annotated labels. OCR
systems score below Qwen3 even in the shared text scope. These results support
prioritization, not a claim that fine-tuning will solve the remaining misses.

The [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) declares
Apache 2.0; review model and data terms before adaptation. Do not train on the
50 repeatedly inspected regression images or their duplicates. Collect independently
reviewed boxes/text, define development/validation/unseen test splits, preserve
baseline metrics and conduct a small forward/backward and checkpoint-reload test.
First distinguish schema/prompt failures from annotation/policy differences; no
complete benchmark rerun is needed to audit existing predictions.

Florence needs a controlled single-image FP32/native-size/tile diagnostic and Llama
needs account authorization. Neither is a training candidate now. The
[training-readiness plan](docs/TRAINING_READINESS.md) covers the proposed LoRA/QLoRA
method, memory/storage measurement and checkpoint policy. No training has started.

## Storage and backup evidence

Downloaded exports live locally under `vlm_<run_id>/<run_id>`,
`ocr_<run_id>/<run_id>` and direct `<run_id>/` folders; their exact paths are in the public audit summary.
They are intentionally absent from a GitHub source clone.
Saved notebook logs report final mounted-Drive read-back verification of **943 VLM
full-run files at 07:17:59 UTC** and **970 OCR full-run files at 08:50:33 UTC**;
the dataset has 103 files. Destination remains
`MyDrive/Leak_Benchmark_Results/benchmark_v2`. These are saved execution logs, not a
new live cloud verification. No outputs were deleted or overwritten by this audit.
