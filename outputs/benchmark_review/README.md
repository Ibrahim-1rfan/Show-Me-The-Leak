# Review of the saved benchmark run

These are measured results from the latest saved runs in `Leak_Benchmark_Results`, not a new GPU run.
The sample contains 50 images: 25 document and 25 general. Source runs and their original protocol
IDs are listed in `provenance.json`. No archived outputs were modified.

## Full-taxonomy results on each model's successful images

| Model | Scored images | Precision | Recall | F1 | Missed labels |
|---|---:|---:|---:|---:|---:|
| qwen3_vl | 50 | 0.642 | 0.125 | 0.210 | 237 |
| qwen25_vl | 45 | 0.778 | 0.031 | 0.060 | 219 |
| llama32_vision | 5 | 0.500 | 0.059 | 0.105 | 32 |
| chandra | 46 | 0.667 | 0.033 | 0.063 | 235 |
| dots_ocr | 0 | unavailable | unavailable | unavailable | unavailable |
| florence2 | 0 | unavailable | unavailable | unavailable | unavailable |

**Do not rank these rows directly:** their image coverage differs. Failures are excluded from
quality scores and counted separately. Llama has only five valid outputs, none on documents.
Dots and Florence never reached successful inference; they have no quality scores.

## Paired Qwen comparison: 45 identical successful images

| Model | Scored images | Precision | Recall | F1 | Missed labels |
|---|---:|---:|---:|---:|---:|
| qwen3_vl | 45 | 0.690 | 0.088 | 0.157 | 206 |
| qwen25_vl | 45 | 0.778 | 0.031 | 0.060 | 219 |

## Paired Qwen/Chandra comparison: 43 identical successful images

| Model | Scored images | Precision | Recall | F1 | Missed labels |
|---|---:|---:|---:|---:|---:|
| qwen3_vl | 43 | 0.679 | 0.086 | 0.153 | 202 |
| qwen25_vl | 43 | 0.778 | 0.032 | 0.061 | 214 |
| chandra | 43 | 0.600 | 0.027 | 0.052 | 215 |

Chandra is an OCR-plus-rule pipeline. The general VLMs directly predict visual privacy labels;
the full-taxonomy scores include non-textual attributes that the OCR rules cannot detect.

## Shared text-label scope (per-model successful images)

| Model | Scored images | Precision | Recall | F1 | Missed labels |
|---|---:|---:|---:|---:|---:|
| qwen3_vl | 50 | 0.611 | 0.218 | 0.321 | 79 |
| qwen25_vl | 45 | 0.750 | 0.076 | 0.138 | 73 |
| llama32_vision | 5 | 0.000 | 0.000 | 0.000 | 5 |
| chandra | 46 | 0.667 | 0.091 | 0.160 | 80 |
| dots_ocr | 0 | unavailable | unavailable | unavailable | unavailable |
| florence2 | 0 | unavailable | unavailable | unavailable | unavailable |

## Document images (per-model successful images)

| Model | Scored images | Precision | Recall | F1 | Missed labels |
|---|---:|---:|---:|---:|---:|
| qwen3_vl | 25 | 0.651 | 0.200 | 0.306 | 112 |
| qwen25_vl | 21 | 0.750 | 0.062 | 0.114 | 91 |
| llama32_vision | 0 | unavailable | unavailable | unavailable | unavailable |
| chandra | 21 | 0.727 | 0.071 | 0.130 | 104 |
| dots_ocr | 0 | unavailable | unavailable | unavailable | unavailable |
| florence2 | 0 | unavailable | unavailable | unavailable | unavailable |

## General images (per-model successful images)

| Model | Scored images | Precision | Recall | F1 | Missed labels |
|---|---:|---:|---:|---:|---:|
| qwen3_vl | 25 | 0.600 | 0.046 | 0.085 | 125 |
| qwen25_vl | 24 | 1.000 | 0.008 | 0.015 | 128 |
| llama32_vision | 5 | 0.500 | 0.059 | 0.105 | 32 |
| chandra | 25 | 0.000 | 0.000 | 0.000 | 131 |
| dots_ocr | 0 | unavailable | unavailable | unavailable | unavailable |
| florence2 | 0 | unavailable | unavailable | unavailable | unavailable |

## Failure diagnosis and repair

- **Dots:** the recorded error says Transformers 4.57.1 was installed, while its adapter needs
  4.56.1. It stopped at setup for all 50 images. The saved L4 GPU is hardware-compatible.
- **Florence:** the first run found Transformers 4.57.1 instead of 4.41.2. Later attempts failed
  during imports after installing 4.41.2 with `--no-deps` in Python 3.13. The generic wrapper
  concealed the nested import error, so its exact message is not recoverable from these files.
  Transformers 4.41.2 requires tokenizers >=0.19,<0.20; installing the package alone is insufficient.
- **Llama:** 45/50 outputs failed validation. Representative outputs are natural-language image
  descriptions rather than the requested JSON. These are not legitimate empty PII predictions.
  The dependency repair does not claim to solve this instruction-following problem.
- **Qwen2.5:** five images failed schema/evidence validation. Qwen3 completed all 50 after its
  configured retries.
- **Chandra:** four images failed; saved metadata shows incomplete generation at the token cap.
  Raising its cap would require a separately disclosed protocol/run, not editing these scores.

The repaired notebook uses isolated Python 3.11 model workers and checks imports, versions, and
GPU availability before inference. It keeps complete dependency resolution and retains nested
import errors. The notebook can include older equal-setting runs explicitly, with code differences
disclosed in provenance, so only the failed profiles need to be retried.

## Interpretation limits

Precision is agreement with the supplied image-level labels, which may be incomplete. Missed
counts are image/attribute occurrences, not individual physical instances. Raw privacy values
are deliberately omitted from this report. Per-image/per-attribute omissions are in the CSVs.
There are no reference boxes or transcripts here, so IoU, ROUGE-L, CER, and WER remain unavailable.
No six-model paired ranking is possible until dots and Florence have valid outputs.

Sources: [Transformers 4.41.2 dependency requirements](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/dependency_versions_table.py),
[uv Python environment management](https://docs.astral.sh/uv/guides/install-python/).
