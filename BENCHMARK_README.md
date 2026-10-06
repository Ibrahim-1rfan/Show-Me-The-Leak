# Running the six-model PII benchmark

Open `Model_Testing.ipynb`. Keep its four `benchmark_*.py` helpers and
`taxonomy_crosswalk.csv` in the same project folder as `train2017/`.
The images and per-image JSON annotations are not included in the code bundle.

1. Use a GPU notebook, mount your persistent dataset folder if using Colab, and
   set `DATA_ROOT` to that folder. Set `CODE_ROOT` separately if the helper files
   are stored elsewhere.
2. Run the notebook with `INSTALL_PROFILE = 'vlm'`. It runs Qwen3-VL 8B,
   Qwen2.5-VL 7B, and Llama 3.2 Vision 11B sequentially. Llama requires accepted
   model terms and an authorized Hugging Face login.
3. In fresh runtimes, repeat with `INSTALL_PROFILE` set to `'chandra'`, `'dots'`,
   and `'florence'`. Retain the same dataset folder and benchmark settings.
   These profiles preserve the original requested checkpoints and their different
   library requirements. Dots requires a supported Ampere-or-newer GPU, such as
   an A100 or L4, and a working CUDA toolchain for FlashAttention; use its
   unquantized profile. The notebook records setup failures rather than scores.
4. Run the comparison and inspection sections. They load actual saved outputs
   across profiles, with no need to keep all models in memory.

The default is 50 fixed images: 25 labeled with a VISPR Documents-group attribute
and 25 other images. All six models receive both groups. Larger samples are
supported; the notebook refuses samples smaller than 50.

## What the comparison measures

- The three general VLMs predict privacy attribute IDs and bounding boxes.
- Chandra and dots use native document parsing; Florence uses its native
  `<OCR_WITH_REGION>` task and postprocessor. A shared conservative text-rule
  mapper then turns recognized text into privacy attribute detections.
- Full-taxonomy and shared-text-label scores are reported separately, with
  overall/document/general slices, precision, recall, F1, exact match, label-set
  overlap, per-attribute/category misses, and inference coverage.
- The main paired comparison uses images successfully processed by every
  selected model. Per-model diagnostic tables also include its other successes.
  Check coverage before interpreting rankings.
- Native outputs, parsing warnings, model/version metadata, redacted images,
  and residual audits remain inspectable for each image.

OCR-pipeline scores reflect both recognition and the rule mapper. An omitted
label does not establish what a model considers private. A detection absent
from the supplied JSON is an unannotated prediction, not necessarily a true
false positive. Image-level labels cannot count missed individual instances.

## Optional IoU and OCR text metrics

The supplied JSON files have image-level labels only. To enable additional
metrics, copy the generated `reference_template.json` to `references.json`
inside the sample directory and add independent, manually checked references:

- Original upright-image pixel boxes, their attribute IDs, a declared
  `box_attribute_scope`, and `boxes_complete=true` only when every instance of
  those attributes is annotated. This enables same-label one-to-one box matching,
  localization F1 at IoU >= 0.5, and mean IoU among matched boxes.
- Exact visible-text transcripts in `reference_text`. This enables ROUGE-L,
  character error rate, and word error rate for the OCR outputs. The general
  VLM PII task does not emit full transcripts, so its text metrics remain unavailable.

Leave unavailable references `null`; empty lists/strings mean confirmed empty.
References enter evaluation only and are never provided to models.

## Saved outputs and verification

Results live under `outputs/multimodel_pii/sample_<size>_<fingerprint>/`.
Every model has a separate timestamped run directory. Comparison CSV/JSON files
record the exact source runs and evaluation configuration. Never mix settings
or alter the rule mapper between models in the same comparison.

Offline tests validate parsing, redaction, bookkeeping, notebook wiring, and
metric arithmetic. Run them with `python -m unittest discover -s tests -v`
in an environment with Pillow and pandas. The notebook's model integration has
not been GPU-executed during authoring. No model scores or baseline results
have been fabricated; the previous Qwen-only notebook was not run.
