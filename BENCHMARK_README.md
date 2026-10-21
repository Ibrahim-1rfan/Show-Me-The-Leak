> Historical snapshot. Current results: [BENCHMARK_REPORT](BENCHMARK_REPORT.md); current usage: [GUIDE](GUIDE.md). Later runs supersede model/coverage statements below.

# Running the six-model PII benchmark

## October 10 loader fixes

- Dots' checkpoint processor drops the video processor required by Transformers
  4.56.1. The adapter now assembles the parent processor explicitly, retaining
  dots' image settings, native chat template and `<|imgpad|>` token. It checks
  image patch/token alignment before loading weights. Passing `video_processor`
  to the old `AutoProcessor` call alone does not fix the checkpoint constructor.
- Florence uses `eager` attention and **does not need flash-attn installed**.
  Its Transformers 4.41 dependency scanner mistakes a conditional import for a
  requirement. A temporary, Florence-only loader shim excludes that import only
  after checking its availability guard. Other dependencies remain checked;
  checkpoint source and weights are unchanged.

Upload the updated `Model_Testing.ipynb` to Colab and run its bootstrap cell to
refresh the embedded helpers. Rerun configuration, environment setup, helper
and execution cells. In an existing runtime whose selected worker environment
already installed successfully, set `INSTALL_PACKAGES=False` to skip downloads
and installation; imports are still checked. Use `True` in a new runtime or for
an environment that never finished installing. Switch `INSTALL_PROFILE` between
`'florence'` and `'dots'`, and rerun the configuration cell each time.

Do not install FlashAttention for Florence. Dots still uses its existing
FlashAttention profile: reuse the installed dots worker instead of rebuilding
it. The two loader fixes themselves require no dependency reinstalls. Installing
packages into the notebook kernel does not install them into a worker environment.
Offline checks cover these changes; a Colab GPU rerun is still required.

Sources: [dots processor constructor](https://huggingface.co/rednote-hilab/dots.ocr/blob/main/configuration_dots.py),
[Qwen2.5 processor API](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py),
[legacy import scanner](https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/dynamic_module_utils.py).

## Setup and execution

Open `Model_Testing.ipynb`. Keep its `benchmark_*.py` helpers and
`taxonomy_crosswalk.csv` in the same project folder as `train2017/`.
The images and per-image JSON annotations are not included in the code bundle.

1. Use a GPU notebook, mount your persistent dataset folder if using Colab, and
   set `DATA_ROOT` to that folder. Set `CODE_ROOT` separately if the helper files
   are stored elsewhere.
2. For the failed models, start with `INSTALL_PROFILE = 'florence'`,
   `INSTALL_PACKAGES = True`, and `RUN_GPU_INFERENCE = True`. The notebook now
   creates an isolated Python 3.11 environment and runs the model in that
   interpreter. It does not replace Torch or Transformers in Colab's Python 3.13
   kernel. The fixed helpers are embedded in the notebook, so the original
   GitHub setup link cannot silently download outdated helpers.
3. Repeat with `INSTALL_PROFILE = 'dots'`. This uses a separate Python 3.11
   environment with Transformers 4.56.1 and matching Torch/Torchvision builds.
   Dots requires an Ampere-or-newer GPU, such as the L4 used in your saved run,
   and a CUDA toolchain for FlashAttention. Keep dependency resolution enabled;
   do not use `--no-deps`. The setup checks actual imports and GPU access before
   starting the model worker. Once an environment passes setup, use
   `INSTALL_PACKAGES=False` to reuse it.
4. Run the comparison and inspection sections. They load actual saved outputs
   across profiles, with no need to keep all models in memory.

The saved `Leak_Benchmark_Results/outputs/multimodel_pii` folder and its Google
Drive counterpart are searched automatically. Equal-setting older runs can be
included explicitly, with code differences shown in provenance. The completed
Qwen/Chandra/Llama runs do not need to be rerun for these environment repairs.
Use `RUN_GPU_INFERENCE=False` for review-only operation. The `'vlm'` and
`'chandra'` worker profiles remain available for future experiments. Llama still
requires accepted model terms and an authorized Hugging Face login.

Your executed notebook was preserved at
`outputs/notebook_before_runtime_repair.ipynb`. The measured saved-run review is
in `outputs/benchmark_review/README.md`; it distinguishes inference failures
from privacy-detection misses. Llama returned valid benchmark JSON for only
5/50 images, and dots/Florence have no successful saved inference yet.

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
have been fabricated. The review uses your supplied completed runs; the repaired
dots and Florence environments still require a GPU rerun.
