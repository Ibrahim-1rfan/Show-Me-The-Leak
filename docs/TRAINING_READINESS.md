# Training readiness plan

No model training is launched by these notebooks. A final best VLM or OCR model
cannot be selected until the repaired benchmark has measured results on the new
cohort. Candidate decisions must compare both coverage and same-image quality.

## Candidates and support checks

Model cards and repository metadata checked 20 October 2026; exact revisions and
declared licenses are in `configs/models.json`. This is a technical readiness
assessment, not a statement that training has been validated on the user's GPU.

| Candidate | Why retain it | Training readiness |
|---|---|---|
| Qwen3 VL 8B | Primary VLM candidate; the old paired sample favored it over Qwen2.5, with low recall overall | Transformer VLM; evaluate supervised LoRA/QLoRA on an isolated training environment. Apache 2.0 model card. Validate target modules and gradients before a full run. |
| Qwen2.5 VL 7B | VLM alternative if resource use or output reliability is better | Apache 2.0; same requirements for image-aware batching and adapter compatibility. |
| Llama 3.2 Vision 11B | Alternative architecture | Custom Llama 3.2 terms and gated access. Fix output compliance first. Larger inference memory does not establish training feasibility. |
| Florence 2 large ft | Smaller OCR/region candidate | MIT model card; encoder-decoder vision architecture. Native task tokens, region sequence targets and compatible remote code are required. Its inference adapter is not a trainer. |
| Chandra 9B | OCR candidate once truncation and field mapping are understood | Original checkpoint declares OpenRAIL; inspect the full license and usage constraints before adaptation/distribution. Vision-language training data and original tokenizer/template must match. |
| dots.ocr 3B | OCR/layout candidate | Card declares MIT and repository also includes license/notice files; review these at the pinned revision. Custom vision code and FlashAttention need a successful forward/backward compatibility pilot. |

Sources: [Qwen3](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
[Llama](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct),
[Florence](https://huggingface.co/microsoft/Florence-2-large-ft),
[Chandra](https://huggingface.co/datalab-to/chandra),
[dots](https://huggingface.co/dots-studio/dots.ocr),
[PEFT quantized training guidance](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).

## Data requirements and leakage controls

The 50-image curated cohort is evaluation-only and already inspected during
development. Exclude its original IDs, byte hashes, decoded-pixel hashes and
near-duplicate groups from all training data. It is not an untouched final test.
Reserve an additional unseen test partition before any fine-tuning. Assign
duplicates and related captures to the same partition. Use fixed, documented
train/validation/test manifests and validate disjointness before creating batches.

VISPR image labels can supervise image-level classification. They cannot teach
accurate boxes or OCR transcripts without new annotations. Collect complete
in-scope boxes, native OCR text/layout targets, and difficult negative examples;
review ambiguous labels independently. Never turn a model's own output into
unquestioned evaluation ground truth. Improve OCR text-to-PII mapping separately
from OCR recognition so effects can be attributed correctly.

## Training experiment design

1. Freeze baseline configurations, dataset version, checkpoint and measured
   metrics. Compare missed categories, label micro F1 and recall, supported macro
   F1, output coverage, document/general slices, runtime and GPU memory.
2. Select one candidate per category. First compare prompt/mapping changes on
   development data. Train only if the remaining errors justify it.
3. Start with a small gradient/serialization smoke test, microbatch 1 and gradient
   accumulation. Choose BF16 only if supported. Consider 4-bit base weights plus
   trainable adapters where the architecture supports them; 4-bit inference alone
   does not prove training support. Confirm optimizer state and activation memory.
4. Provisional LoRA settings are in `configs/training_plan.json`. Resolve target
   modules by inspecting the chosen architecture; they are intentionally unset.
   Configure checkpointing, adapter-only checkpoints, random seeds and early
   stopping against validation performance. Do not tune on the 50-image cohort.
5. Measure peak GPU memory and time for a few real steps before budgeting the
   remaining experiment. Colab Pro does not guarantee a particular GPU or capacity.
   Avoid training all six models or promising wall-clock times before measurement.
6. Evaluate the original and adapted model on identical frozen evaluation inputs.
   Report quality, failure and runtime differences with paired uncertainty where
   support permits. Add localization/coverage/text metrics only with references.
   Preserve negative results and document whether gains justify the extra cost.

Training may proceed only after benchmark review, candidate selection, license
review, suitable supervised annotations, disjoint splits and the GPU smoke test.
Store checkpoints under `Leak_Benchmark_Results/training/<experiment_id>/`, separate
from baseline results. Training artifacts and raw private examples stay out of Git.
