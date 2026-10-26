# Training readiness plan

Planning only; `configs/training_plan.json` remains `enabled: false` and the third
notebook never launches training. Current decision evidence is in
[the benchmark report](../BENCHMARK_REPORT.md); this document defines the proposed
experiment, not a validated training implementation.

## Candidate and approval gates

Recommended first candidate (training not yet enabled): **Qwen/Qwen3-VL-8B-Instruct**.
It produced localized pilot detections and successful bounded ID correction. The
audited full export has 48/50 valid predictions and the strongest F1 on the
34-image common valid subset. See BENCHMARK_REPORT for scores and failure coverage.
Qwen2.5's empty pilot and higher full failure rate do not make it a better first candidate. OCR plus rules has very low recall;
first separate recognition failures from text-rule coverage. Florence repetition
and Llama authorization remain blockers, so neither enters the initial training plan.
No model is approved until baseline review, suitable annotations, disjoint splits,
license review and a real forward/backward smoke test pass.

The [Qwen card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) declares Apache 2.0
(checked 2026-10-21). Preserve pinned checkpoint/license notices and review dataset
rights separately. Initial proposed method: supervised LoRA or QLoRA for image-level PII labels
using the 28-label selection. Bounding-box supervision is deferred until independent
reviewed regions exist. Preserve exact JSON/schema and image-token alignment. The
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
supports adapter training on quantized bases in compatible architectures; it does
not prove this project's Qwen3 collator or target modules work. Training dependencies
(PEFT, a compatible trainer and image-aware batching) are intentionally not added
to the benchmark environments until that integration is tested.

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
2. Select one primary candidate after reviewing the full evidence. First compare prompt/mapping changes on
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

## Resource and experiment plan (not measured training requirements)

Use a separate environment and experiment ID. Start with microbatch 1, gradient
accumulation and activation checkpointing. Provisional adapter rank 16, alpha 32
and learning rate 1e-4 are hypotheses in the config, not tuned values; inspect actual
module names rather than assuming all-linear injection is suitable for vision blocks.
Freeze the vision encoder initially as an experimental choice, then verify gradients
reach intended adapters. Do not mask image tokens as if they were answer tokens.

The saved inference memory footprint cannot predict training memory. Prefer an
available A100 for the first controlled smoke test; an L4 attempt is conditional on
measured peak allocation and a conservative image/token budget. No minimum VRAM,
training duration or throughput has been established. Record free disk before
downloading the base checkpoint; budget its actual downloaded size, data, optimizer
state and at least two adapter checkpoints before continuing. Stop on insufficient
space rather than deleting historical evidence.

Save seed, split hashes, annotation revision, package lock, base checkpoint revision,
LoRA config, loss curves, elapsed time and peak memory per experiment. Save adapter,
optimizer/scheduler and RNG state for resume, plus an explicit best-checkpoint
selection criterion based on validation. Verify checkpoint reload before a long run.
Back up under the existing Drive training destination with manifest/hash readback.

Evaluate baseline and adapter on the same frozen 50-image regression cohort and a
new untouched test set; report coverage/failures, paired label metrics, per-category
misses, localization and retained readable PII where independent references support
those claims. Use validation only for tuning. A gain on this repeatedly reviewed
50-image cohort alone is insufficient evidence of improvement.

Preparation update, 22 October 2026: [attribute-selection report](ATTRIBUTE_SELECTION.md) defines the 28-label first experiment. Prepared manifests are provisional; preparation alone does not satisfy training gates.
