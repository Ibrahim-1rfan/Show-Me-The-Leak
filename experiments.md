# Experiment configurations

## VISPR-28 preparation, 22 October 2026

Seed 421; 28 positive labels; target 1,200 examples (840 general proxies/360 document proxies), including 120 safe negatives; 80/20 train/validation. No model execution. [Protocol](docs/ATTRIBUTE_SELECTION.md); outputs and limitations are recorded in `docs/attribute_selection/summary.json`. Existing measured model comparisons remain in [BENCHMARK_REPORT](BENCHMARK_REPORT.md).

Completed local preparation: 9,996 decodable images; four missing image references; ten Safe conflicts; two duplicate-candidate groups. Proposed subset reaches 1,200 and 960/240 splits. Image-type and near-duplicate review remain pending; no training readiness claim.

## Manual annotation preparation — 24 October 2026

Same frozen 1,200-image selection and 28 labels; assignment seed 421. Three primary annotators, a 27-image training-only shared pilot and 120-image cyclic secondary review sample. No model run or human annotation is claimed. [Rules and provenance](docs/manual_annotation/annotation_guidelines.md); package aggregate: `docs/manual_annotation/package_summary.json`.
