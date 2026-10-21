# Show Me the Leak

Show Me the Leak evaluates whether open vision-language models (VLMs) and OCR
models can locate private information in images and hide it with bounding-box
redaction. It compares predictions with VISPR image labels and preserves raw
outputs, masks, errors, timings and model reviews for inspection.

**Start here:** [Usage and reproduction](GUIDE.md) ·
[Measured results](BENCHMARK_REPORT.md) · [Progress and known issues](TRACKER.md).

## Current status

Research benchmark, not a privacy guarantee. The October 21 exports contain real
single, pilot and full-stage evidence. Both OCR models processed all 50 images,
but each has four invalid predictions. Qwen3 has 48/50 usable outputs and Qwen2.5
has 40/50. All six modes have been audited from downloaded artifacts. Florence
still has a generation failure; Llama is blocked by Hugging Face authorization.
No six-model benchmark is complete and no model is approved for training.
Qwen3 is the recommended first candidate; see the benchmark report for evidence.

## Quick start

1. Import [VLM benchmark](notebooks/01_vlm_benchmark.ipynb) or
   [OCR benchmark](notebooks/02_ocr_benchmark.ipynb) into Google Colab.
2. Select a GPU and run Sections 1–4. Dots requires L4/A100 or another supported
   Ampere-or-newer GPU. Llama requires approved model access and the `HF_TOKEN`
   Colab secret. The notebook creates isolated model environments.
3. Select **single** or **pilot** in Section 5, execute Section 6, then inspect
   Section 8. Full runs require explicit four-image review. Follow [GUIDE](GUIDE.md)
   for dataset download, backup, reproducibility and troubleshooting.

## Repository map

| Path | Purpose |
|---|---|
| `notebooks/` | VLM, OCR and training-readiness entry points |
| `src/leakbench/` | Maintained inference, evaluation, redaction and backup code |
| `configs/` | Pinned models, 68-attribute taxonomy and fixed 50-image selection |
| `scripts/`, `tests/` | Builders, read-only audits and offline regression tests |
| `docs/` | Supporting evidence, historical investigations and training plan |
| `data/`, `results/`, `dist/` | Generated local data, runs and bundles; excluded from Git |

Root `Model_Testing.ipynb`, `benchmark_*.py` and related utilities are legacy
compatibility material, not the recommended entry points. Imported results and
private archives are excluded from Git; see the [cleanup plan](docs/AUDIT_AND_MIGRATION.md).
Local development: install `requirements-dev.txt`, run
`python scripts/build_notebooks.py`, then `python -m unittest discover -s tests -v`. GPU inference targets Linux/Colab;
training remains disabled. No credentials, model weights or dataset images are
required in a source checkout.
