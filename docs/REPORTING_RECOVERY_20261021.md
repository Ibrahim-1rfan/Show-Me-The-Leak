# Recover the partial-result reporting crash

The supplied Colab log records Qwen3 completing its single-image worker on
`2017_12973862` and verifying all seven image artifact copies in mounted Drive.
The prediction was **partial**, with `needs_review` and a failed mask check.
Report generation then rejected the new status with `status must be ok or error`.
The same unguarded report call in `finally` raised again, blocking other models.
This is a reporting integration defect; the log does not show an installation,
GPU memory or model-loading failure for this attempt.

## Recover in the existing Colab session

1. Upload `dist/repair_colab_reporting.py` through Colab's Files panel.
2. Add a cell below the failed execution cell and run:

   ```python
   %run -i /content/repair_colab_reporting.py
   ```

3. Run the summary, inspection and download cells to inspect/export the saved run.

The repair uses the notebook's existing `PROJECT` and `run_dir`. If those variables
are absent, it targets `/content/Show-Me-The-Leak-v2` and the reported run
`results/vlm/20261020T230625225013Z`. If the Colab runtime has reset, restore the
project and saved run from Drive first and set those two paths. Do not rerun
inference merely to regenerate a report.

The script verifies its embedded payload, saves the previous reporting /
orchestration files under `reporting_repairs/<timestamp>/`, records before/after
hashes, adds the execution-record helper, refreshes loaded modules and regenerates reports from saved predictions.
It then attempts the usual Drive backup. It does not install packages, load
models, start inference, change predictions, change `run_config.json`, rewrite
the original run source snapshot or call `start_run`. Existing historical failure
files remain as evidence, with known reporting failures identified in the repair
record. A backup failure is printed separately and can be retried via the usual
backup cell. Reports record the actual reporting source hashes.

Partial detections remain inspectable. They are excluded from successful coverage,
quality aggregates, common-successful comparisons and pilot acceptance. Coverage
exposes `partial_images` and `error_images`; the existing `failed_images` field
continues to mean their sum. Unknown statuses still fail validation.

## Future execution

The maintained VLM and OCR notebooks include the repair. New reporting errors are
saved as `reporting_failure.json`, separately from model failures. Remaining
models and backups can proceed; a persistent reporting error is raised clearly
after those attempts. A recovered reporting error is marked resolved.

Recovering reports does not make Qwen's partial redaction pass the pilot gate.
Inspect its saved rejected entries and mask issues before retrying that image.
Reexecuting the inference cell would retry a partial image; the recovery script
deliberately only regenerates reports. Source changes produce a new protocol when
`start_run` is called, so do not rerun the configuration cell to recover this run.

Validation uses offline parser-to-report, mixed-status, both-scope, orchestration
failure and live-module recovery regression tests. No new GPU inference or live
Drive verification is performed locally.
