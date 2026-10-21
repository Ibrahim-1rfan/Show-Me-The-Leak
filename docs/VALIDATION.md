> Current handoff verification is recorded in [TRACKER Section L](../TRACKER.md) and [GUIDE](../GUIDE.md). This file retains the older validation history.

# Validation and handoff

Historical handoff prepared 20 October 2026. **Superseded for runtime acceptance**
by `REGRESSION_REPORT_20261021.md` and the live `TRACKER.md`: the imported pilots
exposed failures these earlier offline checks did not catch. The 98-test result
below describes the previous release, not proof of successful model redaction.

## Completed locally

- The offline suite contains 98 passing tests: 79 existing checks plus 19 checks
  for the new workflow and notebook packaging. These use synthetic fixtures for
  orchestration tests; none are presented as model benchmark measurements.
- New checks cover fixed-cohort validation, changed inputs and labels, verified
  resume, preservation of failed attempts, CUDA placement rejection, backup
  verification and recovery, subprocess timeouts, failed pilots, worker crashes,
  missing/corrupt outputs, credential-safe logging and protocol changes.
- Each notebook's code parses, has no saved execution output, and restores the
  exact maintained package and configurations independently in a clean directory.
- The actual cohort passes readability, original-byte and annotation SHA-256,
  count, category, source mapping, format/extension, decoded-pixel uniqueness and
  perceptual duplicate checks: 25 documents and 25 general images, 50 total.
- Selection fingerprint:
  `0beae8a380049243b74c8c9e2605ef3d01ecbdfcdfaa14f99dc74e8a3cb50b72`.
- All 21 hashed pre-existing top-level files checked against the initial inventory
  remain unchanged, excluding the intentional `.gitignore` addition. All 17 saved
  historical result files still match their audit hashes. Large directories were
  retained; this is not a claim that every byte of every legacy directory was
  independently hashed.

## Run next in Colab

1. Open `01_vlm_benchmark.ipynb` or `02_ocr_benchmark.ipynb` on a GPU runtime.
   For dots, select L4/A100 or another supported Ampere-or-newer GPU.
2. Optionally upload `vispr_eval_50.zip` into `/content` to avoid downloading the
   full source ZIP. The notebook still validates the frozen selection and hashes.
3. Run from the top with `RUN_STAGE='single'`. Add the `HF_TOKEN` Colab secret
   for authorized Llama access. Failed models remain quarantined.
4. Review single-image output, then use `RUN_STAGE='pilot'`. Inspect all four
   pilot outputs and record image-specific review notes before `RUN_STAGE='full'`.
5. Review coverage alongside quality, and verify the Drive backup. The last cell
   can retry copying without repeating inference.

## Pending measurements and actions

Fresh GPU dependency installation, real inference, dots/Florence GPU compatibility,
pilot timing, the full benchmark, verified Drive persistence and final model
selection remain pending. Static compatibility fixes and offline tests do not
prove those GPU runs succeed. There is no guaranteed runtime estimate.

The historical results do not establish the cause of an hour-long wait per image.
Reports compare recorded historical detection-call medians with new timings as
context, not as paired speedup evidence: the cohort, hardware and protocol differ.

No files have been deleted or relocated as part of legacy cleanup. That migration
is deferred until real cloud verification. No training, branch creation, commit,
push or history rewrite has been performed. Region IoU, redaction coverage and
transcript metrics require independently verified reference annotations.
