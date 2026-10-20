# Proposed Git workflow only

No branches or Git history were changed for this proposal.

Use `main` for reviewed code, runnable notebooks, tests and small public
configuration/manifests. `initial-testing` can collect dataset preparation and
baseline work; `training` can collect fine-tuning support; `evaluation` can collect
post-training comparison work. Each stage branch should start from a known `main`
commit and regularly incorporate reviewed shared-code improvements from `main`.

Keep `src/leakbench` shared through commits merged into `main`, rather than copying
independently edited versions between stage branches. Make individual changes on
short-lived feature branches, such as `codex/progress-reporting`. Review tests and
configuration changes before merging. Merge stable infrastructure back to `main`
early; keep unvalidated training experiments isolated. Resolve conflicts in source
and regenerate notebook bundles rather than hand-merging encoded notebook payloads.

Long-lived stage branches are possible, but they can drift. A practical alternative
is short-lived stage/feature branches plus tags such as `baseline-v2` and
`training-ready-v1`. Tags identify code snapshots; they do not contain model weights
or datasets. Record the code commit/tag, dependency lock, model revision, dataset
fingerprint, seed, protocol hash and unique run ID in each cloud experiment.

Branches alone are insufficient for experiment management. Two runs on the same
branch can use different images, thresholds, weights or hardware. Version the
small configurations in Git and retain immutable results and checkpoints in the
existing Drive project root. Do not put images, archives, notebooks with private
output cells, model checkpoints, tokens, environment files or raw OCR values into
commits. The new broad ignore rules help with future additions but cannot erase
already tracked files or historical secrets. Review staged changes before commits;
any future history cleanup should be a separate explicit decision.
