# Tool choice and working procedure

Recommendation: a local Label Studio instance with one primary project per member, plus three independent pilot projects. It combines image-level Choices and instance RectangleLabels, keeps stable attribute IDs, and exports raw JSON. Separate projects make ownership explicit without assuming edition-specific automatic assignment features.

Alternative: CVAT provides image tags, boxes, per-object attributes and CVAT XML backups. It is a good option if your team already operates it, but its exports would need a different converter. We have prepared Label Studio XML and JSON, not a verified CVAT import. Label Studio's setup is simpler for this particular mixed form; CVAT is well suited to box-heavy work. Neither tool's export constitutes independent approval.

Official documentation checked 24 October 2026:
- Installation: https://labelstud.io/guide/install.html
- Local file serving: https://labelstud.io/guide/storage_local
- Choices: https://labelstud.io/tags/choices
- Rectangle labels: https://labelstud.io/tags/rectanglelabels
- JSON export and percent coordinates: https://labelstud.io/guide/export.html
- CVAT image format: https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/

## Local installation — 25 October 2026

Label Studio 1.23.2 is installed in `.venv-annotation/` with Python 3.10.11. `pip check` passed and the installed backend accepted the annotation XML. To start this local installation, double-click `Start Annotation.cmd` and open http://127.0.0.1:8080. App data and logs are under `data/label_studio/`, outside the served image directory. No account password is stored in source. The dependency freeze is kept privately in `data/label_studio/requirements-lock.txt`.

For a fresh machine, create a separate environment and install `requirements-annotation.txt`; keep training environments separate. Record the actual Python/package versions and run the pilot before scaling. The installed Google connector dependency warns that Python 3.10 is past its support window; no cloud connector is used here. Prefer a currently supported Python for a future fresh setup after verifying dependencies. This does not claim that another Python version has been tested locally.

On a fresh Windows checkout with Python 3.10 and the prepared image package,
run from the repository root (skip installation on this already prepared computer):

```powershell
py -3.10 -m venv .venv-annotation
.\.venv-annotation\Scripts\python.exe -m pip install -r requirements-annotation.txt
.\.venv-annotation\Scripts\python.exe -m pip check
& '.\Start Annotation.cmd'
```

You can also double-click the launcher. It resolves paths from the checkout, enables local image serving and uses the private database directory. Bind locally; do not expose the service to the internet. For a hosted internal team instance, use approved access control and storage. No public cloud upload is required. The package can be copied privately to a team machine; the import URLs remain relative to the package root.

## Import, annotate and export

1. Create a project named Team Member 1 (likewise 2 and 3). In Settings > Labeling Interface > Code, paste guidelines/label_studio_config.xml. Resolve any configuration error before proceeding; live UI compatibility remains unverified until the pilot.
2. Import only assignments/member_N_label_studio.json into that primary project. Register the image folders using `.\.venv-annotation\Scripts\python.exe scripts/configure_annotation_storage.py --project PROJECT_NUMBER --apply` from the repository root, replacing PROJECT_NUMBER with the numeric ID in its URL. Label Studio 1.23.2 requires matching project storage entries as well as enabled local-file serving; missing entries return 404 even for existing files. This helper backs up the database, registers images/ and annotation_views/ with a match-nothing filter, and verifies task images without syncing or creating tasks. Do not separately sync images as new tasks or change that filter. Check a task loads and its orientation matches the provided view.
3. Create separate pilot projects and import assignments/pilot_label_studio.json into each. Repeat the storage registration for each new project. Keep their exports under annotations/reviews/pilot/member_N/. Do not feed pilot exports to the primary importer.
4. Inspect the entire image. Select image type, quality, yes/no/uncertain PII presence and all present selected category IDs. Draw one region per instance with the matching stable ID and select region visibility. For identifiable but unlocalizable categories select the image-level category and explain it in Notes; no invented box. Do not transcribe private text.
5. Tick full_image_checked only after inspecting the whole image for omissions. Use notes without actual private values. Uncertain presence must not be answered no merely to complete the form.
6. Export submitted/completed tasks only, in raw Label Studio JSON. Do not use COCO/YOLO as the canonical export: they can lose image-level labels and uncertainty. Unsubmitted/cancelled tasks are intentionally rejected by the converter.
7. From the source repository run:

```text
python scripts/import_manual_label_studio.py path/to/export.json --member 1
python scripts/validate_manual_annotation.py
```

The converter checks task ownership, dimensions, labels and axis-aligned boxes. Percent [x,y,width,height] becomes normalized xyxy. It refuses existing canonical files and ambiguous multiple annotations. Raw exports are preserved by content hash. Converted records are needs_review and every instance starts manually_verified=false. No approval is inferred from clicking Submit. A batch with invalid tasks is rejected before canonical writing; fix those tasks and re-export.

Review in the tool against the image and correct boxes there. Preserve earlier export/canonical versions under annotations/reviews/ before replacing a corrected canonical file manually. The reviewer records their identity/time, verifies each instance, checks omissions, and updates the canonical review/status fields. Do not batch-set approval without inspecting each image. Null-box instances remain excluded from localization training even after their category has been reviewed.

The supplied XML is backend-validated and the conversion logic is tested with synthetic exports. The user imported 27 pilot tasks into project 1. After adding the missing local storage entries on 25 October 2026, all 27 authenticated image-view responses returned 200 and matched the source bytes. Task and annotation records were unchanged. Browser rendering after refresh and real annotation export still need user verification. Run the pilot to validate this interface and orientation end to end before scaling.
