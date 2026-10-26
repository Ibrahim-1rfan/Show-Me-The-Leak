# Start here: three-person VISPR annotation

This package contains 1,200 original images and 28 selected attribute definitions. Nothing has been annotated or approved yet. Every primary member owns 400 images (280 general and 120 document candidates); train/validation splits stay unchanged. Image types and expected labels are hints, not verified answers.

1. Read guidelines/annotation_guidelines.md and guidelines/tool_workflow.md.
2. Use your assignments/member_1.csv, member_2.csv or member_3.csv. The JSON with the same member number imports your 400 tasks into Label Studio. Keep paths relative to this package root.
3. First complete the 27-image shared pilot independently in a separate project. Agree on rules before full annotation; pilot ownership does not change primary assignments.
4. Annotate image type, quality, category presence and each visible instance. Draw boxes; explicitly flag unlocalizable regions. Do not transcribe PII.
5. Export submitted tasks as Label Studio JSON. Keep the raw export in annotations/raw_exports/. Use the repository import script to create needs_review canonical files, or copy a template into annotations/completed/ and fill it consistently.
6. Follow assignments/cross_review.csv for the secondary sample and review all ambiguous and gold-validation examples. Record approval individually; samples do not approve the rest.
7. From the repository run `python scripts/validate_manual_annotation.py`. Add `--require-complete` only when every image has an independently approved canonical file. Structure passing is not proof of annotation completeness.

## Folder map

- images/: stable original filenames, byte-identical copies.
- annotation_views/: upright PNG derivatives only when EXIF orientation requires them.
- annotations/original_vispr/: preserved original JSONs, no boxes.
- annotations/templates/: pending records, no prefilled ground truth.
- annotations/completed/: primary canonical submissions; initially empty.
- annotations/reviews/: pilot versions, reviewer records and adjudication history.
- annotations/raw_exports/: original tool exports retained before conversion.
- assignments/: three primary manifests/imports, shared pilot and cyclic review.
- manifests/: immutable source/master snapshots, original/selected taxonomy, benchmark exclusions and package summary.
- guidelines/: rules, schema, fictional examples, tool instructions and training-gap report.

The master manifest is an initial assignment/status snapshot. Canonical completed records are the current status source; do not silently edit immutable ownership, hashes or split columns. The validation command reports current submitted/approved/missing totals. There is no final unseen test set in this package; the existing 50-image benchmark is excluded.

No images were uploaded and no annotation software was installed. Keep all package contents private. Original source dataset remains untouched. Do not regenerate over an existing package; preserve all manual work and use a versioned migration when guidelines change.

Preparation verified on 24 October 2026: all copies and assignments passed checks; 41 upright derivatives were pixel-checked. The 27 pilot images cover all 28 expected labels from VISPR (not independently confirmed). Live Label Studio setup and human annotations are still pending. Per-member label counts are in manifests/assignment_label_balance.csv.
