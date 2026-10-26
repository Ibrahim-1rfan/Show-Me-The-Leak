# VISPR attribute selection and provisional training labels

22 October 2026. This is a classification dataset preparation experiment, not a trained model or verified spatial annotation set.

## Inputs and interpretation

`attributes.tsv` has idx, attribute_id, group and description columns: 68 rows including Safe, hence 67 positive attributes. Its description is a short display name, not a detailed labeling manual; no richer definitions are invented here. `Images Classification.txt` describes a 9,999-image manifest. Percentages are images containing a label divided by 9,999, rounded to one decimal; labels overlap and percentages need not total 100. Its final Top 67 includes Safe but omits a29_ausweis; the bottom-frequency section supplies its count of 20. Safe-only (3,701) differs from Safe-tagged (3,711), indicating conflicting labels in that older manifest.

The pipeline reconstructs counts from 10,000 original JSON annotations, counts each label once per image, and preserves the report counts separately. Annotation population and usable-image population are distinct: a missing or corrupt image remains in annotation frequency but cannot enter training. Counts and ranks below use all 10,000 annotations; zero-shot benchmark statistics are not changed.

## Selection method and trade-offs

Recommendation: 28 labels. Apply the same qualitative decision order to every label: (1) direct identification/contact/financial exposure or actionable privacy linkage; (2) feasible visual detection or OCR with contextual interpretation; (3) available support; (4) diversity and redundancy within the 28-label budget. Importance is a design judgment, not a measured score. Frequency determines sampling and evaluation confidence, not whether a dangerous identifier matters.

The recommendation deliberately retains scarce national IDs, fingerprints, driver/student IDs, credit cards, email and home-address labels. These are high-value exceptions, not well-supported classes. Eight selected labels have fewer than 50 annotated positives. Fewer than 50 is a review flag, not a scientific eligibility cutoff: an ordinary 20% validation allocation would supply fewer than ten examples before deduplication. The additional 150-example flag indicates fewer than thirty expected validation examples. Per-split support supersedes these heuristics.

Common appearance and demographic labels are deferred because frequency alone is insufficient and reliable redaction targets are ambiguous. Sensitive health, sexuality, political and legal labels remain important but require explicit evidence and policy-specific annotation. Exclusion never means safe. Handwriting is deferred in favor of signatures; vehicle ownership in favor of plates; birthplace in favor of birth date and actionable location. Keep separate original labels for full/partial faces, names, addresses and plates; a policy may group them, but evaluation should preserve their distinctions. Document labels describe context, not proof that every field is PII.

## Selected labels

| ID | Actual name | Images | % of 10,000 | Processing |
|---|---|---:|---:|---|
| a7_fingerprint | Fingerprint | 24 | 0.24 | specialized detector + VLM review |
| a8_signature | Signature | 175 | 1.75 | specialized detector + VLM review |
| a9_face_complete | Face (Complete) | 3484 | 34.84 | specialized detector + VLM review |
| a10_face_partial | Face (Partial) | 2789 | 27.89 | specialized detector + VLM review |
| a11_tattoo | Tattoo | 180 | 1.80 | specialized detector + VLM review |
| a19_name_full | Full Name | 542 | 5.42 | VLM context + OCR |
| a20_name_first | First Name | 96 | 0.96 | VLM context + OCR |
| a21_name_last | Last Name | 85 | 0.85 | VLM context + OCR |
| a24_birth_date | Date of Birth | 72 | 0.72 | VLM context + OCR |
| a29_ausweis | National Identification | 20 | 0.20 | VLM context + OCR |
| a30_credit_card | Credit Card | 46 | 0.46 | VLM context + OCR |
| a31_passport | Passport | 127 | 1.27 | VLM context + OCR |
| a32_drivers_license | Drivers License | 28 | 0.28 | VLM context + OCR |
| a33_student_id | Student ID | 33 | 0.33 | VLM context + OCR |
| a35_mail | Mail | 71 | 0.71 | VLM context + OCR |
| a37_receipt | Receipts | 99 | 0.99 | VLM context + OCR |
| a38_ticket | Tickets | 250 | 2.50 | VLM context + OCR |
| a49_phone | Phone number | 83 | 0.83 | VLM context + OCR |
| a73_landmark | Visited Landmark or Street Sign | 889 | 8.89 | VLM context + OCR |
| a74_address_current_complete | Visited Location (Complete) | 67 | 0.67 | VLM context + OCR |
| a75_address_current_partial | Visited Location (Partial) | 810 | 8.10 | VLM context + OCR |
| a78_address_home_complete | Home address (Complete) | 41 | 0.41 | VLM context + OCR |
| a79_address_home_partial | Home address (Partial) | 21 | 0.21 | VLM context + OCR |
| a82_date_time | Date/Time of Activity | 761 | 7.61 | VLM context + OCR |
| a85_username | Username | 53 | 0.53 | VLM context + OCR |
| a90_email | Email address | 40 | 0.40 | VLM context + OCR |
| a103_license_plate_complete | License Plate (Complete) | 207 | 2.07 | VLM context + OCR |
| a104_license_plate_partial | License Plate (Partial) | 89 | 0.89 | VLM context + OCR |

## Complete 67-label inventory

Every selected and excluded label, including its group, privacy role, count, frequency rank, rationale and processing method, is available in [CSV](attribute_selection/all_attributes.csv) and [JSON](attribute_selection/all_attributes.json). The reusable selected taxonomy is [selected_attributes.json](../configs/selected_attributes.json). Pairwise overlap counts are saved to `docs/attribute_selection/selected_overlap.csv` after the audit completes.

| ID | Name | Group | Count | % | Rank | Select | Rationale |
|---|---|---|---:|---:|---:|---|---|
| a1_age_approx | Age Group | Personal Description | 3998 | 39.98 | 5 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a2_weight_approx | Weight Group | Personal Description | 2193 | 21.93 | 9 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a3_height_approx | Height Group | Personal Description | 1695 | 16.95 | 10 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a4_gender | Gender | Personal Description | 4625 | 46.25 | 2 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a5_eye_color | Eye Color | Personal Description | 2755 | 27.55 | 8 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a6_hair_color | Hair Color | Personal Description | 4405 | 44.05 | 3 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a7_fingerprint | Fingerprint | Personal Description | 24 | 0.24 | 63 | Yes | Visual identifying feature not recoverable by OCR alone; spatial review required. |
| a8_signature | Signature | Personal Description | 175 | 1.75 | 31 | Yes | Visual identifying feature not recoverable by OCR alone; spatial review required. |
| a9_face_complete | Face (Complete) | Personal Description | 3484 | 34.84 | 6 | Yes | Visual identifying feature not recoverable by OCR alone; spatial review required. |
| a10_face_partial | Face (Partial) | Personal Description | 2789 | 27.89 | 7 | Yes | Visual identifying feature not recoverable by OCR alone; spatial review required. |
| a11_tattoo | Tattoo | Personal Description | 180 | 1.80 | 30 | Yes | Visual identifying feature not recoverable by OCR alone; spatial review required. |
| a12_semi_nudity | Semi-nudity | Personal Description | 588 | 5.88 | 19 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a13_full_nudity | Complete nudity | Personal Description | 21 | 0.21 | 65 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a16_race | Race | Personal Description | 4321 | 43.21 | 4 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a17_color | Skin Color | Personal Description | 4710 | 47.10 | 1 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a18_ethnic_clothing | Traditional clothing | Personal Description | 174 | 1.74 | 32 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a19_name_full | Full Name | Personal Description | 542 | 5.42 | 20 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a20_name_first | First Name | Personal Description | 96 | 0.96 | 41 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a21_name_last | Last Name | Personal Description | 85 | 0.85 | 44 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a23_birth_city | Place of Birth | Personal Description | 30 | 0.30 | 60 | No | Useful quasi-identifier, deferred for narrower identity/contact coverage within 28 labels; handwriting is not equivalent to a signature. |
| a24_birth_date | Date of Birth | Personal Description | 72 | 0.72 | 48 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a25_nationality | Nationality | Personal Description | 112 | 1.12 | 38 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a26_handwriting | Handwriting | Personal Description | 267 | 2.67 | 26 | No | Useful quasi-identifier, deferred for narrower identity/contact coverage within 28 labels; handwriting is not equivalent to a signature. |
| a27_marital_status | Marital status | Personal Description | 79 | 0.79 | 46 | No | Appearance/demographic or broad personal context; high frequency does not justify initial redaction priority or sensitive-trait inference. |
| a29_ausweis | National Identification | Documents | 20 | 0.20 | 67 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a30_credit_card | Credit Card | Documents | 46 | 0.46 | 54 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a31_passport | Passport | Documents | 127 | 1.27 | 36 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a32_drivers_license | Drivers License | Documents | 28 | 0.28 | 62 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a33_student_id | Student ID | Documents | 33 | 0.33 | 58 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a35_mail | Mail | Documents | 71 | 0.71 | 49 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a37_receipt | Receipts | Documents | 99 | 0.99 | 39 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a38_ticket | Tickets | Documents | 250 | 2.50 | 27 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a39_disability_physical | Physical disability | Health | 76 | 0.76 | 47 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a41_injury | Medical Treatment | Health | 167 | 1.67 | 33 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a43_medicine | Medical History | Health | 32 | 0.32 | 59 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a46_occupation | Occupation | Employment | 844 | 8.44 | 13 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a48_occassion_work | Work Occassion | Employment | 365 | 3.65 | 23 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a49_phone | Phone number | Personal Life | 83 | 0.83 | 45 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a55_religion | Religion | Personal Life | 114 | 1.14 | 37 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a56_sexual_orientation | Sexual Orientation | Personal Life | 208 | 2.08 | 28 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a57_culture | Culture | Personal Life | 919 | 9.19 | 11 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a58_hobbies | Hobbies | Personal Life | 166 | 1.66 | 34 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a59_sports | Sports | Personal Life | 382 | 3.82 | 22 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a60_occassion_personal | Personal Occassion | Personal Life | 87 | 0.87 | 43 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a61_opinion_general | General Opinion | Personal Life | 337 | 3.37 | 24 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a62_opinion_political | Political Opinion | Personal Life | 36 | 0.36 | 57 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a70_education_history | Education history | Personal Life | 98 | 0.98 | 40 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a99_legal_involvement | Legal involvement | Personal Life | 22 | 0.22 | 64 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a64_rel_personal | Personal Relationships | Relationship | 334 | 3.34 | 25 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a65_rel_social | Social Circle | Relationship | 694 | 6.94 | 16 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a66_rel_professional | Professional Circle | Relationship | 651 | 6.51 | 18 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a67_rel_competitors | Competitors | Relationship | 144 | 1.44 | 35 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a68_rel_spectators | Spectators | Relationship | 658 | 6.58 | 17 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a69_rel_views | Similar view | Relationship | 438 | 4.38 | 21 | No | Broad activity/relationship/context label with ambiguous redaction target; deferred to keep identifier and localization work manageable. |
| a73_landmark | Visited Landmark or Street Sign | Whereabouts | 889 | 8.89 | 12 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a74_address_current_complete | Visited Location (Complete) | Whereabouts | 67 | 0.67 | 51 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a75_address_current_partial | Visited Location (Partial) | Whereabouts | 810 | 8.10 | 14 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a78_address_home_complete | Home address (Complete) | Whereabouts | 41 | 0.41 | 55 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a79_address_home_partial | Home address (Partial) | Whereabouts | 21 | 0.21 | 66 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a82_date_time | Date/Time of Activity | Whereabouts | 761 | 7.61 | 15 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |
| a85_username | Username | Internet | 53 | 0.53 | 53 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a90_email | Email address | Internet | 40 | 0.40 | 56 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a92_email_content | Email content | Internet | 30 | 0.30 | 61 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a97_online_conversation | Online conversations | Internet | 63 | 0.63 | 52 | No | Potentially high-risk sensitive content; deferred scope, needs explicit evidence and policy-specific annotation. Exclusion does not imply safe. |
| a102_vehicle_ownership | Vehicle Ownership | Automobile | 68 | 0.68 | 50 | No | Useful quasi-identifier, deferred for narrower identity/contact coverage within 28 labels; handwriting is not equivalent to a signature. |
| a103_license_plate_complete | License Plate (Complete) | Automobile | 207 | 2.07 | 29 | Yes | Direct identity/contact/financial exposure; retain despite limited support. |
| a104_license_plate_partial | License Plate (Partial) | Automobile | 89 | 0.89 | 42 | Yes | Location/time linkage or document context complements direct identifiers; context alone does not identify a text field. |

## Imbalance and limitations

The largest selected class is Face (Complete), 3,484 annotated images (34.84%); the smallest is National Identification, 20 (0.20%): a 174.2:1 ratio. The eight selected classes below 50 positives are Fingerprint, National Identification, Credit Card, Drivers License, Student ID, Home address (Complete), Home address (Partial), and Email address. These are retained for practical risk, not because the dataset supplies strong statistical support. The 28-label scope is suitable for an exploratory classification experiment, not certification of every category. Targeted selection cannot fix missing examples or supply spatial ground truth.

Pairwise overlap is reported independently of marginal counts. Never add positive counts to estimate the number of unique sensitive images. Inspect the exported per-split counts before freezing the split; reserve independent reviewed examples for scarce classes where feasible, without copying the same person/document into multiple splits.

## Preparation protocol and annotation status

Run `python scripts/prepare_training_data.py --root .` from the repository, with Pillow installed. Notebook `04_dataset_preparation.ipynb` invokes the same script. Original JSON and images are read only. Derived local artifacts go to ignored `data/training_preparation/`; public aggregates go to `docs/attribute_selection/`.

The script attempts to decode every available matching image, computes byte and upright RGB-pixel hashes, and caches checks by path/size/mtime. Delete only that cache or use a fresh output location when independent revalidation of unchanged metadata is needed. Exact byte/pixel duplicates and equal dHash candidates form connected groups; equal perceptual hashes can group unrelated simple images. This conservative grouping does not detect every near-duplicate. Human review of related captures is still required.

Metadata with any Documents-group label proposes document type; all other images propose general type. This is not equivalent to the visually reviewed benchmark definition. All type labels remain `metadata_proxy_requires_review`. Review ambiguous rows before declaring the 70/30 ratio verified.

Select 360 document-proxy and 840 general-proxy images with seed 421; reserve 120 Safe-only general negatives, then greedily favor labels with lower selected coverage. Select at most one member per duplicate group, block benchmark IDs and their connected groups, and quarantine contradictory Safe-plus-private annotations. Split each image type approximately 80/20 into training/validation. This composition ratio is independent of the split ratio. These splits are provisional until type and near-duplicate review and per-label support checks pass. Do not silently move validation examples after viewing model results. A larger unseen test set is a separate future milestone; none is fabricated here.

Targets retain original_labels and selected_labels, label_source=original_vispr_image_labels, regions=null and region_status=not_annotated. Empty selected_labels means no selected VISPR label, not proven absence of all PII. Example target: `{"selected_labels":["a19_name_full"],"regions":null,"region_status":"not_annotated"}`. In a future chat-format example, supply the image and selected taxonomy in the user message and only these original image-level labels in the assistant response. Do not insert guessed names, text or coordinates.

## Training and evaluation recommendations

Start with multi-label category prediction and zero/few-shot comparison. Use training examples only for demonstrations. Track micro/macro F1, per-class support/precision/recall and invalid output rate. Sparse validation classes require wide uncertainty statements; zero-support classes have unavailable recall, not a perfect or zero measured recall. The selection report exposes actual train/validation positives for each label.

The frequent-face versus rare-ID imbalance remains substantial after taxonomy selection. Targeted selection helps but does not create new independent examples. Consider bounded rare-class sampling within training and evaluate its effect on validation; do not blindly apply classification class weights to autoregressive token loss. Include safe negatives and hard negatives. Avoid training to infer sensitive demographic attributes from appearance.

Use VLM document context to assist field interpretation, with OCR for text and coordinates; do not make document classification a gate that prevents OCR on general scenes. Faces, fingerprints, tattoos and signatures require visual localization plus review. Retain raw detections separately from user policy decisions. Missing localization must block claims of complete redaction. Add IoU/region coverage only after independent human region annotation.

Keep stable original IDs and versioned selections; adding labels later must not rewrite historical metrics. Training is still disabled. No GPU inference or quality improvement is claimed by this preparation work.

Review `data/training_preparation/review_queue.csv` alongside the original images. Fill reviewed_type, reviewed_selected_labels, reviewer and review_status; these edits are review records and are not automatically promoted into verified training targets. Keep reviewed boxes in a separately versioned annotation set.

## Completed local preparation

Audited 10,000 annotations; 9,996 matching images decoded successfully. Four annotation IDs have no matching image. Ten images have Safe-plus-private conflicts and are quarantined from selection. Two duplicate-candidate groups were identified; this does not prove all near-duplicates have been found. EXIF warnings occurred during decoding, without an image-decode failure among the available files.

Prepared 1,200 provisional classification targets: 840 general proxies, 360 document proxies; 960 training and 240 validation; 120 Safe-only negatives. All 50 existing benchmark IDs and their connected groups are excluded. No held-out final test cohort was created.

| Selected attribute | Training positives | Validation positives |
|---|---:|---:|
| a7_fingerprint | 19 | 4 |
| a8_signature | 118 | 24 |
| a9_face_complete | 351 | 92 |
| a10_face_partial | 269 | 68 |
| a11_tattoo | 96 | 23 |
| a19_name_full | 239 | 63 |
| a20_name_first | 71 | 16 |
| a21_name_last | 63 | 21 |
| a24_birth_date | 51 | 17 |
| a29_ausweis | 10 | 7 |
| a30_credit_card | 29 | 8 |
| a31_passport | 46 | 12 |
| a32_drivers_license | 25 | 2 |
| a33_student_id | 22 | 9 |
| a35_mail | 51 | 7 |
| a37_receipt | 51 | 14 |
| a38_ticket | 67 | 14 |
| a49_phone | 66 | 17 |
| a73_landmark | 229 | 56 |
| a74_address_current_complete | 53 | 13 |
| a75_address_current_partial | 297 | 70 |
| a78_address_home_complete | 35 | 5 |
| a79_address_home_partial | 18 | 3 |
| a82_date_time | 305 | 77 |
| a85_username | 40 | 13 |
| a90_email | 31 | 8 |
| a103_license_plate_complete | 87 | 27 |
| a104_license_plate_partial | 67 | 20 |

These are counts in the proposed manifests, not measured model results. Driver-license validation support is only two; claims for such categories must remain exploratory. Review type assignments and near-duplicate candidates before freezing the split.
