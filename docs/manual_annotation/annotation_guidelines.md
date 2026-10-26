# Annotation guidelines — VISPR-28 v1

Purpose: create independent human category and region annotations for visual privacy detection/redaction. The original VISPR labels are hints, not answers to copy. Inspect the entire image for missed instances, including small background objects. No model output is verified ground truth.

## Image inspection and labels

Confirm document vs general: document means a readable/recognizable card, page, ticket, receipt or mailing label is the dominant subject; a person or scene containing a small document stays general. For mixed/unclear images choose uncertain and explain. Confirm clear/unclear/corrupt/ambiguous quality. No selected PII means an inspected image has none of these 28 categories; it does not mean universally safe. Uninspected images have null presence, not false. If unsure whether PII exists, keep presence unknown and request review.

Original names below come from attributes.tsv. Its short descriptions are not detailed annotation rules: the operational boundaries in the last column are project rules proposed for the pilot. Resolve disagreements and version changes before the main round.

| ID | Original name/definition | Positive-instance and boundary rule |
|---|---|---|
| a7_fingerprint | Fingerprint | Visible fingerprint impression or ridge pattern, including an actual print on a form. Box the print, not an ordinary hand or fingertip with no discernible ridges. |
| a8_signature | Signature | A visible handwritten signature or identifiable signature mark. Box the signature strokes. A blank signature line and ordinary printed name are not signatures. |
| a9_face_complete | Face (Complete) | A visible face with its principal facial area available, including a profile with the expected visible side. Box the face, not the whole person. A document portrait can qualify. Do not identify the person. |
| a10_face_partial | Face (Partial) | A cropped or occluded face with only part of its facial area visible. Box the visible facial extent; do not reconstruct hidden boundaries. Blur alone does not make an otherwise complete face partial; mark quality/visibility separately. Use complete OR partial for the same face, not both. |
| a11_tattoo | Tattoo | A visible tattoo design or permanent ink mark on skin. Box the visible design; do not infer one under clothing or confuse printed clothing patterns. |
| a19_name_full | Full Name | A full personal name supported by visible text/context. One box per full-name field. Do not also split that same field into first/last-name boxes; annotate those labels only for separate fields. |
| a20_name_first | First Name | A standalone given-name field or clearly identifiable first name. Do not assume an arbitrary word is a person name. Do not duplicate a full-name instance. |
| a21_name_last | Last Name | A standalone family-name field supported by context. Apply the same non-duplication rule as Full Name. |
| a24_birth_date | Date of Birth | A date explicitly identified as a birth date. Box its value, including adjacent label only if needed to make the region intelligible. An arbitrary date is not a birth date. |
| a29_ausweis | National Identification | A visible national identification document/card recognizable by document context. Box the physical document as a document-object instance. Separately annotate visible selected fields such as name, DOB, face and signature. |
| a30_credit_card | Credit Card | A recognizable payment credit card, including a partial card when context is sufficient. Box the card object; do not invent a separate card-number class. A generic bank logo alone is insufficient. |
| a31_passport | Passport | A recognizable passport cover or identity page. Box the passport object/page; add distinct selected field instances. There is no passport-number class in this taxonomy: do not label a passport number as a name or username; note the coverage gap without transcribing it. |
| a32_drivers_license | Drivers License | A recognizable driving licence. Box the document object and separately annotate selected fields. Do not infer document type solely from a face photograph. |
| a33_student_id | Student ID | A recognizable student identification card. Box the document object and separately annotate supported selected fields. |
| a35_mail | Mail | A visible addressed mail item/envelope or mailing label. Box the mail object/label as document context and independently annotate selected personal fields. A generic mailbox without an addressed item does not qualify. |
| a37_receipt | Receipts | A recognizable transaction receipt. Box the receipt object; separately mark selected fields. Receipt detection alone does not establish that every price/product is private. |
| a38_ticket | Tickets | A recognizable travel/admission ticket or boarding pass. Box the ticket object; separately mark supported selected fields. Do not infer a passenger identity from a generic transport scene. |
| a49_phone | Phone number | A visible phone-number field supported by format/context. Box the number. Do not call an arbitrary digit string a phone number. |
| a73_landmark | Visited Landmark or Street Sign | A recognizable visited landmark or readable street sign providing a location clue. Box the sign or landmark itself, and mark this as contextual evidence; do not assert that a person lives there or infer an unseen address. |
| a74_address_current_complete | Visited Location (Complete) | Visible evidence of a sufficiently specific visited/current location (for example a full street address), linked by scene/document context. Box the location text or identifying region. Do not label it home without evidence. |
| a75_address_current_partial | Visited Location (Partial) | Visible partial current/visited-location clue (for example a city or incomplete address). Box the clue. Complete and partial are mutually exclusive for the same location field; a landmark can also have its own object instance. |
| a78_address_home_complete | Home address (Complete) | A complete home/residential address explicitly linked to a person/residence by context. Box the address block. A business or destination address is not automatically a home address. |
| a79_address_home_partial | Home address (Partial) | A partial home/residential address with explicit residential context. Do not infer a home from an arbitrary street sign. Do not duplicate the same field as complete and partial. |
| a82_date_time | Date/Time of Activity | A visible date/time tied to a personal activity, journey, transaction or image event. Box that field. Do not infer dates from scenery or use a birth date again as activity time. |
| a85_username | Username | A visible personal account handle or username supported by context. Box the handle. Generic website names or logos do not qualify; an email address uses its own category. |
| a90_email | Email address | A visible email-address field. Box the address; exact transcription is unnecessary. Do not label unrelated text merely because it contains an at-sign. |
| a103_license_plate_complete | License Plate (Complete) | An entire licence-plate identifier is visibly available on a plate. Box the plate/identifier, not the vehicle. Unreadable resolution is quality uncertainty, not automatic partial visibility. |
| a104_license_plate_partial | License Plate (Partial) | A licence plate with only part of its identifier visible because of cropping/occlusion. Box the visible plate. Do not double-label a single plate as complete and partial. |

## Instances and boxes

Draw each visible occurrence separately, including repeated names in different places. Use tight axis-aligned boxes with minimal margin; do not include unrelated content. Keep a multiline address as one block; separate unrelated fields. Document categories describe the whole document object, while names, dates, faces and signatures get their own field/object boxes. Do not convert every field into a whole-document box. Overlap between a document object and its constituent fields is intentional and must be preserved; downstream redaction policy decides which categories to hide.

For partially visible text/signatures/fingerprints, box only the visible region when the category is supported by context, mark partially_visible or obscured, and never guess missing characters. If a category is known but cannot be located, record a null box, localization=unlocalizable/uncertain and a reason. An uncertain category should be referred for adjudication rather than confidently invented. For several unlocalizable occurrences use distinct instance IDs and notes; the import converter can create one category-level placeholder, which the reviewer must expand where needed.

Example correct: separate name and signature boxes on a passport, plus a passport document-object box. Incorrect: a Full Name box covering the whole passport. Correct: partial-face label for an occluded face. Incorrect: complete and partial labels for the same face. Correct: unknown location when an image-level label cannot be spatially confirmed. Incorrect: default [0,0,1,1] box to make a record pass.

## Coordinate convention and orientation

Canonical bbox=[x_min,y_min,x_max,y_max], normalized to [0,1], origin top-left of the EXIF-upright image. Require 0<=x_min<x_max<=1 and 0<=y_min<y_max<=1. Original files in images/ are byte-identical copies. Annotation views use upright dimensions; nontrivial EXIF orientation gets a separate PNG in annotation_views/. Always annotate the task's supplied image view, never a rotated/cropped screenshot. Disable image/box rotation in the tool.

For pixel boxes [left,top,right,bottom], divide x coordinates by upright width and y coordinates by upright height. Pixel corners for a mask are floor(x_min*W), floor(y_min*H), ceil(x_max*W), ceil(y_max*H), clipped to the image; right/bottom are exclusive. Label Studio exports percentages x,y,width,height: divide by 100, then x_max=x+width and y_max=y+height. Its coordinates are not already [0,1].

## Privacy, review and submission

Do not transcribe names, numbers or email contents. v1 has no transcription field. Record category and location only; avoid actual PII in notes. Any future transcript experiment requires a separate access-restricted extension and justification. Keep package/images/exports private, on approved team storage; do not upload them to public annotation services or GitHub.

Statuses: pending -> in_progress -> annotated -> needs_review -> approved. Export/import alone never approves anything. Reviewer must inspect all image content for omissions, validate class boundaries and boxes, resolve ambiguity, and record identity/time. Every approved instance must explicitly be manually_verified=true. Approved localization-unknown records can support reviewed category tasks but must be excluded from complete-box/redaction training.

Complete the shared 27-image training-only pilot independently in three separate pilot projects (do not consult colleagues' labels first). Compare presence, per-category disagreement, instance counts and matched box IoU; disagreements are discussion triggers, not automatic majority truth. Adjudicate and freeze v1 rules before scaling. The pilot supplements and does not replace primary ownership. Transfer adjudicated decisions to the primary record only after explicit review.

The cyclic cross_review.csv assigns 40 primary records per member for a secondary audit (120 total). In addition, independently review every uncertain case, scarce category and every record used as gold validation. A sampled review does not approve the unsampled records; only individually approved records may enter gold exports. Keep primary assignments unchanged. Use annotations/reviews/ for pilot/reviewer versions and notes. Preserve earlier versions when correcting annotations; do not regenerate the package over them.
