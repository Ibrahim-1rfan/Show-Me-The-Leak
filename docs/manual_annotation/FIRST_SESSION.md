# Your first Label Studio session

Use this computer's local instance at http://127.0.0.1:8080. Other team members cannot open this address from their own computers to reach your instance.

1. Start the application by double-clicking `Start Annotation.cmd` in the repository. Once it reports ready, open the address above. Keep your annotation database; restarting does not require reinstalling.
2. On the first visit, create a local account using your own email and password. Enter the password directly in Label Studio; do not paste it into chat. On later visits, log in.
3. Create a project named **Pilot â€” Team Member 1** (use 2 or 3 for the appropriate member). This pilot is separate from your main 400-image assignment.
4. In the project's labeling-interface settings, switch to Code and replace the example configuration with the contents of `manual_annotation_dataset/guidelines/label_studio_config.xml`. Save it.
5. Import `manual_annotation_dataset/assignments/pilot_label_studio.json`. It contains 27 tasks and points to local images. Register the image folders for this project using the command below before opening tasks. Do not upload the image folder separately or sync it as additional tasks.
6. Open the first task. Check that the image loads upright. Choose general/document/uncertain, its quality, and yes/no/uncertain for selected PII. Read the category guide rather than copying the expected VISPR labels.
7. Select every category present. For each visible instance choose its attribute ID and draw a tight box. Select its visibility. For an identifiable category you cannot locate, select the category and explain the uncertainty in Notes instead of drawing a guessed box. Do not transcribe names or numbers.
8. Check the whole image for missing instances, including backgrounds. Tick full_image_checked only after that inspection. Submit to save the annotation. Submission is not independent approval.
9. Initially do only a few pilot images, then check that categories, boxes and visibility were saved correctly. Finish the shared pilot independently; compare disagreements with the other two members and agree on rules before the main round.
10. Export submitted tasks as **JSON**. Keep pilot exports under `manual_annotation_dataset/annotations/reviews/pilot/member_1/` (adjust member number). Do not run the primary-import converter on pilot files.

After pilot agreement, create **Team Member 1 â€” Main**, reuse the same XML, and import `assignments/member_1_label_studio.json` for the 400 primary tasks. Use the matching member number. Main JSON exports can be passed to `scripts/import_manual_label_studio.py --member 1` as described in tool_workflow.md. They remain needs_review until independently checked.

### Required project storage setup

After importing tasks into each new project, read its number from the browser URL
(for example, `/projects/1/` means project 1). From PowerShell in the repository:

```powershell
.\.venv-annotation\Scripts\python.exe scripts/configure_annotation_storage.py --project 1 --apply
```

Replace `1` with that project's number. This backs up the local database, registers
only `images/` and `annotation_views/`, checks every imported image against the
original file bytes, and leaves task/annotation records unchanged. Storage uses a
match-nothing filter so syncing cannot create duplicate tasks. Do not change that
filter or click Sync. Project 1 was repaired on 25 October 2026; refresh its page.

If an image does not load, do not mark it no PII. A 404 can mean the project storage
registration above is missing even when the file exists. Check that the application
was started with Start Annotation.cmd and the package has not moved. If a form cannot
save, record the displayed error before changing the annotation. Leave unclear cases
for review rather than guessing.

Local app data lives under `data/label_studio/` and contains the database and settings. Back it up privately with the server stopped, together with annotation exports and the image package. Do not commit it or upload it publicly. Images and app data stay outside Git.
