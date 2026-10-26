# VISPR local application guide

## Start the application

From the repository root, use Python 3.10 or newer. Create a separate environment
so the application does not alter the annotation or research environments:

```powershell
python -m venv .venv-app
.venv-app\Scripts\python -m pip install -r requirements-app.txt
.venv-app\Scripts\python scripts/start_vispr.py
```

Dependency installation is a separate online setup operation; it does not download
model weights. For an offline machine, install these dependencies from a previously
prepared local wheel directory. After setup, no internet connection is needed.
Open **http://127.0.0.1:8765**. Frontend and backend run together; there is no Node
build, CDN or second frontend server. On Windows, `Start VISPR.cmd` uses `.venv-app`
if present, otherwise the current Python installation. Stop with Ctrl+C.

The application was launched and tested using the machine's existing packages;
no new dependency environment or model weights were installed during development.
Use a single server process. Do not expose it to a network or launch multiple
workers sharing its preferences directory.

## Connect LM Studio

1. In LM Studio, manually obtain a compatible **Qwen3-VL Instruct** GGUF model
   and its matching vision projector, if not already present. This is an explicit
   model-download operation you choose in LM Studio; the app never initiates it.
2. Load the model, including its projector, in LM Studio. Start the local server
   in its Developer panel, binding to loopback. Note the port.
3. Select **LM Studio** in VISPR and enter, for example,
   `http://127.0.0.1:1234/v1`. Click **Check connection**. Choose a loaded vision
   model ID from the model field's suggestions, then check again.
4. A loaded vision-model indication verifies availability and native capability
   metadata, not localization accuracy. Processing requests the JSON schema and
   image input together; unsupported schema/image requests fail without fallback.
5. Test on a synthetic image, inspect every predicted box and downloaded mask,
   and record the exact runtime/model version before a demonstration.

The adapter discovers loaded instances using `/api/v1/models`, with native v0
discovery only when v1 returns 404. v0 requires `type: vlm` and `state: loaded`.
It sends images to `/v1/chat/completions` on the same loopback server. It never
loads an unverified model based solely on its name. For authentication-required
LM Studio servers, supply the token in the `VISPR_LM_TOKEN` environment variable
before launching the backend. It is never exposed to the frontend or saved in
preferences. Keep existing server security settings. Redirects, HTTP proxies and
remote endpoints are forbidden.

Reference: [LM Studio model discovery](https://lmstudio.ai/docs/developer/rest/list),
[native v0 metadata](https://lmstudio.ai/docs/developer/rest/endpoints), and
[structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output).

### This machine's inspected state — 26 October 2026

- LM Studio is installed under the user's local profile; `lms server status`
  reported that its server was not running. The default URL is offered as a
  configurable suggestion, not claimed as a verified running endpoint.
- The Qwen3-VL-8B GGUF and projector are `.part` downloads, not ready weights.
  A complete Qwen2.5-VL GGUF exists but was not silently substituted.
- RAM: 15.7 GiB; GPU: Intel UHD Graphics, reported 1 GiB adapter RAM; CUDA absent.
  GPU shared memory is not equivalent to dedicated VRAM.
- Start by considering a manually obtained quantized Qwen3-VL 2B Instruct plus
  matching projector in LM Studio, then measure actual latency/memory. Its fit
  and localization quality on this machine remain untested. 8B is not the default
  recommendation for this memory-constrained CPU/integrated-GPU environment.

## Direct local Qwen3-VL

Use a separate model environment and install `requirements-app-model.txt` there.
The existing global Torch is 2.4.1 and Transformers is 5.2.0; Transformers import
fails because its installed Hugging Face Hub version conflicts. Do not repair
the global environment merely to run this app.

Manually prepare a complete local Transformers checkpoint (config, safetensors,
tokenizer, processor and chat template), following the
[official Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
No complete Qwen3-VL Transformers checkpoint was found in the standard local
Hugging Face cache during inspection. GGUF is not a Transformers checkpoint.

Select **Local Qwen3-VL**, enter the directory, and check it. Processing lazily
loads and reuses the model. Both offline environment flags and
`local_files_only=True` are enforced; custom remote code is disabled. Missing
files, dependency conflicts and memory failures produce errors, never downloads.
Direct inference is implemented but has not been executed with actual weights
here. CPU inference may be very slow and consume substantial memory.

## Future fine-tuned checkpoint

Select **Fine-tuned Qwen3-VL** and supply a complete, merged Qwen3-VL checkpoint
directory. It uses the same isolated adapter contract and strict output parser.
An unmerged LoRA/PEFT adapter directory is not supported yet. No fine-tuned
checkpoint currently exists in this integration and none has been validated.
If training changes the output contract, implement a parser/adapter conversion
to the documented internal schema; the frontend does not need to change.

## Upload, personalize, process, review

1. Drop or browse for JPEG, PNG or WebP still images. Up to five at a time,
   20 MB and 20 megapixels each. Animated inputs and unreadable files are rejected.
   Remove a queued image with its × button; add its replacement normally.
2. Search the 28 canonical IDs/names in **Your privacy rules**. ON masks that
   category; OFF reports it but leaves its exclusive region visible. All 28 are
   enabled by default. Enable all, Disable all and Reset defaults are explicit.
   Changes are saved locally and affect subsequent batches, never an active one.
3. Opaque black masking is the implemented strategy. Optional **Margin (px)**
   adds 0–20 pixels to valid rectangles, clipped to the image. Default: 0.
   Blur and pixelation are intentionally not implemented in this release.
4. Click **Process images**. Images run sequentially. The UI reports real stages
   with no invented inference percentage. Completed images remain reviewable
   while later jobs run. Duplicate batch submissions are rejected.
5. **Cancel batch** requests cancellation at the next safe boundary. It does
   not terminate a model halfway through generation. LM Studio requests time
   out after 300 seconds; direct generation can take longer. Queued jobs are
   skipped, and cancelled jobs cannot be downloaded.
6. Select a result tab. Compare original and output side by side, toggle output
   only, and toggle diagnostic boxes. Inspect attribute counts and individual
   confidence/coordinate records. The saved preference snapshot and model appear
   with each result. A disabled region may still be masked by an overlapping
   enabled detection.
7. Download an individual PNG or all completed PNGs in a ZIP. ZIPs include a
   review-required note and warning counts. Images with some unlocalized
   detections are explicitly incomplete, even if valid boxes were masked.

The PNG contains actual opaque pixel modifications, not diagnostic outlines.
EXIF orientation is applied before inference, outputs omit source metadata, and
saved pixels/dimensions are verified. None of these checks establish that every
PII instance was found. Empty predictions also require human review.

## Storage and cleanup

App data lives only in ignored `data/vispr_app/`. Preferences are in
`preferences.json`; upright uploads and outputs are in `session-<random-id>/`.
Job metadata and detections remain in process memory. No extracted personal text
or raw model response is saved. Filenames are displayed only in the local UI;
download filenames use random IDs. Maximum retained jobs: 50 per session.

**Clear session** removes this session's images/results while retaining preferences.
It is disabled while jobs are active. Normal shutdown cleans the session too.
After crashes, app-owned session folders older than 24 hours are removed on the
next startup. Younger abandoned folders can be removed manually after all app
processes have stopped. Downloads saved elsewhere remain yours to manage.
This is ordinary local file deletion, not forensic secure erasure or encryption.

Set `VISPR_BATCH_LIMIT` to an integer from 1 to 10 before launch to change the
batch size. Limits above 10 are capped. Avoid running multiple app instances.

## Troubleshooting and evidence

- **Unavailable LM Studio:** check its Developer server, loopback port, loaded
  model/projector and native metadata endpoint. The app never changes backend.
- **Text-only/unloaded model:** processing is blocked. Choose a loaded VLM.
- **JSON or truncated output:** no output is produced for a broken top-level
  contract. Check model schema support and context size. No textual description
  is converted into invented boxes.
- **Invalid individual boxes:** valid entries can still be masked; invalid
  entries remain listed with explicit warnings and incomplete localization.
- **No checkpoint/dependency conflict:** use the separate environment and local
  complete weights. Do not pass a remote model identifier or an unfinished download.
- **Browser backend connection lost:** keep the launch terminal running and reload
  after restart. Upload queue previews live in browser memory until sent.
- **Corrupt saved preferences:** startup stops rather than silently changing your
  rules. Repair the JSON or explicitly delete that file to restore defaults.
- **Windows sandbox test hang:** development sandbox blocked asyncio's internal
  socketpair; the approved unsandboxed retry passed. Ordinary local launch worked.

Run application tests (only synthetic images and injected fake adapters):

```powershell
python -m unittest discover -s tests -p test_vispr_app.py -v
node --check src/vispr_app/static/app.js
```

For a visibly labeled synthetic results UI, run
`python tests/vispr_ui_fixture.py` and open http://127.0.0.1:8766. This test-only
server injects fixed predictions and uses a temporary directory; it must never
be presented as model inference. Stop it when finished. The production launcher
has no mock-model option.

The application tests cover pipeline geometry, personalization, exact saved
pixels, metadata stripping, uploads, job states/cancellation, downloads,
loopback guards and mocked LM Studio protocol behavior. These are not a real
model localization benchmark. See `TRACKER.md` for actual counts and browser
checks. Full research tests also require `requirements-dev.txt`; two pre-existing
environment/notebook failures were recorded, not hidden or repaired by replacing
notebooks.
