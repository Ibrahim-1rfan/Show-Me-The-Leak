# Local VISPR application

The existing Python research and annotation workflows remain unchanged. The new
`src/vispr_app` package serves a dependency-free HTML/CSS/JavaScript frontend and
a FastAPI API from one loopback origin. No CDN, analytics, or cloud inference is used.

## Design (26 October 2026)

- Registry: load the exact 28 IDs/names from `configs/selected_attributes.json`.
- API: bounded image uploads, one sequential worker, observable job stages,
  cooperative cancellation, PNG downloads and ZIP exports.
- Adapters: LM Studio with loaded vision-model checks and JSON schema requests;
  direct local Transformers Qwen3-VL; a separate fine-tuned configuration using
  the same adapter contract. No mock adapter is exposed in the application.
- Pipeline: upright RGB image → inference of all 28 categories → strict JSON and
  normalized xyxy validation → preference snapshot → pixel masking → saved PNG
  reopened and compared against expected pixels. Diagnostic overlays are frontend-only.
- Coordinates: model boxes use 0–1000 xyxy in the upright image. One utility maps
  these to half-open integer pixel rectangles with floor/ceil coverage. Overflows
  up to 1% are clipped and reported; larger overflows are rejected.
- Privacy: literal loopback inference URLs, disabled proxies and redirects,
  offline local model loading, same-origin mutation checks, host validation and
  a per-process browser token. Sensitive payloads are never logged.
- Storage: `data/vispr_app/preferences.json` and per-process session folders.
  Upload originals are normalized and metadata stripped. Results remain until
  Clear session, normal shutdown, or automatic stale-session cleanup at startup.
  Preferences survive cleanup. Local disk storage is not encrypted or secure-erased.

The existing benchmark adapters can download dependencies/weights and use a
different experimental output contract, so they are deliberately not imported by
the offline app. The application follows their native PIL/chat-template approach.
Model-generated coordinates remain predictions requiring human review; schema
conformance and pixel verification cannot establish complete PII detection.
