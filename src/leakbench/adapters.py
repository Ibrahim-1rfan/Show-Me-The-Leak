"""Lazy GPU adapters for the PII benchmark (no model loads on import).

Each adapter returns raw output plus explicitly typed, non-PII OCR regions where
appropriate. The caller owns taxonomy mapping and scoring. In particular,
Florence's task tokens are not a substitute for the taxonomy JSON instruction.

Source APIs: the official Hugging Face model cards linked in MODEL_SPECS. Pins
are source-based compatibility profiles, not a claim of GPU validation here.
Use a fresh kernel after switching between the VLM and legacy Florence profiles.
"""
from __future__ import annotations

import gc
import ast
import importlib
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEPENDENCY_PROFILES = {
    "vlm": ["transformers==4.57.1", "accelerate>=1.0,<2", "bitsandbytes>=0.46,<1", "Pillow>=10"],
    # Microsoft's original remote architecture was authored for 4.41; using the
    # community-converted checkpoint would change the requested model identity.
    "florence": ["transformers==4.41.2", "accelerate>=0.30,<2", "einops>=0.8,<1", "timm==1.0.7", "Pillow>=10"],
}

MODEL_SPECS = {
    "qwen3_vl": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct", "display_name": "Qwen3-VL 8B",
        "profile": "vlm", "role": "direct_pii_vlm", "output_kind": "pii_json",
        "class_name": "Qwen3VLForConditionalGeneration", "gated": False,
        "source": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
    },
    "qwen25_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "display_name": "Qwen2.5-VL 7B",
        "profile": "vlm", "role": "direct_pii_vlm", "output_kind": "pii_json",
        "class_name": "Qwen2_5_VLForConditionalGeneration", "gated": False,
        "source": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "llama32_vision": {
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "display_name": "Llama 3.2 Vision 11B",
        "profile": "vlm", "role": "direct_pii_vlm", "output_kind": "pii_json",
        "class_name": "MllamaForConditionalGeneration", "gated": True,
        "source": "https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct",
    },
    "florence2": {
        "model_id": "microsoft/Florence-2-large-ft", "display_name": "Florence-2 large ft (0.77B)",
        "profile": "florence", "role": "ocr_plus_rules", "output_kind": "ocr_regions",
        "class_name": "AutoModelForCausalLM", "gated": False,
        "source": "https://huggingface.co/microsoft/Florence-2-large-ft",
        # Official revision containing the safetensors and weight-tying fixes.
        "revision": "4a12a2b54b7016a48a22037fbd62da90cd566f2a",
    },
}


def prepare_image(image: Any, max_input_edge: int = 1280) -> Any:
    """Cap longest edge without upscaling; preserve the caller's upright frame."""
    from PIL import Image
    if max_input_edge is None:return image.convert('RGB')
    if isinstance(max_input_edge, bool) or not isinstance(max_input_edge, int) or max_input_edge < 1:
        raise ValueError("max_input_edge must be a positive integer")
    converted = image.convert("RGB")
    if min(converted.size) < 1:
        raise ValueError("Image dimensions must be positive")
    scale = min(1.0, max_input_edge / max(converted.size))
    if scale < 1:
        converted = converted.resize(
            (max(1, round(converted.width * scale)), max(1, round(converted.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return converted


def florence_regions(native_output: dict, input_size: tuple, original_size: tuple,
                     task: str = "<OCR_WITH_REGION>") -> list[dict]:
    """Convert native quads/boxes to upright ORIGINAL-image pixel coordinates.

    Invalid native output raises instead of silently looking like zero PII.
    Quads remain available for audit, while their axis-aligned envelopes support
    the common rectangle redaction/IoU pipeline.
    """
    if not isinstance(native_output, dict) or task not in native_output:
        raise ValueError(f"Florence output is missing task {task}")
    payload = native_output[task]
    if not isinstance(payload, dict):
        raise ValueError("Florence region output must be a dictionary")
    key = "quad_boxes" if task == "<OCR_WITH_REGION>" else "bboxes"
    boxes, labels = payload.get(key), payload.get("labels")
    if not isinstance(boxes, list) or not isinstance(labels, list) or len(boxes) != len(labels):
        raise ValueError("Florence labels and regions must be equally sized lists")
    iw, ih = input_size
    ow, oh = original_size
    if min(iw, ih, ow, oh) <= 0:
        raise ValueError("Image dimensions must be positive")
    sx, sy = ow / iw, oh / ih
    regions = []
    for label, box in zip(labels, boxes):
        if not isinstance(label, str) or not isinstance(box, (list, tuple)) or len(box) != (8 if key == "quad_boxes" else 4):
            raise ValueError("Malformed Florence region")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("Florence coordinates must be finite numbers")
        if key == "quad_boxes":
            xs, ys = box[0::2], box[1::2]
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        else:
            left, top, right, bottom = box
        if right <= left or bottom <= top:
            raise ValueError("Florence region has zero or negative area")
        if not 0<=left<right<=iw or not 0<=top<bottom<=ih:
            raise ValueError('Florence region is outside its model-input coordinate frame')
        pixel_box = [max(0.0, min(ow, left * sx)), max(0.0, min(oh, top * sy)),
                     max(0.0, min(ow, right * sx)), max(0.0, min(oh, bottom * sy))]
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            raise ValueError("Florence region lies outside the image")
        region = {"text": label, "bbox_pixels": pixel_box}
        if key == "quad_boxes":
            region["quad_pixels"] = [v * (sx if index % 2 == 0 else sy) for index, v in enumerate(box)]
        regions.append(region)
    return regions


def generation_status(generated_ids: list[int], limit: int, eos_token_id: Any = None,
                      forced_eos_token_id: Any = None) -> dict:
    """Distinguish natural EOS from hitting the generation budget.

    A forced EOS at the limit (used by Florence) cannot prove completion: it is
    conservatively marked truncated. Generated IDs exclude prompt/start tokens.
    """
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id])
    forced_ids = set(forced_eos_token_id if isinstance(forced_eos_token_id, (list, tuple, set)) else [forced_eos_token_id])
    eos_ids.discard(None)
    forced_ids.discard(None)
    count = len(generated_ids)
    hit_limit = count >= limit
    ended_with_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
    forced_at_limit = bool(hit_limit and generated_ids and generated_ids[-1] in forced_ids)
    return {"generated_token_count": count, "hit_token_limit": hit_limit,
            "ended_with_eos": ended_with_eos, "forced_eos_at_limit": forced_at_limit,
            "truncated": bool(hit_limit and (not ended_with_eos or forced_at_limit))}


def _runtime(profile: str) -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import the {profile!r} runtime: {type(exc).__name__}: {exc}. "
            "Use the notebook's isolated Python 3.11 worker setup; installing with --no-deps does not repair incompatible packages."
        ) from exc
    expected = "4.41.2" if profile == "florence" else "4.57.1"
    if transformers.__version__ != expected:
        raise RuntimeError(
            f"This adapter profile requires transformers=={expected}; found {transformers.__version__}. "
            "Install its notebook dependency profile and restart the kernel. Results from other profiles "
            "can be compared from their saved files without loading those models."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available. Select a GPU notebook runtime before loading this model.")
    return torch, transformers


class _Adapter:
    def __init__(self, model_key: str, *, revision: str | None = None,
                 max_new_tokens: int = 2048, max_input_edge: int = 1280,
                 load_in_4bit: bool = True, token: str | None = None):
        self.model_key = model_key
        self.spec = MODEL_SPECS[model_key]
        self.model_id = self.spec["model_id"]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if max_input_edge is not None and (isinstance(max_input_edge, bool) or not isinstance(max_input_edge, int) or max_input_edge < 1):
            raise ValueError("max_input_edge must be a positive integer")
        self.max_new_tokens = max_new_tokens
        self.max_input_edge = max_input_edge
        self.load_in_4bit = load_in_4bit
        self.revision = revision or self.spec.get("revision") or "main"
        self.model = None
        self.processor = None
        self.torch, self.transformers = _runtime(self.spec["profile"])
        from .hardware import native_dtype
        self.dtype = native_dtype(self.torch)
        self._hub_kwargs = {"revision": self.revision}
        if token is not None:
            self._hub_kwargs["token"] = token
        # Omitted token lets Hugging Face use cached login/HF_TOKEN. Credentials
        # never appear in returned metadata or benchmark files.

    def _metadata(self, original: Any, model_input: Any, generated_ids: list[int], limit: int) -> dict:
        # Florence delegates generation to its inner language model, whose
        # forced-EOS setting must be used instead of guessing from decoded text.
        generator = self.model.language_model if self.model_key == "florence2" else self.model
        generation_config = generator.generation_config
        status = generation_status(generated_ids, limit,
                                   getattr(generation_config, "eos_token_id", None),
                                   getattr(generation_config, "forced_eos_token_id", None))
        metadata = {
            "model_key": self.model_key, "model_id": self.model_id,
            "revision_requested": self.revision,
            "revision_resolved": getattr(self.model.config, "_commit_hash", None),
            "transformers_version": self.transformers.__version__,
            "torch_version": self.torch.__version__,
            "dtype": str(self.dtype), "load_in_4bit": self.load_in_4bit,
            "device": str(self.model.device), "gpu_name": self.torch.cuda.get_device_name(),
            "original_size": list(original.size), "input_size": list(model_input.size),
            "max_input_edge": self.max_input_edge,
            "processor_resizing": "Model-specific processor may resize/tile the capped input further.",
            "coordinate_frame":"upright_original",
            "input_to_original_scale":[original.width/model_input.width,original.height/model_input.height],
            "pre_resize_policy":"processor_native" if self.max_input_edge is None else "aspect_ratio_cap_no_upscale",
            "do_sample": False, "max_new_tokens": limit,
            "output_kind": self.spec["output_kind"], "role": self.spec["role"],
        }
        metadata.update(status)
        return metadata

    def close(self) -> None:
        self.model = None
        self.processor = None
        self._hub_kwargs = {}
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class VLMAdapter(_Adapter):
    def __init__(self, model_key: str, **kwargs):
        super().__init__(model_key, **kwargs)
        tf = self.transformers
        model_class = getattr(tf, self.spec["class_name"])
        load_args = dict(self._hub_kwargs, torch_dtype=self.dtype, device_map="auto",
                         trust_remote_code=False, use_safetensors=True)
        if self.load_in_4bit:
            load_args["quantization_config"] = tf.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        try:
            self.processor = tf.AutoProcessor.from_pretrained(self.model_id, **self._hub_kwargs)
            self.model = model_class.from_pretrained(self.model_id, **load_args)
            self.model.eval()
        except Exception as exc:
            self.close()
            if self.spec["gated"] and isinstance(exc, OSError):
                raise RuntimeError(
                    "Llama loading failed. Accept Meta's model license on its Hugging Face page and "
                    "log in with an authorized Hugging Face account (notebook_login or HF_TOKEN). "
                    "Also check storage/network access. This is a load failure, not a benchmark score."
                ) from exc
            raise
        self._hub_kwargs = {}

    def infer(self, image: Any, prompt: str) -> dict:
        if self.model is None:
            raise RuntimeError("Adapter has been closed")
        model_input = prepare_image(image, self.max_input_edge)
        if self.model_key == "llama32_vision":
            # Mllama's card supplies the PIL image separately and explicitly
            # disables duplicate special tokens after formatting the template.
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            formatted = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=model_input, text=formatted, add_special_tokens=False, return_tensors="pt")
        else:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": model_input}, {"type": "text", "text": prompt},
            ]}]
            # The native processor handles PIL images; qwen-vl-utils/video
            # decoders are unnecessary for this single-image benchmark.
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        response = outputs[:, prompt_tokens:]
        raw = self.processor.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        metadata = self._metadata(image, model_input, response[0].tolist(), self.max_new_tokens)
        metadata['vision_input_diagnostics']={k:list(v.shape) for k,v in inputs.items()
            if k in ('pixel_values','image_grid_thw','aspect_ratio_ids','aspect_ratio_mask') and hasattr(v,'shape')}
        if 'image_grid_thw' in inputs:metadata['image_grid_thw']=inputs['image_grid_thw'].detach().cpu().tolist()
        return {"raw_text": raw, "output_kind": "pii_json", "text_regions": [],
                "native_output": None, "metadata": metadata}


def florence_eager_imports(filename, imports):
    """Ignore only guarded optional FlashAttention in Florence's legacy scan."""
    if Path(filename).name != 'modeling_florence2.py' or 'flash_attn' not in imports:
        return imports
    tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
    def flash_import(node):
        if isinstance(node, ast.ImportFrom):
            return (node.module or '').split('.')[0] == 'flash_attn'
        return isinstance(node, ast.Import) and any(x.name.split('.')[0] == 'flash_attn' for x in node.names)
    all_imports = {id(node) for node in ast.walk(tree) if flash_import(node)}
    guarded = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Name)
                and node.test.func.id == 'is_flash_attn_2_available'
                and not node.test.args and not node.test.keywords):
            guarded.update(id(child) for statement in node.body for child in ast.walk(statement) if flash_import(child))
    if all_imports and all_imports <= guarded:
        return [name for name in imports if name != 'flash_attn']
    return imports


@contextmanager
def florence_eager_loader():
    """Temporary import-discovery shim; weights and remote source stay intact.

    Used only in the single-model worker while loading eager Florence. Other
    missing dependencies still raise normally. Restore the scanner on failure.
    """
    from transformers import dynamic_module_utils
    original = dynamic_module_utils.get_imports
    dynamic_module_utils.get_imports = lambda filename: florence_eager_imports(filename, original(filename))
    try:
        yield
    finally:
        dynamic_module_utils.get_imports = original


class FlorenceAdapter(_Adapter):
    def __init__(self, model_key: str = "florence2", precision='float16', **kwargs):
        if kwargs.get("load_in_4bit", False):
            raise ValueError("Florence uses its float16 reference profile; set load_in_4bit=False")
        kwargs["load_in_4bit"] = False
        super().__init__(model_key, **kwargs)
        if precision not in ('float16','float32'):raise ValueError('Florence precision must be float16 or float32')
        self.dtype = getattr(self.torch,precision)
        # Remote code is necessary for the exact original Microsoft checkpoint.
        # The default revision is pinned above. Only the legacy dependency scan
        # is adapted for eager attention; model source and weights stay intact.
        args = dict(self._hub_kwargs, trust_remote_code=True)
        try:
            with florence_eager_loader():
                self.processor = self.transformers.AutoProcessor.from_pretrained(self.model_id, **args)
                self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_id, **args, torch_dtype=self.dtype, attn_implementation="eager", use_safetensors=True,
                )
            self.model.to("cuda").eval()
        except Exception:
            self.close()
            raise
        self._hub_kwargs = {}

    def _task(self, image: Any, task: str, text_input: str = "") -> dict:
        if self.model is None:
            raise RuntimeError("Adapter has been closed")
        model_input = prepare_image(image, self.max_input_edge)
        inputs = self.processor(text=task + text_input, images=model_input, return_tensors="pt")
        inputs = inputs.to(self.model.device, self.dtype)
        # Original Florence decoder position limit is 1024, including its start
        # token. Keep a 1-token margin and report the effective budget.
        limit = min(self.max_new_tokens, int(self.model.config.text_config.max_position_embeddings) - 1)
        with self.torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=limit, do_sample=False, num_beams=3,
            )
        raw = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        metadata = self._metadata(image, model_input, outputs[0].tolist()[1:], limit)
        metadata.update({"task": task, "num_beams": 3, "native_coordinate_size": list(model_input.size),
                         "attention_implementation": "eager", "loader_compatibility": "guarded_optional_flash_import",
                         "regions_coordinate_size": list(image.size), "native_json_required": False,
                         "processor_pixel_values_shape":list(inputs['pixel_values'].shape),
                         "processor_resize":getattr(self.processor.image_processor,'size',None),
                         "requested_max_new_tokens":self.max_new_tokens,
                         "effective_limit_reason":"decoder_position_bound_including_start_token"})
        # Preserve raw generation even when the processor or one region fails.
        native=None;regions=[];issues=[]
        try:
            native=self.processor.post_process_generation(raw,task=task,image_size=model_input.size)
            payload=native[task];key='quad_boxes' if task=='<OCR_WITH_REGION>' else 'bboxes'
            boxes,labels=payload[key],payload['labels']
            if len(boxes)!=len(labels):raise ValueError('Florence labels/boxes length mismatch')
            for index,(box,label) in enumerate(zip(boxes,labels)):
                try:
                    if not label.strip():raise ValueError('Empty Florence text/region label')
                    regions.extend(florence_regions({task:{key:[box],'labels':[label]}},model_input.size,image.size,task))
                except (ValueError,TypeError) as exc:issues.append({'region_index':index,'error':str(exc)})
        except (ValueError,KeyError,TypeError) as exc:issues.append({'error':str(exc)})
        if metadata['truncated']:issues.append({'error':'Output truncated at decoder generation budget'})
        if not regions and raw.replace('</s>','').replace('<s>','').replace('<pad>','').strip():
            issues.append({'error':'Nonempty generation without valid spatial text regions'})
        result = {"raw_text": raw, "output_kind": "ocr_regions", "native_output": native,
                  "text_regions": regions, "metadata": metadata,'parse_success':not issues,'parse_warnings':issues}
        if task == "<OCR_WITH_REGION>":
            result["predicted_text"] = "\n".join(region["text"] for region in regions)
        else:
            # Grounded object phrases are NOT recognized text, and should never
            # enter the OCR-to-PII rule mapper by accident.
            result["output_kind"] = "grounded_regions"
            result["grounded_regions"] = result.pop("text_regions")
            metadata["output_kind"] = "grounded_regions"
        return result

    def infer(self, image: Any, prompt: str = "") -> dict:
        """Native OCR task; the free-form taxonomy prompt is deliberately unused."""
        return self._task(image, "<OCR_WITH_REGION>")

    def extract(self, image: Any) -> dict:
        return self.infer(image)

    def ground(self, image: Any, phrase: str) -> dict:
        """Optional separately reported grounding diagnostic, not OCR or PII labels."""
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError("Provide a nonempty phrase to ground")
        return self._task(image, "<CAPTION_TO_PHRASE_GROUNDING>", phrase)


def load_adapter(model_key: str, *, revision: str | None = None, max_new_tokens: int = 2048,
                 max_input_edge: int = 1280, load_in_4bit: bool | None = None,
                 token: str | None = None, precision='float16') -> VLMAdapter | FlorenceAdapter:
    """Load one model; close it before loading the next. Never installs packages."""
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key {model_key!r}; choose one of {sorted(MODEL_SPECS)}")
    quantize = model_key != "florence2" if load_in_4bit is None else load_in_4bit
    adapter_class = FlorenceAdapter if model_key == "florence2" else VLMAdapter
    extra={'precision':precision} if model_key=='florence2' else {}
    return adapter_class(model_key, revision=revision, max_new_tokens=max_new_tokens,
                         max_input_edge=max_input_edge, load_in_4bit=quantize, token=token,**extra)
