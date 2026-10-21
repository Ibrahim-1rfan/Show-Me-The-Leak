"""GPU inference and native-layout parsing for the original Chandra and dots.ocr.

These are OCR extractors, not PII classifiers. ``extract`` returns native output,
ordered text regions, and coordinates on the upright, original image. The common
benchmark applies its fixed PII mapper afterwards. Importing this module never
downloads a model or imports torch; the pure parsers can be tested on a CPU.

Sources checked 2026-10-05:
* https://huggingface.co/datalab-to/chandra
* https://github.com/datalab-to/chandra/tree/v0.1.7
* https://github.com/datalab-to/chandra/blob/v0.1.7/chandra/output.py
* https://huggingface.co/rednote-hilab/dots.ocr
* https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/parser.py
* https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/layout_utils.py

Chandra 1 uses HTML data-bbox JSON arrays on a 0..1024 coordinate grid. Chandra
2 uses a DIFFERENT model, prompt and coordinate convention; do not upgrade its
package silently. dots.ocr boxes are pixels in the model's resized input, NOT
normalized coordinates. We derive its actual size from processor patch grids.
"""

from __future__ import annotations

import gc
import json
import math
import re
import time
from html import unescape
from html.parser import HTMLParser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


OCR_MODEL_SPECS = {
    "chandra": {
        "model_id": "datalab-to/chandra", "display_name": "Chandra 9B",
        "kind": "ocr", "profile": "chandra", "supports_4bit": True,
        "native_format": "HTML layout; 0..1024 boxes", "max_new_tokens": 8192,
        "source": "https://huggingface.co/datalab-to/chandra",
    },
    "dots_ocr": {
        "model_id": "rednote-hilab/dots.ocr", "display_name": "dots.ocr ~3B",
        "kind": "ocr", "profile": "dots", "supports_4bit": False,
        "native_format": "JSON layout; resized-input pixel boxes", "max_new_tokens": 8192,
        "source": "https://huggingface.co/rednote-hilab/dots.ocr",
    },
}

# Separate fresh GPU runtimes: dots pins an older Transformers and custom Flash
# Attention model code. Run one profile at a time; persist results between runs.
OCR_DEPENDENCY_PROFILES = {
    "chandra": {
        "pip": ["torch==2.8.0", "torchvision==0.23.0", "chandra-ocr==0.1.7",
                "transformers==4.57.1", "accelerate>=1.11.0", "bitsandbytes>=0.46.1"],
        "commands": [
            "python -m pip install torch==2.8.0 torchvision==0.23.0",
            'python -m pip install chandra-ocr==0.1.7 transformers==4.57.1 "accelerate>=1.11.0" "bitsandbytes>=0.46.1"',
        ],
        "notes": "Original 9B checkpoint, not Chandra 2. NF4 is optional; BF16 full precision needs more GPU memory. Restart kernel after installation.",
    },
    "dots": {
        "pip": ["transformers==4.56.1", "qwen-vl-utils>=0.0.14", "accelerate",
                "huggingface_hub", "Pillow", "packaging", "ninja"],
        "commands": [
            "python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128",
            'python -m pip install transformers==4.56.1 "qwen-vl-utils>=0.0.14" accelerate huggingface_hub Pillow packaging ninja',
            "python -m pip install flash-attn==2.8.0.post2 --no-build-isolation",
        ],
        "notes": "Fresh Linux GPU runtime, BF16-capable Ampere+ GPU (e.g. A100/L4); Flash Attention build needs CUDA toolchain. T4 is unsupported by this profile. Use load_in_4bit=False. Restart kernel after installation.",
    },
}


def _unfence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json|html|markdown)?\s*\n?(.*?)\n?```", text, re.S | re.I)
    return match.group(1).strip() if match else text


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip_depth += 1
        if tag in {"br", "p", "div", "tr", "td", "th", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "tr", "td", "th", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def markup_to_text(text: str) -> str:
    """Read HTML/Markdown as inert data; image alt descriptions are not OCR text."""
    # Keep ordinary '<' characters (e.g. inequalities) rather than parsing them
    # as markup unless a known HTML tag is present.
    if re.search(r"</?(?:div|p|table|tr|td|th|span|b|i|a|img|script|style|math|br)\b", text, re.I):
        parser = _TextParser()
        parser.feed(text)
        text = "".join(parser.parts)
    else:
        text = unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(\*\*|__|`)", "", text)
    return " ".join(text.split())


def _box_to_original(box, source_size, original_size):
    """Validate first; clip then floor/ceil to a half-open original-pixel box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(value, bool) for value in box):
        return None
    try:
        x0, y0, x1, y1 = map(float, box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        return None
    sw, sh = source_size
    ow, oh = original_size
    if min(sw, sh, ow, oh) <= 0:
        raise ValueError("Image coordinate dimensions must be positive")
    result = [max(0, min(ow, math.floor(x0 * ow / sw))),
              max(0, min(oh, math.floor(y0 * oh / sh))),
              max(0, min(ow, math.ceil(x1 * ow / sw))),
              max(0, min(oh, math.ceil(y1 * oh / sh)))]
    return result if result[2] > result[0] and result[3] > result[1] else None


def _payload(raw_text, regions, warnings, success=True, **metadata):
    return {"raw_text": raw_text, "text_regions": regions,
            "predicted_text": "\n".join(region["text"] for region in regions),
            "parse_warnings": warnings, "parse_success": success,
            "metadata": metadata}


class _ChandraLayoutParser(HTMLParser):
    """Read outer layout divs without duplicating nested content."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.active = None
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            if self.active is None and ("data-bbox" in attrs or "data-label" in attrs):
                self.active = {"attrs": attrs, "parts": []}
                self.depth = 1
                return
            if self.active is not None:
                self.depth += 1
        if self.active is not None:
            self.active["parts"].append(self.get_starttag_text())

    def handle_endtag(self, tag):
        if self.active is None:
            return
        if tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self.blocks.append(self.active)
                self.active = None
                return
        self.active["parts"].append(f"</{tag}>")

    def handle_data(self, data):
        if self.active is not None:
            # Escape data so a literal '<' does not become a second-stage tag.
            from html import escape
            self.active["parts"].append(escape(data))


def parse_chandra_output(raw_text: str, image_size: tuple[int, int]) -> dict:
    """Parse Chandra 9B HTML, retaining missing-box text without inventing boxes."""
    parser = _ChandraLayoutParser()
    parser.feed(_unfence(raw_text))
    warnings, regions = [], []
    if parser.active is not None:
        warnings.append("unclosed_layout_div")
    if not parser.blocks:
        return _payload(raw_text, [], warnings + ["no_complete_chandra_layout_blocks"], False)
    ignored = 0
    for index, block in enumerate(parser.blocks):
        category = block["attrs"].get("data-label", "Text")
        if category in {"Image", "Figure", "Blank-Page"}:
            ignored += 1
            continue
        text = markup_to_text("".join(block["parts"]))
        if not text:
            continue
        try:
            native_box = json.loads(block["attrs"].get("data-bbox", "null"))
        except (json.JSONDecodeError, TypeError):
            native_box = None
        box = _box_to_original(native_box, (1024, 1024), image_size)
        if box is None:
            warnings.append(f"region_{index}:missing_or_invalid_bbox")
        regions.append({"region_id": f"r{index}", "text": text,
                        "bbox_pixels": box, "category": category})
    return _payload(raw_text, regions, warnings, parser.active is None,
                    native_bbox_space="normalized_1024", ignored_nontext_blocks=ignored)


def parse_dots_output(raw_text: str, image_size: tuple[int, int], input_size: tuple[int, int]) -> dict:
    """dots.ocr normally emits a JSON list with HTML/Markdown text inside cells."""
    try:
        cells = json.loads(_unfence(raw_text))
    except (json.JSONDecodeError, TypeError):
        return _payload(raw_text, [], ["invalid_dots_layout_json"], False)
    # The official prompt says 'JSON object', although its parser expects a list.
    # Accept a single cell or explicit common layout wrappers without coercion.
    if isinstance(cells, dict):
        if "bbox" in cells:
            cells = [cells]
        else:
            cells = next((cells[key] for key in ("layout", "cells", "elements")
                          if isinstance(cells.get(key), list)), None)
    if not isinstance(cells, list):
        return _payload(raw_text, [], ["dots_layout_is_not_a_cell_list"], False)
    warnings, regions = [], []
    ignored = 0
    success = True
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            warnings.append(f"region_{index}:invalid_cell")
            success = False
            continue
        category = str(cell.get("category", "Text"))
        if category == "Picture":
            ignored += 1
            continue
        # An empty list is valid no-text output; an error/malformed cell is not.
        if "text" not in cell:
            warnings.append(f"region_{index}:missing_text_field")
            success = False
            continue
        text = cell["text"]
        if not isinstance(text, str):
            warnings.append(f"region_{index}:non_string_text")
            success = False
            continue
        text = markup_to_text(text)
        if not text:
            continue
        box = _box_to_original(cell.get("bbox"), input_size, image_size)
        if box is None:
            warnings.append(f"region_{index}:missing_or_invalid_bbox")
        regions.append({"region_id": f"r{index}", "text": text,
                        "bbox_pixels": box, "category": category})
    return _payload(raw_text, regions, warnings, success,
                    native_bbox_space="resized_input_pixels", input_size=list(input_size),
                    ignored_nontext_blocks=ignored)


def generation_was_truncated(token_ids, max_new_tokens: int, eos_token_ids) -> bool:
    """Reaching the budget is truncation only if generation did not end in EOS."""
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
    return len(token_ids) >= max_new_tokens and (not token_ids or int(token_ids[-1]) not in eos)


def _require_version(package: str, expected: str):
    try:
        actual = version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"Install the matching OCR profile first: missing {package}=={expected}") from exc
    if actual != expected:
        raise RuntimeError(f"This adapter requires {package}=={expected}, found {actual}. Use a fresh matching profile and restart its kernel.")


class _GPUOCRAdapter:
    def _configure(self, model_id, max_new_tokens, max_input_edge, load_in_4bit):
        if max_new_tokens < 1 or (max_input_edge is not None and max_input_edge < 28):
            raise ValueError("max_new_tokens must be positive and max_input_edge must be >=28")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_input_edge = max_input_edge
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None

    def _prepare(self, image):
        from PIL import Image, ImageOps
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = image.size
        if self.max_input_edge is not None:image.thumbnail((self.max_input_edge, self.max_input_edge), Image.Resampling.LANCZOS)
        return image, original_size

    def _generate(self, image, prompt):
        import torch
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[rendered], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt")
        grid = inputs.get("image_grid_thw")
        if grid is None or len(grid) != 1:
            raise RuntimeError("Cannot determine the single image's processed size; refusing guessed box scaling")
        patch = self.processor.image_processor.patch_size
        input_size = (int(grid[0][2]) * patch, int(grid[0][1]) * patch)
        inputs = inputs.to(self.model.device)
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        # Include the chat turn terminator if the tokenizer defines it.
        vocabulary = self.processor.tokenizer.get_vocab()
        if "<|im_end|>" in vocabulary and vocabulary["<|im_end|>"] not in eos_ids:
            eos_ids.append(vocabulary["<|im_end|>"])
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False, "use_cache": True}
        if eos_ids:
            kwargs["eos_token_id"] = eos_ids
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        token_ids = generated[0, inputs.input_ids.shape[1]:].detach().cpu().tolist()
        raw = self.processor.batch_decode([token_ids], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        metadata = {"model_id": self.model_id, "generation_seconds": elapsed,
                    "generated_tokens": len(token_ids), "max_new_tokens": self.max_new_tokens,
                    "truncated": generation_was_truncated(token_ids, self.max_new_tokens, eos_ids),
                    "input_size": list(input_size), "load_in_4bit": self.load_in_4bit,
                    "model_revision": getattr(self, "revision", None),
                    "transformers_version": version("transformers"),
                    "torch_version": torch.__version__,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "dtype": str(self.model.dtype),
                    "max_input_edge": self.max_input_edge}
        return raw, input_size, metadata

    def close(self):
        """Release this model before loading another profile/model."""
        self.model = None
        self.processor = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ChandraAdapter(_GPUOCRAdapter):
    def __init__(self, model_id="datalab-to/chandra", max_new_tokens=8192,
                 max_input_edge=1280, load_in_4bit=True, revision=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported Chandra options: {sorted(kwargs)}")
        if model_id != "datalab-to/chandra":
            raise ValueError("This parser targets the original datalab-to/chandra 9B checkpoint")
        self._configure(model_id, max_new_tokens, max_input_edge, load_in_4bit)
        _require_version("chandra-ocr", "0.1.7")
        _require_version("transformers", "4.57.1")
        import torch
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        from chandra.prompts import PROMPT_MAPPING
        if not torch.cuda.is_available():
            raise RuntimeError("Chandra benchmark inference requires a CUDA GPU runtime")
        from .hardware import native_dtype
        dtype = native_dtype(torch)
        load_kwargs = {"device_map": "auto", "dtype": dtype, "attn_implementation": "sdpa"}
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, revision=revision, **load_kwargs)
            self.model.eval()
            self.processor = Qwen3VLProcessor.from_pretrained(model_id, revision=revision)
            self.prompt = PROMPT_MAPPING["ocr_layout"]
            self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        except Exception:
            self.close()
            raise

    def extract(self, image):
        image, original_size = self._prepare(image)
        raw, _, metadata = self._generate(image, self.prompt)
        result = parse_chandra_output(raw, original_size)
        result["metadata"].update(metadata)
        return result


# Preserve the original model's layout task and categories. It recognizes text;
# taxonomy names, ground-truth labels and PII instructions are never supplied.
DOTS_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
   - Picture: For the 'Picture' category, the text field should be omitted.
   - Formula: Format its text as LaTeX.
   - Table: Format its text as HTML.
   - All Others (Text, Title, etc.): Format their text as Markdown.
4. Constraints:
   - The output text must be the original text from the image, with no translation.
   - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.
"""


def load_dots_processor(model_path):
    """Preserve dots' image preprocessing/template while satisfying TF 4.56.1.

    The checkpoint's DotsVLProcessor constructor drops video_processor when
    calling its Qwen2.5 parent. Adding a kwarg to AutoProcessor cannot fix it.
    Assemble the same parent explicitly; the video component is unused for our
    still-image task. Never change the checkpoint's custom <|imgpad|> token.
    """
    from transformers import (AutoTokenizer, Qwen2VLImageProcessor,
                              Qwen2_5_VLProcessor, Qwen2VLVideoProcessor)
    model_path = Path(model_path)
    template = json.loads((model_path / 'chat_template.json').read_text(encoding='utf-8'))['chat_template']
    if not isinstance(template, str) or '<|imgpad|>' not in template:
        raise ValueError('Unexpected dots chat template; refusing a substituted image format')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(str(model_path))
    processor = Qwen2_5_VLProcessor(
        image_processor=image_processor, tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(), chat_template=template)
    processor.image_token = '<|imgpad|>'
    processor.image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    if processor.image_token_id != 151665:
        raise ValueError('Unexpected dots image token ID; expected 151665')
    return processor


def check_dots_image_processor(processor):
    """Exercise the real image/token path before allocating model weights."""
    from PIL import Image
    image = Image.new('RGB', (56, 56), 'white')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': 'Read the image.'}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt')
    grid = inputs['image_grid_thw'][0]
    expected = int(grid.prod()) // processor.image_processor.merge_size ** 2
    actual = int((inputs['input_ids'] == processor.image_token_id).sum())
    if expected <= 0 or actual != expected:
        raise RuntimeError(f'dots image token/grid mismatch: {actual} tokens for {expected} patches')


class DotsOCRAdapter(_GPUOCRAdapter):
    def __init__(self, model_id="rednote-hilab/dots.ocr", max_new_tokens=8192,
                 max_input_edge=1280, load_in_4bit=False, revision=None,
                 local_model_dir="benchmark_model_cache/DotsOCR", **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported dots.ocr options: {sorted(kwargs)}")
        if load_in_4bit:
            raise ValueError("The dots.ocr profile uses the official BF16/custom-vision path. Set load_in_4bit=False.")
        self._configure(model_id, max_new_tokens, max_input_edge, load_in_4bit)
        _require_version("transformers", "4.56.1")
        import torch
        from .downloads import snapshot_with_retry
        from transformers import AutoModelForCausalLM
        if (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8
                or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("dots.ocr needs a BF16-capable GPU (A100/L4/Ampere+), not a T4, for this official Flash Attention profile")
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install flash-attn==2.8.0.post2 --no-build-isolation in the dots profile first") from exc
        local_dir = Path(local_model_dir)
        if "." in local_dir.name:
            raise ValueError("dots.ocr custom code needs a model directory name without periods, such as DotsOCR")
        if not revision:raise ValueError('A pinned dots revision is required')
        self.revision = revision
        model_path = snapshot_with_retry(repo_id=model_id, revision=self.revision, local_dir=str(local_dir),local_files_only=True,
            allow_patterns=['*.safetensors','*.json','*.py','*.txt','*.model','*.jinja'])
        try:
            self.processor = load_dots_processor(model_path)
            check_dots_image_processor(self.processor)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16, device_map="auto")
            self.model.eval()
        except Exception:
            self.close()
            raise

    def extract(self, image):
        image, original_size = self._prepare(image)
        raw, input_size, metadata = self._generate(image, DOTS_LAYOUT_PROMPT)
        result = parse_dots_output(raw, original_size, input_size)
        result["metadata"].update(metadata)
        result["metadata"]["processor_compatibility"] = "explicit_qwen25_components_with_dots_image_token"
        return result


def create_ocr_adapter(model_key: str, **kwargs):
    if model_key in {"chandra", "chandra9b", "chandra_9b"}:
        return ChandraAdapter(**kwargs)
    if model_key in {"dots_ocr", "dots", "dots.ocr"}:
        return DotsOCRAdapter(**kwargs)
    raise ValueError(f"Unknown OCR model {model_key!r}; choose chandra or dots_ocr")
