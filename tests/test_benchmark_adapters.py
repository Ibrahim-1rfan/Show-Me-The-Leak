"""Offline adapter-contract tests. No weights, GPU, or synthetic quality scores."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import benchmark_adapters as adapters


class FakeRow(list):
    def tolist(self):
        return list(self)


class FakeBatch(dict):
    def to(self, *args):
        return self


class FakeOutput(list):
    def __init__(self, rows):
        super().__init__([FakeRow(row) for row in rows])
        self.shape = (len(rows), len(rows[0]))


class AdapterTests(unittest.TestCase):
    def test_resize_preserves_frame_and_does_not_upscale(self):
        image = Image.new("RGBA", (1200, 600), (0, 0, 0, 255))
        resized = adapters.prepare_image(image, 300)
        self.assertEqual(resized.size, (300, 150))
        self.assertEqual(resized.mode, "RGB")
        self.assertEqual(image.size, (1200, 600))
        self.assertEqual(adapters.prepare_image(resized, 1200).size, resized.size)
        with self.assertRaises(ValueError):
            adapters.prepare_image(image, 0)

    def test_florence_quads_become_original_pixel_envelopes(self):
        native = {"<OCR_WITH_REGION>": {"quad_boxes": [[10, 20, 40, 10, 50, 30, 20, 40]], "labels": ["Name"]}}
        result = adapters.florence_regions(native, (100, 50), (1000, 500))
        self.assertEqual(result[0]["bbox_pixels"], [100, 100, 500, 400])
        self.assertEqual(result[0]["quad_pixels"], [100, 200, 400, 100, 500, 300, 200, 400])
        self.assertEqual(result[0]["text"], "Name")

    def test_malformed_native_ocr_is_not_an_empty_success(self):
        empty = {"<OCR_WITH_REGION>": {"quad_boxes": [], "labels": []}}
        self.assertEqual(adapters.florence_regions(empty, (100, 50), (100, 50)), [])
        bad = [
            {}, {"<OCR_WITH_REGION>": "not parsed"},
            {"<OCR_WITH_REGION>": {"quad_boxes": [], "labels": ["unpaired"]}},
            {"<OCR_WITH_REGION>": {"quad_boxes": [[0, 0, 1, 1, 2, 2, float("nan"), 3]], "labels": ["bad"]}},
            {"<OCR_WITH_REGION>": {"quad_boxes": [[2, 2] * 4], "labels": ["zero area"]}},
        ]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                adapters.florence_regions(payload, (100, 50), (100, 50))

    def test_truncation_distinguishes_natural_and_forced_eos(self):
        natural = adapters.generation_status([10, 11, 2], 3, eos_token_id=[2, 3])
        self.assertTrue(natural["ended_with_eos"])
        self.assertTrue(natural["hit_token_limit"])
        self.assertFalse(natural["truncated"])
        self.assertTrue(adapters.generation_status([10, 11, 12], 3, eos_token_id=2)["truncated"])
        forced = adapters.generation_status([10, 11, 2], 3, eos_token_id=2, forced_eos_token_id=2)
        self.assertTrue(forced["truncated"])
        self.assertTrue(forced["forced_eos_at_limit"])
        early = adapters.generation_status([10, 2], 10, eos_token_id=2, forced_eos_token_id=2)
        self.assertFalse(early["truncated"])

    def test_florence_uses_native_task_parser_not_taxonomy_json(self):
        adapter = adapters.FlorenceAdapter.__new__(adapters.FlorenceAdapter)
        adapter.max_input_edge = 100
        adapter.max_new_tokens = 4
        adapter.dtype = "float16"
        adapter.model_key = "florence2"
        adapter.model_id = adapters.MODEL_SPECS["florence2"]["model_id"]
        adapter.spec = adapters.MODEL_SPECS["florence2"]
        adapter.revision = "fixture"
        adapter.load_in_4bit = False
        adapter.transformers = SimpleNamespace(__version__="fixture")
        cuda = SimpleNamespace(get_device_name=lambda: "fixture GPU")
        adapter.torch = SimpleNamespace(inference_mode=nullcontext, __version__="fixture", cuda=cuda)
        adapter.model = SimpleNamespace(
            device="cuda:0", config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=1024)),
            language_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2, forced_eos_token_id=2)),
            generate=MagicMock(return_value=FakeOutput([[2, 10, 11, 12, 2]])),
        )
        processor = MagicMock()
        processor.return_value = FakeBatch(input_ids="tokens", pixel_values="pixels")
        processor.batch_decode.return_value = ["<s>NAME<loc_1></s>"]
        processor.post_process_generation.return_value = {
            "<OCR_WITH_REGION>": {"quad_boxes": [[10, 5, 30, 5, 30, 15, 10, 15]], "labels": ["NAME"]},
        }
        adapter.processor = processor
        result = adapter.infer(Image.new("RGB", (400, 200)), "Ignored taxonomy JSON request")
        self.assertEqual(processor.call_args.kwargs["text"], "<OCR_WITH_REGION>")
        self.assertEqual(processor.call_args.kwargs["images"].size, (100, 50))
        processor.post_process_generation.assert_called_once_with(
            "<s>NAME<loc_1></s>", task="<OCR_WITH_REGION>", image_size=(100, 50),
        )
        self.assertEqual(result["predicted_text"], "NAME")
        self.assertEqual(result["text_regions"][0]["bbox_pixels"], [40, 20, 120, 60])
        self.assertTrue(result["metadata"]["truncated"])
        self.assertEqual(result["metadata"]["gpu_name"], "fixture GPU")

    def test_non_os_load_errors_release_both_adapter_types(self):
        for adapter_class, key in [(adapters.VLMAdapter, "qwen3_vl"), (adapters.FlorenceAdapter, "florence2")]:
            with self.subTest(key=key):
                torch = MagicMock()
                torch.cuda.is_available.return_value = True
                torch.cuda.is_bf16_supported.return_value = False
                tf = MagicMock()
                loader = getattr(tf, adapters.MODEL_SPECS[key]["class_name"])
                loader.from_pretrained.side_effect = RuntimeError("simulated CUDA out of memory")
                instance = adapter_class.__new__(adapter_class)
                with patch.object(adapters, "_runtime", return_value=(torch, tf)):
                    with self.assertRaisesRegex(RuntimeError, "out of memory"):
                        instance.__init__(key, load_in_4bit=False)
                self.assertIsNone(instance.model)
                self.assertIsNone(instance.processor)
                self.assertEqual(instance._hub_kwargs, {})
                torch.cuda.empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
