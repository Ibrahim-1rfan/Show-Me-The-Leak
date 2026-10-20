"""Regression tests for dots video_processor and Florence optional imports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark_adapters import florence_eager_imports, florence_eager_loader
from benchmark_ocr_adapters import load_dots_processor


class FlorenceImportTests(unittest.TestCase):
    def scan(self, source, name='modeling_florence2.py'):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(source, encoding='utf-8')
            return florence_eager_imports(path, ['torch', 'flash_attn', 'einops'])

    def test_guarded_flash_import_is_optional_and_other_dependencies_remain(self):
        self.assertEqual(self.scan('if is_flash_attn_2_available():\n    from flash_attn.bert_padding import pad_input\n'), ['torch', 'einops'])

    def test_unconditional_or_other_model_import_is_never_suppressed(self):
        source = 'if is_flash_attn_2_available():\n    import flash_attn\n'
        self.assertIn('flash_attn', self.scan(source + 'from flash_attn import flash_attn_func\n'))
        self.assertIn('flash_attn', self.scan(source, 'modeling_dots_vision.py'))
        self.assertIn('flash_attn', self.scan('if True:\n    import flash_attn\n'))
        self.assertIn('flash_attn', self.scan('if is_flash_attn_2_available():\n    pass\nelse:\n    import flash_attn\n'))

    def test_loader_restores_scanner_even_after_error(self):
        original = Mock(return_value=['torch'])
        module = SimpleNamespace(get_imports=original)
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(dynamic_module_utils=module)}):
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                with florence_eager_loader():
                    self.assertIsNot(module.get_imports, original)
                    self.assertEqual(module.get_imports('other.py'), ['torch'])
                    raise RuntimeError('load failed')
            self.assertIs(module.get_imports, original)


class DotsProcessorTests(unittest.TestCase):
    def build(self, path, token_id=151665):
        image = SimpleNamespace(patch_size=14, temporal_patch_size=1, merge_size=2)
        tokenizer = SimpleNamespace(convert_tokens_to_ids=Mock(return_value=token_id))
        video = object()
        def processor(**kwargs):
            if kwargs.get('video_processor') is None:
                raise TypeError('Received a NoneType for argument video_processor')
            return SimpleNamespace(**kwargs)
        tf = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            Qwen2VLImageProcessor=SimpleNamespace(from_pretrained=Mock(return_value=image)),
            Qwen2_5_VLVideoProcessor=Mock(return_value=video), Qwen2_5_VLProcessor=processor)
        with patch.dict(sys.modules, {'transformers': tf}):
            result = load_dots_processor(path)
        return result, image, video

    def test_explicit_components_keep_native_template_grid_and_image_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            template = '<|user|><|img|><|imgpad|><|endofimg|><|assistant|>'
            (path / 'chat_template.json').write_text(json.dumps({'chat_template': template}))
            result, image, video = self.build(path)
            self.assertIs(result.video_processor, video)
            self.assertIs(result.image_processor, image)
            self.assertEqual(result.chat_template, template)
            self.assertEqual(result.image_token, '<|imgpad|>')
            self.assertEqual(result.image_token_id, 151665)
            with self.assertRaisesRegex(ValueError, 'image token ID'):
                self.build(path, 99)

    def test_unrecognized_template_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'chat_template.json').write_text(json.dumps({'chat_template': '<|image_pad|>'}))
            with self.assertRaisesRegex(ValueError, 'chat template'):
                self.build(path)


if __name__ == '__main__':
    unittest.main()
