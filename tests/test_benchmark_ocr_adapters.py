"""Offline validation of native OCR formats; no model weights are downloaded."""

import json
import unittest

from benchmark_ocr_adapters import (
    generation_was_truncated,
    markup_to_text,
    parse_chandra_output,
    parse_dots_output,
)


class NativeOCRParsingTests(unittest.TestCase):
    def test_original_chandra_uses_1024_not_chandra2_1000(self):
        raw = '<div data-bbox="[256, 128, 768, 512]" data-label="Text"><p>Email: me@example.com</p></div>'
        result = parse_chandra_output(raw, (2000, 1000))
        self.assertTrue(result["parse_success"])
        self.assertEqual(result["text_regions"][0]["bbox_pixels"], [500, 125, 1500, 500])
        self.assertEqual(result["predicted_text"], "Email: me@example.com")

    def test_chandra_nested_blocks_not_duplicated_and_headers_retained(self):
        raw = '<div data-bbox="[0,0,512,512]" data-label="Page-Header"><div><p>Jane</p><p>Doe</p></div></div>'
        result = parse_chandra_output(raw, (1024, 1024))
        self.assertEqual(len(result["text_regions"]), 1)
        self.assertEqual(result["predicted_text"], "Jane Doe")

    def test_chandra_invalid_bbox_retains_unlocalized_text(self):
        result = parse_chandra_output('<div data-bbox="invalid" data-label="Text">A &amp; B</div>', (100, 100))
        self.assertTrue(result["parse_success"])
        self.assertIsNone(result["text_regions"][0]["bbox_pixels"])
        self.assertEqual(result["predicted_text"], "A & B")

    def test_chandra2_box_syntax_is_not_guessed(self):
        result = parse_chandra_output('<div data-bbox="0 0 1000 1000" data-label="Text">Name</div>', (100, 100))
        self.assertIsNone(result["text_regions"][0]["bbox_pixels"])

    def test_chandra_image_descriptions_are_not_ocr(self):
        result = parse_chandra_output('<div data-bbox="[0,0,1024,1024]" data-label="Image"><img alt="passport"/>A passport</div>', (100, 100))
        self.assertTrue(result["parse_success"])
        self.assertEqual(result["predicted_text"], "")

    def test_incomplete_chandra_is_parse_failure(self):
        result = parse_chandra_output('<div data-bbox="[0,0,100,100]" data-label="Text">Name', (100, 100))
        self.assertFalse(result["parse_success"])

    def test_dots_scales_from_actual_input_pixels(self):
        raw = json.dumps([{"bbox": [14, 28, 140, 112], "category": "Text", "text": "Phone: 123"}])
        result = parse_dots_output(raw, (2000, 1000), (280, 140))
        self.assertTrue(result["parse_success"])
        self.assertEqual(result["text_regions"][0]["bbox_pixels"], [100, 200, 1000, 800])

    def test_dots_html_table_preserves_cell_boundaries(self):
        raw = '```json\n[{"bbox":[0,0,280,140],"category":"Table","text":"<table><tr><td>Name</td><td>Jane</td></tr></table>"}]\n```'
        result = parse_dots_output(raw, (2000, 1000), (280, 140))
        self.assertEqual(result["predicted_text"], "Name Jane")
        self.assertEqual(result["text_regions"][0]["bbox_pixels"], [0, 0, 2000, 1000])

    def test_dots_missing_box_not_full_image(self):
        result = parse_dots_output('[{"category":"Text","text":"Jane"}]', (100, 100), (112, 112))
        self.assertIsNone(result["text_regions"][0]["bbox_pixels"])
        self.assertEqual(result["predicted_text"], "Jane")

    def test_dots_clips_box_and_rejects_nan_and_reversed(self):
        cells = [{"bbox": [-10, -10, 120, 120], "text": "Valid"},
                 {"bbox": [50, 50, 10, 10], "text": "Reversed"},
                 {"bbox": [0, 0, float("nan"), 90], "text": "NaN"}]
        result = parse_dots_output(json.dumps(cells), (100, 100), (100, 100))
        self.assertEqual(result["text_regions"][0]["bbox_pixels"], [0, 0, 100, 100])
        self.assertIsNone(result["text_regions"][1]["bbox_pixels"])
        self.assertIsNone(result["text_regions"][2]["bbox_pixels"])

    def test_dots_empty_detection_is_valid_but_malformed_is_failure(self):
        self.assertTrue(parse_dots_output("[]", (100, 100), (112, 112))["parse_success"])
        self.assertFalse(parse_dots_output('[{"text":', (100, 100), (112, 112))["parse_success"])
        self.assertFalse(parse_dots_output('[{"error":"unreadable"}]', (100, 100), (112, 112))["parse_success"])
        self.assertTrue(parse_dots_output('[{"category":"Picture"}]', (100, 100), (112, 112))["parse_success"])

    def test_untrusted_markup_is_inert(self):
        self.assertEqual(markup_to_text('<p>Email</p><script>alert(1)</script><p>me@example.com</p>'), "Email me@example.com")

    def test_token_budget_accounts_for_eos(self):
        self.assertFalse(generation_was_truncated([1, 2, 99], 3, [99]))
        self.assertTrue(generation_was_truncated([1, 2, 3], 3, [99]))
        self.assertFalse(generation_was_truncated([1], 3, [99]))


if __name__ == "__main__":
    unittest.main()
