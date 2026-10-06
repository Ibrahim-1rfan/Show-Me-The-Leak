"""Offline integration checks using tiny synthetic images and fake adapters.

These test plumbing, independence of annotations, and failure accounting, not
model detection quality. No weights, GPU, or network are needed.
"""

import ast
import contextlib
import io
import json
import math
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmark_metrics import build_comparison
from benchmark_workflow import (
    attach_references,
    collect_runs,
    load_references,
    make_detector,
    ocr_to_pii,
    prepare_sample,
    run_one_model,
    scope_records,
    stratified_sample,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = {
    "a0_safe": {"vispr_native_group": "Safe", "description": "Safe", "taxonomy_category": "Safe"},
    "a9_face_complete": {"vispr_native_group": "Personal Description", "description": "Face", "taxonomy_category": "Biometric"},
    "a31_passport": {"vispr_native_group": "Documents", "description": "Passport", "taxonomy_category": "Document"},
    "a90_email": {"vispr_native_group": "Contact", "description": "Email", "taxonomy_category": "Contact"},
    "a19_name_full": {"vispr_native_group": "Name", "description": "Full name", "taxonomy_category": "Identity"},
    "a20_name_first": {"vispr_native_group": "Name", "description": "First name", "taxonomy_category": "Identity"},
}


def notebook_helpers():
    """Load the real notebook pipeline without evaluating installs or inference."""
    namespace = {"Path": Path, "Image": Image, "ImageDraw": ImageDraw, "ImageOps": ImageOps,
                 "json": json, "math": math, "re": re, "FORMAT_RETRIES": 0,
                 "generate_qwen": lambda image, prompt: ("", False)}
    wanted = {"read_image", "write_json", "pixel_box", "redact_and_draw", "evaluate_labels",
              "build_prompt", "parse_response", "detect_pii", "process_record"}
    notebook = json.loads((ROOT / "Model_Testing.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or "install" in cell.get("metadata", {}).get("tags", []):
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_helpers", "exec"), namespace)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EXPLICIT_TEXT_ONLY" for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook_helpers", "exec"), namespace)
    missing = wanted - namespace.keys()
    if missing:
        raise AssertionError(f"Missing required notebook pipeline helpers: {sorted(missing)}")
    return namespace


def create_records(root, count_per_stratum=30):
    records = []
    train = root / "train"
    train.mkdir()
    for index in range(count_per_stratum * 2):
        image_id = f"synthetic_{index:03d}"
        image_path, annotation_path = train / f"{image_id}.png", train / f"{image_id}.json"
        label = "a31_passport" if index < count_per_stratum else "a9_face_complete"
        Image.new("RGB", (12, 8), (index, 200, 230)).save(image_path)
        annotation_path.write_text(json.dumps({"id": image_id, "labels": [label]}), encoding="utf-8")
        records.append({"id": image_id, "image_path": image_path, "annotation_path": annotation_path,
                        "labels": [label], "stratum": "document" if index < count_per_stratum else "general"})
    return records


class SampleTests(unittest.TestCase):
    def test_at_least_fifty_images_balanced_reproducible_and_not_silently_shrunk(self):
        records = [{"id": f"image{i:03}", "labels": ["a31_passport"] if i < 40 else ["a9_face_complete"]}
                   for i in range(80)]
        first = stratified_sample(records, TAXONOMY)
        second = stratified_sample(list(reversed(records)), TAXONOMY)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual(Counter(row["stratum"] for row in first), {"document": 25, "general": 25})
        self.assertEqual(len(stratified_sample(records, TAXONOMY, size=51)), 51)
        with self.assertRaises(ValueError):
            stratified_sample(records, TAXONOMY, size=49)
        with self.assertRaises(ValueError):
            stratified_sample(records[:50], TAXONOMY)

    def test_real_tiny_images_fingerprint_and_reference_template_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = create_records(root)
            selected, sample, folder = prepare_sample(records, TAXONOMY, root, root / "outputs")
            self.assertEqual(len(selected), 50)
            self.assertEqual(len(sample["images"]), 50)
            self.assertTrue(all(row["image_size"] == [12, 8] for row in sample["images"]))
            template = folder / "reference_template.json"
            template.write_text("preserve manual working copy", encoding="utf-8")
            _, repeated, repeated_folder = prepare_sample(list(reversed(records)), TAXONOMY, root, root / "outputs")
            self.assertEqual(repeated["sample_fingerprint"], sample["sample_fingerprint"])
            self.assertEqual(repeated_folder, folder)
            self.assertEqual(template.read_text(encoding="utf-8"), "preserve manual working copy")
            Image.new("RGB", (12, 8), "red").save(selected[0]["image_path"])
            _, changed, changed_folder = prepare_sample(records, TAXONOMY, root, root / "outputs")
            self.assertNotEqual(changed["sample_fingerprint"], sample["sample_fingerprint"])
            self.assertNotEqual(changed_folder, folder)


class OCRBridgeTests(unittest.TestCase):
    def test_native_ocr_regions_do_not_require_a_json_model_response(self):
        native = {"raw_text": "<ocr>native format</ocr>", "text_regions": [
            {"text": "contact@example.test", "bbox_pixels": [1, 2, 10, 7]}]}
        detections = ocr_to_pii(native, (12, 8), TAXONOMY)
        self.assertEqual([d["attribute_id"] for d in detections], ["a90_email"])
        self.assertEqual(detections[0]["bbox_1000"], [83, 250, 834, 875])
        self.assertEqual(detections[0]["evidence_source"], "explicit_text")
        self.assertEqual(ocr_to_pii({"raw_text": "contact@example.test", "text_regions": []}, (12, 8), TAXONOMY), [])

    def test_invalid_or_truncated_ocr_is_error_not_empty_prediction(self):
        invalid_outputs = [
            {"text_regions": "invalid"},
            {"text_regions": [{"text": "contact@example.test", "bbox_pixels": [0, 0, 13, 8]}]},
            {"text_regions": [{"text": 123, "bbox_pixels": None}]},
            {"text_regions": [], "metadata": {"truncated": True}},
            {"text_regions": [], "parse_success": False},
        ]
        for native in invalid_outputs:
            class Adapter:
                def extract(self, image):
                    return native
            with self.subTest(native=native):
                result = make_detector(Adapter(), "florence2", None)(Image.new("RGB", (12, 8)), TAXONOMY)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["detections"])

    def test_first_name_is_not_automatically_a_full_name(self):
        detections = ocr_to_pii({"text_regions": [{"text": "First name: Example", "bbox_pixels": None}]}, (12, 8), TAXONOMY)
        self.assertEqual([d["attribute_id"] for d in detections], ["a20_name_first"])
        self.assertIsNone(detections[0]["bbox_1000"])


class ReferenceTests(unittest.TestCase):
    def test_manual_reference_scope_controls_localization_without_changing_label_predictions(self):
        source = {"image_id": "image", "model_key": "qwen3_vl", "stratum": "general", "status": "ok",
                  "ground_truth_attributes": ["a9_face_complete", "a90_email"],
                  "predicted_attributes": ["a9_face_complete", "a90_email"],
                  "detections": [{"attribute_id": attr, "bbox_pixels": [0, 0, 5, 5]}
                                 for attr in ["a9_face_complete", "a90_email"]]}
        refs = {"image": {"boxes_complete": True, "box_attribute_scope": ["a9_face_complete"],
                          "ground_truth_boxes": [{"attribute_id": "a9_face_complete", "bbox_pixels": [0, 0, 5, 5]}]}}
        attached = attach_references([source], refs)
        self.assertEqual(attached[0]["predicted_attributes"], source["predicted_attributes"])
        self.assertEqual(len(source["detections"]), 2)
        self.assertEqual(len(attached[0]["detections"]), 1)
        scoped = scope_records(attached, ["a90_email"])
        report = build_comparison(scoped, ["a90_email"])
        score = next(row for row in report["summaries"] if row["stratum"] == "all" and row["scope"] == "all_successful")
        self.assertEqual(score["micro_f1"], 1)
        self.assertEqual(score["n_localization_images"], 0)

    def test_reference_fingerprint_and_bounds_are_validated(self):
        sample = {"sample_fingerprint": "expected", "images": [{"image_id": "image", "image_size": [12, 8]}]}
        ref = {"image_size": [12, 8], "boxes_complete": True, "box_attribute_scope": ["a90_email"],
               "ground_truth_boxes": [{"attribute_id": "a90_email", "bbox_pixels": [0, 0, 12, 8]}], "reference_text": "example"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "references.json"
            payload = {"sample_fingerprint": "expected", "images": {"image": ref}}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn("image", load_references(path, sample, TAXONOMY))
            payload["sample_fingerprint"] = "other"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_references(path, sample, TAXONOMY)
            payload["sample_fingerprint"] = "expected"
            ref["ground_truth_boxes"][0]["bbox_pixels"] = [0, 0, 13, 8]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_references(path, sample, TAXONOMY)


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = notebook_helpers()

    def test_native_ocr_geometry_is_preserved_for_iou_and_redaction(self):
        image = Image.new('RGB', (2000, 1000), 'white')
        native = {'text_regions': [{'text': 'Email: sample@example.test',
                                   'bbox_pixels': [101.5, 51.5, 199.5, 99.5]}]}
        detections = ocr_to_pii(native, image.size, TAXONOMY)
        redacted, _, enriched = self.helpers['redact_and_draw'](image, detections, 0.0)
        self.assertEqual(enriched[0]['bbox_pixels'], [101.5, 51.5, 199.5, 99.5])
        self.assertEqual(enriched[0]['redaction_box_pixels'], [101, 51, 200, 100])
        self.assertEqual(redacted.getpixel((101, 51)), (0, 0, 0))
        self.assertEqual(redacted.getpixel((200, 100)), (255, 255, 255))

    def test_annotations_are_read_after_blind_inference_and_redaction(self):
        state = {"inferred": False, "closed": False, "prompts": []}

        class GuardedRecord(dict):
            def __getitem__(self, key):
                if key == "labels" and not state["inferred"]:
                    raise AssertionError("Ground truth read before blind inference")
                return super().__getitem__(key)

        class Adapter:
            def infer(self, image, prompt):
                self.assert_image = image.size
                state["prompts"].append(prompt)
                state["inferred"] = True
                return {"raw_text": json.dumps({"detections": [
                    {"attribute_id": "a90_email", "bbox_1000": [0, 0, 500, 500],
                     "location": "upper left", "evidence_source": "explicit_text"}]})}

            def close(self):
                state["closed"] = True

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            selected = [GuardedRecord(create_records(root, 1)[0])]
            run_dir, reports = run_one_model("qwen3_vl", lambda key: Adapter(), selected, TAXONOMY,
                                            root / "runs", {"protocol_id": "test"},
                                            self.helpers["process_record"], self.helpers["detect_pii"], audit=False)
            self.assertEqual(reports[0]["status"], "ok")
            self.assertTrue(state["closed"])
            self.assertEqual(len(state["prompts"]), 1)
            self.assertNotIn(str(selected[0]["annotation_path"]), state["prompts"][0])
            self.assertTrue((run_dir / selected[0]["id"] / "redacted.png").exists())
            self.assertEqual(reports[0]["ground_truth_attributes"], ["a31_passport"])
            self.assertEqual(reports[0]["predicted_attributes"], ["a90_email"])

    def test_loading_failure_is_distinct_from_a_model_not_run(self):
        def unavailable(model_key):
            raise RuntimeError("synthetic unavailable model")

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            selected = create_records(root, 1)
            run_one_model("chandra", unavailable, selected, TAXONOMY, root / "runs", {"protocol_id": "test"}, None, None)
            records, provenance = collect_runs(root / "runs", ["chandra", "dots_ocr"], "test")
            report = build_comparison(records, TAXONOMY, ["chandra", "dots_ocr"], [r["id"] for r in selected])
            rows = {r["model_key"]: r for r in report["coverage"] if r["stratum"] == "all"}
            self.assertEqual(rows["chandra"]["failed_images"], 2)
            self.assertEqual(rows["dots_ocr"]["failed_images"], 0)
            self.assertEqual(rows["dots_ocr"]["not_attempted_images"], 2)
            self.assertEqual(report["common_image_ids"], [])
            self.assertIsNone(next(p for p in provenance if p["model_key"] == "dots_ocr")["run_directory"])

    def test_cleanup_failure_preserves_finished_results(self):
        class Adapter:
            def extract(self, image):
                return {"text_regions": [], "predicted_text": ""}

            def close(self):
                raise RuntimeError("synthetic cleanup failure")

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            selected = create_records(root, 1)
            run_dir, reports = run_one_model("florence2", lambda key: Adapter(), selected, TAXONOMY,
                                            root / "runs", {"protocol_id": "test"},
                                            self.helpers["process_record"], self.helpers["detect_pii"], audit=False)
            self.assertEqual(len(reports), 2)
            self.assertTrue(all(r["status"] == "ok" for r in reports))
            self.assertEqual(len(json.loads((run_dir / "results.json").read_text(encoding="utf-8"))), 2)
            manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertIn("cleanup", json.dumps(manifest).lower())


if __name__ == "__main__":
    unittest.main()
