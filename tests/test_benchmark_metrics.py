"""Synthetic metric arithmetic checks; these do not simulate model quality."""

import json
import math
import unittest

from benchmark_metrics import (
    box_iou,
    build_comparison,
    label_metrics,
    localization_metrics,
    text_metrics,
)


LABELS = ["a0_safe", "face", "name", "email"]


def record(model="first", image="image1", truth=None, prediction=None,
           stratum="document", status="ok", **extra):
    return {"model_key": model, "image_id": image, "stratum": stratum,
            "status": status, "ground_truth_attributes": truth if truth is not None else ["face", "name"],
            "predicted_attributes": prediction if prediction is not None else ["face"], **extra}


def box(attribute="face", bounds=None):
    return {"attribute_id": attribute, "bbox_pixels": [0, 0, 10, 10] if bounds is None else bounds}


def summary(report, model="first", scope="all_successful", stratum="all"):
    return next(row for row in report["summaries"] if row["model_key"] == model
                and row["scope"] == scope and row["stratum"] == stratum)


class LabelMetricTests(unittest.TestCase):
    def test_unique_attributes_safe_exclusion_and_unannotated_predictions(self):
        metrics = label_metrics(["a0_safe", "face", "face", "name"],
                                ["face", "face", "email"], LABELS)
        self.assertEqual(metrics["matched_attributes"], ["face"])
        self.assertEqual(metrics["missed_attributes"], ["name"])
        self.assertEqual(metrics["unannotated_attributes"], ["email"])
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["f1"]), (0.5, 0.5, 0.5))

    def test_empty_ground_truth_and_predictions_have_explicit_convention(self):
        metrics = label_metrics(["a0_safe"], [], LABELS)
        self.assertTrue(metrics["exact_match"])
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(metrics["jaccard"], 1)
        self.assertEqual(label_metrics([], ["face"], LABELS)["unannotated_count"], 1)

    def test_unknown_labels_and_invalid_container_fail_loudly(self):
        with self.assertRaises(ValueError):
            label_metrics(["unknown"], [], LABELS)
        with self.assertRaises(ValueError):
            label_metrics([], "face", LABELS)


class LocalizationTests(unittest.TestCase):
    def test_exclusive_box_iou_geometry(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_duplicate_and_wrong_label_boxes_count_as_unmatched(self):
        metrics = localization_metrics([box(), box(), box("name")], [box()], ground_truth_complete=True)
        self.assertEqual((metrics["localization_tp"], metrics["localization_fp"], metrics["localization_fn"]), (1, 2, 0))
        self.assertEqual(metrics["localization_f1"], 0.5)
        self.assertEqual(len(metrics["matches"]), 1)

    def test_augmenting_matching_recovers_match_missed_by_greedy(self):
        truth = [box(bounds=[0, 0, 10, 10]), box(bounds=[8, 0, 18, 10])]
        prediction = [box(bounds=[0, 0, 18, 10]), box(bounds=[0, 0, 10, 10])]
        metrics = localization_metrics(prediction, truth, ground_truth_complete=True)
        self.assertEqual(metrics["localization_tp"], 2)
        self.assertEqual(metrics["localization_f1"], 1)
        self.assertEqual({item["ground_truth_index"] for item in metrics["matches"]}, {0, 1})
        self.assertEqual({item["prediction_index"] for item in metrics["matches"]}, {0, 1})

    def test_missing_localization_is_not_a_free_correct_detection(self):
        metrics = localization_metrics([{"attribute_id": "face", "bbox_pixels": None}], [box()],
                                       ground_truth_complete=True)
        self.assertEqual((metrics["localization_tp"], metrics["localization_fp"], metrics["localization_fn"]), (0, 1, 1))
        self.assertEqual(metrics["unlocalized_prediction_count"], 1)
        self.assertIsNone(metrics["mean_matched_iou"])

    def test_missing_or_partial_truth_does_not_produce_fake_iou(self):
        for truth, complete in [(None, True), ([box()], False), ([], False)]:
            with self.subTest(truth=truth, complete=complete):
                metrics = localization_metrics([box()], truth, ground_truth_complete=complete)
                self.assertFalse(metrics["localization_available"])
                self.assertIsNone(metrics["localization_f1"])
                self.assertIsNone(metrics["mean_matched_iou"])

    def test_complete_empty_reference_counts_predictions_as_false_positives(self):
        metrics = localization_metrics([box()], [], ground_truth_complete=True)
        self.assertTrue(metrics["localization_available"])
        self.assertEqual(metrics["localization_fp"], 1)
        self.assertEqual(metrics["localization_fn"], 0)

    def test_invalid_geometry_and_threshold_are_rejected(self):
        for bounds in [[0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, math.nan, 1], [False, 0, 1, 1]]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                box_iou(bounds, [0, 0, 1, 1])
        for threshold in [0, -1, 1.1, math.nan, True]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                localization_metrics([], [], threshold, True)


class TextMetricTests(unittest.TestCase):
    def test_rouge_l_uses_sequence_not_set_overlap(self):
        metrics = text_metrics("three two one", "one two three")
        self.assertEqual(metrics["lcs_tokens"], 1)
        self.assertAlmostEqual(metrics["rouge_l_f1"], 1 / 3)
        self.assertAlmostEqual(metrics["wer"], 2 / 3)

    def test_character_and_word_edits(self):
        metrics = text_metrics("cat sat", "cat bat")
        self.assertEqual(metrics["character_edits"], 1)
        self.assertEqual(metrics["word_edits"], 1)
        self.assertAlmostEqual(metrics["cer"], 1 / 7)
        self.assertEqual(metrics["wer"], 0.5)

    def test_normalization_and_case_conventions(self):
        self.assertEqual(text_metrics("\uff21   B\nC", "A B C")["rouge_l_f1"], 1)
        self.assertEqual(text_metrics("A", "a")["wer"], 1)
        self.assertEqual(text_metrics("A", "a", casefold=True)["wer"], 0)

    def test_missing_is_distinct_from_an_empty_ocr_output(self):
        self.assertFalse(text_metrics(None, "hello")["text_available"])
        self.assertFalse(text_metrics("hello", None)["text_available"])
        metrics = text_metrics("", "hello")
        self.assertTrue(metrics["text_available"])
        self.assertEqual(metrics["wer"], 1)
        self.assertEqual(metrics["rouge_l_f1"], 0)
        with self.assertRaises(ValueError):
            text_metrics("hello", ["face", "name"])

    def test_empty_reference_keeps_insertions_but_rate_is_undefined(self):
        metrics = text_metrics("hello", "")
        self.assertEqual(metrics["word_edits"], 1)
        self.assertEqual(metrics["character_edits"], 5)
        self.assertIsNone(metrics["wer"])
        self.assertIsNone(metrics["rouge_l_f1"])


class ComparisonTests(unittest.TestCase):
    def test_micro_macro_supported_and_all_label_denominators(self):
        report = build_comparison([record(), record(image="image2", truth=["face"],
                                                   prediction=["face", "email"], stratum="general")], LABELS)
        result = summary(report)
        self.assertEqual(result["n_evaluated_labels"], 3)
        self.assertEqual(result["n_supported_labels"], 2)
        self.assertAlmostEqual(result["micro_f1"], 2 / 3)
        self.assertEqual(result["macro_f1_supported"], 0.5)
        self.assertAlmostEqual(result["macro_f1_all_labels"], 1 / 3)
        self.assertAlmostEqual(summary(report, stratum="document")["micro_recall"], 0.5)
        missed = next(row for row in report["per_attribute"] if row["scope"] == "all_successful"
                      and row["stratum"] == "all" and row["attribute_id"] == "name")
        self.assertEqual(missed["missed_count"], 1)

    def test_failures_are_not_fake_empty_predictions(self):
        report = build_comparison([record(), record(image="failed", status="error", error="out of memory")], LABELS,
                                  sample_image_ids=["image1", "failed", "unattempted"])
        result = summary(report)
        self.assertEqual(result["n_images"], 1)
        self.assertEqual(result["missed_count"], 1)
        coverage = next(row for row in report["coverage"] if row["stratum"] == "all")
        self.assertEqual((coverage["expected_images"], coverage["attempted_images"],
                          coverage["failed_images"], coverage["not_attempted_images"]), (3, 2, 1, 1))
        self.assertAlmostEqual(coverage["coverage_of_sample"], 1 / 3)
        failed = next(row for row in report["per_image"] if row["status"] == "error")
        self.assertIsNone(failed["f1"])
        self.assertIsNone(failed["missed_attributes"])
        self.assertEqual(report["unobserved_stratum_image_ids"], ["unattempted"])

    def test_common_subset_is_shared_across_every_requested_model(self):
        records = [record(), record(image="image2", stratum="general"),
                   record(model="second", prediction=["face", "name"]),
                   record(model="second", image="image2", stratum="general", status="error")]
        report = build_comparison(records, LABELS, ["first", "second"])
        self.assertEqual(report["common_image_ids"], ["image1"])
        self.assertEqual(summary(report, "first", "common_successful")["n_images"], 1)
        self.assertEqual(summary(report, "second", "common_successful")["n_images"], 1)
        self.assertIsNone(summary(report, "first", "common_successful", "general")["micro_f1"])

    def test_missing_requested_model_makes_common_subset_empty(self):
        report = build_comparison([record()], LABELS, ["first", "missing"])
        self.assertEqual(report["common_image_ids"], [])
        self.assertIsNone(summary(report, "first", "common_successful")["micro_f1"])
        self.assertIsNone(summary(report, "missing")["macro_f1_all_labels"])
        missing = next(row for row in report["coverage"] if row["model_key"] == "missing" and row["stratum"] == "all")
        self.assertEqual(missing["not_attempted_images"], 1)

    def test_empty_run_still_reports_requested_model_coverage(self):
        report = build_comparison([], LABELS, ["first"], ["image1", "image2"])
        self.assertEqual(report["common_image_ids"], [])
        self.assertIsNone(summary(report)["micro_f1"])
        self.assertEqual(report["coverage"][0]["not_attempted_images"], 2)

    def test_planned_strata_are_visible_before_any_inference(self):
        strata = {f"image{i:02d}": "document" if i < 25 else "general" for i in range(50)}
        report = build_comparison([], LABELS, ["first", "second"], list(strata), sample_strata=strata)
        self.assertEqual(report["unobserved_stratum_image_ids"], [])
        for row in report["coverage"]:
            expected = 50 if row["stratum"] == "all" else 25
            self.assertEqual(row["expected_images"], expected)
            self.assertEqual(row["not_attempted_images"], expected)
            self.assertEqual(row["attempted_images"], 0)
            self.assertEqual(row["failed_images"], 0)
        with self.assertRaises(ValueError):
            build_comparison([record()], LABELS, sample_strata={"image1": "general"})

    def test_duplicate_results_inconsistent_ground_truth_and_unknown_labels_rejected(self):
        invalid = [
            [record(), record()],
            [record(), record(model="second", truth=["email"])],
            [record(), record(model="second", stratum="general")],
            [record(prediction=["invented"])],
        ]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                build_comparison(records, LABELS)

    def test_optional_metrics_are_unavailable_without_references(self):
        report = build_comparison([record(detections=[box()], predicted_text="hello")], LABELS)
        result = summary(report)
        self.assertEqual(result["n_localization_images"], 0)
        self.assertEqual(result["n_text_images"], 0)
        self.assertIsNone(result["localization_f1"])
        self.assertIsNone(result["corpus_cer"])

    def test_manual_boxes_and_corpus_text_scores(self):
        report = build_comparison([
            record(detections=[box()], ground_truth_boxes=[box()], boxes_complete=True,
                   predicted_text="cat", reference_text="cat"),
            record(image="blank", truth=[], prediction=[], stratum="general", detections=[],
                   ground_truth_boxes=[], box_ground_truth_complete=True, predicted_text="x", reference_text=""),
        ], LABELS)
        result = summary(report)
        self.assertEqual(result["n_localization_images"], 2)
        self.assertEqual(result["localization_f1"], 1)
        self.assertEqual(result["mean_matched_iou"], 1)
        self.assertEqual(result["n_text_images"], 2)
        self.assertAlmostEqual(result["corpus_cer"], 1 / 3)
        self.assertEqual(result["corpus_wer"], 1)
        self.assertEqual(result["mean_rouge_l_f1"], 1)
        json.dumps(report, allow_nan=False)

    def test_category_counts_remain_label_level(self):
        report = build_comparison([record()], LABELS,
                                  attribute_to_category={"face": "biometric", "name": "identity", "email": "identity"})
        identity = next(row for row in report["per_category"] if row["scope"] == "all_successful"
                        and row["stratum"] == "all" and row["category"] == "identity")
        self.assertEqual(identity["missed_count"], 1)
        self.assertEqual(identity["micro_recall"], 0)
        self.assertNotIn("mean_image_f1", identity)


if __name__ == "__main__":
    unittest.main()
