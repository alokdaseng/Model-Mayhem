"""Unit tests for competition metric evaluation, WBF fusion, geometry, and CSV validation."""

import csv
from pathlib import Path
import tempfile
import unittest

from infer import fuse_prediction_csvs, is_interior_seam_clipped
from scripts.prepare_data import extract_stem_and_group
from src.constants import CLASS_NAMES, CSV_COLUMNS, IOU_THRESHOLDS
from src.metrics import compute_coco101_ap, evaluate_predictions
from src.utils import compute_iou, nms_boxes, rotate_xyxy_inverse, wbf_boxes, yolo_to_xyxy_pixels
from validate_submission import validate_submission_csv


class TestMetricsAndGeometry(unittest.TestCase):
    def test_iou_thresholds_match_spec(self) -> None:
        expected = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
        self.assertEqual(IOU_THRESHOLDS, expected)

    def test_compute_iou_exact_and_disjoint(self) -> None:
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (10, 10, 20, 20)), 0.0)
        # Overlap 5x10=50 over union 100+100-50=150 -> 1/3
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)

    def test_coco101_ap_exact_values(self) -> None:
        # 1 TP at conf=0.9 -> AP = 1.0
        self.assertAlmostEqual(compute_coco101_ap([1.0], [1.0]), 1.0, places=6)
        # 1 FP then 1 TP -> precision envelope stays 0.5 across [0.0, 1.0] -> AP = 0.5
        self.assertAlmostEqual(compute_coco101_ap([0.0, 1.0], [0.0, 0.5]), 0.5, places=6)

    def test_evaluate_predictions_map_50_90_and_macro_recall(self) -> None:
        # 2 classes present: missing_hole (exact IoU=1.0) and mouse_bite (IoU=0.64 -> passes 0.50,0.55,0.60 only)
        gt = {
            "pcb1.jpg": {
                0: [(100.0, 100.0, 200.0, 200.0)],
                1: [(0.0, 0.0, 100.0, 100.0)],
            }
        }
        # Box (0,0,80,80) vs (0,0,100,100): IoU = 6400/10000 = 0.64 -> matches 3/9 IoU thresholds
        preds = [
            ("pcb1.jpg", 0, 0.95, 100.0, 100.0, 200.0, 200.0),
            ("pcb1.jpg", 1, 0.90, 0.0, 0.0, 80.0, 80.0),
        ]
        res = evaluate_predictions(gt, preds)
        # Class 0 AP=1.0 at all 9 thresholds; Class 1 AP=1.0 at 3/9 thresholds
        expected_map = (1.0 + (3.0 / 9.0)) / 2.0
        self.assertAlmostEqual(res["mAP@50:90"], expected_map, places=6)  # type: ignore[arg-type]
        self.assertAlmostEqual(res["mAP@50"], 1.0, places=6)  # type: ignore[arg-type]
        self.assertAlmostEqual(res["macro_recall@50"], 1.0, places=6)  # type: ignore[arg-type]

    def test_yolo_to_xyxy_clamping(self) -> None:
        box = yolo_to_xyxy_pixels(0.5, 0.5, 0.2, 0.4, 1000, 500)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box  # type: ignore[misc]
        self.assertAlmostEqual(x1, 400.0)
        self.assertAlmostEqual(y1, 150.0)
        self.assertAlmostEqual(x2, 600.0)
        self.assertAlmostEqual(y2, 350.0)

    def test_wbf_consensus_and_tile_stitch(self) -> None:
        m1 = [(0, 0.90, 10.0, 10.0, 50.0, 50.0)]
        m2 = [(0, 0.70, 14.0, 12.0, 54.0, 52.0)]
        fused = wbf_boxes([m1, m2], iou_thr=0.55, mode="consensus")
        self.assertEqual(len(fused), 1)
        cid, score, x1, y1, x2, y2 = fused[0]
        self.assertEqual(cid, 0)
        self.assertAlmostEqual(score, 0.80, places=5)
        self.assertGreater(x1, 10.0)
        self.assertLess(x1, 14.0)

        # In tile_stitch mode, a box seen in only 1 of 4 tiles keeps its confidence
        stitched = wbf_boxes([m1, [], [], []], iou_thr=0.55, mode="tile_stitch")
        self.assertEqual(len(stitched), 1)
        self.assertAlmostEqual(stitched[0][1], 0.90, places=5)

    def test_nms_boxes_suppresses_overlap(self) -> None:
        boxes = [
            (2, 0.92, 10.0, 10.0, 50.0, 50.0),
            (2, 0.75, 11.0, 11.0, 51.0, 51.0),
            (3, 0.85, 11.0, 11.0, 51.0, 51.0),  # Different class kept
        ]
        kept = nms_boxes(boxes, iou_thr=0.60)
        self.assertEqual(len(kept), 2)

    def test_stem_extraction(self) -> None:
        stem, grp = extract_stem_and_group("01_spur_10_jpg.rf.94fe7a84e062753ff1f1298ee9074e37.jpg")
        self.assertEqual(stem, "01_spur_10")
        self.assertEqual(grp, "01_spur")

    def test_interior_seam_clip_filter(self) -> None:
        # Center tile [800..1824] on a 3000x3000 image: left=2.0 touches interior seam -> drop
        self.assertTrue(is_interior_seam_clipped(2.0, 200.0, 30.0, 230.0, 800, 800, 1824, 1824, 3000, 3000))
        # Outer left border tile [0..1024]: left=2.0 is real image border -> keep!
        self.assertFalse(is_interior_seam_clipped(2.0, 200.0, 30.0, 230.0, 0, 0, 1024, 1024, 3000, 3000))
        # Center box far from interior tile seams -> keep!
        self.assertFalse(is_interior_seam_clipped(300.0, 300.0, 340.0, 340.0, 800, 800, 1824, 1824, 3000, 3000))

    def test_submission_csv_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_file = Path(tmp) / "preds.csv"
            with csv_file.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(list(CSV_COLUMNS))
                w.writerow(["pcb_001.jpg", CLASS_NAMES[0], 120, 85, 160, 125, 0.94])
            ok, errs, _ = validate_submission_csv(csv_file)
            self.assertTrue(ok, errs)

            # Corrupt confidence > 1.0 must fail
            bad_csv = Path(tmp) / "bad.csv"
            with bad_csv.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(list(CSV_COLUMNS))
                w.writerow(["pcb_001.jpg", "invalid_cls", 120, 85, 160, 125, 1.25])
            ok_bad, errs_bad, _ = validate_submission_csv(bad_csv)
            self.assertFalse(ok_bad)
            self.assertGreaterEqual(len(errs_bad), 2)

    def test_rotate_xyxy_inverse_all_orientations(self) -> None:
        # Original box [10, 5, 30, 15] on 100x60 image
        orig_w, orig_h = 100, 60
        expected = (10.0, 5.0, 30.0, 15.0)
        # k=0: identity
        self.assertEqual(rotate_xyxy_inverse(10.0, 5.0, 30.0, 15.0, orig_w, orig_h, 0), expected)
        # k=1 (90CW): x'=H-y in [45,55], y'=x in [10,30]
        self.assertEqual(rotate_xyxy_inverse(45.0, 10.0, 55.0, 30.0, orig_w, orig_h, 1), expected)
        # k=2 (180): x'=W-x in [70,90], y'=H-y in [45,55]
        self.assertEqual(rotate_xyxy_inverse(70.0, 45.0, 90.0, 55.0, orig_w, orig_h, 2), expected)
        # k=3 (270CW): x'=y in [5,15], y'=W-x in [70,90]
        self.assertEqual(rotate_xyxy_inverse(5.0, 70.0, 15.0, 90.0, orig_w, orig_h, 3), expected)

    def test_fuse_prediction_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "m1.csv"
            p2 = Path(tmp) / "m2.csv"
            out = Path(tmp) / "fused.csv"
            for p, row in [
                (p1, ["pcb.jpg", CLASS_NAMES[0], 10.0, 10.0, 50.0, 50.0, 0.90]),
                (p2, ["pcb.jpg", CLASS_NAMES[0], 14.0, 12.0, 54.0, 52.0, 0.70]),
            ]:
                with p.open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(list(CSV_COLUMNS))
                    w.writerow(row)
            fuse_prediction_csvs([p1, p2], out, weights=[1.0, 1.0], validate=True)
            with out.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["confidence"]), 0.80, places=4)


if __name__ == "__main__":
    unittest.main()

