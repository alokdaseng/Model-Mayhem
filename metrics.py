"""Exact Model Mayhem evaluator: mAP@50:90, mAP@50, and Macro Recall."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, IMAGE_EXTENSIONS, IOU_THRESHOLDS
from src.utils import compute_iou, read_image_size, yolo_to_xyxy_pixels

# PredRecord: (image_name, class_id, confidence, x_min, y_min, x_max, y_max)
PredRecord = Tuple[str, int, float, float, float, float, float]
# GtMap: {image_name: {class_id: [(x_min, y_min, x_max, y_max), ...]}}
GtMap = Dict[str, Dict[int, List[Tuple[float, float, float, float]]]]


def load_ground_truth_yolo(images_dir: Path, labels_dir: Path) -> GtMap:
    """Load YOLO ground-truth annotations converted to original pixel XYXY coordinates."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    gt: GtMap = {}

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for img_path in image_files:
        img_w, img_h = read_image_size(img_path)
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        per_class: Dict[int, List[Tuple[float, float, float, float]]] = {}
        if lbl_path.exists():
            for raw_line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                xyxy = yolo_to_xyxy_pixels(xc, yc, bw, bh, img_w, img_h)
                if xyxy is not None:
                    per_class.setdefault(cls_id, []).append(xyxy)
        gt[img_path.name] = per_class
    return gt


def load_predictions_csv(csv_path: Path) -> List[PredRecord]:
    """Load submission predictions CSV into normalized PredRecord tuples."""
    preds: List[PredRecord] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cls_name = row["class_name"].strip()
            if cls_name not in CLASS_TO_ID:
                raise ValueError(f"Unknown class_name '{cls_name}' in {csv_path}")
            preds.append(
                (
                    row["image_name"].strip(),
                    CLASS_TO_ID[cls_name],
                    float(row["confidence"]),
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                )
            )
    return preds


def compute_coco101_ap(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """Compute COCO-style 101-point interpolated Average Precision."""
    if not recalls or not precisions:
        return 0.0

    # Reverse cumulative max yields the upper precision envelope (matches pycocotools searchsorted)
    env_p = list(precisions)
    for idx in range(len(env_p) - 2, -1, -1):
        if env_p[idx] < env_p[idx + 1]:
            env_p[idx] = env_p[idx + 1]

    ap_sum = 0.0
    j = 0
    n = len(recalls)
    for k in range(101):
        target_r = k / 100.0
        while j < n and recalls[j] < target_r:
            j += 1
        if j < n:
            ap_sum += env_p[j]
    return ap_sum / 101.0


def evaluate_class_at_iou(
    class_preds: Sequence[Tuple[str, float, Tuple[float, float, float, float]]],
    class_gt_by_image: Dict[str, List[Tuple[float, float, float, float]]],
    iou_thr: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (AP, Recall) for one class at one IoU threshold, or (None, None) if 0 GT."""
    total_gt = sum(len(boxes) for boxes in class_gt_by_image.values())
    if total_gt == 0:
        return None, None
    if not class_preds:
        return 0.0, 0.0

    sorted_preds = sorted(class_preds, key=lambda x: x[1], reverse=True)
    matched: Dict[str, List[bool]] = {
        img: [False] * len(boxes) for img, boxes in class_gt_by_image.items()
    }

    tp_cum = 0
    fp_cum = 0
    recalls: List[float] = []
    precisions: List[float] = []

    for img_name, _conf, pred_box in sorted_preds:
        gt_boxes = class_gt_by_image.get(img_name, [])
        best_iou = iou_thr
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if matched[img_name][gt_idx]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou >= best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched[img_name][best_gt_idx] = True
            tp_cum += 1
        else:
            fp_cum += 1

        recalls.append(tp_cum / total_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))

    ap = compute_coco101_ap(recalls, precisions)
    recall = tp_cum / total_gt
    return ap, recall


def evaluate_predictions(
    gt_map: GtMap,
    predictions: Sequence[PredRecord],
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> Dict[str, object]:
    """Compute mAP@50:90, mAP@50, Macro Recall@50, and per-class breakdowns."""
    # Group predictions by class_id
    preds_by_cls: Dict[int, List[Tuple[str, float, Tuple[float, float, float, float]]]] = {
        cid: [] for cid in range(len(CLASS_NAMES))
    }
    for img_name, cls_id, conf, x1, y1, x2, y2 in predictions:
        preds_by_cls.setdefault(cls_id, []).append((img_name, conf, (x1, y1, x2, y2)))

    per_class_summary: Dict[str, Dict[str, float]] = {}
    ap_matrix: List[List[float]] = []  # [class_idx][thr_idx]
    ap50_list: List[float] = []
    recall50_list: List[float] = []
    recall_50_90_list: List[float] = []

    for cls_id, cls_name in ID_TO_CLASS.items():
        cls_gt = {img: per_cls.get(cls_id, []) for img, per_cls in gt_map.items()}
        cls_preds = preds_by_cls.get(cls_id, [])
        thr_aps: List[float] = []
        thr_recalls: List[float] = []

        ap_at_50 = 0.0
        rec_at_50 = 0.0
        has_gt = True
        for thr in iou_thresholds:
            ap, rec = evaluate_class_at_iou(cls_preds, cls_gt, thr)
            if ap is None or rec is None:
                has_gt = False
                break
            thr_aps.append(ap)
            thr_recalls.append(rec)
            if abs(thr - 0.50) < 1e-6:
                ap_at_50 = ap
                rec_at_50 = rec

        if not has_gt:
            continue

        mean_ap_cls = sum(thr_aps) / len(thr_aps)
        mean_rec_cls = sum(thr_recalls) / len(thr_recalls)
        ap_matrix.append(thr_aps)
        ap50_list.append(ap_at_50)
        recall50_list.append(rec_at_50)
        recall_50_90_list.append(mean_rec_cls)

        per_class_summary[cls_name] = {
            "gt_count": float(sum(len(b) for b in cls_gt.values())),
            "pred_count": float(len(cls_preds)),
            "mAP@50:90": mean_ap_cls,
            "mAP@50": ap_at_50,
            "recall@50": rec_at_50,
            "recall@50:90": mean_rec_cls,
        }

    overall_map_50_90 = (
        sum(sum(row) / len(row) for row in ap_matrix) / len(ap_matrix) if ap_matrix else 0.0
    )
    overall_map_50 = sum(ap50_list) / len(ap50_list) if ap50_list else 0.0
    macro_recall_50 = sum(recall50_list) / len(recall50_list) if recall50_list else 0.0
    macro_recall_50_90 = (
        sum(recall_50_90_list) / len(recall_50_90_list) if recall_50_90_list else 0.0
    )

    return {
        "mAP@50:90": overall_map_50_90,
        "mAP@50": overall_map_50,
        "macro_recall@50": macro_recall_50,
        "macro_recall@50:90": macro_recall_50_90,
        "per_class": per_class_summary,
        "num_images": len(gt_map),
        "num_predictions": len(predictions),
    }


def format_evaluation_report(results: Dict[str, object]) -> str:
    """Format an aligned text summary of evaluation metrics."""
    lines = [
        "=" * 74,
        "MODEL MAYHEM PCB EVALUATION REPORT",
        "=" * 74,
        f"Images Evaluated   : {results['num_images']}",
        f"Total Detections   : {results['num_predictions']}",
        f"Primary Metric     : mAP@50:90 = {results['mAP@50:90']:.4f}",
        f"Tie-Breaker 1      : mAP@50    = {results['mAP@50']:.4f}",
        f"Tie-Breaker 2      : Macro Rec = {results['macro_recall@50']:.4f} "
        f"(mean@50:90 = {results['macro_recall@50:90']:.4f})",
        "-" * 74,
        f"{'Class':<18} {'GT':>6} {'Preds':>7} {'mAP@50:90':>11} {'mAP@50':>9} {'Rec@50':>9}",
        "-" * 74,
    ]
    per_class: Dict[str, Dict[str, float]] = results["per_class"]  # type: ignore[assignment]
    for cls_name in CLASS_NAMES:
        if cls_name not in per_class:
            continue
        m = per_class[cls_name]
        lines.append(
            f"{cls_name:<18} {int(m['gt_count']):>6} {int(m['pred_count']):>7} "
            f"{m['mAP@50:90']:>11.4f} {m['mAP@50']:>9.4f} {m['recall@50']:>9.4f}"
        )
    lines.append("=" * 74)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CSV predictions against YOLO labels.")
    parser.add_argument("--pred", required=True, type=Path, help="Predictions CSV file path.")
    parser.add_argument("--images", required=True, type=Path, help="Ground truth images folder.")
    parser.add_argument("--labels", required=True, type=Path, help="Ground truth YOLO labels folder.")
    args = parser.parse_args()

    gt_map = load_ground_truth_yolo(args.images, args.labels)
    preds = load_predictions_csv(args.pred)
    results = evaluate_predictions(gt_map, preds)
    print(format_evaluation_report(results))


if __name__ == "__main__":
    main()
