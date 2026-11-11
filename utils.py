"""Shared geometry, image metadata, NMS, and Weighted Boxes Fusion utilities."""

from pathlib import Path
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Box representation: (class_id, confidence, x_min, y_min, x_max, y_max)
Box = Tuple[int, float, float, float, float, float]


def read_image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) in pixels using PIL/cv2 if present, else pure-Python headers."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            return im.size
    except ImportError:
        pass

    data = Path(image_path).read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)

    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        idx = 2
        n = len(data)
        while idx < n - 9:
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h, w = struct.unpack(">HH", data[idx + 5 : idx + 9])
                return int(w), int(h)
            if marker in (0xD8, 0xD9, 0x01) or (0xD0 <= marker <= 0xD7):
                idx += 2
            else:
                if idx + 4 > n:
                    break
                seg_len = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
                if seg_len < 2:
                    break
                idx += 2 + seg_len

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except ImportError:
        pass

    raise ValueError(f"Unsupported or unreadable image dimensions: {image_path}")


def clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Clamp XYXY coordinates to [0, img_w] x [0, img_h]; return None if degenerate."""
    cx1 = max(0.0, min(float(x1), float(img_w)))
    cy1 = max(0.0, min(float(y1), float(img_h)))
    cx2 = max(0.0, min(float(x2), float(img_w)))
    cy2 = max(0.0, min(float(y2), float(img_h)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def yolo_to_xyxy_pixels(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert normalized YOLO (xc, yc, w, h) coordinates to clamped original pixels."""
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - 0.5 * bw
    y1 = (yc * img_h) - 0.5 * bh
    x2 = x1 + bw
    y2 = y1 + bh
    return clip_xyxy(x1, y1, x2, y2, img_w, img_h)


def rotate_xyxy_inverse(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    orig_w: int,
    orig_h: int,
    rot_k: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Map XYXY box from k*90deg CW rotated image back to original (orig_w, orig_h) coords."""
    k = rot_k % 4
    if k == 0:
        bx1, by1, bx2, by2 = x1, y1, x2, y2
    elif k == 1:
        bx1, by1, bx2, by2 = y1, orig_h - x2, y2, orig_h - x1
    elif k == 2:
        bx1, by1, bx2, by2 = orig_w - x2, orig_h - y2, orig_w - x1, orig_h - y1
    else:
        bx1, by1, bx2, by2 = orig_w - y2, x1, orig_w - y1, x2
    return clip_xyxy(bx1, by1, bx2, by2, orig_w, orig_h)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms_boxes(boxes: Sequence[Box], iou_thr: float = 0.60) -> List[Box]:
    """Perform class-aware greedy Non-Maximum Suppression."""
    by_class: Dict[int, List[Box]] = {}
    for b in boxes:
        by_class.setdefault(b[0], []).append(b)

    kept: List[Box] = []
    for cls_id in sorted(by_class.keys()):
        candidates = sorted(by_class[cls_id], key=lambda x: x[1], reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            best_xyxy = best[2:]
            candidates = [
                c for c in candidates if compute_iou(best_xyxy, c[2:]) < iou_thr
            ]
    kept.sort(key=lambda x: x[1], reverse=True)
    return kept


def wbf_boxes(
    model_box_lists: Sequence[Sequence[Box]],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    weights: Optional[Sequence[float]] = None,
    mode: str = "consensus",
) -> List[Box]:
    """Fuse detections via class-aware Weighted Boxes Fusion.

    Args:
        model_box_lists: Per-model or per-tile lists of (cls_id, conf, x1, y1, x2, y2).
        iou_thr: Minimum IoU to merge a candidate box into an existing cluster.
        skip_box_thr: Discard input boxes below this confidence before clustering.
        weights: Optional non-negative weights matching length of model_box_lists.
        mode: "consensus" (multi-model/TTA ensemble dividing sum by total weights) or
              "tile_stitch" (overlapping crop merger using cluster max confidence).
    """
    if mode not in ("consensus", "tile_stitch"):
        raise ValueError(f"Invalid WBF mode: {mode}")
    n_sources = len(model_box_lists)
    if n_sources == 0:
        return []

    src_weights = [1.0] * n_sources if weights is None else [float(w) for w in weights]
    if len(src_weights) != n_sources:
        raise ValueError("weights length must match model_box_lists length")
    total_weight = sum(src_weights) or 1.0

    # Group filtered boxes by class: (cls_id, weighted_conf, raw_conf, weight, x1, y1, x2, y2, src_idx)
    by_class: Dict[int, List[Tuple[int, float, float, float, float, float, float, float, int]]] = {}
    for src_idx, box_list in enumerate(model_box_lists):
        w = src_weights[src_idx]
        if w <= 0.0:
            continue
        for cls_id, conf, x1, y1, x2, y2 in box_list:
            if conf < skip_box_thr or x2 <= x1 or y2 <= y1:
                continue
            by_class.setdefault(cls_id, []).append(
                (cls_id, conf * w, conf, w, x1, y1, x2, y2, src_idx)
            )

    fused_results: List[Box] = []
    for cls_id in sorted(by_class.keys()):
        entries = sorted(by_class[cls_id], key=lambda item: item[1], reverse=True)
        # Each cluster tracks: members list and current weighted average box [x1, y1, x2, y2]
        clusters: List[Dict[str, object]] = []

        for entry in entries:
            _, wconf, conf, w, x1, y1, x2, y2, src_idx = entry
            best_idx = -1
            best_iou = iou_thr
            for c_idx, cluster in enumerate(clusters):
                iou = compute_iou((x1, y1, x2, y2), cluster["avg_box"])  # type: ignore[arg-type]
                if iou >= best_iou:
                    best_iou = iou
                    best_idx = c_idx

            if best_idx == -1:
                clusters.append(
                    {
                        "members": [entry],
                        "avg_box": [x1, y1, x2, y2],
                        "sum_wconf": wconf,
                    }
                )
            else:
                cluster = clusters[best_idx]
                members: List = cluster["members"]  # type: ignore[assignment]
                members.append(entry)
                sum_wc = float(cluster["sum_wconf"]) + wconf  # type: ignore[arg-type]
                cluster["sum_wconf"] = sum_wc
                cluster["avg_box"] = [
                    sum(m[4] * m[1] for m in members) / sum_wc,
                    sum(m[5] * m[1] for m in members) / sum_wc,
                    sum(m[6] * m[1] for m in members) / sum_wc,
                    sum(m[7] * m[1] for m in members) / sum_wc,
                ]

        for cluster in clusters:
            members = cluster["members"]  # type: ignore[assignment]
            avg_box: List[float] = cluster["avg_box"]  # type: ignore[assignment]
            if mode == "consensus":
                # Keep at most the highest-confidence prediction per source when scaling confidence
                per_src_best: Dict[int, float] = {}
                for m in members:
                    src_i, wc = m[8], m[1]
                    if wc > per_src_best.get(src_i, -1.0):
                        per_src_best[src_i] = wc
                score = min(1.0, sum(per_src_best.values()) / total_weight)
            else:
                # Overlapping tiles should not drop confidence just because only 1 tile covered the object
                score = min(1.0, max(m[2] for m in members))

            fused_results.append(
                (cls_id, score, avg_box[0], avg_box[1], avg_box[2], avg_box[3])
            )

    fused_results.sort(key=lambda b: b[1], reverse=True)
    return fused_results

