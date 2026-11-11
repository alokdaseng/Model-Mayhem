"""Inference script producing exact Model Mayhem competition CSV predictions."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.constants import CLASS_NAMES, CLASS_TO_ID, CSV_COLUMNS, IMAGE_EXTENSIONS
from src.utils import (
    Box,
    clip_xyxy,
    nms_boxes,
    read_image_size,
    rotate_xyxy_inverse,
    wbf_boxes,
)
from validate_submission import validate_submission_csv


def compute_tile_windows(img_w: int, img_h: int, tile_size: int = 960, overlap: float = 0.22) -> List[Tuple[int, int, int, int]]:
    """Compute overlapping (x0, y0, x1, y1) crop windows covering the image cleanly."""
    if max(img_w, img_h) <= tile_size:
        return [(0, 0, img_w, img_h)]

    step = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, img_w - tile_size + 1), step))
    if not xs or xs[-1] + tile_size < img_w:
        xs.append(max(0, img_w - tile_size))

    ys = list(range(0, max(1, img_h - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < img_h:
        ys.append(max(0, img_h - tile_size))

    windows: List[Tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(img_w, x0 + tile_size)
            y1 = min(img_h, y0 + tile_size)
            windows.append((x0, y0, x1, y1))
    return windows


def _extract_boxes(res) -> List[Box]:
    """Convert one Ultralytics Results object to a list of Box tuples."""
    boxes_out: List[Box] = []
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return boxes_out
    xyxy_list = res.boxes.xyxy.cpu().tolist()
    conf_list = res.boxes.conf.cpu().tolist()
    cls_list = res.boxes.cls.cpu().tolist()
    for (x1, y1, x2, y2), score, cls_idx in zip(xyxy_list, conf_list, cls_list):
        cid = int(cls_idx)
        if 0 <= cid < len(CLASS_NAMES):
            boxes_out.append((cid, float(score), float(x1), float(y1), float(x2), float(y2)))
    return boxes_out


def predict_single_array(
    model,
    img_bgr,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    tta: bool,
    device: str,
    half: bool = False,
    agnostic_nms: bool = False,
) -> List[Box]:
    """Run Ultralytics prediction on an image array and return Box tuples."""
    del half  # Ultralytics 8.4 deprecated `half` in predict(); let runtime handle precision cleanly
    results = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        augment=tta,
        agnostic_nms=agnostic_nms,
        device=device,
        verbose=False,
    )
    return _extract_boxes(results[0]) if results else []


def predict_rot4_wbf(
    model,
    img_bgr,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    tta: bool,
    wbf_iou: float,
    device: str,
    agnostic_nms: bool = False,
) -> List[Box]:
    """Run 4-angle (0/90/180/270 deg) batched prediction and fuse via consensus WBF."""
    import cv2  # type: ignore

    img_h, img_w = img_bgr.shape[:2]
    views = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    results = model.predict(
        source=views,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        augment=tta,
        agnostic_nms=agnostic_nms,
        device=device,
        verbose=False,
    )
    per_view: List[List[Box]] = []
    for k, res in enumerate(results):
        view_boxes: List[Box] = []
        for cid, score, x1p, y1p, x2p, y2p in _extract_boxes(res):
            inv = rotate_xyxy_inverse(x1p, y1p, x2p, y2p, orig_w=img_w, orig_h=img_h, rot_k=k)
            if inv is not None:
                view_boxes.append((cid, score, *inv))
        per_view.append(view_boxes)

    fused = wbf_boxes(per_view, iou_thr=wbf_iou, skip_box_thr=0.0, mode="consensus")
    return nms_boxes(fused, iou_thr=iou)[:max_det]


def is_interior_seam_clipped(
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    img_w: int,
    img_h: int,
    margin: float = 4.0,
) -> bool:
    """Return True if a local tile detection sits on an interior crop seam (covered by overlap)."""
    tile_w = float(x1 - x0)
    tile_h = float(y1 - y0)
    if x0 > 0 and bx1 <= margin:
        return True
    if y0 > 0 and by1 <= margin:
        return True
    if x1 < img_w and bx2 >= tile_w - margin:
        return True
    if y1 < img_h and by2 >= tile_h - margin:
        return True
    return False


def predict_image_for_model(
    model,
    img_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    tta: bool,
    tile_threshold: int,
    tile_size: int,
    tile_overlap: float,
    device: str,
    half: bool = False,
    rot_tta: bool = False,
    wbf_iou: float = 0.55,
    agnostic_nms: bool = False,
) -> Tuple[List[Box], Tuple[int, int]]:
    """Run full-image (or 4-angle WBF + optional tiling) prediction for one model."""
    import cv2  # type: ignore

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        w, h = read_image_size(img_path)
        raise RuntimeError(f"Failed to read image with OpenCV ({w}x{h}): {img_path}")

    img_h, img_w = img_bgr.shape[:2]

    # Always run full-image pass so global context is preserved
    if rot_tta:
        full_boxes = predict_rot4_wbf(
            model=model,
            img_bgr=img_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            tta=tta,
            wbf_iou=wbf_iou,
            device=device,
            agnostic_nms=agnostic_nms,
        )
    else:
        full_boxes = predict_single_array(
            model=model,
            img_bgr=img_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            tta=tta,
            device=device,
            half=half,
            agnostic_nms=agnostic_nms,
        )

    clipped_full: List[Box] = []
    for cid, score, x1, y1, x2, y2 in full_boxes:
        box = clip_xyxy(x1, y1, x2, y2, img_w, img_h)
        if box is not None:
            clipped_full.append((cid, score, *box))

    if tile_threshold <= 0 or max(img_w, img_h) <= tile_threshold:
        return clipped_full, (img_w, img_h)

    # High-resolution adaptive tiling keeps tiny PCB defects sharp when explicitly enabled
    tile_lists: List[List[Box]] = [clipped_full]
    for x0, y0, x1, y1 in compute_tile_windows(img_w, img_h, tile_size=tile_size, overlap=tile_overlap):
        crop = img_bgr[y0:y1, x0:x1]
        crop_preds = predict_single_array(
            model=model,
            img_bgr=crop,
            imgsz=min(tile_size, imgsz),
            conf=conf,
            iou=iou,
            max_det=max_det,
            tta=tta,
            device=device,
            half=half,
            agnostic_nms=agnostic_nms,
        )
        shifted: List[Box] = []
        for cid, score, bx1, by1, bx2, by2 in crop_preds:
            if is_interior_seam_clipped(bx1, by1, bx2, by2, x0, y0, x1, y1, img_w, img_h):
                continue
            box = clip_xyxy(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, img_w, img_h)
            if box is not None:
                shifted.append((cid, score, *box))
        tile_lists.append(shifted)

    stitched = wbf_boxes(tile_lists, iou_thr=0.55, skip_box_thr=conf, mode="tile_stitch")
    stitched = nms_boxes(stitched, iou_thr=iou)[:max_det]
    return stitched, (img_w, img_h)


def draw_and_save_visual(img_path: Path, boxes: Sequence[Box], out_path: Path) -> None:
    """Save visual inspection overlay with labeled defect boxes."""
    import cv2  # type: ignore

    palette = [
        (0, 215, 255),
        (50, 205, 50),
        (0, 140, 255),
        (0, 60, 255),
        (255, 190, 0),
        (220, 80, 220),
    ]
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for cid, conf, x1, y1, x2, y2 in boxes:
        if conf < 0.25:
            continue
        color = palette[cid % len(palette)]
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{CLASS_NAMES[cid]} {conf:.2f}"
        cv2.putText(
            img,
            label,
            (p1[0], max(14, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def _write_box_rows(
    writer,
    image_name: str,
    boxes: Sequence[Box],
    img_w: Optional[int],
    img_h: Optional[int],
    int_coords: bool,
) -> int:
    """Sort boxes by confidence, clamp/round coordinates, write CSV rows, and return count."""
    ordered = sorted(boxes, key=lambda b: b[1], reverse=True)
    written = 0
    for cid, score, x1, y1, x2, y2 in ordered:
        if img_w is not None and img_h is not None:
            clipped = clip_xyxy(x1, y1, x2, y2, img_w, img_h)
            if clipped is None:
                continue
            cx1, cy1, cx2, cy2 = clipped
        else:
            cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), max(0.0, x2), max(0.0, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

        if int_coords:
            rx1, ry1, rx2, ry2 = int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
        else:
            rx1, ry1, rx2, ry2 = round(cx1, 2), round(cy1, 2), round(cx2, 2), round(cy2, 2)
        if rx2 <= rx1 or ry2 <= ry1:
            continue

        conf_val = min(1.0, max(0.0, round(float(score), 4)))
        writer.writerow([image_name, CLASS_NAMES[cid], rx1, ry1, rx2, ry2, conf_val])
        written += 1
    return written


def _validate_if_requested(
    output_csv: Path, images_dir: Optional[Path], total_rows: int, validate: bool
) -> None:
    """Validate submission CSV schema/bounds when enabled."""
    if not validate:
        return
    ok, errors, _ = validate_submission_csv(output_csv, images_dir=images_dir, allow_empty=(total_rows == 0))
    if not ok:
        raise RuntimeError(f"Submission validation failed: {errors}")
    print("[infer] Submission CSV validation: PASSED")


def fuse_prediction_csvs(
    csv_paths: Sequence[Path],
    output_csv: Path,
    weights: Optional[Sequence[float]] = None,
    wbf_iou: float = 0.55,
    iou: float = 0.60,
    max_det: int = 300,
    post_conf: float = 0.0,
    images_dir: Optional[Path] = None,
    int_coords: bool = False,
    validate: bool = True,
) -> Path:
    """Fuse predictions from multiple submission CSVs (e.g. different model resolutions) via WBF + NMS."""
    if not csv_paths:
        raise ValueError("At least one input CSV path is required for fusion.")

    per_csv_boxes: List[Dict[str, List[Box]]] = []
    all_image_names = set()
    for csv_path in csv_paths:
        by_img: Dict[str, List[Box]] = {}
        with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cls_name = row["class_name"].strip()
                if cls_name not in CLASS_TO_ID:
                    continue
                img_name = row["image_name"].strip()
                all_image_names.add(img_name)
                by_img.setdefault(img_name, []).append(
                    (
                        CLASS_TO_ID[cls_name],
                        float(row["confidence"]),
                        float(row["x_min"]),
                        float(row["y_min"]),
                        float(row["x_max"]),
                        float(row["y_max"]),
                    )
                )
        per_csv_boxes.append(by_img)

    dims_map: Dict[str, Tuple[int, int]] = {}
    img_dir_resolved = images_dir.parent if (images_dir is not None and images_dir.is_file()) else images_dir
    if img_dir_resolved is not None and img_dir_resolved.exists():
        for p in sorted(img_dir_resolved.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                dims_map[p.name] = read_image_size(p)
                all_image_names.add(p.name)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(CSV_COLUMNS))
        for img_name in sorted(all_image_names):
            model_lists = [src.get(img_name, []) for src in per_csv_boxes]
            fused = wbf_boxes(
                model_lists,
                iou_thr=wbf_iou,
                skip_box_thr=0.0,
                weights=weights,
                mode="consensus",
            )
            fused = [b for b in nms_boxes(fused, iou_thr=iou) if b[1] >= post_conf][:max_det]
            dims = dims_map.get(img_name)
            total_rows += _write_box_rows(
                writer,
                img_name,
                fused,
                dims[0] if dims else None,
                dims[1] if dims else None,
                int_coords=int_coords,
            )

    print(f"[infer] Fused {len(csv_paths)} CSV(s) -> {total_rows} predictions -> {output_csv}")
    _validate_if_requested(output_csv, img_dir_resolved, total_rows, validate)
    return output_csv


def run_inference(
    source: Path,
    weights: Sequence[Path],
    output_csv: Path,
    imgsz: int = 1024,
    conf: float = 0.01,
    iou: float = 0.60,
    max_det: int = 300,
    tta: bool = False,
    rot_tta: bool = False,
    tile_threshold: int = 0,
    tile_size: int = 1024,
    tile_overlap: float = 0.22,
    model_weights: Optional[Sequence[float]] = None,
    wbf_iou: float = 0.55,
    post_conf: float = 0.0,
    agnostic_nms: bool = False,
    device: str = "",
    int_coords: bool = False,
    recursive: bool = False,
    save_vis: Optional[Path] = None,
    validate: bool = True,
) -> Path:
    """Run inference over image(s) and write Model Mayhem submission CSV."""
    import torch  # type: ignore
    from tqdm import tqdm  # type: ignore
    from ultralytics import YOLO  # type: ignore

    for w in weights:
        if not Path(w).exists():
            raise FileNotFoundError(f"Checkpoint not found: {w}")

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Inference source not found: {source}")

    if source.is_file():
        image_paths = [source]
    else:
        iterator = source.rglob("*") if recursive else source.iterdir()
        image_paths = sorted(
            p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        raise RuntimeError(f"No images found under {source}")

    resolved_device = device if device else ("0" if torch.cuda.is_available() else "cpu")

    models = [YOLO(str(w)) for w in weights]
    print(
        f"[infer] Loaded {len(models)} model(s) on device={resolved_device} | "
        f"images={len(image_paths)} | imgsz={imgsz} | conf={conf} | iou={iou} | "
        f"tta={tta} | rot_tta={rot_tta} | tile_thr={tile_threshold}"
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(CSV_COLUMNS))

        model_conf = min(conf, 0.01) if (len(models) > 1 or rot_tta) else conf
        for img_path in tqdm(image_paths, desc="Inferring"):
            per_model_boxes: List[List[Box]] = []
            img_dims = (0, 0)
            for model in models:
                boxes, dims = predict_image_for_model(
                    model=model,
                    img_path=img_path,
                    imgsz=imgsz,
                    conf=model_conf,
                    iou=iou,
                    max_det=max_det,
                    tta=tta,
                    tile_threshold=tile_threshold,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    device=resolved_device,
                    rot_tta=rot_tta,
                    wbf_iou=wbf_iou,
                    agnostic_nms=agnostic_nms,
                )
                img_dims = dims
                per_model_boxes.append(boxes)

            if len(models) == 1:
                min_final = post_conf if rot_tta else max(conf, post_conf)
                final_boxes = [b for b in per_model_boxes[0] if b[1] >= min_final][:max_det]
            else:
                fused = wbf_boxes(
                    per_model_boxes,
                    iou_thr=wbf_iou,
                    skip_box_thr=0.0,
                    weights=model_weights,
                    mode="consensus",
                )
                final_boxes = [b for b in nms_boxes(fused, iou_thr=iou) if b[1] >= post_conf][:max_det]

            img_w, img_h = img_dims
            final_boxes.sort(key=lambda b: b[1], reverse=True)
            total_rows += _write_box_rows(
                writer, img_path.name, final_boxes, img_w, img_h, int_coords=int_coords
            )

            if save_vis is not None:
                draw_and_save_visual(img_path, final_boxes, save_vis / img_path.name)

    print(f"[infer] Wrote {total_rows} predictions across {len(image_paths)} images -> {output_csv}")
    img_dir = source.parent if source.is_file() else source
    _validate_if_requested(output_csv, img_dir, total_rows, validate)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCB defect inference and output submission CSV.")
    parser.add_argument("--source", required=False, type=Path, default=None, help="Image file or folder of test images.")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=Path,
        default=[Path("weights/best.pt")],
        help="One or more trained .pt weights (multiple weights trigger WBF fusion).",
    )
    parser.add_argument("--fuse-csvs", nargs="+", type=Path, default=None, help="Optional CSV files to fuse via WBF instead of running weights.")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Output CSV path.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution.")
    parser.add_argument("--conf", type=float, default=0.01, help="Detector confidence threshold.")
    parser.add_argument("--post-conf", type=float, default=0.0, help="Post-WBF minimum confidence filter.")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image.")
    parser.add_argument("--tta", action="store_true", help="Enable Ultralytics multi-scale/flip TTA.")
    parser.add_argument("--rot-tta", action="store_true", help="Enable 4-angle (0/90/180/270 deg) WBF rotation TTA.")
    parser.add_argument("--agnostic-nms", action="store_true", help="Suppress overlapping boxes across classes.")
    parser.add_argument(
        "--tile-threshold",
        type=int,
        default=0,
        help="Apply adaptive tiling if max(W, H) > threshold (0 disables tiling by default).",
    )
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile width/height when tiling.")
    parser.add_argument("--tile-overlap", type=float, default=0.22, help="Fractional tile overlap.")
    parser.add_argument("--model-weights", nargs="+", type=float, default=None, help="WBF weights per model or CSV.")
    parser.add_argument("--wbf-iou", type=float, default=0.55, help="WBF clustering IoU threshold.")
    parser.add_argument("--device", type=str, default="", help="CUDA device id or 'cpu'.")
    parser.add_argument("--int-coords", action="store_true", help="Round pixel coordinates to integers.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively.")
    parser.add_argument("--save-vis", type=Path, default=None, help="Optional directory for annotated overlays.")
    parser.add_argument("--no-validate", action="store_true", help="Skip final CSV schema validation.")
    args = parser.parse_args()

    if args.fuse_csvs:
        fuse_prediction_csvs(
            csv_paths=args.fuse_csvs,
            output_csv=args.output,
            weights=args.model_weights,
            wbf_iou=args.wbf_iou,
            iou=args.iou,
            max_det=args.max_det,
            post_conf=args.post_conf,
            images_dir=args.source,
            int_coords=args.int_coords,
            validate=not args.no_validate,
        )
        return

    if args.source is None:
        parser.error("--source is required unless --fuse-csvs is provided.")

    run_inference(
        source=args.source,
        weights=args.weights,
        output_csv=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        tta=args.tta,
        rot_tta=args.rot_tta,
        tile_threshold=args.tile_threshold,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        model_weights=args.model_weights,
        wbf_iou=args.wbf_iou,
        post_conf=args.post_conf,
        agnostic_nms=args.agnostic_nms,
        device=args.device,
        int_coords=args.int_coords,
        recursive=args.recursive,
        save_vis=args.save_vis,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()

