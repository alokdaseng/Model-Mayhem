"""Reproducible PCB defect detection training & exact mAP@50:90 evaluation script."""

import argparse
from pathlib import Path
import shutil
from typing import Optional

from infer import run_inference
from src.metrics import evaluate_predictions, format_evaluation_report, load_ground_truth_yolo, load_predictions_csv


def resolve_data_yaml(preferred: Optional[Path]) -> Path:
    """Resolve dataset YAML path cleanly."""
    if preferred is not None and preferred.exists():
        return preferred
    for candidate in (Path("data/clean/data.yaml"), Path("data/raw/data.yaml"), Path("data.yaml")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No dataset YAML found. Run scripts/prepare_data.py first or pass --data."
    )


def extract_val_dirs(data_yaml: Path) -> tuple[Path, Path]:
    """Read val images directory from data.yaml and derive matching labels directory."""
    import yaml  # type: ignore

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val_rel = cfg["val"]
    val_img_dir = (root / val_rel).resolve()
    val_lbl_dir = val_img_dir.parent / "labels"
    return val_img_dir, val_lbl_dir


def train_model(args: argparse.Namespace) -> Path:
    """Train YOLO model tuned for tiny PCB defects and high-IoU localization."""
    import torch  # type: ignore
    from ultralytics import YOLO  # type: ignore

    data_yaml = resolve_data_yaml(args.data)
    device = args.device if args.device else ("0" if torch.cuda.is_available() else "cpu")
    print(f"[train] Starting training: model={args.model} | data={data_yaml} | imgsz={args.imgsz} | epochs={args.epochs} | device={device}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        optimizer="AdamW",
        cos_lr=True,
        lr0=args.lr0,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=args.box_gain,
        cls=args.cls_gain,
        dfl=1.5,
        # Tilt, shear, and flip invariance matching multi-resolution PCB boards without destroying tiny defects
        degrees=args.degrees,
        shear=args.shear,
        perspective=args.perspective,
        translate=0.08,
        scale=args.scale,
        fliplr=0.5,
        flipud=0.5,
        mosaic=args.mosaic,
        close_mosaic=min(args.close_mosaic, max(1, args.epochs // 2)),
        mixup=0.0,
        erasing=0.0,
        hsv_h=0.015,
        hsv_s=0.30,
        hsv_v=0.30,
        workers=args.workers,
        cache=args.cache,
        amp=True,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best_src = save_dir / "weights" / "best.pt"
    if not best_src.exists():
        best_src = save_dir / "weights" / "last.pt"

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    named_dst = weights_dir / f"{args.name}_best.pt"
    shutil.copy2(best_src, named_dst)
    if not args.no_overwrite_best:
        shutil.copy2(best_src, weights_dir / "best.pt")

    print(f"[train] Saved best checkpoint -> {named_dst} (and weights/best.pt)")

    if not args.skip_eval:
        val_img_dir, val_lbl_dir = extract_val_dirs(data_yaml)
        if val_img_dir.exists() and val_lbl_dir.exists():
            val_csv = save_dir / "val_predictions.csv"
            run_inference(
                source=val_img_dir,
                weights=[named_dst],
                output_csv=val_csv,
                imgsz=args.imgsz,
                conf=args.eval_conf,
                iou=args.eval_iou,
                tta=args.eval_tta,
                tile_threshold=0,
                device=device,
                validate=True,
            )
            gt_map = load_ground_truth_yolo(val_img_dir, val_lbl_dir)
            preds = load_predictions_csv(val_csv)
            report = evaluate_predictions(gt_map, preds)
            print(format_evaluation_report(report))

    return named_dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model Mayhem PCB defect detector.")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base checkpoint (e.g. yolo11m.pt, yolo11s.pt, yolov8m.pt).")
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML path (defaults to data/clean/data.yaml).")
    parser.add_argument("--imgsz", type=int, default=768, help="Training resolution.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 auto-tunes to GPU VRAM).")
    parser.add_argument("--device", type=str, default="", help="CUDA device id or 'cpu'.")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Training runs folder.")
    parser.add_argument("--name", type=str, default="pcb_yolo11m_768", help="Run name.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial AdamW learning rate.")
    parser.add_argument("--box-gain", type=float, default=7.5, help="Box regression loss gain.")
    parser.add_argument("--cls-gain", type=float, default=1.0, help="Classification loss gain for calibrated confidence.")
    parser.add_argument("--degrees", type=float, default=15.0, help="Random rotation degrees (+/-).")
    parser.add_argument("--shear", type=float, default=6.0, help="Random shear degrees (+/-).")
    parser.add_argument("--perspective", type=float, default=0.0003, help="Random perspective warp.")
    parser.add_argument("--mosaic", type=float, default=0.35, help="Mosaic augmentation probability.")
    parser.add_argument("--scale", type=float, default=0.25, help="Random scale gain.")
    parser.add_argument("--close-mosaic", type=int, default=8, help="Final epochs without mosaic.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--cache", type=str, default="False", choices=["ram", "disk", "False"], help="Ultralytics image cache mode.")
    parser.add_argument("--eval-conf", type=float, default=0.01, help="Confidence threshold for post-train metric evaluation.")
    parser.add_argument("--eval-iou", type=float, default=0.60, help="NMS IoU threshold for post-train evaluation.")
    parser.add_argument("--eval-tta", action="store_true", help="Enable TTA during post-train evaluation.")
    parser.add_argument("--no-overwrite-best", action="store_true", help="Do not overwrite weights/best.pt.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip standalone mAP@50:90 report after training.")
    args = parser.parse_args()
    if args.cache == "False":
        args.cache = False
    return args


if __name__ == "__main__":
    train_model(parse_args())
