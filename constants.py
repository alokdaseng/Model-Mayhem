"""Shared constants matching the Model Mayhem PCB Defect Detection specification."""

from typing import Dict, Tuple

CLASS_NAMES: Tuple[str, ...] = (
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
)

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

CSV_COLUMNS: Tuple[str, ...] = (
    "image_name",
    "class_name",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "confidence",
)

# Primary metric mAP@50:90 uses 9 IoU thresholds: 0.50, 0.55, ..., 0.90
IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(9))

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

