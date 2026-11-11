"""Strict validator verifying predictions CSV against all Model Mayhem submission rules."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.constants import CLASS_NAMES, CSV_COLUMNS, IMAGE_EXTENSIONS
from src.utils import read_image_size


def index_image_sizes(images_dir: Optional[Path]) -> Dict[str, Tuple[int, int]]:
    """Build {filename: (width, height)} map if an images directory is supplied."""
    if images_dir is None:
        return {}
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    sizes: Dict[str, Tuple[int, int]] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            sizes[p.name] = read_image_size(p)
    return sizes


def validate_submission_csv(
    csv_path: Path,
    images_dir: Optional[Path] = None,
    allow_empty: bool = False,
) -> Tuple[bool, List[str], Dict[str, int]]:
    """Validate prediction CSV headers, class labels, box coordinates, and confidences."""
    errors: List[str] = []
    class_counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    seen_images = set()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False, [f"CSV file does not exist: {csv_path}"], class_counts

    img_sizes = index_image_sizes(images_dir)
    valid_classes = set(CLASS_NAMES)
    row_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return False, ["CSV file is empty or missing header row."], class_counts
        if tuple(reader.fieldnames) != CSV_COLUMNS:
            return (
                False,
                [f"Invalid CSV header {reader.fieldnames}; expected {list(CSV_COLUMNS)}"],
                class_counts,
            )

        for row_idx, row in enumerate(reader, start=2):
            row_count += 1
            if any(row.get(col) is None or str(row.get(col)).strip() == "" for col in CSV_COLUMNS):
                errors.append(f"Row {row_idx}: missing/blank value in {row}")
                if len(errors) >= 25:
                    break
                continue

            img_name = row["image_name"].strip()
            cls_name = row["class_name"].strip()
            seen_images.add(img_name)

            if cls_name not in valid_classes:
                errors.append(f"Row {row_idx}: invalid class_name '{cls_name}'")
            else:
                class_counts[cls_name] += 1

            try:
                x1 = float(row["x_min"])
                y1 = float(row["y_min"])
                x2 = float(row["x_max"])
                y2 = float(row["y_max"])
                conf = float(row["confidence"])
            except ValueError:
                errors.append(f"Row {row_idx}: non-numeric coordinate or confidence")
                continue

            if not (0.0 <= conf <= 1.0):
                errors.append(f"Row {row_idx}: confidence {conf} outside [0, 1]")
            if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
                errors.append(
                    f"Row {row_idx}: invalid box ({x1}, {y1}, {x2}, {y2}); expected 0 <= min < max"
                )

            if img_sizes:
                if img_name not in img_sizes:
                    errors.append(f"Row {row_idx}: image '{img_name}' not found in {images_dir}")
                else:
                    w, h = img_sizes[img_name]
                    if x2 > w + 1e-3 or y2 > h + 1e-3:
                        errors.append(
                            f"Row {row_idx}: box ({x2}, {y2}) exceeds image dimensions ({w}, {h})"
                        )

            if len(errors) >= 25:
                break

    if row_count == 0 and not allow_empty:
        errors.append("CSV has 0 prediction rows.")

    stats = {
        "rows": row_count,
        "unique_images": len(seen_images),
        "total_images_indexed": len(img_sizes),
    }
    return len(errors) == 0, errors, {**stats, **class_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Model Mayhem submission CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to predictions CSV.")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Optional test images folder to verify pixel bounds.",
    )
    parser.add_argument("--allow-empty", action="store_true", help="Allow 0 prediction rows.")
    args = parser.parse_args()

    ok, errors, stats = validate_submission_csv(args.csv, args.images, allow_empty=args.allow_empty)
    print(f"Submission audit for {args.csv}: {'PASSED' if ok else 'FAILED'}")
    print(f"Summary stats: {stats}")
    if errors:
        print("Errors found:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

