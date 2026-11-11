"""Extract and organize the Roboflow PCB defect dataset with leak-free stem grouping."""

import argparse
import collections
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Tuple
import zipfile

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import CLASS_NAMES, IMAGE_EXTENSIONS


def extract_stem_and_group(filename: str) -> Tuple[str, str]:
    """Return (unique_pcb_stem, stratification_group) from a Roboflow filename."""
    stem = filename.split("_jpg.rf.")[0].split("_png.rf.")[0]
    if "." in stem:
        stem = Path(stem).stem
    # Filename format is typically <board_id>_<defect_type>_<idx>, e.g. 01_spur_10
    parts = stem.split("_")
    if len(parts) >= 3:
        strat_key = f"{parts[0]}_{'_'.join(parts[1:-1])}"
    elif len(parts) == 2:
        strat_key = parts[0]
    else:
        strat_key = "default"
    return stem, strat_key


def write_yolo_yaml(yaml_path: Path, dataset_root: Path, train_rel: str, val_rel: str, test_rel: str = "") -> None:
    """Write an Ultralytics-compatible data.yaml without requiring external PyYAML."""
    lines = [
        f"path: {dataset_root.resolve()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
    ]
    if test_rel:
        lines.append(f"test: {test_rel}")
    lines.extend(
        [
            f"nc: {len(CLASS_NAMES)}",
            "names:",
        ]
    )
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    """Create a relative symlink (zero disk overhead) or copy file."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        os.symlink(rel_src, dst)


def collect_samples(raw_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    """Collect (img_path, lbl_path, stem, strat_key) across raw splits."""
    samples: List[Tuple[Path, Path, str, str]] = []
    seen_filenames = set()
    for split in ("train", "valid", "test"):
        img_dir = raw_dir / split / "images"
        lbl_dir = raw_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if img_path.name in seen_filenames:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            seen_filenames.add(img_path.name)
            stem, strat_key = extract_stem_and_group(img_path.name)
            samples.append((img_path, lbl_path, stem, strat_key))
    return samples


def build_clean_grouped_split(
    raw_dir: Path,
    clean_dir: Path,
    val_ratio: float = 0.14,
    seed: int = 42,
    copy_files: bool = False,
) -> Dict[str, int]:
    """Build a leak-free train/val split where no augmented stem crosses splits."""
    samples = collect_samples(raw_dir)
    if not samples:
        raise RuntimeError(f"No samples found under {raw_dir}")

    # Group samples by stem and assign each stem to its stratification bucket
    by_stem: Dict[str, List[Tuple[Path, Path]]] = collections.defaultdict(list)
    stem_strat: Dict[str, str] = {}
    for img_path, lbl_path, stem, strat_key in samples:
        by_stem[stem].append((img_path, lbl_path))
        stem_strat[stem] = strat_key

    by_strat: Dict[str, List[str]] = collections.defaultdict(list)
    for stem, strat_key in stem_strat.items():
        by_strat[strat_key].append(stem)

    rng = random.Random(seed)
    val_stems = set()
    train_stems = set()

    for strat_key in sorted(by_strat.keys()):
        stems = sorted(by_strat[strat_key])
        rng.shuffle(stems)
        n_val = max(1, int(round(len(stems) * val_ratio))) if len(stems) >= 3 else 0
        for idx, stem in enumerate(stems):
            if idx < n_val:
                val_stems.add(stem)
            else:
                train_stems.add(stem)

    assert train_stems.isdisjoint(val_stems), "Stem leakage detected between train and val!"

    if clean_dir.exists():
        shutil.rmtree(clean_dir)

    counts = {"train_images": 0, "val_images": 0, "train_stems": len(train_stems), "val_stems": len(val_stems)}
    for split_name, stem_set in (("train", train_stems), ("val", val_stems)):
        out_img_dir = clean_dir / split_name / "images"
        out_lbl_dir = clean_dir / split_name / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem in sorted(stem_set):
            for img_path, lbl_path in by_stem[stem]:
                link_or_copy(img_path, out_img_dir / img_path.name, copy_files=copy_files)
                link_or_copy(lbl_path, out_lbl_dir / lbl_path.name, copy_files=copy_files)
                counts[f"{split_name}_images"] += 1

    write_yolo_yaml(clean_dir / "data.yaml", clean_dir, "train/images", "val/images")
    return counts


def prepare_dataset(
    zip_path: Path,
    out_dir: Path,
    val_ratio: float = 0.14,
    seed: int = 42,
    copy_files: bool = False,
    force_extract: bool = False,
) -> Dict[str, int]:
    """Extract ZIP into out_dir/raw and create leak-free split in out_dir/clean."""
    raw_dir = out_dir / "raw"
    clean_dir = out_dir / "clean"

    if force_extract or not (raw_dir / "train" / "images").exists():
        if not zip_path.exists():
            raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prepare_data] Extracting {zip_path.name} -> {raw_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    else:
        print(f"[prepare_data] Using existing extracted raw dataset at {raw_dir}")

    write_yolo_yaml(
        raw_dir / "data.yaml",
        raw_dir,
        "train/images",
        "valid/images",
        "test/images",
    )

    counts = build_clean_grouped_split(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        val_ratio=val_ratio,
        seed=seed,
        copy_files=copy_files,
    )
    print(
        f"[prepare_data] Clean leak-free split ready at {clean_dir}:\n"
        f"  Train: {counts['train_images']} images ({counts['train_stems']} unique PCB stems)\n"
        f"  Val  : {counts['val_images']} images ({counts['val_stems']} unseen PCB stems)\n"
        f"  YAML : {clean_dir / 'data.yaml'}"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Model Mayhem PCB dataset.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("pcb defect detection.v2i.yolov8.zip"),
        help="Path to Roboflow YOLOv8 dataset zip.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output dataset directory (creates raw/ and clean/).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.14, help="Fraction of unique stems for clean val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stem split.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Re-extract zip even if raw/ exists.")
    args = parser.parse_args()

    prepare_dataset(
        zip_path=args.zip,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_files=args.copy,
        force_extract=args.force,
    )


if __name__ == "__main__":
    main()
