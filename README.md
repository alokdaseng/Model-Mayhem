# Model Mayhem — PCB Defect Detection

Object detection and localization pipeline for detecting six classes of Printed Circuit Board (PCB) manufacturing defects (`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`), optimized for **mAP@50:90**, **mAP@50**, and **Macro Recall**.

---

## 1. Key Engineering Highlights

1. **Leak-Free Stem-Grouped Validation (`scripts/prepare_data.py`)**
   - Audits of the 2,586-image Roboflow v2 export revealed 693 unique underlying PCB board stems, of which 473 cross across the default `train`/`valid`/`test` splits due to rotated/resized clones.
   - `prepare_data.py` builds a stratified, **zero-leakage** `train`/`val` split grouped by PCB stem (`data/clean/`) so validation metrics accurately measure generalization to unseen boards.

2. **Small-Object & High-IoU Optimization (`train.py`)**
   - PCB defects are small (`~2.9%` median normalized width/height) and judged heavily on tight localization (`mAP@50:90` across `0.50:0.05:0.90`).
   - Trains **YOLO11 (`yolo11m` / `yolo11s`)** with AdamW cosine LR scheduling, zero random erasing (`erasing=0.0` so tiny defects are never masked), orientation/shear augmentation (`--degrees`, `--shear`, `--perspective`), calibrated classification gain (`--cls-gain`), and late-stage mosaic turn-off (`--close-mosaic`) for tight bounding-box refinement.

3. **4-Angle Rotation WBF, Optional High-Res Tiling & Multi-CSV Fusion (`infer.py`, `src/utils.py`)**
   - Uses global inference by default (`--tile-threshold 0`) with optional overlapping high-resolution sliding crops when `--tile-threshold > 0` (`mode="tile_stitch"`).
   - Supports **4-angle (`0°/90°/180°/270°`) rotation-consensus WBF (`--rot-tta`)**, standard flip/scale TTA (`--tta`), direct multi-model WBF (`--weights modelA.pt modelB.pt`), and cross-resolution CSV fusion (`--fuse-csvs`).

4. **Exact Competition Metric Suite & Submission Audit (`src/metrics.py`, `validate_submission.py`)**
   - Standalone COCO-101 evaluator implementing the exact `mAP@50:90` across `[0.50, 0.55, ..., 0.90]`, `mAP@50`, and `Macro Recall`.
   - Built-in strict CSV validator enforcing original pixel coordinates, valid labels, `[0, 1]` confidences, and exact header structure.

---

## 2. Project Structure

```text
Model Mayhem/
├── requirements.txt
├── README.md
├── train.py                          # Reproducible training + auto evaluation
├── infer.py                          # Submission CSV generator (single/ensemble/TTA/tiling)
├── validate_submission.py            # Strict submission CSV rule checker
├── scripts/
│   └── prepare_data.py               # Extracts ZIP + builds leak-free grouped split
├── src/
│   ├── constants.py                  # Canonical class names, CSV columns, IoU thresholds
│   ├── metrics.py                    # Standalone mAP@50:90, mAP@50, Macro Recall evaluator
│   └── utils.py                      # Image metadata, pixel geometry, NMS, and WBF
├── notebooks/
│   └── ModelMayhem_Colab_GPU.ipynb   # One-click Google Colab GPU training & benchmarking
└── tests/
    └── test_pipeline.py              # Unit tests for metrics, WBF, geometry, validation
```

---

## 3. Quickstart & Execution Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare the Dataset
Place `pcb defect detection.v2i.yolov8.zip` in the repository root and run:
```bash
python3 scripts/prepare_data.py --zip "pcb defect detection.v2i.yolov8.zip" --out data
```
This creates `data/raw/` and a leak-free split at `data/clean/data.yaml`.

### Step 3: Train the Model
**GPU (recommended / Google Colab T4):**
```bash
python3 train.py --model yolo11s.pt --data data/clean/data.yaml --imgsz 1024 --epochs 40 --batch -1 --box-gain 8.5 --degrees 0 --shear 0 --name pcb_yolo11s_1024
```
*(Orientation-hardened backbone option with tilt/shear invariance:)*
```bash
python3 train.py --model yolo11m.pt --data data/clean/data.yaml --imgsz 768 --epochs 28 --batch -1 --cls-gain 1.0 --degrees 15.0 --shear 6.0 --perspective 0.0003 --mosaic 0.35 --scale 0.25 --close-mosaic 8 --name pcb_yolo11m_768
```

### Step 4: Generate Submission Predictions CSV
Run inference on any folder of unseen test PCB images (including nested folders via `--recursive`):
```bash
python3 infer.py \
  --source path/to/test_images \
  --weights weights/best.pt \
  --output predictions.csv \
  --imgsz 1024 \
  --conf 0.01 \
  --iou 0.60 \
  --rot-tta \
  --tile-threshold 0 \
  --recursive
```
*(Optional multi-resolution CSV WBF fusion across separate model runs:)*
```bash
python3 infer.py \
  --fuse-csvs runs/model_a.csv runs/model_b.csv \
  --model-weights 1.2 1.0 \
  --wbf-iou 0.55 \
  --iou 0.60 \
  --source path/to/test_images \
  --output predictions.csv
```

### Step 5: Verify Submission CSV & Metrics
Audit CSV compliance against all competition rules:
```bash
python3 validate_submission.py --csv predictions.csv --images path/to/test_images
```
Evaluate against YOLO ground-truth labels using exact `mAP@50:90`:
```bash
python3 -m src.metrics --pred predictions.csv --images data/clean/val/images --labels data/clean/val/labels
```

### Unit Tests
```bash
python3 -m unittest discover -s tests -v
```

---

## 4. Verified Validation Results (`weights/best.pt` — `YOLO11s @ 1024`, Stage 2)

| Evaluation Subset | Infer Config | `mAP@50:90` | `mAP@50` | `Macro Recall@50` |
|---|---|---:|---:|---:|
| **`clean/val` Overall (`320` unseen imgs)** | `imgsz=1024`, `conf=0.01`, `iou=0.60`, `tile=0` | `0.3061` | `0.6075` | `0.9325` |
| **`clean/val` HiRes (`85` imgs `>640px`)** | `imgsz=1024`, `conf=0.01`, `iou=0.60`, `tile=0` | `0.5850` | `0.9671` | `0.9738` |
| **`clean/val` Upright `640×640` (`10` imgs)** | `imgsz=1024`, `conf=0.01`, `iou=0.60`, `tile=0` | `0.6433` | `1.0000` | `1.0000` |
| **`raw/test` (`189` tilted `640×640` imgs)** | `imgsz=1024`, `conf=0.01`, `iou=0.60`, `--rot-tta` | `0.1762` | `0.4561` | `0.9656` |

