"""Offline Copy-Paste augmentation for few-shot defect detection.

Pipeline:
    1. Scan the train split:
       - Negative images (non-empty label .txt)  → extract defect crops via polygon mask.
       - Positive images (empty label .txt)      → use as backgrounds.
    2. For each positive background, synthesize K augmented variants:
       paste N random defect crops with random scale + 90° rotation, feathered blend.
    3. Write to dataset/images/train/aug_*.jpg and dataset/labels/train/aug_*.txt.

The val split is *never* read or modified — avoids leakage.

Run:
    python src/augment_copy_paste.py --per-positive 3 --defects-per-image 1 3
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import CLASS_NAMES, CLASS_TO_ID, LEAKED_STEMS

ROOT = Path(__file__).resolve().parent.parent
DATASET_TRAIN_IMG = ROOT / "dataset" / "images" / "train"
DATASET_TRAIN_LBL = ROOT / "dataset" / "labels" / "train"
ORIGINAL_NEG_ROOT = ROOT / "初赛数据" / "训练集" / "负样本"


# ---------- IO helpers (cv2 can't handle Chinese paths on Windows) ----------

def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"imencode failed for {path}")
    buf.tofile(str(path))


# ---------- defect crop extraction ----------

@dataclass
class DefectCrop:
    class_id: int
    label: str
    rgb: np.ndarray   # H x W x 3
    mask: np.ndarray  # H x W, uint8 0/255

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]  # (W, H)


def build_json_index() -> dict[str, Path]:
    """Map image stem → original LabelMe JSON path. Skips leaked test stems."""
    idx: dict[str, Path] = {}
    for jp in ORIGINAL_NEG_ROOT.rglob("*.json"):
        if jp.stem in LEAKED_STEMS:
            continue
        idx[jp.stem] = jp
    return idx


def extract_defects_from_json(json_path: Path) -> list[DefectCrop]:
    """Open the original image + JSON, return one DefectCrop per polygon."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    img_path = json_path.with_suffix(".jpg")
    img = imread_unicode(img_path)
    if img is None:
        return []
    H, W = img.shape[:2]

    crops: list[DefectCrop] = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_TO_ID:
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        # build full-image mask then crop
        full_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(full_mask, [pts], 255)
        rgb_crop = img[y1:y2, x1:x2].copy()
        mask_crop = full_mask[y1:y2, x1:x2].copy()
        if mask_crop.sum() == 0:
            continue

        # feather the mask edge for softer blend
        mask_crop = cv2.GaussianBlur(mask_crop, (5, 5), 0)
        crops.append(DefectCrop(class_id=CLASS_TO_ID[label], label=label,
                                rgb=rgb_crop, mask=mask_crop))
    return crops


# ---------- placement helpers ----------

def transform_crop(crop: DefectCrop, scale: float, rot90: int) -> DefectCrop:
    new_w = max(4, int(crop.rgb.shape[1] * scale))
    new_h = max(4, int(crop.rgb.shape[0] * scale))
    rgb = cv2.resize(crop.rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(crop.mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if rot90:
        rgb = np.rot90(rgb, rot90).copy()
        mask = np.rot90(mask, rot90).copy()
    return DefectCrop(class_id=crop.class_id, label=crop.label, rgb=rgb, mask=mask)


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def paste(bg: np.ndarray, crop: DefectCrop, x: int, y: int) -> None:
    """Alpha-blend crop into bg at top-left (x, y). Mutates bg in place."""
    h, w = crop.rgb.shape[:2]
    alpha = crop.mask.astype(np.float32) / 255.0
    alpha = alpha[..., None]  # H x W x 1
    region = bg[y:y + h, x:x + w].astype(np.float32)
    blended = region * (1.0 - alpha) + crop.rgb.astype(np.float32) * alpha
    bg[y:y + h, x:x + w] = blended.astype(np.uint8)


# ---------- main augmentation loop ----------

def to_yolo_line(class_id: int, x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> str:
    xc = (x1 + x2) / 2.0 / W
    yc = (y1 + y2) / 2.0 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def split_train_images() -> tuple[list[Path], list[Path]]:
    """Return (positive_bgs, negative_imgs) from the existing train split."""
    positives, negatives = [], []
    for img_path in sorted(DATASET_TRAIN_IMG.glob("*.jpg")):
        if img_path.name.startswith("aug_"):
            continue  # don't recurse on prior aug output
        lbl_path = DATASET_TRAIN_LBL / (img_path.stem + ".txt")
        text = lbl_path.read_text(encoding="utf-8").strip() if lbl_path.exists() else ""
        (positives if text == "" else negatives).append(img_path)
    return positives, negatives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-positive", type=int, default=3,
                   help="how many augmented variants per positive background")
    p.add_argument("--defects-per-image", type=int, nargs=2, default=[1, 3],
                   help="min/max defects to paste per augmented image")
    p.add_argument("--scale-range", type=float, nargs=2, default=[0.7, 1.4])
    p.add_argument("--max-place-tries", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear-existing", action="store_true",
                   help="delete previous aug_*.jpg/txt before generating")
    p.add_argument("--class-weights", type=float, nargs=4, default=None,
                   metavar=("collision", "dirt", "particle", "scratch"),
                   help="oversampling weights per class for crop selection (default: uniform). "
                        "e.g. --class-weights 1 20 1 1 makes dirt 20x more likely per draw.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    if args.clear_existing:
        n_removed = 0
        for f in list(DATASET_TRAIN_IMG.glob("aug_*.jpg")) + list(DATASET_TRAIN_LBL.glob("aug_*.txt")):
            f.unlink()
            n_removed += 1
        print(f">>> removed {n_removed} prior aug_* files")

    print(">>> indexing original LabelMe JSONs")
    json_index = build_json_index()
    print(f"    {len(json_index)} JSONs available")

    positives, negatives = split_train_images()
    print(f">>> train split: {len(positives)} backgrounds, {len(negatives)} defect images")

    print(">>> extracting defect crops from train negatives")
    crops: list[DefectCrop] = []
    for img_path in negatives:
        jp = json_index.get(img_path.stem)
        if jp is None:
            print(f"[warn] no JSON for {img_path.name}")
            continue
        crops.extend(extract_defects_from_json(jp))

    from collections import Counter
    print(f"    {len(crops)} defect crops total, per-class:",
          dict(Counter(c.label for c in crops)))

    if not crops:
        raise SystemExit("no defect crops extracted — abort")

    # build per-crop sampling weights — default uniform; --class-weights rebalances
    if args.class_weights:
        cw = {i: w for i, w in enumerate(args.class_weights)}
        crop_weights = [cw[c.class_id] for c in crops]
        print(f"    class weights: {dict(zip(CLASS_NAMES, args.class_weights))}")
    else:
        crop_weights = [1.0] * len(crops)

    min_n, max_n = args.defects_per_image
    s_lo, s_hi = args.scale_range

    print(f">>> generating {args.per_positive}× variants per background")
    n_written = 0
    n_pasted = 0
    class_counter: Counter[str] = Counter()

    for bg_path in positives:
        bg_img = imread_unicode(bg_path)
        if bg_img is None:
            print(f"[warn] cannot read {bg_path}")
            continue
        H, W = bg_img.shape[:2]

        for variant in range(args.per_positive):
            canvas = bg_img.copy()
            n_defects = rng.randint(min_n, max_n)
            placed: list[tuple[int, int, int, int]] = []
            yolo_lines: list[str] = []

            for _ in range(n_defects):
                crop = rng.choices(crops, weights=crop_weights, k=1)[0]
                scale = rng.uniform(s_lo, s_hi)
                rot = rng.randint(0, 3)
                tcrop = transform_crop(crop, scale, rot)
                cw, ch = tcrop.size
                if cw >= W - 2 or ch >= H - 2:
                    continue

                for _ in range(args.max_place_tries):
                    x = rng.randint(0, W - cw)
                    y = rng.randint(0, H - ch)
                    box = (x, y, x + cw, y + ch)
                    if any(boxes_overlap(box, b) for b in placed):
                        continue
                    placed.append(box)
                    paste(canvas, tcrop, x, y)
                    yolo_lines.append(to_yolo_line(tcrop.class_id, *box, W, H))
                    class_counter[tcrop.label] += 1
                    break

            if not yolo_lines:
                continue  # no defects placed → don't bother writing
            out_stem = f"aug_{bg_path.stem}_v{variant}"
            imwrite_unicode(DATASET_TRAIN_IMG / f"{out_stem}.jpg", canvas)
            (DATASET_TRAIN_LBL / f"{out_stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
            n_written += 1
            n_pasted += len(yolo_lines)

    print(f">>> wrote {n_written} augmented images, {n_pasted} pasted defects")
    print(f">>> per-class boxes added: {dict(class_counter)}")


if __name__ == "__main__":
    main()
