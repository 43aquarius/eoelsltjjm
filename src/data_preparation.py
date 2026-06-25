"""Convert LabelMe annotations to YOLO format and build the train/val dataset.

Reads from `初赛数据/训练集/{正样本,负样本}` and writes a YOLO-style dataset to
`dataset/{images,labels}/{train,val}` along with `dataset/data.yaml`.

Run:
    python src/data_preparation.py
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import CLASS_NAMES, CLASS_TO_ID, LEAKED_STEMS

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT / "初赛数据" / "训练集"
NEG_ROOT = RAW_ROOT / "负样本"
POS_ROOT = RAW_ROOT / "正样本"
DATASET_ROOT = ROOT / "dataset"

VAL_RATIO = 0.2
SEED = 42


def polygon_to_yolo_bbox(points: list[list[float]], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Return (x_center, y_center, width, height) normalized to [0, 1]."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(min(xs), 0.0), min(max(xs), float(img_w))
    y1, y2 = max(min(ys), 0.0), min(max(ys), float(img_h))
    if x2 <= x1 or y2 <= y1:
        return None  # degenerate; caller must skip
    x_c = (x1 + x2) / 2.0 / img_w
    y_c = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_c, y_c, w, h


def load_negative_samples() -> list[dict]:
    """Each item: {image_path, yolo_lines: [str], primary_class: str}."""
    items = []
    n_skipped_leak = 0
    for json_path in sorted(NEG_ROOT.rglob("*.json")):
        if json_path.stem in LEAKED_STEMS:
            n_skipped_leak += 1
            continue
        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            print(f"[warn] missing image for {json_path}, skipping")
            continue

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        img_w, img_h = data["imageWidth"], data["imageHeight"]

        yolo_lines = []
        label_counter: Counter[str] = Counter()
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in CLASS_TO_ID:
                print(f"[warn] unknown label '{label}' in {json_path}, skipping shape")
                continue
            bbox = polygon_to_yolo_bbox(shape["points"], img_w, img_h)
            if bbox is None:
                continue
            cls_id = CLASS_TO_ID[label]
            yolo_lines.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
            label_counter[label] += 1

        if not yolo_lines:
            print(f"[warn] no valid shapes for {json_path}, skipping image")
            continue

        primary = label_counter.most_common(1)[0][0]
        items.append({"image_path": img_path, "yolo_lines": yolo_lines, "primary_class": primary})
    if n_skipped_leak:
        print(f"    [compliance] skipped {n_skipped_leak} negative images whose stems match test set")
    return items


def load_positive_samples() -> list[dict]:
    """Positive (defect-free) images get empty label files — YOLO treats them as background."""
    items = []
    n_skipped_leak = 0
    for img_path in sorted(POS_ROOT.glob("*.jpg")):
        if img_path.stem in LEAKED_STEMS:
            n_skipped_leak += 1
            continue
        items.append({"image_path": img_path, "yolo_lines": [], "primary_class": "background"})
    if n_skipped_leak:
        print(f"    [compliance] skipped {n_skipped_leak} positive images whose stems match test set")
    return items


def stratified_split(items: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for it in items:
        by_class.setdefault(it["primary_class"], []).append(it)
    train, val = [], []
    for cls, group in by_class.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) >= 5 else max(1, len(group) // 5)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_split(items: list[dict], split: str) -> None:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for it in items:
        src_img: Path = it["image_path"]
        # Use the bare filename — original names are unique across subfolders (timestamps + IDs).
        dst_img = img_dir / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        # Always (re)write the label, even if empty (background image).
        (lbl_dir / (src_img.stem + ".txt")).write_text("\n".join(it["yolo_lines"]), encoding="utf-8")


def write_data_yaml() -> None:
    data = {
        "path": str(DATASET_ROOT).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    with open(DATASET_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def summarize(name: str, items: list[dict]) -> None:
    cls_counter: Counter[str] = Counter()
    n_boxes = 0
    for it in items:
        cls_counter[it["primary_class"]] += 1
        n_boxes += len(it["yolo_lines"])
    print(f"  [{name}] images={len(items)}  boxes={n_boxes}  per-primary={dict(cls_counter)}")


def main() -> None:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    DATASET_ROOT.mkdir()

    print(">>> scanning negative samples (with annotations)")
    negatives = load_negative_samples()
    print(f"    {len(negatives)} annotated images")

    print(">>> scanning positive samples (background)")
    positives = load_positive_samples()
    print(f"    {len(positives)} background images")

    all_items = negatives + positives
    train_items, val_items = stratified_split(all_items, VAL_RATIO, SEED)

    print(">>> split summary")
    summarize("train", train_items)
    summarize("val", val_items)

    print(">>> writing splits")
    write_split(train_items, "train")
    write_split(val_items, "val")
    write_data_yaml()
    print(f">>> done. dataset at {DATASET_ROOT}")


if __name__ == "__main__":
    main()
