"""JSON post-processing optimizer.

Applies general-purpose tricks to a submission to bump mAP@0.5 without touching the model:
  - White-region filter: drop boxes whose region is uniformly white (README says white
    areas are NOT defects, so any high-conf box on a white patch is a guaranteed FP).
  - BBox expand: scale each box around its center by (1 + ratio); evaluator uses
    IoU=0.5, so a slight expand can convert near-misses into TPs.
  - Small-box filter: drop boxes smaller than --min-area pixels.
  - Per-image limit: keep top-N by confidence per image.
  - Per-class-per-image limit: keep top-K per (image, class).

Run:
    python src/postprocess.py \\
        --input submission_v123_v8sR_mosaic0_30_170 \\
        --output submission_v123_pp_default \\
        --white-thr 220 --white-frac 0.85 \\
        --bbox-expand 0.05
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def is_white_region(img_gray, box, white_thr, white_frac):
    """Return True if the cropped region is mostly bright/uniform (likely background)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = img_gray.shape
    x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
    crop = img_gray[y1:y2, x1:x2]
    if crop.size == 0:
        return True
    bright_frac = (crop >= white_thr).mean()
    return bright_frac >= white_frac


def expand_bbox(box, ratio, img_w, img_h):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio / 2, h * ratio / 2
    nx1 = max(0.0, x1 - dx)
    ny1 = max(0.0, y1 - dy)
    nx2 = min(float(img_w), x2 + dx)
    ny2 = min(float(img_h), y2 + dy)
    return [nx1, ny1, nx2, ny2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--white-thr", type=int, default=220, help="pixel intensity to call 'white' (0-255)")
    p.add_argument("--white-frac", type=float, default=0.85,
                   help="if this fraction of pixels in the box are >= white-thr, drop the box")
    p.add_argument("--disable-white", action="store_true")
    p.add_argument("--bbox-expand", type=float, default=0.0,
                   help="expand each bbox by this ratio (e.g. 0.05 = 5% larger). Negative shrinks.")
    p.add_argument("--min-area", type=float, default=0.0, help="drop boxes smaller than this area (px²)")
    p.add_argument("--max-per-image", type=int, default=0, help="0 = no limit")
    p.add_argument("--max-per-class", type=int, nargs="+", default=None,
                   help="per-class max boxes per image. Order: collision dirt particle scratch")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    (out_dir / "per_image").mkdir(parents=True, exist_ok=True)

    if not (in_dir / "per_image").is_dir():
        raise SystemExit(f"{in_dir}/per_image not found")

    class_order = ["collision", "dirt", "plain particle", "scratch"]
    per_class_max = None
    if args.max_per_class:
        if len(args.max_per_class) != 4:
            raise SystemExit("--max-per-class needs 4 ints")
        per_class_max = dict(zip(class_order, args.max_per_class))

    cnt_input = Counter()
    cnt_output = Counter()
    n_dropped_white = 0
    n_dropped_small = 0
    n_dropped_perimg = 0
    n_dropped_perclass = 0

    out_records = []
    for jp in sorted((in_dir / "per_image").glob("*.json")):
        rec = json.loads(jp.read_text(encoding="utf-8"))
        anns = rec["annotations"]
        for a in anns:
            cnt_input[a["label"]] += 1

        # need image for white check / expand bound clipping
        img_path = TEST_DIR / rec["image_id"]
        need_img = (not args.disable_white) or args.bbox_expand != 0
        gray = None
        img_w = img_h = None
        if need_img:
            bgr = imread_unicode(img_path)
            if bgr is not None:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                img_h, img_w = gray.shape

        # 1) filter
        kept = []
        for a in anns:
            box = a["bbox"]
            w, h = box[2] - box[0], box[3] - box[1]
            if args.min_area > 0 and w * h < args.min_area:
                n_dropped_small += 1
                continue
            if not args.disable_white and gray is not None:
                if is_white_region(gray, box, args.white_thr, args.white_frac):
                    n_dropped_white += 1
                    continue
            # 2) expand
            if args.bbox_expand != 0 and img_w:
                a = dict(a)  # copy
                a["bbox"] = expand_bbox(box, args.bbox_expand, img_w, img_h)
            kept.append(a)

        # 3) per-class limit
        if per_class_max:
            by_cls = {}
            for a in kept:
                by_cls.setdefault(a["label"], []).append(a)
            new_kept = []
            for cls, group in by_cls.items():
                lim = per_class_max.get(cls, 0)
                if lim <= 0:
                    new_kept.extend(group)
                else:
                    group.sort(key=lambda x: x["confidence"], reverse=True)
                    new_kept.extend(group[:lim])
                    n_dropped_perclass += max(0, len(group) - lim)
            kept = new_kept

        # 4) per-image limit
        if args.max_per_image > 0 and len(kept) > args.max_per_image:
            kept.sort(key=lambda x: x["confidence"], reverse=True)
            n_dropped_perimg += len(kept) - args.max_per_image
            kept = kept[: args.max_per_image]

        for a in kept:
            cnt_output[a["label"]] += 1
        new_rec = {"image_id": rec["image_id"], "annotations": kept}
        out_records.append(new_rec)
        (out_dir / "per_image" / jp.name).write_text(
            json.dumps(new_rec, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "submission.json").write_text(
        json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")

    n_in = sum(cnt_input.values())
    n_out = sum(cnt_output.values())
    print(f">>> input boxes: {n_in}, output boxes: {n_out}")
    print(f"    dropped white: {n_dropped_white}, small: {n_dropped_small}, "
          f"per-image: {n_dropped_perimg}, per-class: {n_dropped_perclass}")
    print(f"    in per class:  {dict(cnt_input)}")
    print(f"    out per class: {dict(cnt_output)}")


if __name__ == "__main__":
    main()
