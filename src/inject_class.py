"""Inject high-conf predictions of a specific class from one submission into another.

Use case: v39 has good particle/scratch/collision but only 35 low-conf dirt boxes.
Take v8s rescue's high-conf dirt boxes and add them to v39 — boost dirt class
without changing other classes.

Run:
    python src/inject_class.py \\
        --base submission_v39_v8s_30_170 \\
        --source submission_v8s_5scale_tta \\
        --class dirt --min-conf 0.20 \\
        --out submission_v39_plus_dirt_boost
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


def iou(a, b):
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - inter
    return inter / ua


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--class", dest="cls", required=True)
    p.add_argument("--min-conf", type=float, default=0.20)
    p.add_argument("--out", required=True)
    p.add_argument("--iou-merge", type=float, default=0.5,
                   help="if a source box overlaps a base box of same class by this IoU, skip (avoid dup)")
    args = p.parse_args()

    base_dir = Path(args.base)
    src_dir = Path(args.source)
    out_dir = Path(args.out)
    (out_dir / "per_image").mkdir(parents=True, exist_ok=True)

    base_sub = json.load(open(base_dir / "submission.json", encoding="utf-8"))
    src_sub = json.load(open(src_dir / "submission.json", encoding="utf-8"))

    src_by_id = {r["image_id"]: r["annotations"] for r in src_sub}

    n_injected = 0
    n_kept_base = 0
    out_records = []
    for r in base_sub:
        anns = list(r["annotations"])
        n_kept_base += len(anns)
        existing_cls = [a["bbox"] for a in anns if a["label"] == args.cls]
        for a in src_by_id.get(r["image_id"], []):
            if a["label"] != args.cls:
                continue
            if a["confidence"] < args.min_conf:
                continue
            # skip if overlaps existing of same class significantly
            if any(iou(a["bbox"], b) >= args.iou_merge for b in existing_cls):
                continue
            anns.append(a)
            n_injected += 1
        new_rec = {"image_id": r["image_id"], "annotations": anns}
        out_records.append(new_rec)
        stem = Path(r["image_id"]).stem
        (out_dir / "per_image" / f"{stem}.json").write_text(
            json.dumps(new_rec, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "submission.json").write_text(
        json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"injected {n_injected} {args.cls} boxes (conf >= {args.min_conf}) into base")
    print(f"base kept: {n_kept_base}")
    print(f"new totals per class:")
    c = Counter(a["label"] for r in out_records for a in r["annotations"])
    for k, v in sorted(c.items()): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
