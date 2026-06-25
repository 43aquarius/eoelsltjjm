"""Visualize a small set of predictions for sanity-checking the model.

Picks samples from `submission/submission.json`:
- top-N by max-confidence per image
- mid-confidence samples (around the median)
- a couple of empty-detection images

Draws bbox + class + conf, writes to `submission/viz/`.

Run:
    python src/visualize.py
    python src/visualize.py --topn 3 --midn 3 --emptyn 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = ROOT / "submission" / "submission.json"
DEFAULT_TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"
DEFAULT_OUT = ROOT / "submission" / "viz"

# class → BGR color
COLORS = {
    "collision":      (0, 0, 255),     # red
    "dirt":           (0, 165, 255),   # orange
    "plain particle": (0, 255, 255),   # yellow
    "scratch":        (0, 255, 0),     # green
}


def imread_unicode(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise RuntimeError(f"imencode failed for {path}")
    buf.tofile(str(path))


def draw(img: np.ndarray, annotations: list[dict]) -> np.ndarray:
    out = img.copy()
    for a in annotations:
        x1, y1, x2, y2 = map(int, a["bbox"])
        color = COLORS.get(a["label"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{a['label']} {a['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    p.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--topn", type=int, default=3, help="top-N by max confidence")
    p.add_argument("--midn", type=int, default=3, help="N around median max-confidence")
    p.add_argument("--emptyn", type=int, default=2, help="N empty-detection samples")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.test_dir)

    with open(args.submission, encoding="utf-8") as f:
        records = json.load(f)

    with_det = [r for r in records if r["annotations"]]
    empty = [r for r in records if not r["annotations"]]

    # rank by max conf in the image
    with_det.sort(key=lambda r: max(a["confidence"] for a in r["annotations"]), reverse=True)

    picks: list[tuple[str, dict]] = []
    picks += [("top", r) for r in with_det[: args.topn]]
    if with_det and args.midn:
        mid_start = len(with_det) // 2
        picks += [("mid", r) for r in with_det[mid_start: mid_start + args.midn]]
    picks += [("empty", r) for r in empty[: args.emptyn]]

    print(f">>> drawing {len(picks)} samples to {out_dir}")
    for tag, rec in picks:
        img_path = test_dir / rec["image_id"]
        img = imread_unicode(img_path)
        if img is None:
            print(f"[warn] cannot read {img_path}")
            continue
        drawn = draw(img, rec["annotations"])
        out_name = f"{tag}_{Path(rec['image_id']).stem}.jpg"
        imwrite_unicode(out_dir / out_name, drawn)
        n = len(rec["annotations"])
        max_c = max((a["confidence"] for a in rec["annotations"]), default=0.0)
        print(f"  [{tag}] {rec['image_id']}  boxes={n}  max_conf={max_c:.2f}")


if __name__ == "__main__":
    main()
