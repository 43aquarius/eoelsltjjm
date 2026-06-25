"""Filter an existing submission.json by confidence threshold(s).

For each threshold, write a new submission file and print stats.

Run:
    python src/filter_by_conf.py
    python src/filter_by_conf.py --thresholds 0.1 0.2 0.3 0.4
    python src/filter_by_conf.py --input submission/submission.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "submission" / "submission.json"
DEFAULT_OUT_DIR = ROOT / "submission"


def filter_one(records: list[dict], threshold: float) -> list[dict]:
    out = []
    for r in records:
        kept = [a for a in r["annotations"] if a["confidence"] >= threshold]
        out.append({"image_id": r["image_id"], "annotations": kept})
    return out


def summarize(name: str, records: list[dict]) -> None:
    n_with = sum(1 for r in records if r["annotations"])
    boxes = [a for r in records for a in r["annotations"]]
    per_class = Counter(a["label"] for a in boxes)
    print(f"  [{name}] images_with_det={n_with}/{len(records)} ({n_with/len(records)*100:.1f}%)  "
          f"boxes={len(boxes)}  per_class={dict(per_class)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = p.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8") as f:
        records = json.load(f)

    summarize("source", records)
    for thr in args.thresholds:
        filtered = filter_one(records, thr)
        out_path = out_dir / f"submission_conf_{thr:.2f}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        summarize(f"conf>={thr:.2f}", filtered)
        print(f"    → {out_path}")


if __name__ == "__main__":
    main()
