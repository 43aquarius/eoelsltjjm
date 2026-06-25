"""One-shot post-training pipeline: WBF inference -> threshold variants -> packaging.

Run after a training job finishes to produce all submission ZIPs from a single weights file.

Usage:
    python src/post_train_pipeline.py \\
        --weights runs/defect_yolov8s_dirt_rescue/weights/best.pt \\
        --work-suffix "YOLOv8s_dirt救援"
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], log_path: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}")
    if log_path:
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd).returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--work-suffix", required=True,
                   help="suffix added to the submission zip name (will be wrapped as YOLOv8x_<suffix>)")
    p.add_argument("--team", default="你们说的队")
    p.add_argument("--tag", default=None,
                   help="short tag for the submission folder name (default: derived from weights path)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[448, 512, 576])
    p.add_argument("--no-flip", action="store_true")
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")

    tag = args.tag or weights.parent.parent.name  # runs/<name>/weights/best.pt
    out_dir = ROOT / f"submission_{tag}_wbf"

    # 1) WBF inference
    flip_arg = [] if args.no_flip else ["--flip"]
    cmd = [PY, "src/predict_wbf.py",
           "--weights", str(weights),
           "--conf", "0.05",
           "--imgsz", *[str(x) for x in args.imgsz],
           "--wbf-iou", "0.55",
           "--out-dir", str(out_dir)] + flip_arg
    code = run(cmd)
    if code != 0:
        sys.exit(f"predict_wbf failed (exit {code})")

    # 2) threshold variants
    src_json = out_dir / "submission.json"
    src = json.load(open(src_json, encoding="utf-8"))
    for thr in [0.10, 0.15, 0.20]:
        thr_dir = ROOT / f"submission_{tag}_thr{thr:.2f}"
        (thr_dir / "per_image").mkdir(parents=True, exist_ok=True)
        out = []
        for r in src:
            keep = [a for a in r["annotations"] if a["confidence"] >= thr]
            rec = {"image_id": r["image_id"], "annotations": keep}
            out.append(rec)
            stem = Path(r["image_id"]).stem
            (thr_dir / "per_image" / f"{stem}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(thr_dir / "submission.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        n_box = sum(len(r["annotations"]) for r in out)
        print(f"  thr={thr}: wrote {n_box} boxes to {thr_dir}")

    # 3) package the main version + threshold variants
    work_names = [
        (out_dir, args.work_suffix),
        (ROOT / f"submission_{tag}_thr0.10", f"{args.work_suffix}_thr0.10"),
        (ROOT / f"submission_{tag}_thr0.15", f"{args.work_suffix}_thr0.15"),
        (ROOT / f"submission_{tag}_thr0.20", f"{args.work_suffix}_thr0.20"),
    ]
    for src_dir, work in work_names:
        cmd = [PY, "src/package_submission.py",
               "--source", str(src_dir / "per_image"),
               "--team", args.team,
               "--work", work,
               "--out-dir", str(ROOT / "release")]
        code = run(cmd)
        if code != 0:
            print(f"[warn] packaging failed for {work}")
    print("\n=== post-train pipeline complete ===")


if __name__ == "__main__":
    main()
