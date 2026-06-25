"""Simple single-pass inference (no TTA, no WBF). Robust to mid-run kills via incremental write.

Usage:
    python src/predict_simple.py --weights ... --out ... --imgsz 512
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.45)
    args = p.parse_args()

    out_dir = Path(args.out)
    per_dir = out_dir / "per_image"
    per_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(Path(TEST_DIR).glob("*.jpg"))
    print(f">>> {len(images)} test images, weights={Path(args.weights).name}, imgsz={args.imgsz}")

    # Skip already-processed images (incremental resume)
    done_stems = {p.stem for p in per_dir.glob("*.json")}
    todo = [p for p in images if p.stem not in done_stems]
    print(f">>> already done: {len(done_stems)}, todo: {len(todo)}")

    model = YOLO(args.weights)

    for img_path in tqdm(todo, desc="predict"):
        r = model.predict(source=str(img_path), conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, device="cpu", verbose=False)[0]
        annotations = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for box, c, k in zip(xyxy, confs, clses):
                annotations.append({
                    "label": names[int(k)],
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": float(c),
                })
        rec = {"image_id": img_path.name, "annotations": annotations}
        (per_dir / f"{img_path.stem}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # assemble submission.json
    combined = []
    for img_path in images:
        jp = per_dir / f"{img_path.stem}.json"
        if jp.exists():
            combined.append(json.loads(jp.read_text(encoding="utf-8")))
    (out_dir / "submission.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f">>> submission.json with {len(combined)} records")


if __name__ == "__main__":
    main()
