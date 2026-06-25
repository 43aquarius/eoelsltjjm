"""Run inference on the test set and write a submission JSON.

Each test image's predictions are written to one JSON object::

    {
      "image_id": "<filename>.jpg",
      "annotations": [
        {"label": "collision", "bbox": [x1, y1, x2, y2], "confidence": 0.93},
        ...
      ]
    }

By default writes one JSON per image to ``submission/`` and a combined
``submission.json`` (list of all per-image objects).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"
DEFAULT_WEIGHTS = ROOT / "runs" / "defect_yolov8n_cpu" / "weights" / "best.pt"
OUT_DIR = ROOT / "submission"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--test-dir", default=str(TEST_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--tta", action="store_true",
                   help="enable Ultralytics built-in TTA (multi-scale + flip); ~4x slower")
    p.add_argument("--limit", type=int, default=0, help="if > 0, only process the first N images (smoke test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights} — train first or pass --weights")

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir = out_dir / "per_image"
    per_image_dir.mkdir(exist_ok=True)

    images = sorted(test_dir.glob("*.jpg"))
    if args.limit > 0:
        images = images[: args.limit]
    print(f">>> {len(images)} test images, weights={weights.name}")

    model = YOLO(str(weights))

    combined: list[dict] = []
    for img_path in tqdm(images, desc="predict"):
        result = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device="cpu",
            augment=args.tta,
            verbose=False,
        )[0]

        names = result.names  # {id: name}
        annotations = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(xyxy, confs, clses):
                annotations.append({
                    "label": names[int(cls)],
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": float(conf),
                })

        record = {"image_id": img_path.name, "annotations": annotations}
        combined.append(record)
        with open(per_image_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    with open(out_dir / "submission.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    n_with_det = sum(1 for r in combined if r["annotations"])
    n_boxes = sum(len(r["annotations"]) for r in combined)
    print(f">>> wrote {len(combined)} records ({n_with_det} with detections, {n_boxes} boxes total)")
    print(f">>> per-image JSONs: {per_image_dir}")
    print(f">>> combined: {out_dir / 'submission.json'}")


if __name__ == "__main__":
    main()
