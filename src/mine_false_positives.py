"""Run inference on the positive train images (known defect-free) — every detection is a FP.

Outputs:
- A report (printed) of FP counts per image, total FPs, mean confidence per class.
- A mining JSON listing FP boxes with class+conf+image_id.
- Optional: write visualization for the top-K FP images.

Run:
    python src/mine_false_positives.py --weights runs/defect_yolov8n_ft/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
POS_DIR = ROOT / "初赛数据" / "训练集" / "正样本"
DEFAULT_OUT = ROOT / "fp_mining"


def imread_unicode(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


COLORS = {0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 255), 3: (0, 255, 0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--pos-dir", default=str(POS_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--viz-topk", type=int, default=6, help="visualize top-K worst FP images")
    args = p.parse_args()

    pos_dir = Path(args.pos_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    images = sorted(pos_dir.glob("*.jpg"))
    print(f">>> {len(images)} positive (defect-free) images")

    model = YOLO(args.weights)
    names: dict[int, str] = {}

    fp_records: list[dict] = []
    per_image_count = Counter()
    per_image_max_conf: dict[str, float] = {}
    per_class_confs = defaultdict(list)

    for img_path in tqdm(images, desc="mine"):
        r = model.predict(source=str(img_path), conf=args.conf, iou=0.45,
                          imgsz=args.imgsz, device="cpu", verbose=False)[0]
        names = r.names if r.names else names
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls in zip(xyxy, confs, clses):
            label = names[int(cls)]
            fp_records.append({
                "image_id": img_path.name,
                "label": label,
                "bbox": [float(b) for b in box],
                "confidence": float(conf),
            })
            per_image_count[img_path.name] += 1
            per_image_max_conf[img_path.name] = max(per_image_max_conf.get(img_path.name, 0.0), float(conf))
            per_class_confs[label].append(float(conf))

    print()
    print(f">>> {len(fp_records)} total FPs across {len(per_image_count)} positive images "
          f"({len(per_image_count)/len(images)*100:.1f}% of positives produce at least one FP)")
    print(">>> per-class FP stats (conf @ {:.2f} cut):".format(args.conf))
    for cls, confs in per_class_confs.items():
        arr = np.array(confs)
        print(f"    {cls:>15}  n={len(arr):>4}  mean={arr.mean():.3f}  max={arr.max():.3f}  "
              f">=0.3: {(arr>=0.3).sum()}  >=0.5: {(arr>=0.5).sum()}")

    # write mining JSON
    with open(out_dir / "fp_records.json", "w", encoding="utf-8") as f:
        json.dump(fp_records, f, ensure_ascii=False, indent=2)

    # viz top-K worst (by max conf)
    worst = sorted(per_image_max_conf.items(), key=lambda kv: kv[1], reverse=True)[: args.viz_topk]
    print(f">>> top {len(worst)} FP-worst positives (drawing to {out_dir / 'viz'}):")
    for fname, mc in worst:
        img = imread_unicode(pos_dir / fname)
        if img is None:
            continue
        for rec in fp_records:
            if rec["image_id"] != fname:
                continue
            x1, y1, x2, y2 = map(int, rec["bbox"])
            cls_id = next((k for k, v in names.items() if v == rec["label"]), 0)
            color = COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            tag = f"{rec['label']} {rec['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, tag, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        imwrite_unicode(out_dir / "viz" / fname, img)
        print(f"    {fname}  max_conf={mc:.3f}  n_fp={per_image_count[fname]}")


if __name__ == "__main__":
    main()
