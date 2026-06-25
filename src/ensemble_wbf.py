"""Cross-model WBF ensemble: fuse predictions from multiple submission folders.

Each input folder must contain per_image/*.json in our standard format
(image_id + annotations[{label, bbox xyxy, confidence}]).

Output: a NEW submission folder with the same structure, where each image's
annotations are the WBF fusion of all input folders' predictions for that image.

Run:
    python src/ensemble_wbf.py \\
        --inputs submission_v3_wbf submission_v4_clean_wbf \\
        --out submission_v5_ensemble \\
        --iou-thr 0.55 --weights 1.0 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import CLASS_NAMES, CLASS_TO_ID

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"

ID_TO_CLASS = {i: n for n, i in CLASS_TO_ID.items()}


def imread_dims(path: Path) -> tuple[int, int]:
    """Return (W, H) without decoding the full image — use cv2 imdecode header read."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img.shape[1], img.shape[0]


def load_per_image(folder: Path) -> dict[str, dict]:
    """stem → {annotations: [...]}."""
    out = {}
    for jp in (folder / "per_image").glob("*.json"):
        with open(jp, encoding="utf-8") as f:
            out[jp.stem] = json.load(f)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="submission folders to ensemble; each must have per_image/")
    p.add_argument("--out", required=True, help="output submission folder")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="per-input model weights for WBF; defaults to equal")
    p.add_argument("--iou-thr", type=float, default=0.55)
    p.add_argument("--skip-thr", type=float, default=0.03)
    p.add_argument("--final-conf", type=float, default=0.05)
    args = p.parse_args()

    inputs = [Path(p) for p in args.inputs]
    for ip in inputs:
        if not (ip / "per_image").is_dir():
            raise SystemExit(f"{ip} has no per_image/ folder")
    weights = args.weights or [1.0] * len(inputs)
    if len(weights) != len(inputs):
        raise SystemExit("--weights length must match --inputs length")

    out_dir = Path(args.out)
    (out_dir / "per_image").mkdir(parents=True, exist_ok=True)

    test_files = sorted(TEST_DIR.glob("*.jpg"))
    print(f">>> ensembling {len(inputs)} models on {len(test_files)} test images")

    per_model = [load_per_image(ip) for ip in inputs]

    combined = []
    for img_path in tqdm(test_files, desc="ensemble"):
        stem = img_path.stem
        W, H = imread_dims(img_path)

        boxes_list, scores_list, labels_list = [], [], []
        for pm in per_model:
            rec = pm.get(stem)
            if rec is None or not rec["annotations"]:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue
            b, s, l = [], [], []
            for a in rec["annotations"]:
                cls = CLASS_TO_ID.get(a["label"])
                if cls is None:
                    continue
                x1, y1, x2, y2 = a["bbox"]
                b.append([max(0.0, x1 / W), max(0.0, y1 / H),
                          min(1.0, x2 / W), min(1.0, y2 / H)])
                s.append(float(a["confidence"]))
                l.append(cls)
            boxes_list.append(b)
            scores_list.append(s)
            labels_list.append(l)

        if not any(boxes_list):
            rec = {"image_id": img_path.name, "annotations": []}
        else:
            fb, fs, fl = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights, iou_thr=args.iou_thr,
                skip_box_thr=args.skip_thr,
            )
            annotations = []
            for box, score, lbl in zip(fb, fs, fl):
                if score < args.final_conf:
                    continue
                annotations.append({
                    "label": ID_TO_CLASS[int(lbl)],
                    "bbox": [float(box[0] * W), float(box[1] * H),
                             float(box[2] * W), float(box[3] * H)],
                    "confidence": float(score),
                })
            rec = {"image_id": img_path.name, "annotations": annotations}

        combined.append(rec)
        with open(out_dir / "per_image" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    with open(out_dir / "submission.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    n_with = sum(1 for r in combined if r["annotations"])
    n_box = sum(len(r["annotations"]) for r in combined)
    print(f">>> ensemble done: {n_with}/{len(combined)} images with dets, {n_box} fused boxes")
    print(f">>> output: {out_dir}/submission.json")


if __name__ == "__main__":
    main()
