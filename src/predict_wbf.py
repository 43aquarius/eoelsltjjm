"""Inference with manual multi-scale + flip TTA, fused with Weighted Boxes Fusion.

Runs predict at multiple imgsz + with horizontal flip, then fuses all
candidate boxes via WBF (usually +1-2% mAP vs NMS).

Run:
    python src/predict_wbf.py \\
        --weights runs/defect_yolov8n_ft/weights/best.pt \\
        --conf 0.05 --iou-thr 0.55 \\
        --imgsz 448 512 576 --flip \\
        --out-dir submission_v3_wbf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--test-dir", default=str(TEST_DIR))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--conf", type=float, default=0.05, help="per-pass conf threshold (kept loose)")
    p.add_argument("--iou", type=float, default=0.5, help="per-pass NMS iou")
    p.add_argument("--imgsz", type=int, nargs="+", default=[448, 512, 576])
    p.add_argument("--flip", action="store_true", help="also run horizontal-flipped pass at each scale")
    # WBF tuning
    p.add_argument("--wbf-iou", type=float, default=0.55, help="WBF cluster iou")
    p.add_argument("--wbf-skip", type=float, default=0.03, help="boxes below this are dropped pre-WBF")
    p.add_argument("--final-conf", type=float, default=0.05, help="post-WBF conf threshold")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def run_pass(model: YOLO, img_path: Path, imgsz: int, flip: bool, conf: float, iou: float):
    """One inference pass; returns (xyxy_normalized, conf, cls, W, H). Empty arrays if no boxes."""
    source = str(img_path)
    if flip:
        # flip the source — easier than dealing with mid-pipeline flipping
        import cv2
        data = np.fromfile(source, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = np.ascontiguousarray(img[:, ::-1, :])
        source = img  # ultralytics accepts ndarray
    r = model.predict(source=source, imgsz=imgsz, conf=conf, iou=iou,
                      device="cpu", verbose=False)[0]
    H, W = r.orig_shape  # (H, W) of original image
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int), W, H
    xyxy = r.boxes.xyxy.cpu().numpy()  # in original-image coords
    if flip:
        # un-flip x coords back to original orientation
        x1, y1, x2, y2 = xyxy[:, 0].copy(), xyxy[:, 1], xyxy[:, 2].copy(), xyxy[:, 3]
        xyxy[:, 0] = W - x2
        xyxy[:, 2] = W - x1
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    # normalize to [0, 1] for WBF
    norm = xyxy.copy().astype(np.float32)
    norm[:, [0, 2]] /= W
    norm[:, [1, 3]] /= H
    np.clip(norm, 0.0, 1.0, out=norm)
    return norm, confs, clses, W, H


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    out_dir = Path(args.out_dir)
    (out_dir / "per_image").mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.test_dir)
    images = sorted(test_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    print(f">>> {len(images)} images, weights={weights.name}")

    passes: list[tuple[int, bool]] = [(s, False) for s in args.imgsz]
    if args.flip:
        passes += [(s, True) for s in args.imgsz]
    print(f">>> TTA passes: {passes}")

    model = YOLO(str(weights))
    # warmup name lookup
    names_map: dict[int, str] = {}

    combined: list[dict] = []
    for img_path in tqdm(images, desc="wbf"):
        all_boxes, all_scores, all_labels = [], [], []
        W = H = None
        for imgsz, flip in passes:
            boxes, scores, clses, w, h = run_pass(model, img_path, imgsz, flip, args.conf, args.iou)
            W, H = w, h
            if not names_map:
                # fetch class names on first non-empty result
                tmp = model.predict(source=str(img_path), imgsz=imgsz, conf=0.99, iou=0.5,
                                    device="cpu", verbose=False)[0]
                names_map = tmp.names
            if len(boxes):
                all_boxes.append(boxes.tolist())
                all_scores.append(scores.tolist())
                all_labels.append(clses.tolist())

        if not all_boxes:
            combined.append({"image_id": img_path.name, "annotations": []})
            with open(out_dir / "per_image" / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(combined[-1], f, ensure_ascii=False, indent=2)
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=args.wbf_iou, skip_box_thr=args.wbf_skip,
        )

        annotations = []
        for box, score, lbl in zip(fused_boxes, fused_scores, fused_labels):
            if score < args.final_conf:
                continue
            x1 = float(box[0] * W)
            y1 = float(box[1] * H)
            x2 = float(box[2] * W)
            y2 = float(box[3] * H)
            annotations.append({
                "label": names_map[int(lbl)],
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
            })

        rec = {"image_id": img_path.name, "annotations": annotations}
        combined.append(rec)
        with open(out_dir / "per_image" / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    with open(out_dir / "submission.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    n_with = sum(1 for r in combined if r["annotations"])
    n_box = sum(len(r["annotations"]) for r in combined)
    print(f">>> WBF done: {n_with}/{len(combined)} images with dets, {n_box} fused boxes")
    print(f">>> output: {out_dir}/submission.json")


if __name__ == "__main__":
    main()
