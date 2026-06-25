"""Train YOLOv8n on the prepared dataset (CPU-only).

Default config is tuned for CPU: small model, frozen backbone, small image size, small batch.
Override anything via CLI flags, e.g.::

    python src/train.py --epochs 40 --imgsz 512 --batch 4
    python src/train.py --epochs 2  # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
RUNS_DIR = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt", help="pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--freeze", type=int, default=10, help="freeze first N layers (YOLOv8n backbone = 10)")
    p.add_argument("--name", default="defect_yolov8n_cpu")
    p.add_argument("--workers", type=int, default=0, help="0 = main process; safer on Windows + CPU")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr0", type=float, default=0.01, help="initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01, help="final lr factor (lr0 * lrf)")
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--copy-paste", type=float, default=0.3)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--cos-lr", action="store_true", help="use cosine LR schedule")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml not found at {DATA_YAML} — run src/data_preparation.py first")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        # augmentation tuned for few-shot defects
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        # losses
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # optimizer
        optimizer="SGD",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        seed=args.seed,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # housekeeping
        save=True,
        save_period=-1,  # only save best+last; per-epoch ckpts waste disk on small CPU runs
        verbose=True,
        # (seed set via args)
    )

    best = RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\n>>> training done. best weights: {best}")


if __name__ == "__main__":
    main()
