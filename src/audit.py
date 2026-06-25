"""Comprehensive audit of submission packages + training pipeline integrity.

Checks:
A1. ZIPs in release/ parse, contain expected file count, no BOM.
A2. All bboxes are inside their source image bounds (read original test image dims).
A3. No duplicate bbox+label within a single image.
A4. Confidence distribution sanity (no NaN/inf, all in [0,1]).
A5. Model weight files exist + load.
A6. Data leak check: any test image filename appears in train/val labels.
A7. Augmentation sanity: sample 5 aug_*.jpg + their labels, verify bboxes inside crop bounds.
A8. v1 vs v3 detection-count differences per image (large divergence = potential issue).
"""
from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = ROOT / "release"
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"
DATASET_DIR = ROOT / "dataset"
ALLOWED = {"plain particle", "dirt", "scratch", "collision"}


def banner(s: str) -> None:
    print(f"\n{'='*70}\n{s}\n{'='*70}")


def imread_unicode(p: Path):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def a1_zip_integrity() -> dict:
    banner("A1 — ZIP integrity / encoding / file counts")
    issues = []
    zips = sorted(RELEASE_DIR.glob("*.zip"))
    if not zips:
        issues.append("no zips in release/")
    test_stems = {p.stem for p in TEST_DIR.glob("*.jpg")}
    report = {"zips_checked": len(zips), "issues": []}
    for zp in zips:
        info = {"name": zp.name, "size_kb": zp.stat().st_size / 1024}
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            info["n_json"] = len(names)
            info["matches_876"] = (len(names) == 876)

            n_bom = 0
            n_id_mismatch = 0
            n_extra_stem = 0
            seen_stems = set()
            for n in names:
                stem = Path(n).stem
                seen_stems.add(stem)
                with zf.open(n) as f:
                    raw = f.read()
                if raw.startswith(b"\xef\xbb\xbf"):
                    n_bom += 1
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except Exception as e:
                    issues.append(f"{zp.name}/{n}: parse error {e}")
                    continue
                if rec.get("image_id", "").rsplit(".", 1)[0] != stem:
                    n_id_mismatch += 1
                if stem not in test_stems:
                    n_extra_stem += 1
            info["bom_count"] = n_bom
            info["image_id_mismatches"] = n_id_mismatch
            info["filenames_not_in_test"] = n_extra_stem
            info["missing_test_stems"] = len(test_stems - seen_stems)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        for k in ("bom_count", "image_id_mismatches", "filenames_not_in_test", "missing_test_stems"):
            if info[k] != 0:
                issues.append(f"{zp.name}: {k}={info[k]}")
        if not info["matches_876"]:
            issues.append(f"{zp.name}: expected 876 JSONs, got {info['n_json']}")
    report["issues"] = issues
    print(f"\nA1 issues: {len(issues)}")
    for i in issues:
        print(f"  - {i}")
    return report


def a2_bbox_bounds() -> dict:
    banner("A2 — bbox bounds vs original test image dims (sample 100 images)")
    issues = []
    # cache image dims by reading the source
    test_files = sorted(TEST_DIR.glob("*.jpg"))[:100]
    dims = {}
    for tp in test_files:
        img = imread_unicode(tp)
        if img is not None:
            dims[tp.stem] = (img.shape[1], img.shape[0])  # W, H

    out_of_bounds = 0
    degenerate = 0
    checked = 0
    n_neg = 0
    for sub in ["submission_v1_frozen", "submission_v2_ft_tta", "submission_v3_wbf"]:
        per = ROOT / sub / "per_image"
        if not per.exists():
            continue
        for stem, (W, H) in dims.items():
            jp = per / f"{stem}.json"
            if not jp.exists():
                continue
            rec = json.load(open(jp, encoding="utf-8"))
            for a in rec["annotations"]:
                checked += 1
                x1, y1, x2, y2 = a["bbox"]
                if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
                    issues.append(f"{sub}/{stem}: NaN/inf in bbox")
                    continue
                if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
                    out_of_bounds += 1
                if x1 < 0 or y1 < 0:
                    n_neg += 1
                if x2 <= x1 or y2 <= y1:
                    degenerate += 1
    print(f"checked {checked} boxes across 3 submission folders (100 imgs each)")
    print(f"  out_of_bounds: {out_of_bounds}")
    print(f"  negative_coords: {n_neg}")
    print(f"  degenerate (x2<=x1 or y2<=y1): {degenerate}")
    if out_of_bounds:
        issues.append(f"{out_of_bounds} boxes outside image bounds")
    if degenerate:
        issues.append(f"{degenerate} degenerate boxes")
    return {"checked": checked, "out_of_bounds": out_of_bounds, "degenerate": degenerate, "issues": issues}


def a3_duplicate_boxes() -> dict:
    banner("A3 — duplicate (bbox+label) within a single image")
    n_dup = 0
    n_close = 0
    for vname in ["submission_v3_wbf"]:
        per = ROOT / vname / "per_image"
        if not per.exists():
            continue
        for jp in per.glob("*.json"):
            rec = json.load(open(jp, encoding="utf-8"))
            seen = []
            for a in rec["annotations"]:
                key = (a["label"], round(a["bbox"][0], 1), round(a["bbox"][1], 1),
                       round(a["bbox"][2], 1), round(a["bbox"][3], 1))
                if key in seen:
                    n_dup += 1
                seen.append(key)
                # near-duplicate via IoU>0.95
                for prev_label, *prev_box in seen[:-1]:
                    if prev_label != a["label"]:
                        continue
                    b = a["bbox"]
                    ix1 = max(b[0], prev_box[0]); iy1 = max(b[1], prev_box[1])
                    ix2 = min(b[2], prev_box[2]); iy2 = min(b[3], prev_box[3])
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    aA = (b[2] - b[0]) * (b[3] - b[1])
                    aB = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                    union = aA + aB - inter
                    if union > 0 and inter / union > 0.95:
                        n_close += 1
                        break
    print(f"exact duplicates: {n_dup}")
    print(f"near-duplicates (IoU>0.95): {n_close}")
    return {"exact": n_dup, "near": n_close}


def a4_conf_distribution() -> dict:
    banner("A4 — confidence sanity")
    for vname in ["submission_v1_frozen", "submission_v3_wbf"]:
        p = ROOT / vname / "submission.json"
        if not p.exists():
            continue
        data = json.load(open(p, encoding="utf-8"))
        confs = [a["confidence"] for r in data for a in r["annotations"]]
        if not confs:
            print(f"{vname}: empty")
            continue
        arr = np.array(confs)
        bad = ((arr < 0) | (arr > 1)).sum()
        print(f"{vname}: n={len(arr)}, min={arr.min():.4f}, max={arr.max():.4f}, "
              f"mean={arr.mean():.4f}, out-of-[0,1]={bad}")
    return {}


def a5_weights_check() -> dict:
    banner("A5 — model weight files")
    from ultralytics import YOLO
    paths = [
        ROOT / "runs/defect_yolov8n_cpu/weights/best.pt",
        ROOT / "runs/defect_yolov8n_ft/weights/best.pt",
    ]
    out = {}
    for p in paths:
        if not p.exists():
            out[p.name] = "MISSING"
            continue
        try:
            m = YOLO(str(p))
            n_param = sum(x.numel() for x in m.model.parameters())
            out[str(p.relative_to(ROOT))] = f"OK ({n_param/1e6:.1f}M params)"
        except Exception as e:
            out[str(p.relative_to(ROOT))] = f"LOAD_ERROR: {e}"
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


def a6_data_leak_check() -> dict:
    banner("A6 — data leak (test images in train/val labels)")
    test_stems = {p.stem for p in TEST_DIR.glob("*.jpg")}
    train_stems = {p.stem for p in (DATASET_DIR / "images" / "train").glob("*.jpg")
                   if not p.name.startswith("aug_")}
    val_stems = {p.stem for p in (DATASET_DIR / "images" / "val").glob("*.jpg")}
    aug_count = len(list((DATASET_DIR / "images" / "train").glob("aug_*.jpg")))
    leaked_train = train_stems & test_stems
    leaked_val = val_stems & test_stems
    print(f"train images (non-aug): {len(train_stems)}; aug images: {aug_count}")
    print(f"val images: {len(val_stems)}")
    print(f"leaked into train: {len(leaked_train)}")
    print(f"leaked into val: {len(leaked_val)}")
    return {"leaked_train": len(leaked_train), "leaked_val": len(leaked_val), "aug": aug_count}


def a7_aug_sanity() -> dict:
    banner("A7 — augmented training samples sanity (sample 5)")
    aug_imgs = sorted((DATASET_DIR / "images" / "train").glob("aug_*.jpg"))[:5]
    issues = []
    for ip in aug_imgs:
        img = imread_unicode(ip)
        if img is None:
            issues.append(f"cannot read {ip.name}")
            continue
        H, W = img.shape[:2]
        lp = DATASET_DIR / "labels" / "train" / (ip.stem + ".txt")
        if not lp.exists():
            issues.append(f"no label for {ip.name}")
            continue
        for line in lp.read_text().splitlines():
            cls, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w/2) * W; y1 = (yc - h/2) * H
            x2 = (xc + w/2) * W; y2 = (yc + h/2) * H
            if x1 < -1 or y1 < -1 or x2 > W+1 or y2 > H+1:
                issues.append(f"{ip.name}: box outside ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) vs {W}x{H}")
            if x2 - x1 < 1 or y2 - y1 < 1:
                issues.append(f"{ip.name}: tiny box {x2-x1}x{y2-y1}")
    print(f"sampled {len(aug_imgs)} aug images; issues: {len(issues)}")
    for i in issues:
        print(f"  - {i}")
    return {"sampled": len(aug_imgs), "issues": issues}


def a8_v1_v3_divergence() -> dict:
    banner("A8 — v1 vs v3 detection divergence")
    v1 = {r["image_id"]: r["annotations"] for r in json.load(
        open(ROOT / "submission_v1_frozen/submission.json", encoding="utf-8"))}
    v3 = {r["image_id"]: r["annotations"] for r in json.load(
        open(ROOT / "submission_v3_wbf/submission.json", encoding="utf-8"))}
    both_empty = both_have = only_v1 = only_v3 = 0
    big_div = 0
    for iid in v1:
        v1n = len(v1[iid])
        v3n = len(v3.get(iid, []))
        if v1n == 0 and v3n == 0:
            both_empty += 1
        elif v1n > 0 and v3n > 0:
            both_have += 1
            if abs(v1n - v3n) >= 10:
                big_div += 1
        elif v1n > 0:
            only_v1 += 1
        else:
            only_v3 += 1
    print(f"both empty: {both_empty}")
    print(f"both have detections: {both_have}")
    print(f"only v1: {only_v1}")
    print(f"only v3: {only_v3}")
    print(f"both have, but |v1-v3|>=10 boxes: {big_div}")
    return {"both_empty": both_empty, "both_have": both_have,
            "only_v1": only_v1, "only_v3": only_v3, "big_div": big_div}


def main():
    print("AUDIT REPORT")
    a1 = a1_zip_integrity()
    a2 = a2_bbox_bounds()
    a3 = a3_duplicate_boxes()
    a4 = a4_conf_distribution()
    a5 = a5_weights_check()
    a6 = a6_data_leak_check()
    a7 = a7_aug_sanity()
    a8 = a8_v1_v3_divergence()

    banner("AUDIT SUMMARY")
    blocking = []
    if a1["issues"]:
        blocking.append(f"A1: {len(a1['issues'])} issues")
    if a2["out_of_bounds"] or a2["degenerate"]:
        blocking.append(f"A2: bbox bounds — {a2['out_of_bounds']} OOB, {a2['degenerate']} degenerate")
    if a6["leaked_train"] or a6["leaked_val"]:
        blocking.append(f"A6: data leak — train={a6['leaked_train']}, val={a6['leaked_val']}")
    if a7["issues"]:
        blocking.append(f"A7: aug issues — {len(a7['issues'])}")
    if blocking:
        print("BLOCKING ISSUES:")
        for b in blocking:
            print(f"  X {b}")
    else:
        print("PASS — no blocking issues")
    print("\nNOTES:")
    print(f"  - v1 vs v3 divergence: only_v1={a8['only_v1']}, only_v3={a8['only_v3']}, "
          f"both_have={a8['both_have']} (potential for ensemble gain)")
    print(f"  - A3 near-duplicates v3: {a3['near']} (WBF should suppress most of these)")


if __name__ == "__main__":
    main()
