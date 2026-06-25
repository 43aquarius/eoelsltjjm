"""Package per-image prediction JSONs into the official competition ZIP.

Official format (from 少样本条件下电子产品外观缺陷检测.txt):
- ZIP name: 少样本条件下电子产品外观缺陷检测_{团队名}_{作品名}.zip
- Inner folder: same name (without .zip)
- Each test image gets one JSON; filename = image stem + ".json"
- Encoding: UTF-8 NoBom (no BOM marker)
- bbox: NOT normalized (raw pixel coordinates, [x1, y1, x2, y2])

Run:
    python src/package_submission.py \\
        --source submission_v3_wbf/per_image \\
        --team "你们说的队" \\
        --work "YOLOv8n少样本检测"
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "初赛数据" / "测试集" / "image"
COMPETITION_PREFIX = "少样本条件下电子产品外观缺陷检测"
ALLOWED_LABELS = {"plain particle", "dirt", "scratch", "collision"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="folder containing per-image *.json files")
    p.add_argument("--team", required=True, help="team name (中文/英文)")
    p.add_argument("--work", required=True, help="work name (中文/英文)")
    p.add_argument("--out-dir", default=str(ROOT / "release"), help="where to put the staged folder + zip")
    p.add_argument("--test-dir", default=str(TEST_DIR), help="expected list of test images (for completeness check)")
    return p.parse_args()


def write_json_utf8_nobom(path: Path, obj: dict) -> None:
    """Write JSON as UTF-8 with no BOM and LF line endings (Unix style, cross-platform safe)."""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    # binary mode to control line endings exactly + no BOM
    with open(path, "wb") as f:
        f.write(text.replace("\r\n", "\n").encode("utf-8"))


def validate_record(rec: dict, expected_image_id: str) -> list[str]:
    """Return list of human-readable problems. Empty list = OK."""
    issues = []
    if rec.get("image_id") != expected_image_id:
        issues.append(f"image_id mismatch: have {rec.get('image_id')!r}, expected {expected_image_id!r}")
    anns = rec.get("annotations")
    if not isinstance(anns, list):
        issues.append("annotations is not a list")
        return issues
    for i, a in enumerate(anns):
        if a.get("label") not in ALLOWED_LABELS:
            issues.append(f"ann[{i}] bad label: {a.get('label')!r}")
        bbox = a.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            issues.append(f"ann[{i}] bad bbox: {bbox!r}")
        else:
            x1, y1, x2, y2 = bbox
            if not (x2 > x1 and y2 > y1):
                issues.append(f"ann[{i}] degenerate bbox: {bbox}")
        c = a.get("confidence")
        if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
            issues.append(f"ann[{i}] bad confidence: {c!r}")
    return issues


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"source not a directory: {source}")
    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = f"{COMPETITION_PREFIX}_{args.team}_{args.work}"
    staged_dir = out_dir / name
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir()

    test_images = sorted(test_dir.glob("*.jpg"))
    print(f">>> {len(test_images)} test images to cover")

    expected_stems = {p.stem for p in test_images}
    available_stems = {p.stem for p in source.glob("*.json")}
    missing = expected_stems - available_stems
    extra = available_stems - expected_stems
    if missing:
        print(f"[warn] {len(missing)} test images have no prediction JSON — will write empty annotations")
    if extra:
        print(f"[warn] {len(extra)} extra JSONs in source not corresponding to a test image (skipped)")

    n_written = 0
    n_issues = 0
    total_boxes = 0
    for img_path in test_images:
        stem = img_path.stem
        src_json = source / f"{stem}.json"
        expected_id = img_path.name  # filename with .jpg

        if src_json.exists():
            with open(src_json, encoding="utf-8") as f:
                rec = json.load(f)
            # always normalize the image_id to match the test filename exactly
            rec["image_id"] = expected_id
        else:
            rec = {"image_id": expected_id, "annotations": []}

        issues = validate_record(rec, expected_id)
        if issues:
            n_issues += 1
            print(f"[warn] {src_json.name}: {issues[:3]}")

        write_json_utf8_nobom(staged_dir / f"{stem}.json", rec)
        n_written += 1
        total_boxes += len(rec.get("annotations", []))

    print(f">>> wrote {n_written} JSONs to {staged_dir}  (total boxes: {total_boxes})")
    if n_issues:
        print(f"[warn] {n_issues} records had validation issues — review the warnings above")

    # encoding spot-check
    spot = next(staged_dir.glob("*.json"))
    with open(spot, "rb") as f:
        head = f.read(3)
    if head == b"\xef\xbb\xbf":
        raise SystemExit(f"[fatal] {spot.name} has UTF-8 BOM — packaging is wrong, fix write_json_utf8_nobom")
    print(f">>> encoding spot-check OK: no BOM in {spot.name}")

    # build the zip
    zip_path = out_dir / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for jp in sorted(staged_dir.glob("*.json")):
            zf.write(jp, arcname=f"{name}/{jp.name}")
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f">>> wrote {zip_path}  ({size_mb:.1f} MB)")

    # zip integrity check
    with zipfile.ZipFile(zip_path) as zf:
        n_in_zip = len([n for n in zf.namelist() if n.endswith(".json")])
        first = next(n for n in zf.namelist() if n.endswith(".json"))
        with zf.open(first) as f:
            sample_head = f.read(3)
    print(f">>> zip contains {n_in_zip} JSONs; first entry '{first}' starts with {sample_head!r} (no BOM)")
    print(f"\n=== ready to submit ===")
    print(f"file: {zip_path}")
    print(f"name format OK: {COMPETITION_PREFIX}_<team>_<work>")


if __name__ == "__main__":
    main()
