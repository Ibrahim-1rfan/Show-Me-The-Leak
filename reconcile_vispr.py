"""
VISPR dataset reconciliation script.

Run this LOCALLY on your machine where the 17GB of images and the
test2017_jsons folder actually live (not on a remote/container path).

What it does:
  1. Walks all image folders (test2017-01 ... test2017-09) and indexes
     every image file found, by its filename stem (e.g. "2017_10163146").
  2. Walks test2017_jsons and loads every per-image JSON (one file per image).
  3. Matches JSON entries <-> image files by id/stem.
  4. Opens every matched image to verify it's not corrupted / truncated /
     a fake image (e.g. an HTML error page saved with a .jpg extension).
  5. Computes a perceptual hash (phash) for every valid image to flag
     near-duplicates.
  6. Writes out:
       - manifest.csv          -> clean, valid, deduped rows ready for training
       - issues_report.csv     -> everything that got excluded, with a reason
       - duplicate_groups.csv  -> groups of near-duplicate images (for review)

Usage:
    python reconcile_vispr.py --images_root /path/to/dataset_root \
                               --jsons_dir  /path/to/dataset_root/test2017_jsons \
                               --output_dir /path/to/dataset_root/reconciled

`--images_root` should be the folder that CONTAINS test2017-01 .. test2017-09
(the script auto-discovers any subfolder starting with "test2017-").
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import imagehash

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def find_image_folders(images_root: Path):
    folders = sorted(
        p for p in images_root.iterdir()
        if p.is_dir() and p.name.startswith("test2017-")
    )
    if not folders:
        # fallback: maybe images_root itself is flat
        folders = [images_root]
    return folders


def index_images(images_root: Path):
    """Return dict: stem -> full path. Also flags duplicate stems across folders."""
    stem_to_path = {}
    duplicate_stems = []
    folders = find_image_folders(images_root)
    print(f"Scanning {len(folders)} image folder(s)...")
    for folder in folders:
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS:
                stem = f.stem
                if stem in stem_to_path:
                    duplicate_stems.append((stem, str(stem_to_path[stem]), str(f)))
                else:
                    stem_to_path[stem] = f
    print(f"Found {len(stem_to_path)} unique image files on disk.")
    if duplicate_stems:
        print(f"WARNING: {len(duplicate_stems)} filename collisions across folders (same id, different files).")
    return stem_to_path, duplicate_stems


def index_jsons(jsons_dir: Path):
    """Return dict: stem -> loaded json dict (or None + error if unreadable)."""
    stem_to_json = {}
    bad_jsons = []
    json_files = list(jsons_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files in {jsons_dir}.")
    for jf in json_files:
        stem = jf.stem
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            stem_to_json[stem] = data
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            bad_jsons.append((stem, str(jf), str(e)))
    return stem_to_json, bad_jsons


def verify_image(path: Path, max_dim_check=True):
    """
    Try to fully decode the image (not just open header).
    Returns (ok: bool, reason: str or None, size: (w,h) or None)
    """
    try:
        with Image.open(path) as im:
            im.verify()  # checks file integrity (header + structure)
        # verify() invalidates the file handle for further use, reopen to actually decode pixels
        with Image.open(path) as im:
            im.load()  # forces full pixel decode -> catches truncated files verify() may miss
            w, h = im.size
            if w == 0 or h == 0:
                return False, "zero_dimension", None
            if max_dim_check and (w < 10 or h < 10):
                return False, "suspiciously_tiny", (w, h)
            return True, None, (w, h)
    except UnidentifiedImageError:
        return False, "not_an_image_or_html_error_page", None
    except (OSError, SyntaxError) as e:
        return False, f"corrupted_or_truncated:{e}", None


def compute_phash(path: Path):
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im.convert("RGB")))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True, type=Path)
    ap.add_argument("--jsons_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--skip_phash", action="store_true",
                     help="Skip perceptual hashing (faster, no dedup pass)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step A: index everything on disk ---
    stem_to_path, dup_filename_collisions = index_images(args.images_root)
    stem_to_json, bad_jsons = index_jsons(args.jsons_dir)

    image_stems = set(stem_to_path.keys())
    json_stems = set(stem_to_json.keys())

    only_json = json_stems - image_stems      # labels with no downloaded image
    only_image = image_stems - json_stems     # images with no labels (orphans)
    matched = image_stems & json_stems

    print(f"\nMatched (image + json both present): {len(matched)}")
    print(f"JSON entries with missing image (failed downloads): {len(only_json)}")
    print(f"Images with no matching JSON (orphans): {len(only_image)}")

    # --- Step B: verify each matched image actually decodes ---
    manifest_rows = []
    issues = []

    for stem, jf, path, reason in [
        (s, "missing_image", None, "json_present_image_missing") for s in only_json
    ]:
        issues.append({"id": stem, "path": "", "reason": reason})

    for stem in only_image:
        issues.append({"id": stem, "path": str(stem_to_path[stem]), "reason": "image_present_json_missing"})

    for stem, path, orig, err in dup_filename_collisions:
        issues.append({"id": stem, "path": f"{orig} | {path}", "reason": f"duplicate_filename_across_folders:{err}"})

    for stem, path, err in bad_jsons:
        issues.append({"id": stem, "path": path, "reason": f"unreadable_json:{err}"})

    print(f"\nVerifying {len(matched)} matched images (this is the slow part)...")
    valid_entries = []
    for i, stem in enumerate(sorted(matched)):
        if i % 1000 == 0 and i > 0:
            print(f"  ...{i}/{len(matched)} checked")
        img_path = stem_to_path[stem]
        ok, reason, size = verify_image(img_path)
        if not ok:
            issues.append({"id": stem, "path": str(img_path), "reason": reason})
            continue

        raw = stem_to_json[stem]
        labels = raw.get("labels", [])
        entry = {
            "id": stem,
            "local_path": str(img_path),
            "labels": "|".join(labels),
            "openimages_id": raw.get("openimages_id"),
            "source_url": raw.get("source_url"),
            "width": size[0],
            "height": size[1],
            "is_safe": 1 if labels == ["a0_safe"] else 0,
        }
        valid_entries.append(entry)

    print(f"\nValid, verified images: {len(valid_entries)}")

    # --- Step C: perceptual-hash dedup pass ---
    duplicate_groups_out = []
    if not args.skip_phash:
        print("\nComputing perceptual hashes for dedup (this can take a while for large sets)...")
        hash_to_ids = {}
        for i, entry in enumerate(valid_entries):
            if i % 1000 == 0 and i > 0:
                print(f"  ...{i}/{len(valid_entries)} hashed")
            h = compute_phash(Path(entry["local_path"]))
            entry["phash"] = h
            if h:
                hash_to_ids.setdefault(h, []).append(entry["id"])

        dup_groups = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
        print(f"Found {len(dup_groups)} groups of near/exact-duplicate images "
              f"({sum(len(v) for v in dup_groups.values())} images total involved).")

        for h, ids in dup_groups.items():
            duplicate_groups_out.append({"phash": h, "ids": "|".join(ids), "count": len(ids)})
    else:
        for entry in valid_entries:
            entry["phash"] = ""

    # --- Step D: write outputs ---
    manifest_path = args.output_dir / "manifest.csv"
    issues_path = args.output_dir / "issues_report.csv"
    dupes_path = args.output_dir / "duplicate_groups.csv"

    fieldnames = ["id", "local_path", "labels", "openimages_id", "source_url",
                  "width", "height", "is_safe", "phash"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_entries)

    with open(issues_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "path", "reason"])
        writer.writeheader()
        writer.writerows(issues)

    with open(dupes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phash", "ids", "count"])
        writer.writeheader()
        writer.writerows(duplicate_groups_out)

    print("\n=== SUMMARY ===")
    print(f"Clean manifest:     {manifest_path}  ({len(valid_entries)} rows)")
    print(f"Issues report:      {issues_path}  ({len(issues)} rows)")
    print(f"Duplicate groups:   {dupes_path}  ({len(duplicate_groups_out)} groups)")
    print("\nNext: review issues_report.csv, decide on duplicate_groups.csv "
          "(keep first occurrence per group is a common default), then use "
          "manifest.csv as your source of truth for the taxonomy crosswalk step.")


if __name__ == "__main__":
    sys.exit(main())
