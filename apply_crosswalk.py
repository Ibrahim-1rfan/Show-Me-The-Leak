"""
Apply the 68 -> 14 taxonomy crosswalk to manifest.csv, producing a new
manifest with an added 'categories' column (the deduplicated, mapped
category labels for each image).

Usage:
    python apply_crosswalk.py \
        --manifest /path/to/reconciled/manifest.csv \
        --crosswalk /path/to/taxonomy_crosswalk.csv \
        --output /path/to/reconciled/manifest_with_categories.csv
"""

import argparse
import csv
from pathlib import Path


def load_crosswalk(path: Path):
    mapping = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["attribute_id"]] = row["taxonomy_category"]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--crosswalk", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    mapping = load_crosswalk(args.crosswalk)
    print(f"Loaded crosswalk with {len(mapping)} attribute mappings.")

    rows = []
    unmapped_attrs = set()

    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["categories"]
        for row in reader:
            labels = [l for l in row["labels"].split("|") if l]
            categories = []
            for l in labels:
                cat = mapping.get(l)
                if cat is None:
                    unmapped_attrs.add(l)
                    continue
                if cat not in categories:
                    categories.append(cat)
            row["categories"] = "|".join(categories)
            rows.append(row)

    if unmapped_attrs:
        print(f"WARNING: {len(unmapped_attrs)} attribute(s) in the manifest "
              f"have no entry in the crosswalk and were skipped:")
        for a in sorted(unmapped_attrs):
            print(f"    {a}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")
    print("New 'categories' column added (pipe-separated, deduplicated per image).")


if __name__ == "__main__":
    main()
