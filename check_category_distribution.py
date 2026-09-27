"""
Category-level distribution check on manifest_with_categories.csv.
Run after apply_crosswalk.py.

Usage:
    python check_category_distribution.py --manifest /path/to/manifest_with_categories.csv
"""

import argparse
import csv
import collections
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    print(f"Total images: {total}\n")

    cat_counter = collections.Counter()
    cats_per_image = []
    for r in rows:
        cats = [c for c in r["categories"].split("|") if c]
        cats_per_image.append(len(cats))
        for c in cats:
            cat_counter[c] += 1

    print(f"Avg categories/image: {sum(cats_per_image)/total:.2f}")
    print(f"Max categories on one image: {max(cats_per_image)}\n")

    print("Category frequency (sorted by count):")
    for cat, count in cat_counter.most_common():
        pct = 100 * count / total
        print(f"  {count:5d}  ({pct:5.1f}%)  {cat}")

    low_support = [(c, n) for c, n in cat_counter.items() if n < 100]
    if low_support:
        print(f"\n{len(low_support)} categories have fewer than 100 examples "
              f"— still worth weighting even after aggregation:")
        for c, n in sorted(low_support, key=lambda x: x[1]):
            print(f"    {c}: {n}")


if __name__ == "__main__":
    main()
