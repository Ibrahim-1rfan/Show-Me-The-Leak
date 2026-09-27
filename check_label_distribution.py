"""
Quick sanity check on manifest.csv: label frequency, safe/unsafe ratio,
labels-per-image stats. Run after reconcile_vispr.py.

Usage:
    python check_label_distribution.py --manifest /path/to/reconciled/manifest.csv
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
    print(f"Total images in manifest: {total}\n")

    # Safe vs unsafe
    n_safe = sum(1 for r in rows if r["is_safe"] == "1")
    n_unsafe = total - n_safe
    print(f"Safe (a0_safe only):   {n_safe:5d}  ({100*n_safe/total:.1f}%)")
    print(f"Unsafe (>=1 attribute):{n_unsafe:5d}  ({100*n_unsafe/total:.1f}%)")
    print("(Reference paper: ~4,991/7,977 private ≈ 62.6% unsafe on the full VISPR test set)\n")

    # Label frequency
    label_counter = collections.Counter()
    labels_per_image = []
    for r in rows:
        labels = [l for l in r["labels"].split("|") if l]
        labels_per_image.append(len(labels))
        for l in labels:
            label_counter[l] += 1

    print(f"Total unique attribute labels seen: {len(label_counter)}")
    print(f"Avg labels/image (all):    {sum(labels_per_image)/total:.2f}")

    unsafe_counts = [len(r["labels"].split("|")) for r in rows if r["is_safe"] == "0"]
    if unsafe_counts:
        print(f"Avg labels/image (unsafe): {sum(unsafe_counts)/len(unsafe_counts):.2f}")
    print(f"Max labels on one image:   {max(labels_per_image)}\n")

    print("Top 20 most frequent attributes:")
    for label, count in label_counter.most_common(20):
        pct = 100 * count / total
        print(f"  {count:5d}  ({pct:5.1f}%)  {label}")

    print("\nBottom 20 rarest attributes (long-tail / imbalance check):")
    rarest = sorted(label_counter.items(), key=lambda x: x[1])[:20]
    for label, count in rarest:
        pct = 100 * count / total
        print(f"  {count:5d}  ({pct:5.1f}%)  {label}")

    # Flag categories with very low support -> matters for Step 4 (balancing)
    low_support = [(l, c) for l, c in label_counter.items() if c < 20]
    if low_support:
        print(f"\n{len(low_support)} attributes have fewer than 20 examples total "
              f"— these will need oversampling or weighted loss in Step 4:")
        for l, c in sorted(low_support, key=lambda x: x[1]):
            print(f"    {l}: {c}")


if __name__ == "__main__":
    main()
