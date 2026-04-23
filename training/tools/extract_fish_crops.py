#!/usr/bin/env python3
"""
Extract bounding-box crops of fish from images/train for quality-filter VAE training.

The quality-filter VAE (Component 2) runs on YOLO detection crops at inference time,
so it must be trained on crops — not full frames. A VAE trained on full frames learns
to reconstruct background well; fish crops then get high reconstruction error and would
be incorrectly filtered. Training on crops fixes this: genuine fish should reconstruct
with low error, while false positives (debris, shadows) reconstruct poorly.

Output layout (flat):
    <output_dir>/<original_stem>_<box_index>.jpg

Filenames preserve the original frame stem so that training/vae/train.py can still
identify the river via _river(stem) when stratifying training data.

Usage (from repo root):
    python3 training/tools/extract_fish_crops.py \\
        --images-dir images/train \\
        --labels-dir labels/train \\
        --output-dir data/fish_crops \\
        --max-crops 100000
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = {
    0: 'Coho', 1: 'Bull', 2: 'Rainbow', 3: 'Sockeye', 4: 'Pink',
    5: 'Whitefish', 6: 'Chinook', 7: 'Shiner', 8: 'Pikeminnow',
    9: 'Chum', 10: 'Steelhead', 11: 'Lamprey', 12: 'Cutthroat',
    13: 'Stickleback', 14: 'Sculpin', 15: 'Jack_Coho', 16: 'Jack_Chinook',
}

RIVER_PATTERNS = {
    'kitwanga': ['right_bank', 'left_bank'],
    'bear':     ['salmon_camera'],
}


def _river(stem):
    for river, patterns in RIVER_PATTERNS.items():
        if any(p in stem for p in patterns):
            return river
    return 'other'


def extract_crops(images_dir, labels_dir, output_dir, max_crops, min_per_class,
                  padding, min_side, seed):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    random.seed(seed)

    # Collect label files for valid rivers
    label_files = [
        p for p in labels_dir.glob('*.txt')
        if p.stat().st_size > 0 and _river(p.stem) != 'other'
    ]
    print(f"Found {len(label_files)} non-empty label files")

    by_river = defaultdict(list)
    for lp in label_files:
        by_river[_river(lp.stem)].append(lp)
    for river, paths in sorted(by_river.items()):
        print(f"  {river}: {len(paths)} annotated frames")

    # Expand all annotations into crop entries and group by class
    all_entries = []        # (label_path, cls_id, cx, cy, bw, bh)
    class_to_entries = defaultdict(list)

    random.shuffle(label_files)
    for lp in label_files:
        for line in lp.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            entry = (lp, cls_id, cx, cy, bw, bh)
            all_entries.append(entry)
            class_to_entries[cls_id].append(entry)

    print(f"Total annotations: {len(all_entries)}")
    print("\nAvailable crops per class:")
    for cls_id in sorted(class_to_entries, key=lambda c: len(class_to_entries[c])):
        print(f"  {CLASS_NAMES.get(cls_id, cls_id):<20} {len(class_to_entries[cls_id]):>6}")

    # Phase 1: guarantee min_per_class for each species, rarest first
    selected_indices = set()
    entry_to_idx = {id(e): i for i, e in enumerate(all_entries)}

    for cls_id in sorted(class_to_entries, key=lambda c: len(class_to_entries[c])):
        candidates = [e for e in class_to_entries[cls_id]
                      if entry_to_idx[id(e)] not in selected_indices]
        n_take = min(min_per_class, len(candidates))
        if n_take > 0:
            for e in random.sample(candidates, n_take):
                selected_indices.add(entry_to_idx[id(e)])
            print(f"  Phase 1 guaranteed {n_take} crops for "
                  f"{CLASS_NAMES.get(cls_id, cls_id)} "
                  f"({len(class_to_entries[cls_id])} available)")

    # Phase 2: fill remaining budget from unselected entries
    remaining = (max_crops - len(selected_indices)) if max_crops else None
    if remaining is None or remaining > 0:
        unselected = [e for i, e in enumerate(all_entries)
                      if i not in selected_indices]
        n_fill = min(remaining, len(unselected)) if remaining is not None else len(unselected)
        for e in random.sample(unselected, n_fill):
            selected_indices.add(entry_to_idx[id(e)])

    crop_entries = [all_entries[i] for i in sorted(selected_indices)]
    random.shuffle(crop_entries)
    print(f"\nSelected {len(crop_entries)} crops after stratification")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-class counts for reporting
    class_counts = defaultdict(int)
    saved = 0
    skipped_img = 0
    skipped_small = 0
    n_total = len(crop_entries)

    for i, (lp, cls_id, cx, cy, bw, bh) in enumerate(crop_entries):
        if i % 5000 == 0:
            print(f"  {i}/{n_total} processed, {saved} saved ...", flush=True)

        img_path = images_dir / (lp.stem + '.jpg')
        if not img_path.exists():
            skipped_img += 1
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            skipped_img += 1
            continue

        iw, ih = img.size
        # Convert YOLO normalised coords to pixel xyxy
        x1 = (cx - bw / 2) * iw
        y1 = (cy - bh / 2) * ih
        x2 = (cx + bw / 2) * iw
        y2 = (cy + bh / 2) * ih

        # Add padding (fraction of box dimensions)
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(iw, x2 + pad_x)
        y2 = min(ih, y2 + pad_y)

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < min_side or crop_h < min_side:
            skipped_small += 1
            continue

        crop = img.crop((x1, y1, x2, y2))
        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

        # Preserve original stem so train.py can identify river via _river(stem)
        out_name = f"{lp.stem}_box{saved:06d}.jpg"
        crop.save(output_dir / out_name, quality=90)

        class_counts[cls_name] += 1
        saved += 1

    print(f"\nSaved {saved} crops (skipped {skipped_img} missing images, {skipped_small} too small)")
    print("\nPer-class crop counts:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:<20} {count:>6}")
    print(f"\nCrops saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir',
                        default=str(REPO_ROOT / 'images/train'))
    parser.add_argument('--labels-dir',
                        default=str(REPO_ROOT / 'labels/train'))
    parser.add_argument('--output-dir',
                        default=str(REPO_ROOT / 'data/fish_crops'))
    parser.add_argument('--max-crops', type=int, default=100000,
                        help='Maximum total crops to extract')
    parser.add_argument('--min-per-class', type=int, default=500,
                        help='Minimum crops guaranteed per species (rarest first)')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Fractional padding around each bounding box (default 0.1 = 10%%)')
    parser.add_argument('--min-side', type=int, default=16,
                        help='Minimum crop side length in pixels; smaller crops are skipped')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    extract_crops(args.images_dir, args.labels_dir, args.output_dir,
                  args.max_crops, args.min_per_class,
                  args.padding, args.min_side, args.seed)
