#!/usr/bin/env python3
"""
Samples a class-stratified subset of the test set for baseline evaluation.
Produces a test.txt file listing sampled image paths for use with YOLO val.

Sampling strategy for positive frames:
  Phase 1 — guarantee at least --min-per-class frames per species (rarest first).
  Phase 2 — fill remaining budget randomly from unselected positives.
This ensures rare species (Lamprey, Jack_Coho, etc.) appear in evaluation.

River identification by filename pattern:
  Kitwanga: filenames containing 'right_bank' or 'left_bank' (2019 recordings)
  Bear Creek: filenames containing 'salmon_camera' (2020 recordings)
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

RIVER_PATTERNS = {
    'kitwanga': ['right_bank', 'left_bank'],
    'bear':     ['salmon_camera'],
}

CLASS_NAMES = {
    0: 'Coho', 1: 'Bull', 2: 'Rainbow', 3: 'Sockeye', 4: 'Pink',
    5: 'Whitefish', 6: 'Chinook', 7: 'Shiner', 8: 'Pikeminnow',
    9: 'Chum', 10: 'Steelhead', 11: 'Lamprey', 12: 'Cutthroat',
    13: 'Stickleback', 14: 'Sculpin', 15: 'Jack_Coho', 16: 'Jack_Chinook',
}


def matches_river(stem, river):
    if river is None:
        return True
    patterns = RIVER_PATTERNS.get(river, [])
    return any(p in stem for p in patterns)


def get_classes(label_path):
    classes = set()
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                classes.add(int(line.split()[0]))
    return classes


def sample_subset(images_dir, labels_dir, output_txt, n_positive, n_negative,
                  min_per_class, seed=42, river=None):
    random.seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    positive = []
    negative = []
    frame_classes = {}           # img -> set of class IDs (cached for summary)
    class_to_frames = defaultdict(list)

    for img in sorted(images_dir.glob("*.jpg")):
        if not matches_river(img.stem, river):
            continue
        label = labels_dir / (img.stem + ".txt")
        if label.exists() and label.stat().st_size > 0:
            classes = get_classes(label)
            frame_classes[img] = classes
            positive.append(img)
            for c in classes:
                class_to_frames[c].append(img)
        else:
            negative.append(img)

    # Phase 1: guarantee min_per_class frames per species, rarest first
    selected = set()
    for cls in sorted(class_to_frames, key=lambda c: len(class_to_frames[c])):
        candidates = [f for f in class_to_frames[cls] if f not in selected]
        n_take = min(min_per_class, len(candidates))
        if n_take > 0:
            selected.update(random.sample(candidates, n_take))

    # Phase 2: fill remaining budget from unselected positives
    remaining = n_positive - len(selected)
    if remaining > 0:
        unselected = [f for f in positive if f not in selected]
        selected.update(random.sample(unselected, min(remaining, len(unselected))))

    # Trim if phase 1 guarantees overshot n_positive
    sampled_positive = list(selected)
    if len(sampled_positive) > n_positive:
        sampled_positive = random.sample(sampled_positive, n_positive)

    n_neg = min(n_negative, len(negative))
    sampled = sampled_positive + random.sample(negative, n_neg)
    random.shuffle(sampled)

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w") as f:
        for img in sampled:
            f.write(str(img.resolve()) + "\n")

    river_label = river if river else 'all'
    print(f"River: {river_label} | {len(sampled_positive)} positive + {n_neg} negative = {len(sampled)} frames -> {output_txt}")

    sampled_set = set(sampled_positive)
    print("Per-class frame counts in sample:")
    for cls in sorted(class_to_frames):
        count = sum(1 for f in class_to_frames[cls] if f in sampled_set)
        total = len(class_to_frames[cls])
        name = CLASS_NAMES.get(cls, str(cls))
        print(f"  {name:15s}: {count:4d} sampled / {total:6d} available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a class-stratified subset of test frames.")
    parser.add_argument("--images-dir", default="images/test")
    parser.add_argument("--labels-dir", default="labels/test")
    parser.add_argument("--output-txt", default="data/sample_test.txt")
    parser.add_argument("--n-positive", type=int, default=5000,
                        help="Total positive frames to sample.")
    parser.add_argument("--n-negative", type=int, default=1000,
                        help="Number of background frames to sample.")
    parser.add_argument("--min-per-class", type=int, default=100,
                        help="Minimum frames guaranteed per species class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--river", choices=["kitwanga", "bear"], default=None)
    args = parser.parse_args()

    sample_subset(args.images_dir, args.labels_dir, args.output_txt,
                  args.n_positive, args.n_negative, args.min_per_class,
                  args.seed, args.river)
