#!/usr/bin/env python3
"""
Randomly samples a stratified subset of the test set for baseline evaluation.
Produces a test.txt file listing sampled image paths for use with YOLO val.
"""
import argparse
import random
from pathlib import Path


def sample_subset(images_dir, labels_dir, output_txt, n_positive, n_negative, seed=42):
    random.seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    positive, negative = [], []
    for img in images_dir.glob("*.jpg"):
        label = labels_dir / (img.stem + ".txt")
        if label.exists() and label.stat().st_size > 0:
            positive.append(img)
        else:
            negative.append(img)

    n_pos = min(n_positive, len(positive))
    n_neg = min(n_negative, len(negative))
    sampled = random.sample(positive, n_pos) + random.sample(negative, n_neg)
    random.shuffle(sampled)

    repo_root = images_dir.parent.parent
    with open(output_txt, "w") as f:
        for img in sampled:
            f.write(str(img.relative_to(repo_root)) + "\n")

    print(f"Sampled {n_pos} positive frames (with salmon) and {n_neg} negative frames")
    print(f"Total: {len(sampled)} frames -> {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a stratified subset of test frames.")
    parser.add_argument("--images-dir", default="images/test", help="Path to test images directory.")
    parser.add_argument("--labels-dir", default="labels/test", help="Path to test labels directory.")
    parser.add_argument("--output-txt", default="data/sample_test.txt", help="Output txt file listing sampled image paths.")
    parser.add_argument("--n-positive", type=int, default=1500, help="Number of frames with salmon to sample.")
    parser.add_argument("--n-negative", type=int, default=500, help="Number of background frames to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    sample_subset(args.images_dir, args.labels_dir, args.output_txt, args.n_positive, args.n_negative, args.seed)
