#!/usr/bin/env python3
"""
Run baseline YOLOv8 evaluation across rivers and random seeds, saving results to CSV.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
import yaml


SEEDS = [42, 123, 456]
RIVERS = ['combined', 'kitwanga', 'bear']

REPO_ROOT = Path(__file__).resolve().parents[2]


def sample_and_eval(weights, images_dir, labels_dir, data_yaml, batch, device, river, seed):
    txt_path = REPO_ROOT / 'data' / f'sample_test_{river}_seed{seed}.txt'

    subprocess.run([
        sys.executable,
        Path(__file__).parent / 'sample_subset.py',
        '--images-dir', images_dir,
        '--labels-dir', labels_dir,
        '--output-txt', str(txt_path),
        '--seed', str(seed),
        *([] if river == 'combined' else ['--river', river]),
    ], check=True)

    # Point the yaml at this sample
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    data['train'] = str(txt_path)
    data['val'] = str(txt_path)
    data['test'] = str(txt_path)

    tmp_yaml = REPO_ROOT / 'data' / f'tmp_{river}_seed{seed}.yaml'
    with open(tmp_yaml, 'w') as f:
        yaml.dump(data, f)

    model = YOLO(weights)
    metrics = model.val(data=str(tmp_yaml), split='test', imgsz=640, batch=batch, device=device)
    tmp_yaml.unlink()

    class_names = data['names']
    detected_classes = [class_names[i] for i in metrics.box.ap_class_index]

    rows = [{'river': river, 'seed': seed, 'class': 'all',
             'AP50': metrics.box.map50, 'Precision': metrics.box.mp, 'Recall': metrics.box.mr}]
    for cls, ap, p, r in zip(detected_classes, metrics.box.ap50, metrics.box.p, metrics.box.r):
        rows.append({'river': river, 'seed': seed, 'class': cls, 'AP50': ap, 'Precision': p, 'Recall': r})

    return pd.DataFrame(rows)


def run_all(weights, images_dir, labels_dir, data_yaml, output_csv, batch, device):
    all_results = []

    for river in RIVERS:
        for seed in SEEDS:
            print(f"\n=== River: {river} | Seed: {seed} ===")
            df = sample_and_eval(weights, images_dir, labels_dir, data_yaml, batch, device, river, seed)
            all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    # Print summary: mean ± std of mAP50 per river
    summary = (results[results['class'] == 'all']
               .groupby('river')['AP50']
               .agg(['mean', 'std'])
               .round(4))
    print(f"\n=== Baseline Summary (mAP50) ===\n{summary}")
    print(f"\nFull results saved to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baseline YOLO evaluation across rivers and seeds.')
    parser.add_argument('--weights', default=str(REPO_ROOT / 'training/weights/2024-03-19-yolov8n-model-Kitwanga-BearCreek-Koeye-KwaKwa.pt'))
    parser.add_argument('--images-dir', default=str(REPO_ROOT / 'images/test'))
    parser.add_argument('--labels-dir', default=str(REPO_ROOT / 'labels/test'))
    parser.add_argument('--data', default=str(REPO_ROOT / 'training/subset_salmon.yaml'))
    parser.add_argument('--output-csv', default=str(REPO_ROOT / 'results/baseline_metrics.csv'))
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    run_all(args.weights, args.images_dir, args.labels_dir,
            args.data, args.output_csv, args.batch, args.device)
