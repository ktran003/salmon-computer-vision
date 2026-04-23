#!/usr/bin/env python3
"""
Analyse VAE reconstruction error distributions for TP vs FP detections.

For each YOLO detection on a sample of test frames:
  1. Compute VAE reconstruction error
  2. Match against ground truth (IoU >= 0.5) to label as TP or FP
  3. Plot TP vs FP error distributions overall and per species
  4. Mark the current calibration threshold (µ+2σ)

Usage (from repo root):
    python3 training/tools/plot_recon_error_dist.py \
        --vae-weights training/vae/weights_crops/vae_latest.pt \
        --output-dir results/recon_error_analysis
"""
import argparse
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_DIR   = REPO_ROOT / 'training' / 'vae'

CLASS_NAMES = {
    0: 'Coho', 1: 'Bull', 2: 'Rainbow', 3: 'Sockeye', 4: 'Pink',
    5: 'Whitefish', 6: 'Chinook', 7: 'Shiner', 8: 'Pikeminnow',
    9: 'Chum', 10: 'Steelhead', 11: 'Lamprey', 12: 'Cutthroat',
    13: 'Stickleback', 14: 'Sculpin', 15: 'Jack_Coho', 16: 'Jack_Chinook',
}


def load_vae(weights_path, device):
    sys.path.insert(0, str(VAE_DIR))
    from model import VAE
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = VAE(latent_dim=ckpt.get('latent_dim', 64),
                depth=ckpt.get('depth', 5)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def load_gt(label_path, img_w, img_h):
    classes, boxes = [], []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return classes, boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        classes.append(cls)
        boxes.append([x1, y1, x2, y2])
    return classes, boxes


def compute_threshold(calibration_dir, vae, to_tensor, device, max_samples=5000):
    cal_paths = list(Path(calibration_dir).glob('*.jpg'))
    if not cal_paths:
        raise FileNotFoundError(f"No .jpg in calibration dir: {calibration_dir}")
    if len(cal_paths) > max_samples:
        cal_paths = random.sample(cal_paths, max_samples)
    errors = []
    for p in cal_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        x = to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            x_recon, _, _ = vae(x)
        errors.append(F.mse_loss(x_recon, x, reduction='mean').item())
    mu    = float(np.mean(errors))
    sigma = float(np.std(errors))
    threshold = mu + 2 * sigma
    print(f"Calibration: n={len(errors)}  µ={mu:.4f}  σ={sigma:.4f}  threshold={threshold:.4f}")
    return threshold, mu, sigma


def collect_detections(weights, images_dir, labels_dir, vae, to_tensor, device,
                       img_paths, iou_threshold=0.5):
    """
    Run YOLO + VAE on img_paths. Return list of dicts with per-detection info.
    TP/FP label assigned via IoU matching against ground truth.
    """
    yolo       = YOLO(weights)
    labels_dir = Path(labels_dir)
    records    = []

    for i, img_path in enumerate(img_paths):
        if i % 500 == 0:
            print(f"  {i}/{len(img_paths)} frames processed ...", flush=True)

        img = Image.open(img_path).convert('RGB')
        iw, ih = img.size

        # Ground truth for this frame
        label_path = labels_dir / (Path(img_path).stem + '.txt')
        gt_classes, gt_boxes = load_gt(label_path, iw, ih)

        results = yolo.predict(img_path, verbose=False, device=device)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        boxes = results[0].boxes

        preds = []
        errors = []
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()
            conf = boxes.conf[j].item()
            cls  = int(boxes.cls[j].item())
            crop = img.crop((max(0, x1), max(0, y1),
                             min(iw, x2), min(ih, y2)))
            if crop.width < 4 or crop.height < 4:
                continue
            x = to_tensor(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                x_recon, _, _ = vae(x)
            err = F.mse_loss(x_recon, x, reduction='mean').item()
            preds.append((cls, conf, x1, y1, x2, y2))
            errors.append(err)

        if not preds:
            continue

        # IoU matching to assign TP/FP
        pred_boxes_t = torch.tensor([[p[2], p[3], p[4], p[5]] for p in preds],
                                    dtype=torch.float32)
        matched_pred = set()
        matched_gt   = set()

        if gt_boxes:
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
            iou = box_iou(pred_boxes_t, gt_boxes_t).numpy()
            pairs = sorted(
                [(pi, gi) for pi in range(len(preds)) for gi in range(len(gt_classes))],
                key=lambda x: -iou[x[0], x[1]],
            )
            for pi, gi in pairs:
                if pi in matched_pred or gi in matched_gt:
                    continue
                if iou[pi, gi] >= iou_threshold:
                    # Only count as TP if class matches
                    if preds[pi][0] == gt_classes[gi]:
                        matched_pred.add(pi)
                        matched_gt.add(gi)

        for pi, (cls, conf, x1, y1, x2, y2) in enumerate(preds):
            is_tp = pi in matched_pred
            records.append({
                'species': CLASS_NAMES.get(cls, str(cls)),
                'cls_id':  cls,
                'conf':    conf,
                'error':   errors[pi],
                'tp':      is_tp,
            })

    return records


def plot_overall(records, threshold, output_dir):
    tp_errors = [r['error'] for r in records if r['tp']]
    fp_errors = [r['error'] for r in records if not r['tp']]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, np.percentile([r['error'] for r in records], 99), 60)
    ax.hist(fp_errors, bins=bins, alpha=0.6, color='tomato',  label=f'FP (n={len(fp_errors)})', density=True)
    ax.hist(tp_errors, bins=bins, alpha=0.6, color='steelblue', label=f'TP (n={len(tp_errors)})', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold µ+2σ={threshold:.4f}')
    ax.set_xlabel('VAE reconstruction error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Reconstruction error: TP vs FP (all species)')
    ax.legend()
    fig.tight_layout()
    out = Path(output_dir) / 'overall_error_dist.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary stats
    print(f"\n  TP  n={len(tp_errors):5d}  mean={np.mean(tp_errors):.4f}  "
          f"median={np.median(tp_errors):.4f}  p95={np.percentile(tp_errors, 95):.4f}")
    print(f"  FP  n={len(fp_errors):5d}  mean={np.mean(fp_errors):.4f}  "
          f"median={np.median(fp_errors):.4f}  p95={np.percentile(fp_errors, 95):.4f}")
    pct_fp_above = 100 * np.mean(np.array(fp_errors) > threshold)
    pct_tp_above = 100 * np.mean(np.array(tp_errors) > threshold)
    print(f"\n  FP above threshold: {pct_fp_above:.1f}%")
    print(f"  TP above threshold: {pct_tp_above:.1f}%  (false rejections)")


def plot_per_species(records, threshold, output_dir, min_dets=20):
    by_species = defaultdict(lambda: {'tp': [], 'fp': []})
    for r in records:
        key = 'tp' if r['tp'] else 'fp'
        by_species[r['species']][key].append(r['error'])

    # Filter to species with enough total detections
    species_list = [s for s, d in by_species.items()
                    if len(d['tp']) + len(d['fp']) >= min_dets]
    species_list = sorted(species_list,
                          key=lambda s: -(len(by_species[s]['tp']) + len(by_species[s]['fp'])))

    if not species_list:
        print("No species with enough detections for per-species plot.")
        return

    ncols = 3
    nrows = (len(species_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    all_errors = [r['error'] for r in records]
    x_max = np.percentile(all_errors, 99)

    for ax, species in zip(axes, species_list):
        tp_e = np.array(by_species[species]['tp'])
        fp_e = np.array(by_species[species]['fp'])
        bins = np.linspace(0, x_max, 40)
        if len(fp_e):
            ax.hist(fp_e, bins=bins, alpha=0.6, color='tomato',
                    label=f'FP n={len(fp_e)}', density=True)
        if len(tp_e):
            ax.hist(tp_e, bins=bins, alpha=0.6, color='steelblue',
                    label=f'TP n={len(tp_e)}', density=True)
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.2,
                   label=f'thr={threshold:.3f}')
        ax.set_title(species, fontsize=11)
        ax.set_xlabel('MSE error', fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes[len(species_list):]:
        ax.set_visible(False)

    fig.suptitle('Reconstruction error by species: TP (blue) vs FP (red)', fontsize=13)
    fig.tight_layout()
    out = Path(output_dir) / 'per_species_error_dist.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # Print per-species overlap summary
    print(f"\n{'Species':<15}  {'TP n':>6}  {'FP n':>6}  "
          f"{'TP mean':>8}  {'FP mean':>8}  {'FP>thr%':>8}  {'TP>thr%':>8}")
    print('-' * 75)
    for species in species_list:
        tp_e = np.array(by_species[species]['tp'])
        fp_e = np.array(by_species[species]['fp'])
        tp_mean = np.mean(tp_e) if len(tp_e) else float('nan')
        fp_mean = np.mean(fp_e) if len(fp_e) else float('nan')
        fp_above = 100 * np.mean(fp_e > threshold) if len(fp_e) else float('nan')
        tp_above = 100 * np.mean(tp_e > threshold) if len(tp_e) else float('nan')
        print(f"{species:<15}  {len(tp_e):>6}  {len(fp_e):>6}  "
              f"{tp_mean:>8.4f}  {fp_mean:>8.4f}  {fp_above:>8.1f}  {tp_above:>8.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        default=str(REPO_ROOT / 'training/weights/2024-03-19-yolov8n-model-Kitwanga-BearCreek-Koeye-KwaKwa.pt'))
    parser.add_argument('--images-dir', default=str(REPO_ROOT / 'images/test'))
    parser.add_argument('--labels-dir', default=str(REPO_ROOT / 'labels/test'))
    parser.add_argument('--vae-weights', required=True)
    parser.add_argument('--vae-img-size', type=int, default=256)
    parser.add_argument('--calibration-dir',
                        default=str(REPO_ROOT / 'data/fish_crops'))
    parser.add_argument('--calibration-max', type=int, default=5000)
    parser.add_argument('--river', default='combined',
                        choices=['combined', 'kitwanga', 'bear'],
                        help='Which river subset to sample from (default: combined)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir',
                        default=str(REPO_ROOT / 'results/recon_error_analysis'))
    parser.add_argument('--device', default='mps')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--min-dets', type=int, default=20,
                        help='Min detections per species to include in per-species plot')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    to_tensor = transforms.Compose([
        transforms.Resize((args.vae_img_size, args.vae_img_size)),
        transforms.ToTensor(),
    ])

    print("Loading VAE ...")
    vae = load_vae(args.vae_weights, device)

    print("Computing calibration threshold ...")
    threshold, mu, sigma = compute_threshold(
        args.calibration_dir, vae, to_tensor, device, args.calibration_max)

    # Sample test frames using existing sample_subset.py
    txt_path = REPO_ROOT / 'data' / f'sample_test_{args.river}_seed{args.seed}.txt'
    print(f"\nSampling test frames ({args.river}, seed={args.seed}) ...")
    subprocess.run([
        sys.executable,
        str(REPO_ROOT / 'training' / 'tools' / 'sample_subset.py'),
        '--images-dir', args.images_dir,
        '--labels-dir', args.labels_dir,
        '--output-txt', str(txt_path),
        '--seed', str(args.seed),
        *([] if args.river == 'combined' else ['--river', args.river]),
    ], check=True)
    img_paths = txt_path.read_text().strip().splitlines()
    print(f"  {len(img_paths)} frames sampled")

    print("\nRunning YOLO + VAE inference ...")
    records = collect_detections(
        args.weights, args.images_dir, args.labels_dir,
        vae, to_tensor, device, img_paths, args.iou_threshold,
    )
    print(f"  {len(records)} total detections  "
          f"({sum(r['tp'] for r in records)} TP, "
          f"{sum(not r['tp'] for r in records)} FP)")

    print("\n=== Overall distribution ===")
    plot_overall(records, threshold, output_dir)

    print("\n=== Per-species distribution ===")
    plot_per_species(records, threshold, output_dir, args.min_dets)

    print(f"\nDone. Plots saved to {output_dir}")


if __name__ == '__main__':
    main()
