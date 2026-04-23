#!/usr/bin/env python3
"""
Run YOLOv8 evaluation with VAE probabilistic quality filtering (Component 2).

Pipeline per (river, seed):
  1. sample_subset.py  -> sample_test_{river}_seed{seed}.txt
  2. YOLO predict      -> raw detections per frame
  3. For each detection: crop bounding box -> VAE encode/decode
                         -> reconstruction error r = ||crop - x_hat||^2
  4. Filter detections above the error threshold (top recon_percentile%)
  5. Evaluate filtered detections against ground truth labels

The reconstruction error acts as a confidence calibration signal: genuine
fish detections should reconstruct well (low error), while false positives
(debris, shadows, reflections) should reconstruct poorly (high error).
"""
import argparse
import random
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.ops import box_iou
from ultralytics import YOLO

SEEDS  = [42, 123, 456]
RIVERS = ['combined', 'kitwanga', 'bear']

REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_DIR   = REPO_ROOT / 'training' / 'vae'

CLASS_NAMES = {
    0: 'Coho', 1: 'Bull', 2: 'Rainbow', 3: 'Sockeye', 4: 'Pink',
    5: 'Whitefish', 6: 'Chinook', 7: 'Shiner', 8: 'Pikeminnow',
    9: 'Chum', 10: 'Steelhead', 11: 'Lamprey', 12: 'Cutthroat',
    13: 'Stickleback', 14: 'Sculpin', 15: 'Jack_Coho', 16: 'Jack_Chinook',
}
NC = len(CLASS_NAMES)


def load_vae(weights_path, device):
    sys.path.insert(0, str(VAE_DIR))
    from model import VAE
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = VAE(latent_dim=ckpt.get('latent_dim', 64),
                depth=ckpt.get('depth', 5)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def save_frame_viz(img_path, dets, out_path):
    """
    Save frame with bounding boxes coloured green (kept) or red (filtered).
    dets: list of (cls_id, conf, x1, y1, x2, y2, err, kept)
    """
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for cls_id, conf, x1, y1, x2, y2, err, kept in dets:
        color = 'lime' if kept else 'red'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 + 2)),
                  f"{CLASS_NAMES.get(cls_id, cls_id)} e={err:.3f}", fill=color)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_recon_grid(entries, out_path, thumb=128, max_entries=32, pairs_per_row=4):
    """
    Save a grid of [original crop | VAE reconstruction] pairs.
    entries: list of (crop_pil, recon_pil, cls_name, err, kept)
    Green label = kept, red label = filtered.
    """
    entries = entries[:max_entries]
    gap, label_h = 4, 18
    pair_w = 2 * thumb + gap
    n_rows = (len(entries) + pairs_per_row - 1) // pairs_per_row
    grid_w = pairs_per_row * pair_w + (pairs_per_row - 1) * gap
    grid_h = n_rows * (thumb + label_h + gap)

    grid = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for i, (crop_pil, recon_pil, cls_name, err, kept) in enumerate(entries):
        row, col = i // pairs_per_row, i % pairs_per_row
        x = col * (pair_w + gap)
        y = row * (thumb + label_h + gap)
        grid.paste(crop_pil.resize((thumb, thumb)), (x, y + label_h))
        grid.paste(recon_pil.resize((thumb, thumb)), (x + thumb + gap, y + label_h))
        color = 'lime' if kept else 'red'
        draw.text((x, y + 2),
                  f"{'kept' if kept else 'drop'} {cls_name} e={err:.4f}", fill=color)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def compute_threshold(calibration_dir, vae, to_tensor, device, max_samples=5000):
    """
    Compute µ+2σ threshold from calibration crops (paper eq. 2).
    Runs a sample of crops through the VAE and returns mean + 2*std of errors.
    """
    cal_paths = list(Path(calibration_dir).glob('*.jpg'))
    if not cal_paths:
        raise FileNotFoundError(f"No .jpg files found in calibration dir: {calibration_dir}")
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
    print(f"Calibration threshold (µ+2σ): "
          f"n={len(errors)}  µ={mu:.4f}  σ={sigma:.4f}  threshold={threshold:.4f}")
    return threshold


def load_gt(label_path, img_w, img_h):
    """Load YOLO-format label file, return (classes, boxes_xyxy_pixels)."""
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


def compute_ap(recalls, precisions):
    """101-point interpolated AP (COCO style, matches YOLOv8 model.val())."""
    return sum(
        max((p for r, p in zip(recalls, precisions) if r >= t), default=0.0)
        for t in np.linspace(0, 1, 101)
    ) / 101


def evaluate(detections, all_img_paths, labels_dir, iou_threshold=0.5):
    """
    Compute per-class AP50, Precision, Recall.

    detections: list of (img_path, cls_id, conf, x1, y1, x2, y2)
    all_img_paths: all sampled frames (for correct GT counting)
    """
    # Load GT for all sampled frames
    gt_by_img  = defaultdict(lambda: defaultdict(list))  # img -> cls -> [boxes]
    gt_count   = defaultdict(int)                         # cls -> n_gt

    for img_path in all_img_paths:
        img  = Image.open(img_path)
        w, h = img.size
        label_path = labels_dir / (Path(img_path).stem + '.txt')
        classes, boxes = load_gt(label_path, w, h)
        for cls, box in zip(classes, boxes):
            gt_by_img[img_path][cls].append(box)
            gt_count[cls] += 1

    # Group detections by class, sorted by descending confidence
    det_by_cls = defaultdict(list)
    for img_path, cls_id, conf, x1, y1, x2, y2 in detections:
        det_by_cls[cls_id].append((conf, img_path, x1, y1, x2, y2))
    for cls_id in det_by_cls:
        det_by_cls[cls_id].sort(key=lambda x: -x[0])

    rows = []
    total_tp = total_fp = total_fn = 0

    for cls_id in sorted(set(list(det_by_cls.keys()) + list(gt_count.keys()))):
        n_gt     = gt_count[cls_id]
        cls_dets = det_by_cls[cls_id]
        matched  = defaultdict(set)
        tp_list, fp_list = [], []

        for conf, img_path, x1, y1, x2, y2 in cls_dets:
            pred  = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            gt_bs = gt_by_img[img_path][cls_id]
            best_iou, best_idx = 0.0, -1
            if gt_bs:
                ious = box_iou(pred, torch.tensor(gt_bs, dtype=torch.float32))[0]
                best_iou, best_idx = ious.max().item(), ious.argmax().item()

            if best_iou >= iou_threshold and best_idx not in matched[img_path]:
                tp_list.append(1)
                fp_list.append(0)
                matched[img_path].add(best_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list) if tp_list else np.array([0])
        fp_cum = np.cumsum(fp_list) if fp_list else np.array([0])
        recalls    = (tp_cum / n_gt).tolist()       if n_gt > 0 else [0.0]
        precisions = (tp_cum / (tp_cum + fp_cum + 1e-9)).tolist()

        ap = compute_ap(recalls, precisions)
        p  = precisions[-1]
        r  = recalls[-1]
        fn = n_gt - int(tp_cum[-1])

        total_tp += int(tp_cum[-1])
        total_fp += int(fp_cum[-1])
        total_fn += fn

        name = CLASS_NAMES.get(cls_id, str(cls_id))
        rows.append({'class': name, 'AP50': ap, 'Precision': p, 'Recall': r})

    overall_p = total_tp / (total_tp + total_fp + 1e-9)
    overall_r = total_tp / (total_tp + total_fn + 1e-9)
    map50     = float(np.mean([r['AP50'] for r in rows])) if rows else 0.0
    rows.insert(0, {'class': 'all', 'AP50': map50,
                    'Precision': overall_p, 'Recall': overall_r})
    return rows


def build_confusion_matrix(detections, all_img_paths, labels_dir, iou_threshold=0.5):
    """
    Build (NC+1)×(NC+1) confusion matrix using multi-class IoU matching.
    Rows = predicted class, columns = true class; last index = background.
    """
    matrix = np.zeros((NC + 1, NC + 1), dtype=np.int64)
    labels_dir = Path(labels_dir)

    det_by_img = defaultdict(list)
    for img_path, cls_id, conf, x1, y1, x2, y2 in detections:
        det_by_img[img_path].append((cls_id, x1, y1, x2, y2))

    for img_path in all_img_paths:
        img = Image.open(img_path)
        w, h = img.size
        label_path = labels_dir / (Path(img_path).stem + '.txt')
        gt_classes, gt_boxes = load_gt(label_path, w, h)

        preds        = det_by_img.get(img_path, [])
        pred_classes = [p[0] for p in preds]
        pred_boxes   = [list(p[1:]) for p in preds]

        if not gt_classes and not pred_classes:
            continue
        if not gt_classes:
            for pc in pred_classes:
                matrix[pc, NC] += 1
            continue
        if not pred_classes:
            for gc in gt_classes:
                matrix[NC, gc] += 1
            continue

        iou = box_iou(
            torch.tensor(pred_boxes, dtype=torch.float32),
            torch.tensor(gt_boxes,   dtype=torch.float32),
        ).numpy()  # (n_pred, n_gt)

        matched_pred, matched_gt = set(), set()
        for i, j in sorted(
            [(i, j) for i in range(len(pred_classes))
                    for j in range(len(gt_classes))],
            key=lambda x: -iou[x[0], x[1]],
        ):
            if i in matched_pred or j in matched_gt:
                continue
            if iou[i, j] >= iou_threshold:
                matrix[pred_classes[i], gt_classes[j]] += 1
                matched_pred.add(i)
                matched_gt.add(j)

        for i, pc in enumerate(pred_classes):
            if i not in matched_pred:
                matrix[pc, NC] += 1
        for j, gc in enumerate(gt_classes):
            if j not in matched_gt:
                matrix[NC, gc] += 1

    return matrix


def plot_confusion_matrix(matrix, save_dir, normalize=True):
    """Plot confusion matrix matching ultralytics model.val() styling."""
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    names = [CLASS_NAMES[i] for i in range(NC)] + ['background']
    n     = NC + 1

    array = matrix.astype(float)
    if normalize:
        array = array / (array.sum(0, keepdims=True) + 1e-9)
    array[array < 0.005] = np.nan  # suppress near-zero annotations

    tick_fontsize  = max(6, 15 - 0.1 * n)
    label_fontsize = max(6, 12 - 0.1 * n)
    title_fontsize = max(6, 12 - 0.1 * n)
    btm            = max(0.1, 0.25 - 0.001 * n)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im = ax.imshow(array, cmap='Blues', vmin=0.0, interpolation='none')
        ax.xaxis.set_label_position('bottom')
        if n < 30:
            color_threshold = 0.45 * (1 if normalize else float(np.nanmax(array)))
            for i in range(n):
                for j in range(n):
                    val = array[i, j]
                    if np.isnan(val):
                        continue
                    ax.text(j, i,
                            f'{val:.2f}' if normalize else f'{int(val)}',
                            ha='center', va='center', fontsize=10,
                            color='white' if val > color_threshold else 'black')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)

    title = 'Confusion Matrix' + (' Normalized' if normalize else '')
    ax.set_xlabel('True',      fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel('Predicted', fontsize=label_fontsize, labelpad=10)
    ax.set_title(title, fontsize=title_fontsize, pad=20)

    xy = np.arange(n)
    ax.set_xticks(xy)
    ax.set_yticks(xy)
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', left=True,  right=False, labelleft=True,  labelright=False)
    ax.set_xticklabels(names, fontsize=tick_fontsize, rotation=90, ha='center')
    ax.set_yticklabels(names, fontsize=tick_fontsize)

    for spine in ('left', 'right', 'bottom', 'top'):
        ax.spines[spine].set_visible(False)
        cbar.ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    print(f'  Saved confusion matrix -> {fname}')


def sample_and_eval(weights, images_dir, labels_dir, vae_weights, vae_img_size,
                    threshold, device, river, seed,
                    viz_dir=None, viz_n_frames=20, no_filter=False):
    # Step 1: sample frames
    txt_path = REPO_ROOT / 'data' / f'sample_test_{river}_seed{seed}.txt'
    subprocess.run([
        sys.executable,
        str(REPO_ROOT / 'training' / 'tools' / 'sample_subset.py'),
        '--images-dir', images_dir,
        '--labels-dir', labels_dir,
        '--output-txt', str(txt_path),
        '--seed', str(seed),
        *([] if river == 'combined' else ['--river', river]),
    ], check=True)

    all_img_paths = txt_path.read_text().strip().splitlines()

    # Step 2: load models
    yolo      = YOLO(weights)
    if not no_filter:
        vae       = load_vae(vae_weights, torch.device(device))
    to_pil    = transforms.ToPILImage()
    to_tensor = transforms.Compose([
        transforms.Resize((vae_img_size, vae_img_size)),
        transforms.ToTensor(),
    ])

    # Frames selected for visualisation
    viz_frames = set(random.sample(all_img_paths,
                                   min(viz_n_frames, len(all_img_paths)))) if viz_dir else set()
    viz_pils = {}  # det_index -> (crop_pil, recon_pil)

    # Step 3: YOLO predict + VAE reconstruction error per detection crop
    all_detections = []
    for img_path in all_img_paths:
        img     = Image.open(img_path).convert('RGB')
        results = yolo.predict(img_path, verbose=False, device=device)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = boxes.conf[i].item()
            cls  = int(boxes.cls[i].item())
            crop = img.crop((max(0, x1), max(0, y1),
                             min(img.width, x2), min(img.height, y2)))
            if crop.width < 4 or crop.height < 4:
                continue
            if no_filter:
                err = 0.0
            else:
                x = to_tensor(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    x_recon, _, _ = vae(x)
                err = F.mse_loss(x_recon, x, reduction='mean').item()
            det_idx = len(all_detections)
            all_detections.append((img_path, cls, conf, x1, y1, x2, y2, err))
            if img_path in viz_frames and not no_filter:
                recon_pil = to_pil(x_recon.squeeze(0).cpu().clamp(0, 1))
                viz_pils[det_idx] = (crop.copy(), recon_pil)

    # Step 4: filter detections above µ+2σ threshold (paper eq. 2)
    kept_indices = set()
    if no_filter:
        filtered = [(p, c, cf, x1, y1, x2, y2)
                    for p, c, cf, x1, y1, x2, y2, e in all_detections]
        kept_indices = set(range(len(all_detections)))
        print(f"  No filter applied — {len(filtered)} detections passed through")
    elif all_detections:
        filtered = []
        for idx, (p, c, cf, x1, y1, x2, y2, e) in enumerate(all_detections):
            if e <= threshold:
                filtered.append((p, c, cf, x1, y1, x2, y2))
                kept_indices.add(idx)
        n_removed = len(all_detections) - len(filtered)
        print(f"  Filtered {n_removed}/{len(all_detections)} detections "
              f"(threshold={threshold:.4f})")
    else:
        filtered = []
        print("  No detections found.")

    # Step 5: visualise (optional)
    if viz_dir and all_detections:
        run_viz_dir = Path(viz_dir) / f'{river}_seed{seed}'

        # Annotated frames: green = kept, red = filtered
        by_frame = defaultdict(list)
        for idx, (img_path, cls, conf, x1, y1, x2, y2, err) in enumerate(all_detections):
            if img_path in viz_frames:
                by_frame[img_path].append(
                    (cls, conf, x1, y1, x2, y2, err, idx in kept_indices))
        frames_dir = run_viz_dir / 'frames'
        for img_path, dets in by_frame.items():
            out_name = Path(img_path).stem + '_filtered.jpg'
            save_frame_viz(img_path, dets, frames_dir / out_name)
        print(f"  Saved {len(by_frame)} annotated frames -> {frames_dir}")

        # Reconstruction grid: crop vs VAE reconstruction
        recon_entries = [
            (crop_pil, recon_pil,
             CLASS_NAMES.get(all_detections[i][1], str(all_detections[i][1])),
             all_detections[i][7], i in kept_indices)
            for i, (crop_pil, recon_pil) in viz_pils.items()
        ]
        if recon_entries:
            grid_path = run_viz_dir / 'recon_grid.jpg'
            save_recon_grid(recon_entries, grid_path)
            print(f"  Saved reconstruction grid ({len(recon_entries)} crops) -> {grid_path}")

    # Step 6: evaluate against ground truth
    labels_dir_path = Path(labels_dir)
    rows = evaluate(filtered, all_img_paths, labels_dir_path)
    cm   = build_confusion_matrix(filtered, all_img_paths, labels_dir_path)
    return rows, cm


def run_all(weights, images_dir, labels_dir, vae_weights, vae_img_size,
            calibration_dir, calibration_max, output_csv, device,
            viz_dir=None, viz_n_frames=20, cm_dir=None, no_filter=False):
    # Compute fixed threshold once from calibration crops (µ+2σ, paper eq. 2)
    if no_filter:
        threshold = None
        print("VAE filtering disabled — running YOLO-only evaluation for comparison.")
    else:
        device_t   = torch.device(device)
        vae        = load_vae(vae_weights, device_t)
        to_tensor  = transforms.Compose([
            transforms.Resize((vae_img_size, vae_img_size)),
            transforms.ToTensor(),
        ])
        threshold = compute_threshold(calibration_dir, vae, to_tensor, device_t, calibration_max)

    all_results = []
    cm_by_river = defaultdict(lambda: np.zeros((NC + 1, NC + 1), dtype=np.int64))
    for river in RIVERS:
        for seed in SEEDS:
            print(f"\n=== River: {river} | Seed: {seed} ===")
            rows, cm = sample_and_eval(weights, images_dir, labels_dir, vae_weights,
                                       vae_img_size, threshold, device, river, seed,
                                       viz_dir=viz_dir, viz_n_frames=viz_n_frames,
                                       no_filter=no_filter)
            cm_by_river[river] += cm
            for r in rows:
                r['river'] = river
                r['seed']  = seed
            all_results.extend(rows)

    results = pd.DataFrame(all_results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    summary = (results[results['class'] == 'all']
               .groupby('river')['AP50']
               .agg(['mean', 'std'])
               .round(4))
    print(f"\n=== Quality Filter Baseline Summary (mAP50) ===\n{summary}")
    print(f"\nFull results saved to {output_csv}")

    if cm_dir:
        print(f"\n=== Saving Confusion Matrices -> {cm_dir} ===")
        cm_all = np.zeros((NC + 1, NC + 1), dtype=np.int64)
        for river in RIVERS:
            river_dir = Path(cm_dir) / river
            plot_confusion_matrix(cm_by_river[river], river_dir, normalize=False)
            plot_confusion_matrix(cm_by_river[river], river_dir, normalize=True)
            cm_all += cm_by_river[river]
        overall_dir = Path(cm_dir) / 'overall'
        plot_confusion_matrix(cm_all, overall_dir, normalize=False)
        plot_confusion_matrix(cm_all, overall_dir, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        default=str(REPO_ROOT / 'training/weights/2024-03-19-yolov8n-model-Kitwanga-BearCreek-Koeye-KwaKwa.pt'))
    parser.add_argument('--images-dir', default=str(REPO_ROOT / 'images/test'))
    parser.add_argument('--labels-dir', default=str(REPO_ROOT / 'labels/test'))
    parser.add_argument('--vae-weights', default=None,
                        help='Path to trained VAE checkpoint (vae_best.pt). '
                             'Not required when --no-filter is set.')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip VAE filtering and evaluate all YOLO detections — '
                             'use to produce a comparable baseline via the same eval pipeline')
    parser.add_argument('--vae-img-size', type=int, default=256)
    parser.add_argument('--calibration-dir',
                        default=str(REPO_ROOT / 'data/fish_crops'),
                        help='Directory of fish crop .jpgs used to compute µ+2σ threshold')
    parser.add_argument('--calibration-max', type=int, default=5000,
                        help='Max crops to sample for threshold calibration (default 5000)')
    parser.add_argument('--output-csv',
                        default=str(REPO_ROOT / 'results/quality_filter_metrics.csv'))
    parser.add_argument('--device', default='mps')
    parser.add_argument('--viz-dir', default=None,
                        help='Directory to save annotated frames and reconstruction grids. '
                             'Omit to skip visualisation.')
    parser.add_argument('--viz-n-frames', type=int, default=20,
                        help='Number of frames to visualise per river/seed run (default 20)')
    parser.add_argument('--cm-dir',
                        default=str(REPO_ROOT / 'results/confusion_matrices/quality_filter'),
                        help='Directory to save per-river and overall confusion matrix plots')
    args = parser.parse_args()

    if not args.no_filter and args.vae_weights is None:
        parser.error('--vae-weights is required unless --no-filter is set')

    run_all(args.weights, args.images_dir, args.labels_dir, args.vae_weights,
            args.vae_img_size, args.calibration_dir, args.calibration_max,
            args.output_csv, args.device,
            viz_dir=args.viz_dir, viz_n_frames=args.viz_n_frames,
            cm_dir=args.cm_dir, no_filter=args.no_filter)
