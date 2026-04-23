#!/usr/bin/env python3
"""
Train a VAE on frames from both rivers for site-invariant feature extraction.

The VAE is trained to minimise the ELBO (reconstruction + KL divergence) on
unlabelled frames. By training on data from both Bear Creek and Kitwanga, the
encoder is encouraged to produce latent representations that generalise across
sites rather than memorising site-specific artefacts (lighting, water colour,
background clutter).

Usage (from repo root):
    python3 training/vae/train.py \\
        --images-dirs images/test \\
        --output-dir training/vae/weights \\
        --epochs 50 --batch-size 32 --device mps
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import VAE


RIVER_PATTERNS = {
    'kitwanga': ['right_bank', 'left_bank'],
    'bear':     ['salmon_camera'],
}


def _river(stem):
    for river, patterns in RIVER_PATTERNS.items():
        if any(p in stem for p in patterns):
            return river
    return 'other'


class FrameDataset(Dataset):
    def __init__(self, images_dirs, labels_dirs=None, img_size=256,
                 max_frames=None, fish_ratio=0.7, seed=42):
        """
        fish_ratio: fraction of max_frames that should be annotated (fish) frames.
                    Remaining slots are filled with background frames.
                    Requires labels_dirs to be set; ignored otherwise.
        """
        all_paths = []
        for d in images_dirs:
            all_paths.extend(sorted(Path(d).glob("*.jpg")))

        random.seed(seed)

        # Split into fish/background if label dirs provided
        if labels_dirs:
            label_dirs = [Path(d) for d in labels_dirs]
            fish_paths, bg_paths = [], []
            for p in all_paths:
                if _river(p.stem) == 'other':
                    continue
                label = None
                for ld in label_dirs:
                    candidate = ld / (p.stem + '.txt')
                    if candidate.exists() and candidate.stat().st_size > 0:
                        label = candidate
                        break
                (fish_paths if label else bg_paths).append(p)
            print(f"  Fish frames: {len(fish_paths)} | Background frames: {len(bg_paths)}")
        else:
            fish_paths = [p for p in all_paths if _river(p.stem) != 'other']
            bg_paths   = []

        if max_frames and len(fish_paths) + len(bg_paths) > max_frames:
            if bg_paths:
                n_fish = min(int(max_frames * fish_ratio), len(fish_paths))
                n_bg   = min(max_frames - n_fish, len(bg_paths))
            else:
                n_fish = min(max_frames, len(fish_paths))
                n_bg   = 0

            # Stratify fish frames evenly across rivers
            by_river = {}
            for p in fish_paths:
                by_river.setdefault(_river(p.stem), []).append(p)
            per_river = n_fish // max(len(by_river), 1)
            sampled_fish = []
            for river, paths in by_river.items():
                n = min(per_river, len(paths))
                sampled_fish.extend(random.sample(paths, n))
                print(f"  {river} fish: {n} sampled / {len(paths)} available")

            sampled_bg = random.sample(bg_paths, n_bg) if n_bg else []
            print(f"  Background: {len(sampled_bg)} sampled / {len(bg_paths)} available")
            self.paths = sampled_fish + sampled_bg
        else:
            self.paths = fish_paths + bg_paths

        random.shuffle(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrameDataset(args.images_dirs, args.labels_dirs, args.img_size,
                           args.max_frames, args.fish_ratio)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
    print(f"Dataset: {len(dataset)} frames from {args.images_dirs}")

    model = VAE(latent_dim=args.latent_dim, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, recon_sum, kl_sum = 0.0, 0.0, 0.0

        for batch in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            loss, recon, kl = model.elbo_loss(batch, x_recon, mu, logvar, args.beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total    += loss.item()
            recon_sum += recon.item()
            kl_sum    += kl.item()

        n = len(loader)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={total/n:.2f}  recon={recon_sum/n:.2f}  kl={kl_sum/n:.2f}")

        torch.save({'epoch': epoch, 'latent_dim': args.latent_dim,
                    'depth': args.depth, 'state_dict': model.state_dict()},
                   out_dir / 'vae_latest.pt')

        if total < best_loss:
            best_loss = total
            torch.save({'epoch': epoch, 'latent_dim': args.latent_dim,
                        'depth': args.depth, 'state_dict': model.state_dict()},
                       out_dir / 'vae_best.pt')

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample = next(iter(loader))[:8].to(device)
                recon, _, _ = model(sample)
                save_image(torch.cat([sample, recon]),
                           out_dir / f'recon_epoch{epoch:03d}.png', nrow=8)
            model.train()

    print(f"\nDone. Best model: {out_dir / 'vae_best.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dirs', nargs='+',
                        default=[str(REPO_ROOT / 'images/train')],
                        help='One or more directories of .jpg frames (both rivers)')
    parser.add_argument('--labels-dirs', nargs='+', default=None,
                        help='Label directories to identify fish frames for fish/bg split. '
                             'Omit when training on pre-extracted crops (all files treated as fish).')
    parser.add_argument('--fish-ratio', type=float, default=0.7,
                        help='Fraction of max_frames that should be annotated fish frames')
    parser.add_argument('--output-dir', default='training/vae/weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Channels in spatial latent z')
    parser.add_argument('--depth', type=int, default=4,
                        help='Encoder/decoder depth: 4=16x16 latent (crops), 5=8x8 latent (full frames)')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Resize frames to this size before VAE (256 recommended)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL weight: 1.0 = standard VAE, >1 = beta-VAE (more disentangled)')
    parser.add_argument('--max-frames', type=int, default=50000,
                        help='Randomly subsample dataset to this many frames (None = use all)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    train(args)
