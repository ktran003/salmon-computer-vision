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
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import VAE


class FrameDataset(Dataset):
    def __init__(self, images_dirs, img_size=256):
        self.paths = []
        for d in images_dirs:
            self.paths.extend(sorted(Path(d).glob("*.jpg")))
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

    dataset = FrameDataset(args.images_dirs, args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
    print(f"Dataset: {len(dataset)} frames from {args.images_dirs}")

    model = VAE(latent_dim=args.latent_dim).to(device)
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
                    'state_dict': model.state_dict()},
                   out_dir / 'vae_latest.pt')

        if total < best_loss:
            best_loss = total
            torch.save({'epoch': epoch, 'latent_dim': args.latent_dim,
                        'state_dict': model.state_dict()},
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
    parser.add_argument('--images-dirs', nargs='+', required=True,
                        help='One or more directories of .jpg frames (both rivers)')
    parser.add_argument('--output-dir', default='training/vae/weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Channels in spatial latent z (shape: latent_dim x 8 x 8)')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Resize frames to this size before VAE (256 recommended)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL weight: 1.0 = standard VAE, >1 = beta-VAE (more disentangled)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    train(args)
