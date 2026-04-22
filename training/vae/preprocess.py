#!/usr/bin/env python3
"""
Preprocess a sampled test set through the trained VAE encoder-decoder.

Each image is:
  1. Resized to img_size x img_size
  2. Passed through the VAE encoder -> latent z ~ q(z|x)
  3. Decoded back to image space -> x_hat (the site-invariant reconstruction)
  4. Saved to output_dir

The output .txt file lists the reconstructed image paths, ready for YOLO eval.

Usage (from repo root):
    python3 training/vae/preprocess.py \\
        --weights training/vae/weights/vae_best.pt \\
        --input-txt data/sample_test_combined_seed42.txt \\
        --output-dir data/vae_reconstructed \\
        --output-txt data/vae_sample_test_combined_seed42.txt
"""
import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import VAE


def preprocess(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.weights, map_location=device)
    latent_dim = ckpt.get('latent_dim', args.latent_dim)
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    to_tensor = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    input_paths = Path(args.input_txt).read_text().strip().splitlines()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_paths = []
    with torch.no_grad():
        for path_str in tqdm(input_paths, desc="VAE preprocess"):
            img_path = Path(path_str.strip())
            x = to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            x_recon, _, _ = model(x)
            out_path = out_dir / img_path.name
            save_image(x_recon.squeeze(0), out_path)
            new_paths.append(str(out_path.resolve()))

    out_txt = Path(args.output_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(new_paths) + "\n")
    print(f"Preprocessed {len(new_paths)} images -> {out_dir}")
    print(f"Image list -> {out_txt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to vae_best.pt checkpoint')
    parser.add_argument('--input-txt', required=True, help='Sample txt from sample_subset.py')
    parser.add_argument('--output-dir', required=True, help='Dir to save reconstructed images')
    parser.add_argument('--output-txt', required=True, help='Output txt with reconstructed paths')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    preprocess(args)
