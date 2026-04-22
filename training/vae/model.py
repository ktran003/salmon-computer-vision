import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_ch, out_ch, final=False):
    if final:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VAE(nn.Module):
    """
    Convolutional VAE for site-invariant latent feature extraction.

    Expects input images of shape (B, 3, 256, 256).
    Spatial latent z has shape (B, latent_dim, 8, 8), preserving structure
    for downstream spatial tasks.

    Architecture:
      Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8  (5x stride-2 conv)
      Latent:  mu, logvar each (B, latent_dim, 8, 8)
      Decoder: 8 -> 16 -> 32 -> 64 -> 128 -> 256  (5x stride-2 deconv)
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            _conv_block(3, 32),    # 256 -> 128
            _conv_block(32, 64),   # 128 -> 64
            _conv_block(64, 128),  # 64  -> 32
            _conv_block(128, 256), # 32  -> 16
            _conv_block(256, 512), # 16  -> 8
        )
        self.mu_conv     = nn.Conv2d(512, latent_dim, 1)
        self.logvar_conv = nn.Conv2d(512, latent_dim, 1)

        self.decoder_input = nn.Conv2d(latent_dim, 512, 1)
        self.decoder = nn.Sequential(
            _deconv_block(512, 256),          # 8  -> 16
            _deconv_block(256, 128),          # 16 -> 32
            _deconv_block(128, 64),           # 32 -> 64
            _deconv_block(64, 32),            # 64 -> 128
            _deconv_block(32, 3, final=True), # 128 -> 256
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_conv(h), self.logvar_conv(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def elbo_loss(self, x, x_recon, mu, logvar, beta=1.0):
        recon = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
        return recon + beta * kl, recon, kl
