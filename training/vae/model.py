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


# Channel widths at each encoder depth level (indexed from 0)
_CHANNELS = [32, 64, 128, 256, 512]


class VAE(nn.Module):
    """
    Convolutional VAE for site-invariant latent feature extraction.

    depth=4 (default, crops):
      Encoder: 256->128->64->32->16   (4x stride-2 conv)
      Latent z: (B, latent_dim, 16, 16)
      Decoder: 16->32->64->128->256   (4x stride-2 deconv)

    depth=5 (full frames):
      Encoder: 256->128->64->32->16->8  (5x stride-2 conv)
      Latent z: (B, latent_dim, 8, 8)
      Decoder: 8->16->32->64->128->256  (5x stride-2 deconv)
    """

    def __init__(self, latent_dim=64, depth=4):
        super().__init__()
        assert 2 <= depth <= 5, "depth must be between 2 and 5"
        self.latent_dim = latent_dim
        self.depth = depth

        channels = _CHANNELS[:depth]          # e.g. [32,64,128,256] for depth=4
        pre_latent = channels[-1]

        enc_layers = [_conv_block(3, channels[0])]
        for i in range(1, depth):
            enc_layers.append(_conv_block(channels[i - 1], channels[i]))
        self.encoder = nn.Sequential(*enc_layers)

        self.mu_conv     = nn.Conv2d(pre_latent, latent_dim, 1)
        self.logvar_conv = nn.Conv2d(pre_latent, latent_dim, 1)

        self.decoder_input = nn.Conv2d(latent_dim, pre_latent, 1)

        dec_channels = list(reversed(channels))  # e.g. [256,128,64,32] for depth=4
        dec_layers = []
        for i in range(len(dec_channels) - 1):
            dec_layers.append(_deconv_block(dec_channels[i], dec_channels[i + 1]))
        dec_layers.append(_deconv_block(dec_channels[-1], 3, final=True))
        self.decoder = nn.Sequential(*dec_layers)

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
