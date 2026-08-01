"""Decodability adapter: teacher latent -> chunk tokens, CALM-style robustness.

This is the distillation interface between the frozen teacher's latent space
and the student's token space. It applies CALM's autoencoder recipe (Sec. 2.2)
wholesale, on top of latents we did NOT train:

- variational bottleneck: z -> (mu, sigma); the decoder consumes a reparameterized
  sample, so the space the *predictor* operates in is smoothed the same way
  CALM smooths its own latent manifold;
- KL clipping (lambda_KL per-dimension floor) so no dimension collapses to an
  uninformative prior;
- dual dropout: feature dropout on the latent before the decoder (robustness
  to predictor error) + input-token masking applied at *precompute* time to the
  teacher pass (CBOW-style contextual enrichment), the frozen-teacher port.

The decode is token-level cross-entropy through the student's tied LM head
(SONAR-LLM's anchor): gradients flow token-by-token through the decode path
while the model "thinks" in chunk latents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodabilityAdapter(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        bottleneck_dim: int = 128,
        hidden: int = 512,
        layers: int = 2,
        heads: int = 8,
        max_chunk: int = 16,
        student_hidden: int = 1024,
        kl_beta: float = 0.001,
        kl_clip: float = 0.5,
        dropout_latent: float = 0.15,
    ):
        super().__init__()
        self.max_chunk = max_chunk
        self.bottleneck_dim = bottleneck_dim
        self.kl_beta = kl_beta
        self.kl_clip = kl_clip
        self.dropout_latent = dropout_latent
        self.enc = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.mu = nn.Linear(hidden, bottleneck_dim)
        self.logvar = nn.Linear(hidden, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden)
        self.pos_emb = nn.Parameter(torch.randn(max_chunk, hidden) * (hidden**-0.5))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.decoder = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(hidden, student_hidden)

    def forward(
        self,
        z: torch.Tensor,  # [B, latent_dim]
        reparam: bool = True,
    ) -> dict:
        B = z.shape[0]
        h = self.enc(z)
        mu, logvar = self.mu(h), self.logvar(h)
        if reparam and self.training:
            std = torch.exp(0.5 * logvar)
            z_lat = mu + std * torch.randn_like(std)
        else:
            z_lat = mu
        # KL per dimension, clipped at a floor (CALM eq. 4) — no dead dims
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = kl_per_dim.clamp_min(self.kl_clip).sum(-1).mean()
        z_lat = F.dropout(z_lat, p=self.dropout_latent, training=self.training)
        seq = z_lat.unsqueeze(1).expand(B, self.max_chunk, self.bottleneck_dim)
        x = self.up(seq) + self.pos_emb.unsqueeze(0)
        x = self.decoder(x)
        return {
            "logits_hidden": self.out(x),  # [B, max_chunk, student_hidden]
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
            "z_latent": z_lat,
        }
