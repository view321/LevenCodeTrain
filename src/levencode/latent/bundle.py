"""LatentBundle: the full multi-granularity JEPA stack glued together.

Owns the RVQ (discrete anchor), the decodability adapter (teacher-latent ->
tokens), and the per-level heads (AR code priors + residual heads + energy
scorers). Exposes the two training steps the trainer runs:

- `latent_step`: the mixture of JEPAs — coarse AR prior over plan codes at
  1/32 token rate, fine AR prior at 1/8 rate *conditioned on the quantized
  coarse latent*, energy-trained residual heads for both, and CFG-trained
  energy scorers (condition dropout everywhere).
- `decodability_step`: token-CE decode of teacher latents through the student's
  tied LM head with the variational + KL-clip + dual-dropout recipe.

The teacher itself is never loaded here — training consumes the precomputed
latent store only (that is what fits a ~2h demo run on a single 5090)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import DecodabilityAdapter
from .heads import LatentHeads
from .losses import code_ce_loss, energy_loss, scorer_loss
from .rvq import RVQ


class LatentBundle(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        student_hidden: int,
        codebook_size: int = 2048,
        rvq_books: int = 2,
        ar_dim: int = 512,
        ar_layers: int = 3,
        residual_blocks: int = 3,
        energy_hidden: int = 256,
        bottleneck_dim: int = 128,
        adapter_layers: int = 2,
        adapter_hidden: int = 512,
        adapter_heads: int = 8,
        max_chunk: int = 16,
        kl_beta: float = 0.001,
        kl_clip: float = 0.5,
        dropout_latent: float = 0.15,
        cfg_dropout: float = 0.2,
        residual_n_samples: int = 8,
        residual_m_targets: int = 100,
        residual_target_noise: float = 0.02,
        residual_mse: float = 0.1,
        scorer_margin: float = 1.0,
        rvq_ema_decay: float = 0.99,
        rvq_commit: float = 0.25,
    ):
        super().__init__()
        self.cfg_dropout = cfg_dropout
        self.residual_n = residual_n_samples
        self.residual_m = residual_m_targets
        self.residual_sigma = residual_target_noise
        self.residual_mse = residual_mse
        self.scorer_margin = scorer_margin
        self.rvq_commit = rvq_commit
        self.rvq = RVQ(rvq_books, codebook_size, latent_dim, ema_decay=rvq_ema_decay, commitment=rvq_commit)
        self.adapter = DecodabilityAdapter(
            latent_dim=latent_dim,
            bottleneck_dim=bottleneck_dim,
            hidden=adapter_hidden,
            layers=adapter_layers,
            heads=adapter_heads,
            max_chunk=max_chunk,
            student_hidden=student_hidden,
            kl_beta=kl_beta,
            kl_clip=kl_clip,
            dropout_latent=dropout_latent,
        )
        self.heads = LatentHeads(
            student_hidden=student_hidden,
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            ar_dim=ar_dim,
            ar_layers=ar_layers,
            residual_blocks=residual_blocks,
            energy_hidden=energy_hidden,
        )
        self.kl_beta = kl_beta

    # ---------- training steps ----------

    def _level_losses(
        self, level: nn.Module, head: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, ctx_embed: torch.Tensor
    ) -> dict:
        """Shared body for coarse/fine: quantize -> AR codes -> residual -> scorer.

        Padded (masked) window rows are dropped up front so zero-padding never
        pollutes the RVQ EMA or the energy targets; they stay only in the AR
        sequence (where the mask already zeroes their CE)."""
        B, T, D = z.shape
        m = mask.reshape(B * T).bool()
        zf = z.reshape(B * T, D)
        zq, codes, stats = self.rvq.quantize(zf[m], update=True)
        codes_full = torch.zeros(B * T, 2, dtype=codes.dtype, device=codes.device)
        codes_full[m] = codes
        codes = codes_full.reshape(B, T, 2)
        out = level(head, codes, mask)
        ce, acc1, acc2 = code_ce_loss(out["b1_logits"], out["b2_logits"], codes, mask)
        r = (zf[m] - zq)  # residual only over real rows
        conds = out["conds"].reshape(B * T, -1)[m]
        n = self.residual_n
        samples = level.residual.sample(conds, n)  # [K, n, D]
        sigma = self.residual_sigma * max(r.norm(dim=-1).mean().item(), 1e-3)
        tgt = r.unsqueeze(1) + sigma * torch.randn(len(r), self.residual_m, D, device=z.device)
        en = energy_loss(samples, tgt)
        mse = (samples.mean(1) - r).pow(2).mean()
        drop = torch.rand(len(r), device=z.device) < self.cfg_dropout
        e_pos = level.energy(conds, zf[m].detach(), dropped=drop)
        perm = torch.randperm(len(r), device=z.device)
        e_neg = level.energy(conds, zf[m].detach()[perm], dropped=drop)
        sc = scorer_loss(e_pos, e_neg, margin=self.scorer_margin)
        return {
            "ce": ce,
            "acc1": acc1,
            "acc2": acc2,
            "energy": en,
            "mse": mse,
            "scorer": sc,
            "commit": stats["commit"],
            "dist": stats["dist"],
            "cfg_drop": drop.float().mean().detach(),
            "codes": codes,
        }

    def latent_step(self, batch: dict, ctx_hidden: torch.Tensor, device: torch.device) -> dict:
        """ctx_hidden: student encoder hiddens over the batch ctx ids [B, L, H]."""
        pooled = ctx_hidden.mean(dim=1)
        ctx_embed = self.heads.ctx_embed(pooled)

        # coarse level: one AR sequence per sample over a window of chunks
        z_c = batch["z_coarse"]  # [B, W, D]
        c_mask = batch["coarse_mask"]
        coarse = self._level_losses(self.heads.coarse, ctx_embed, z_c, c_mask, ctx_embed)

        # fine level: per-row one coarse chunk; cond = quantized coarse latent
        z_f = batch["z_fine"]  # [B, Wf, D]
        f_mask = batch["fine_mask"]
        z_cond = batch["z_coarse_cond"]  # [B, D] teacher latent of the cond chunk
        zq_cond = self.rvq.quantize(z_cond, update=False)[0]
        fine_head = ctx_embed + self.heads.coarse_cond_proj(zq_cond)
        fine = self._level_losses(self.heads.fine, fine_head, z_f, f_mask, ctx_embed)

        out = {
            "coarse_ce": coarse["ce"],
            "coarse_acc_b1": coarse["acc1"],
            "coarse_acc_b2": coarse["acc2"],
            "coarse_energy": coarse["energy"],
            "coarse_mse": coarse["mse"],
            "coarse_scorer": coarse["scorer"],
            "fine_ce": fine["ce"],
            "fine_acc_b1": fine["acc1"],
            "fine_acc_b2": fine["acc2"],
            "fine_energy": fine["energy"],
            "fine_mse": fine["mse"],
            "fine_scorer": fine["scorer"],
            "commit": 0.5 * (coarse["commit"] + fine["commit"]),
            "rvq_dist": 0.5 * (coarse["dist"] + fine["dist"]),
            "cfg_drop": 0.5 * (coarse["cfg_drop"] + fine["cfg_drop"]),
        }
        return out

    def decodability_step(self, batch: dict, lm_head: nn.Module) -> dict:
        z = batch["z"]  # [B, D]
        tokens = batch["tokens"]  # [B, max_chunk]
        out = self.adapter(z, reparam=True)
        logits = lm_head(out["logits_hidden"])  # [B, max_chunk, V]
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V).float(), tokens.reshape(-1))
        kl = out["kl"]
        loss = ce + self.kl_beta * kl
        acc = (logits.argmax(-1) == tokens).float().mean()
        return {
            "decodability": loss,
            "decodability_ce": ce.detach(),
            "decodability_kl": kl.detach(),
            "decodability_acc": acc.detach(),
        }

    # ---------- inference helpers ----------

    @torch.no_grad()
    def predict_coarse_codes(
        self, ctx_embed: torch.Tensor, prev_codes: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the next coarse chunk's plan codes (discrete anchor, cheap
        temperature). prev_codes [B, T, 2] or empty; returns (codes [B, 2],
        cond [B, ar_dim])."""
        from ..sampling.block_sampler import pick_token

        T = prev_codes.shape[1]
        codes = torch.zeros(1, T + 1, 2, dtype=torch.long, device=ctx_embed.device)
        codes[:, :T] = prev_codes
        out = self.heads.coarse(ctx_embed, codes, torch.ones(1, T + 1, device=ctx_embed.device))
        l1, l2 = out["b1_logits"][0, T], out["b2_logits"][0, T]
        b1, _ = pick_token(l1.unsqueeze(0), temperature, top_p, None)
        codes[:, T, 0] = b1
        out2 = self.heads.coarse(ctx_embed, codes, torch.ones(1, T + 1, device=ctx_embed.device))
        l2b = out2["b2_logits"][0, T]
        b2, _ = pick_token(l2b.unsqueeze(0), temperature, top_p, None)
        codes[:, T, 1] = b2
        cond = out2["conds"][0, T]
        return codes[:, T], cond

    @torch.no_grad()
    def predict_coarse_latent(self, ctx_embed: torch.Tensor, prev_codes: torch.Tensor, cfg: dict) -> torch.Tensor:
        """Full coarse latent z = z_q + r via CFG-scored residual candidates."""
        codes, cond = self.predict_coarse_codes(
            ctx_embed, prev_codes, float(cfg.get("plan_temperature", 0.8)), float(cfg.get("plan_top_p", 0.9))
        )
        zq = self.rvq.quantize_codes(codes.unsqueeze(0))[0]
        n = int(cfg.get("residual_candidates", 8))
        cand = self.heads.coarse.residual.sample(cond.unsqueeze(0), n)[0]  # [n, D]
        z_cand = zq + cand
        w = float(cfg.get("cfg_weight", 1.0))
        e_cond = self.heads.coarse.energy(cond.unsqueeze(0).expand(n, -1), z_cand)
        e_null = self.heads.coarse.energy(
            torch.zeros_like(cond).unsqueeze(0).expand(n, -1), z_cand, dropped=torch.ones(n, dtype=torch.bool, device=cond.device)
        )
        score = (1 + w) * e_cond - w * e_null
        best = score.argmin()
        return (zq + cand[best])[0], codes

    @torch.no_grad()
    def predict_fine_codes(
        self, ctx_embed: torch.Tensor, coarse_codes: torch.Tensor, prev_fine_codes: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next fine chunk's codes, conditioned on the coarse plan chunk."""
        from ..sampling.block_sampler import pick_token

        zq_c = self.rvq.quantize_codes(coarse_codes.unsqueeze(0))[0]
        head = ctx_embed + self.heads.coarse_cond_proj(zq_c)
        T = prev_fine_codes.shape[1]
        codes = torch.zeros(1, T + 1, 2, dtype=torch.long, device=ctx_embed.device)
        codes[:, :T] = prev_fine_codes
        out = self.heads.fine(head, codes, torch.ones(1, T + 1, device=ctx_embed.device))
        b1, _ = pick_token(out["b1_logits"][0, T].unsqueeze(0), temperature, top_p, None)
        codes[:, T, 0] = b1
        out2 = self.heads.fine(head, codes, torch.ones(1, T + 1, device=ctx_embed.device))
        b2, _ = pick_token(out2["b2_logits"][0, T].unsqueeze(0), temperature, top_p, None)
        codes[:, T, 1] = b2
        return codes[:, T], out2["conds"][0, T]

    @torch.no_grad()
    def predict_fine_latent(self, ctx_embed: torch.Tensor, coarse_codes: torch.Tensor, prev_fine_codes: torch.Tensor, cfg: dict) -> torch.Tensor:
        codes, cond = self.predict_fine_codes(
            ctx_embed, coarse_codes, prev_fine_codes,
            float(cfg.get("plan_temperature", 0.8)), float(cfg.get("plan_top_p", 0.9)),
        )
        zq = self.rvq.quantize_codes(codes.unsqueeze(0))[0]
        n = int(cfg.get("residual_candidates", 8))
        cand = self.heads.fine.residual.sample(cond.unsqueeze(0), n)[0]
        z_cand = zq + cand
        w = float(cfg.get("cfg_weight", 1.0))
        e_cond = self.heads.fine.energy(cond.unsqueeze(0).expand(n, -1), z_cand)
        e_null = self.heads.fine.energy(
            torch.zeros_like(cond).unsqueeze(0).expand(n, -1), z_cand, dropped=torch.ones(n, dtype=torch.bool, device=cond.device)
        )
        score = (1 + w) * e_cond - w * e_null
        return (zq + cand[score.argmin()])[0], codes
