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
from .losses import IGNORE, code_ce_loss, energy_loss, neg_hinge, scorer_loss
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
        self, level: nn.Module, head: torch.Tensor, z: torch.Tensor, mask: torch.Tensor
    ) -> dict:
        """Shared body for coarse/fine: quantize -> AR codes -> residual -> scorer.

        Padded (masked) window rows are dropped up front so zero-padding never
        pollutes the RVQ EMA or the energy targets; they stay only in the AR
        sequence (where the mask already zeroes their CE)."""
        B, T, D = z.shape
        m = mask.reshape(B * T).bool()
        zf = z.reshape(B * T, D)
        zq, codes, stats = self.rvq.quantize(zf[m], update=True)
        R = codes.shape[-1]
        codes_full = torch.zeros(B * T, R, dtype=codes.dtype, device=codes.device)
        codes_full[m] = codes
        codes = codes_full.reshape(B, T, R)
        out = level(head, codes, mask)
        ce, acc1, acc2 = code_ce_loss(out["b1_logits"], out["b2_logits"], codes, mask)
        r = (zf[m] - zq)  # residual only over real rows
        conds = out["conds"].reshape(B * T, -1)[m]
        n = self.residual_n
        samples = level.residual.sample(conds, n)  # [K, n, D] — grad-carrying
        sigma = self.residual_sigma * max(r.norm(dim=-1).mean().item(), 1e-3)
        tgt = r.unsqueeze(1) + sigma * torch.randn(len(r), self.residual_m, D, device=z.device)
        en = energy_loss(samples, tgt)
        mse = (samples.mean(1) - r).pow(2).mean()
        # CFG-trained scorer: the RANKING hinge only applies to rows whose
        # condition survived dropout — a dropped row has no (cond, z) pairing to
        # rank. Dropped rows instead calibrate the unconditional (null) branch:
        # real latents stay plausible under it (neg_hinge), so at sampling time
        # E(null, .) is ~flat across candidates and the guided score
        # (1+w)*E(cond,z) - w*E(null,z) reduces to the conditional ranking.
        drop = torch.rand(len(r), device=z.device) < self.cfg_dropout
        zp = zf[m].detach()
        e_pos = level.energy(conds, zp, dropped=drop)
        # fixed-point-free shuffle: randperm can map i -> i, ranking a latent
        # against itself (constant-margin loss, zero gradient)
        n_rows = len(r)
        if n_rows > 1:
            shift = int(torch.randint(1, n_rows, (1,)).item())
            perm = (torch.arange(n_rows, device=z.device) + shift) % n_rows
        else:
            perm = torch.zeros(1, dtype=torch.long, device=z.device)
        e_neg = level.energy(conds, zp[perm], dropped=drop)
        keep = ~drop
        sc_terms = []
        if keep.any():
            sc_terms.append(scorer_loss(e_pos[keep], e_neg[keep], margin=self.scorer_margin))
        if drop.any():
            sc_terms.append(neg_hinge(e_pos[drop]))
        sc = torch.stack(sc_terms).mean() if sc_terms else e_pos.sum() * 0.0
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

    def latent_step(
        self,
        batch: dict,
        ctx_hidden: torch.Tensor,
        device: torch.device,
        ctx_att: torch.Tensor | None = None,
    ) -> dict:
        """ctx_hidden: student encoder hiddens over the batch ctx ids [B, L, H].
        ctx_att: 1=real token (the store left-pads short contexts; an unmasked
        mean would let pad embeddings dominate the pooled context)."""
        if ctx_att is None:
            pooled = ctx_hidden.mean(dim=1)
        else:
            a = ctx_att.unsqueeze(-1).to(ctx_hidden.dtype)
            pooled = (ctx_hidden * a).sum(dim=1) / a.sum(dim=1).clamp_min(1.0)
        ctx_embed = self.heads.ctx_embed(pooled)

        # coarse level: one AR sequence per sample over a window of chunks
        z_c = batch["z_coarse"]  # [B, W, D]
        c_mask = batch["coarse_mask"]
        coarse = self._level_losses(self.heads.coarse, ctx_embed, z_c, c_mask)

        # fine level: per-row one coarse chunk; cond = quantized coarse latent
        z_f = batch["z_fine"]  # [B, Wf, D]
        f_mask = batch["fine_mask"]
        z_cond = batch["z_coarse_cond"]  # [B, D] teacher latent of the cond chunk
        zq_cond = self.rvq.quantize(z_cond, update=False)[0]
        fine_head = ctx_embed + self.heads.coarse_cond_proj(zq_cond)
        fine = self._level_losses(self.heads.fine, fine_head, z_f, f_mask)

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
        tokens = batch["tokens"]  # [B, <=max_chunk], IGNORE-padded to the right
        if tokens.shape[1] < self.adapter.max_chunk:
            tokens = F.pad(tokens, (0, self.adapter.max_chunk - tokens.shape[1]), value=IGNORE)
        out = self.adapter(z, reparam=True)
        logits = lm_head(out["logits_hidden"])  # [B, max_chunk, V]
        V = logits.shape[-1]
        valid = tokens != IGNORE
        n_valid = valid.sum().clamp_min(1)
        # shorter chunks are -100 padded; supervising those positions would
        # teach the adapter to always emit a pad token after the chunk ends
        ce = F.cross_entropy(
            logits.reshape(-1, V).float(), tokens.reshape(-1), ignore_index=IGNORE, reduction="sum"
        ) / n_valid
        kl = out["kl"]
        loss = ce + self.kl_beta * kl
        acc = ((logits.argmax(-1) == tokens) & valid).float().sum() / n_valid
        return {
            "decodability": loss,
            "decodability_ce": ce.detach(),
            "decodability_kl": kl.detach(),
            "decodability_acc": acc.detach(),
        }

    # ---------- inference helpers ----------

    @torch.no_grad()
    def _predict_codes(
        self, level: nn.Module, head: torch.Tensor, prev_codes: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the next chunk's plan codes. prev_codes [B, T, 2] or empty;
        returns (codes [B, 2], cond [B, ar_dim])."""
        from ..sampling.block_sampler import pick_token

        T = prev_codes.shape[1]
        codes = torch.zeros(1, T + 1, 2, dtype=torch.long, device=head.device)
        codes[:, :T] = prev_codes
        ones = torch.ones(1, T + 1, device=head.device)
        out = level(head, codes, ones)
        b1, _ = pick_token(out["b1_logits"][0, T].unsqueeze(0), temperature, top_p, None)
        codes[:, T, 0] = b1
        out = level(head, codes, ones)
        b2, _ = pick_token(out["b2_logits"][0, T].unsqueeze(0), temperature, top_p, None)
        codes[:, T, 1] = b2
        # conds must see BOTH sampled books: training reads them after b2_j
        # (test_conds_are_after_b2), so reading from the pass above would
        # condition the residual/energy on the zero placeholder in b2's slot
        out = level(head, codes, ones)
        return codes[:, T], out["conds"][0, T]

    @torch.no_grad()
    def _guided_latent(self, level: nn.Module, cond: torch.Tensor, codes: torch.Tensor, cfg: dict) -> torch.Tensor:
        """Full latent z = z_q + r via CFG-scored residual candidates."""
        zq = self.rvq.quantize_codes(codes.unsqueeze(0))[0]
        n = int(cfg.get("residual_candidates", 8))
        cand = level.residual.sample(cond.unsqueeze(0), n)[0]  # [n, D]
        z_cand = zq + cand
        w = float(cfg.get("cfg_weight", 1.0))
        e_cond = level.energy(cond.unsqueeze(0).expand(n, -1), z_cand)
        e_null = level.energy(
            torch.zeros_like(cond).unsqueeze(0).expand(n, -1), z_cand,
            dropped=torch.ones(n, dtype=torch.bool, device=cond.device),
        )
        score = (1 + w) * e_cond - w * e_null
        return (zq + cand[score.argmin()])[0]

    @torch.no_grad()
    def predict_coarse_codes(
        self, ctx_embed: torch.Tensor, prev_codes: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._predict_codes(self.heads.coarse, ctx_embed, prev_codes, temperature, top_p)

    @torch.no_grad()
    def predict_coarse_latent(
        self, ctx_embed: torch.Tensor, prev_codes: torch.Tensor, cfg: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, cond = self.predict_coarse_codes(
            ctx_embed, prev_codes, float(cfg.get("plan_temperature", 0.8)), float(cfg.get("plan_top_p", 0.9))
        )
        return self._guided_latent(self.heads.coarse, cond, codes, cfg), codes

    def _fine_head(self, ctx_embed: torch.Tensor, coarse_codes: torch.Tensor) -> torch.Tensor:
        zq_c = self.rvq.quantize_codes(coarse_codes.unsqueeze(0))[0]
        return ctx_embed + self.heads.coarse_cond_proj(zq_c)

    @torch.no_grad()
    def predict_fine_codes(
        self, ctx_embed: torch.Tensor, coarse_codes: torch.Tensor, prev_fine_codes: torch.Tensor, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next fine chunk's codes, conditioned on the coarse plan chunk."""
        head = self._fine_head(ctx_embed, coarse_codes)
        return self._predict_codes(self.heads.fine, head, prev_fine_codes, temperature, top_p)

    @torch.no_grad()
    def predict_fine_latent(
        self, ctx_embed: torch.Tensor, coarse_codes: torch.Tensor, prev_fine_codes: torch.Tensor, cfg: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes, cond = self.predict_fine_codes(
            ctx_embed, coarse_codes, prev_fine_codes,
            float(cfg.get("plan_temperature", 0.8)), float(cfg.get("plan_top_p", 0.9)),
        )
        return self._guided_latent(self.heads.fine, cond, codes, cfg), codes
