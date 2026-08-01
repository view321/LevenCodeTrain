"""Muon optimizer (single-device): momentum SGD whose update is replaced by a
Newton-Schulz orthogonalization of the gradient (Jordan et al., 2024).

Use ONLY for >=2D trunk weights — embeddings, norms, routers, and the tied
head belong in AdamW (`LoomLM.param_groups()` makes the split)."""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T for g = USV^T via quintic Newton-Schulz iteration.
    Coefficients tuned for fast convergence at bf16 (Jordan's nanogpt recipe)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise ValueError("Muon requires >=2D parameters; route 1D params to AdamW")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                # flatten conv-style >2D weights into a matrix for NS
                upd2 = upd.reshape(upd.shape[0], -1) if upd.ndim > 2 else upd
                o = zeropower_via_newtonschulz5(upd2, group["ns_steps"]).reshape_as(upd)
                # scale so per-element update RMS is shape-independent
                o = o * max(1.0, o.size(-2) / o.size(-1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(o, alpha=-lr)
        return loss
