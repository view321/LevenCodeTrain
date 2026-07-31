"""EXPERIMENTAL — GRPO/RLVR on the code-repair task.

Policy = the edit process itself (delete decisions, insertion counts, token
fills). Unlike token-diffusion GRPO (d1, coupled-GRPO), every action here has
an exact log-probability under the sampler, so no likelihood estimator is
needed: we record each stochastic action during the rollout and recompute its
log-prob under grad for the update. One update per rollout batch keeps the
algorithm on-policy (REINFORCE with a group-relative baseline; PPO-style
ratio clipping is unnecessary in this regime and left for future work).

Rewards are dense by construction: Levenshtein-distance reduction toward the
known reference fix + syntax validity + exact match."""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import cfg_get
from ..data.corruption import CorruptionCfg, corrupt, make_junk_sampler
from ..data.tokens import TokenizerBundle
from ..model.backbone import load_tokenizer_bundle
from ..model.editor import build_editor
from ..util import resolve_device, set_seed
from ..util.lev import lev_reduction
from .state import RunDir


@dataclass
class Traj:
    del_actions: list = field(default_factory=list)   # (state_ids, keep_flags)
    ins_actions: list = field(default_factory=list)   # (state_ids, counts)
    fill_actions: list = field(default_factory=list)  # (state_ids, positions, tokens)
    out_ids: list = field(default_factory=list)

    def n_actions(self) -> int:
        n = sum(len(s[1]) for s in self.del_actions)
        n += sum(len(s[1]) for s in self.ins_actions)
        n += sum(len(s[1]) for s in self.fill_actions)
        return max(n, 1)


@torch.no_grad()
def rollout(editor, bundle: TokenizerBundle, ids: list[int], rng: random.Random,
            rounds: int, fill_steps: int, temperature: float, device) -> Traj:
    traj = Traj()
    seq = list(ids)
    protected = bundle.protected
    for _ in range(rounds):
        x = torch.tensor([seq], dtype=torch.long, device=device)
        out = editor(x)
        del_p = torch.sigmoid(out["del_logits"][0].float()).tolist()
        keep_flags = [
            1 if (seq[i] in protected or rng.random() >= del_p[i]) else 0
            for i in range(len(seq))
        ]
        traj.del_actions.append((list(seq), keep_flags))
        seq = [t for t, kflag in zip(seq, keep_flags) if kflag]

        x = torch.tensor([seq], dtype=torch.long, device=device)
        out = editor(x)
        n_ins = 0
        if len(seq) >= 2:
            probs = torch.softmax(out["ins_logits"][0].float() / max(temperature, 1e-4), dim=-1)
            counts = torch.multinomial(probs, 1).squeeze(-1).tolist()
            traj.ins_actions.append((list(seq), counts))
            new_seq: list[int] = []
            for i, tok in enumerate(seq):
                new_seq.append(tok)
                if i < len(counts) and counts[i] > 0:
                    new_seq.extend([bundle.mask_id] * int(counts[i]))
                    n_ins += int(counts[i])
            seq = new_seq

        masked = [i for i, t in enumerate(seq) if t == bundle.mask_id]
        for step in range(fill_steps):
            if not masked:
                break
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = editor(x)["mlm_logits"][0].float()
            pos = masked[: max(1, len(masked) // max(fill_steps - step, 1))]
            probs = torch.softmax(logits[pos] / max(temperature, 1e-4), dim=-1)
            toks = torch.multinomial(probs, 1).squeeze(-1).tolist()
            traj.fill_actions.append((list(seq), list(pos), list(toks)))
            for p, tk in zip(pos, toks):
                seq[p] = tk
            masked = [i for i, t in enumerate(seq) if t == bundle.mask_id]

        n_del = sum(1 for kf in keep_flags if kf == 0)
        if n_del == 0 and n_ins == 0:
            break

    traj.out_ids = seq
    return traj


def traj_logprob(editor, bundle: TokenizerBundle, traj: Traj, device) -> torch.Tensor:
    """Recompute the exact log-prob of every recorded action under grad."""
    total = torch.zeros((), device=device)
    for state, keep_flags in traj.del_actions:
        x = torch.tensor([state], dtype=torch.long, device=device)
        dl = editor(x)["del_logits"][0].float()
        logp_del = F.logsigmoid(dl)          # log P(delete)
        logp_keep = F.logsigmoid(-dl)        # log P(keep)
        for i, kflag in enumerate(keep_flags):
            if state[i] in bundle.protected:
                continue  # deterministic keep, prob 1
            total = total + (logp_keep[i] if kflag else logp_del[i])
    for state, counts in traj.ins_actions:
        x = torch.tensor([state], dtype=torch.long, device=device)
        il = editor(x)["ins_logits"][0].float().log_softmax(-1)
        for i, c in enumerate(counts):
            total = total + il[i, min(int(c), il.shape[-1] - 1)]
    for state, pos, toks in traj.fill_actions:
        x = torch.tensor([state], dtype=torch.long, device=device)
        ml = editor(x)["mlm_logits"][0].float().log_softmax(-1)
        for p, tk in zip(pos, toks):
            total = total + ml[p, tk]
    return total / traj.n_actions()


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def reward_fn(bundle: TokenizerBundle, corrupted: list[int], out_ids: list[int],
              clean: list[int], w: dict) -> float:
    r = float(w.get("w_lev", 1.0)) * lev_reduction(corrupted, out_ids, clean)
    r += float(w.get("w_syntax", 0.3)) * (1.0 if syntax_ok(bundle.decode(out_ids)) else 0.0)
    r += float(w.get("w_exact", 1.0)) * (1.0 if list(out_ids) == list(clean) else 0.0)
    return r


def make_repair_tasks(bundle: TokenizerBundle, snippets: list[str], ccfg: CorruptionCfg,
                      seed: int) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    tasks = []
    head = [bundle.bos_id] if bundle.bos_id is not None else [bundle.eos_id]
    for code in snippets:
        ids = head + bundle.encode(code) + [bundle.eos_id]
        junk = make_junk_sampler(bundle.vocab_size, frozenset(bundle.protected | {bundle.mask_id}),
                                 echo_pool=ids)
        c = corrupt(ids, rng, ccfg, junk, protected=bundle.protected)
        if c.n_junk() + c.n_missing() > 0:
            tasks.append((c.corrupted, ids))
    return tasks


def run_grpo(cfg: dict) -> None:
    from ..bench.fixtures import load_snippets

    device = resolve_device(cfg_get(cfg, "run.device", "auto"))
    seed = int(cfg_get(cfg, "run.seed", 1337))
    set_seed(seed)
    rng = random.Random(seed)

    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.repo_id"))
    editor = build_editor(
        cfg.get("init_from") or cfg_get(cfg, "model.repo_id"),
        insert_max=int(cfg_get(cfg, "model.insert_max", 8)),
        device=device, dtype=torch.float32,
    )
    g = cfg["grpo"]
    group = int(g.get("group_size", 8))
    total_steps = int(g.get("total_steps", 300))
    opt = torch.optim.AdamW([p for p in editor.parameters() if p.requires_grad],
                            lr=float(g.get("lr", 1e-6)))

    tasks = make_repair_tasks(bundle, load_snippets(), CorruptionCfg.from_dict(cfg.get("corruption", {})), seed)
    runs_root = Path(cfg_get(cfg, "run.runs_dir", "runs")) / cfg_get(cfg, "run.experiment", "levencode")
    run = RunDir(runs_root / "grpo")
    run.start("grpo", total_steps, config=cfg)

    try:
        for step in range(1, total_steps + 1):
            editor.eval()
            batch_tasks = rng.sample(tasks, min(int(g.get("prompts_per_step", 4)), len(tasks)))
            groups = []
            for corrupted, clean in batch_tasks:
                trajs = [
                    rollout(editor, bundle, corrupted, rng, rounds=2, fill_steps=6,
                            temperature=1.0, device=device)
                    for _ in range(group)
                ]
                rewards = [reward_fn(bundle, corrupted, t.out_ids, clean, g.get("reward", {})) for t in trajs]
                groups.append((trajs, rewards))

            editor.train()
            opt.zero_grad(set_to_none=True)
            n_used = 0
            mean_r = 0.0
            for trajs, rewards in groups:
                r = torch.tensor(rewards, dtype=torch.float32)
                mean_r += r.mean().item()
                if r.std() < 1e-6:
                    continue  # no learning signal in a degenerate group
                adv = (r - r.mean()) / (r.std() + 1e-6)
                for traj, a in zip(trajs, adv.tolist()):
                    lp = traj_logprob(editor, bundle, traj, device)
                    ((-a) * lp / (len(trajs) * len(groups))).backward()
                    n_used += 1
            if n_used:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in editor.parameters() if p.requires_grad],
                    float(g.get("grad_clip", 1.0)),
                )
                opt.step()
            run.progress(step, {"loss": 0.0, "mean_reward": mean_r / max(len(groups), 1),
                                "used_trajs": n_used}, lr=float(g.get("lr", 1e-6)), tok_per_sec=0.0)

        editor.save(run.root / "ckpt" / "final")
        run.finish("completed")
    except Exception:
        run.finish("failed")
        raise
