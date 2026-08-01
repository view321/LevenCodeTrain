import torch

from levencode.sampling.block_sampler import BlockSamplerCfg, generate, pick_token
from levencode.sampling.edit_sampler import EditSamplerCfg, repair

from conftest import EOS, IM_END, MASK, VOCAB

A, B_TOK, C_TOK, JUNK = 100, 101, 102, 666


class ConstantMLM:
    """Predicts token (10 + pos % 7) at every position with high confidence."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        L = x.shape[1]
        logits = torch.zeros(1, L, VOCAB)
        for pos in range(L):
            logits[0, pos, 10 + pos % 7] = 10.0
        return logits


class EosAtSecondBlockMLM:
    """Fills block 1 with token 50; in block 2 predicts IM_END at its first slot."""

    def __init__(self, prompt_len: int, block_size: int):
        self.eos_pos = prompt_len + block_size  # first position of block 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        logits = torch.zeros(1, L, VOCAB)
        logits[:, :, 50] = 10.0
        if L > self.eos_pos:
            logits[0, self.eos_pos, :] = 0.0
            logits[0, self.eos_pos, IM_END] = 20.0
        return logits


def test_block_sampler_fills_all_masks(bundle):
    cfg = BlockSamplerCfg(block_size=8, steps_per_block=4, max_blocks=2, temperature=0.0)
    model = ConstantMLM()
    res = generate(model, bundle, [5, 11, 12], cfg, device="cpu")
    assert MASK not in res.ids
    assert len(res.ids) == 3 + 2 * 8  # no stop token in ConstantMLM's vocab choices
    assert res.blocks == 2
    # commit schedule: at most steps_per_block forwards per block
    assert model.calls <= 2 * 4


def test_block_sampler_stops_at_eos(bundle):
    cfg = BlockSamplerCfg(block_size=8, steps_per_block=2, max_blocks=4, temperature=0.0)
    model = EosAtSecondBlockMLM(prompt_len=3, block_size=8)
    res = generate(model, bundle, [5, 11, 12], cfg, device="cpu")
    assert res.ids[-1] == IM_END
    assert res.blocks == 2
    assert len(res.ids) == 3 + 8 + 1  # block 1 full, block 2 cut at its first token
    assert IM_END not in res.new_ids and EOS not in res.new_ids


def test_block_sampler_deterministic_greedy(bundle):
    cfg = BlockSamplerCfg(block_size=8, steps_per_block=4, max_blocks=1, temperature=0.0)
    r1 = generate(ConstantMLM(), bundle, [5], cfg)
    r2 = generate(ConstantMLM(), bundle, [5], cfg)
    assert r1.ids == r2.ids


def test_pick_token_greedy_and_nucleus():
    logits = torch.tensor([[0.0, 1.0, 5.0, 2.0]])
    tok, conf = pick_token(logits, temperature=0.0, top_p=0.9, generator=None)
    assert tok.item() == 2 and conf.item() > 0.5

    gen = torch.Generator().manual_seed(0)
    picks = set()
    for _ in range(50):
        tok, _ = pick_token(logits, temperature=1.0, top_p=0.5, generator=gen)
        picks.add(tok.item())
    assert 0 not in picks  # lowest-probability token excluded from a 0.5 nucleus


class ScriptedEditor:
    """DEL flags JUNK tokens; INS asks for one mask after token A when the
    sequence has no masks; FILL predicts C_TOK at masks."""

    def __call__(self, x: torch.Tensor) -> dict:
        L = x.shape[1]
        ids = x[0]
        mlm = torch.zeros(1, L, VOCAB)
        mlm[:, :, C_TOK] = 10.0
        dl = torch.full((1, L), -8.0)
        dl[0, ids == JUNK] = 8.0
        ins = torch.zeros(1, max(L - 1, 0), 9)
        ins[:, :, 0] = 5.0
        if MASK not in ids.tolist():
            for i in range(L - 1):
                if ids[i].item() == A and ids[i + 1].item() != C_TOK:
                    ins[0, i, 0] = 0.0
                    ins[0, i, 1] = 9.0
        return {"mlm_logits": mlm, "del_logits": dl, "ins_logits": ins}


def test_edit_sampler_repairs(bundle):
    from conftest import BOS

    cfg = EditSamplerCfg(rounds=3, delete_threshold=0.5, fill_steps=4)
    seq = [BOS, A, JUNK, B_TOK, EOS]
    out, trace = repair(ScriptedEditor(), bundle, seq, cfg)
    assert out == [BOS, A, C_TOK, B_TOK, EOS]
    assert trace.deleted == 1 and trace.inserted == 1
    assert trace.rounds_used <= 3


def test_edit_sampler_ins_zero_penalty(bundle):
    """A conservatively-tied insert head (class 0 barely winning) inserts
    nothing by default, but does insert once the zero-penalty is applied."""
    from conftest import BOS

    class BarelyConservative:
        def __call__(self, x):
            L = x.shape[1]
            ids = x[0]
            mlm = torch.zeros(1, L, VOCAB)
            mlm[:, :, C_TOK] = 10.0
            ins = torch.zeros(1, max(L - 1, 0), 9)
            ins[:, :, 0] = 1.0   # count-0 wins...
            ins[:, :, 1] = 0.5   # ...but count-1 is close
            if MASK in ids.tolist():
                ins[:, :, 0] = 9.0  # never insert next to pending masks
            return {"mlm_logits": mlm, "del_logits": torch.full((1, L), -8.0), "ins_logits": ins}

    seq = [BOS, A, B_TOK, EOS]
    out, trace = repair(BarelyConservative(), bundle, seq, EditSamplerCfg(rounds=1))
    assert trace.inserted == 0 and out == seq

    out, trace = repair(
        BarelyConservative(), bundle, seq, EditSamplerCfg(rounds=1, ins_zero_penalty=1.0)
    )
    assert trace.inserted > 0
    assert C_TOK in out  # inserted masks were filled


def test_edit_sampler_never_deletes_protected(bundle):
    from conftest import BOS

    class DeleteEverything:
        def __call__(self, x):
            L = x.shape[1]
            return {
                "mlm_logits": torch.zeros(1, L, VOCAB),
                "del_logits": torch.full((1, L), 9.0),
                "ins_logits": torch.zeros(1, max(L - 1, 0), 9),
            }

    cfg = EditSamplerCfg(rounds=1)
    seq = [BOS, A, B_TOK, EOS]
    out, trace = repair(DeleteEverything(), bundle, seq, cfg)
    assert out == [BOS, EOS]
    assert trace.deleted == 2
