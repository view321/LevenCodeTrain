import ast

import pytest
import torch

from levencode.bench.fixtures import load_snippets
from levencode.bench.sandbox import run_python
from levencode.bench.tasks import extract_number, numbers_equal, syntax_ok


def test_fixtures_are_valid_python():
    snippets = load_snippets()
    assert len(snippets) >= 30
    for code in snippets:
        ast.parse(code)  # raises on breakage
        assert len(code.split("\n")) >= 2


def test_extract_number():
    assert extract_number("blah blah #### 42") == "42"
    assert extract_number("#### 1,234.5") == "1234.5"
    assert extract_number("the result is 7. #### 7") == "7"
    assert extract_number("I think 3 then 5 so the answer is 8") == "8"
    assert extract_number("no numbers here") is None


def test_numbers_equal():
    assert numbers_equal("42", "42.0")
    assert numbers_equal("1234", "1234")
    assert not numbers_equal("42", "43")
    assert not numbers_equal(None, "1")


def test_syntax_ok():
    assert syntax_ok("def f():\n    return 1\n")
    assert not syntax_ok("def f(:\n    return 1\n")


def test_salvage_code():
    from levencode.bench.tasks import salvage_code

    fenced = "Sure!\n```python\ndef f():\n    return 1\n```\n"
    assert salvage_code(fenced) == "def f():\n    return 1\n"

    prose = "Here is the solution you asked for.\ndef g(x):\n    return x * 2\nHope that helps!"
    out = salvage_code(prose)
    assert out.startswith("def g") and syntax_ok(out)

    trailing = "def h():\n    return 3\nThis works because of reasons."
    assert syntax_ok(salvage_code(trailing))

    hopeless = "I cannot write code today."
    assert salvage_code(hopeless) == hopeless  # falls back to raw text


def test_sandbox_pass_fail():
    ok, detail = run_python("assert 1 + 1 == 2\n", timeout_s=10.0)
    assert ok, detail
    ok, detail = run_python("assert 1 + 1 == 3\n", timeout_s=10.0)
    assert not ok
    assert "AssertionError" in detail or "assert" in detail.lower()


def test_sandbox_timeout():
    ok, detail = run_python("while True:\n    pass\n", timeout_s=1.0)
    assert not ok and detail == "timeout"


@pytest.mark.network
def test_pll_scorer_on_tiny_model(bundle):
    from levencode.bench.tasks import BenchCtx, pll_choice_logprob
    from levencode.model.backbone import tiny_backbone
    from levencode.model.editor import LevencodeEditor

    torch.manual_seed(0)
    editor = LevencodeEditor(tiny_backbone(), insert_max=4)
    editor.eval()
    ctx = BenchCtx(editor=editor, bundle=bundle, cfg={}, device=torch.device("cpu"))
    prompt = bundle.chat_prompt_ids([{"role": "user", "content": "pick one"}])
    lp_short = pll_choice_logprob(ctx, prompt, bundle.encode("blue sky"))
    lp_long = pll_choice_logprob(ctx, prompt, bundle.encode("a much longer answer with many words"))
    assert lp_short < 0 and lp_long < 0
    assert abs(lp_short) < 50 and abs(lp_long) < 50  # length-normalized, sane scale


@pytest.mark.network
def test_repair_code_text_on_tiny_model(bundle):
    from levencode.bench.tasks import BenchCtx, repair_code_text
    from levencode.model.backbone import tiny_backbone
    from levencode.model.editor import LevencodeEditor
    from levencode.sampling.edit_sampler import EditSamplerCfg

    torch.manual_seed(0)
    editor = LevencodeEditor(tiny_backbone(), insert_max=4)
    editor.eval()
    ctx = BenchCtx(editor=editor, bundle=bundle, cfg={}, device=torch.device("cpu"))
    out = repair_code_text(ctx, "def f():\n    return 1\n", EditSamplerCfg(rounds=1, fill_steps=2))
    assert isinstance(out, str)  # random tiny weights give garbage, but no crash


@pytest.mark.network
def test_offline_tasks_run_on_tiny_model(bundle):
    """repair / infill / speed must execute end-to-end on the tiny real
    architecture (garbage quality, but exercised mechanics + report shape)."""
    from levencode.bench.benchmark import run_benchmark
    from levencode.model.backbone import tiny_backbone
    from levencode.model.editor import LevencodeEditor

    torch.manual_seed(0)
    editor = LevencodeEditor(tiny_backbone(), insert_max=4)
    cfg = {
        "stage": "unittest",
        "run": {"seed": 7},
        "bench": {"repair_n": 3, "infill_n": 3, "gen_max_blocks": 1, "exec_timeout_s": 5.0},
        "sampler": {"block_size": 8, "steps_per_block": 2, "max_blocks": 1},
        "edit_sampler": {"rounds": 1, "fill_steps": 2},
        "corruption": {"p_delete": 0.1, "p_insert": 0.05, "p_substitute": 0.05},
        "model": {"max_seq_len": 512},
    }
    results = run_benchmark(editor, bundle, cfg, torch.device("cpu"), only=["repair", "infill", "speed"])
    assert "repair_exact" in results["repair"], results["repair"]
    for key in ("repair_oracle_exact", "repair_noop_rate", "repair_len_ratio",
                "repair_mean_deleted", "repair_mean_inserted"):
        assert key in results["repair"], results["repair"]
    assert "infill_exact" in results["infill"], results["infill"]
    assert "gen_tok_per_sec" in results["speed"], results["speed"]
    for name in ("repair", "infill", "speed"):
        assert "seconds" in results[name]
