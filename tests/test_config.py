from levencode.config import cfg_get, deep_merge, load_config


def test_deep_merge_overrides_nested():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    over = {"a": {"y": 20}, "b": [9]}
    out = deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 20}, "b": [9], "c": 3}
    assert base["a"]["y"] == 2  # no mutation


def test_load_config_extends(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnest:\n  x: 1\n  y: 2\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "_extends: base.yaml\nnest:\n  y: 99\nb: 2\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"a": 1, "b": 2, "nest": {"x": 1, "y": 99}}


def test_cfg_get():
    cfg = {"a": {"b": {"c": 42}}}
    assert cfg_get(cfg, "a.b.c") == 42
    assert cfg_get(cfg, "a.b.zzz", default=None) is None
    try:
        cfg_get(cfg, "a.zzz")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_apply_overrides():
    from levencode.config import apply_overrides

    cfg = {"bench": {"mbpp_n": 50}}
    apply_overrides(cfg, ["bench.mbpp_n=257", "bench.new_flag=true", "run.lr=2.5e-5", "run.name=big"])
    assert cfg["bench"]["mbpp_n"] == 257
    assert cfg["bench"]["new_flag"] is True
    assert cfg["run"]["lr"] == 2.5e-5
    assert cfg["run"]["name"] == "big"
    try:
        apply_overrides(cfg, ["no_equals_sign"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_real_stage_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ["stage1_sft.yaml", "stage2_edit.yaml", "stage3_jepa.yaml", "stage4_grpo.yaml"]:
        cfg = load_config(root / name)
        assert cfg["model"]["repo_id"].startswith("LiquidAI/")
        assert cfg["stage"] in {"sft", "edit", "jepa", "grpo"}
        weights = [e["weight"] for e in cfg["data"]["mix"]]
        assert abs(sum(weights) - 1.0) < 1e-6
