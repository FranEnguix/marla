"""Unit tests for the checkpoint evaluation harness (research/aamas2027).

Covers VALIDATION.md checks 9 (no optimizer/gradient step anywhere in
marla.evaluation) and 10 (a policy's weights are unchanged by evaluation).
"""

import ast
from pathlib import Path

import pytest
import torch
import yaml

from marla.config.loader import config_hash, load_config, redacted_config_dict
from marla.evaluation.checkpoint_eval import build_policy, evaluate_checkpoint, load_run_config
from marla.evaluation.overrides import EvaluationOverrides
from marla.learning.checkpoint import save_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
EVALUATION_PACKAGE = REPO_ROOT / "src" / "marla" / "evaluation"


def make_fake_run_dir(tmp_path, consultation_mode="disabled") -> Path:
    """A minimal but real run directory: a config.yaml an actual training
    run would have written, plus a real checkpoint.pt -- exactly what
    evaluate_checkpoint expects to find."""
    example = "assisted.yaml" if consultation_mode == "learned" else "baseline.yaml"
    config = load_config(REPO_ROOT / "examples" / example)
    # Use an absolute scenario path so reloading config.yaml from tmp_path
    # resolves it correctly regardless of tmp_path's location (see
    # AUDIT.md's note on relative-scenario resolution being tied to the
    # *original* config file's directory, not the run directory).
    config = config.model_copy(
        update={"environment": config.environment.model_copy(update={"scenario": SMALL_SCENARIO})}
    )

    policy = RecurrentPolicy(config.policy, consultation_enabled=(consultation_mode == "learned"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(redacted_config_dict(config)), encoding="utf-8")
    save_checkpoint(
        run_dir / "checkpoint.pt", policy, optimizer, update_count=1, environment_steps=512,
        config_hash=config_hash(config),
    )
    return run_dir


def test_load_run_config_round_trips_scenario_and_policy_shape(tmp_path):
    run_dir = make_fake_run_dir(tmp_path)
    config = load_run_config(run_dir)
    assert config.environment.scenario == SMALL_SCENARIO
    assert config.consultation.mode == "disabled"


@pytest.mark.asyncio
async def test_evaluate_checkpoint_on_baseline_run_returns_requested_episode_count(tmp_path):
    run_dir = make_fake_run_dir(tmp_path, consultation_mode="disabled")
    result = await evaluate_checkpoint(run_dir, seed_start=1, num_episodes=2, device="cpu")
    assert result.consultation_enabled is False
    assert len(result.summaries) == 2
    assert result.cache_hits == 0 and result.cache_misses == 0


@pytest.mark.asyncio
async def test_evaluate_checkpoint_never_mutates_the_loaded_policy_weights(tmp_path):
    run_dir = make_fake_run_dir(tmp_path, consultation_mode="disabled")
    config = load_run_config(run_dir)
    policy = build_policy(config, consultation_enabled=False, device=torch.device("cpu"), checkpoint_path=run_dir / "checkpoint.pt")
    before = {k: v.clone() for k, v in policy.state_dict().items()}

    await evaluate_checkpoint(run_dir, seed_start=1, num_episodes=2, device="cpu", policy_override=policy)

    for key, value in before.items():
        assert torch.equal(value, policy.state_dict()[key]), f"parameter {key!r} changed during evaluation"


@pytest.mark.asyncio
async def test_evaluate_checkpoint_with_scenario_override_uses_that_scenario(tmp_path):
    run_dir = make_fake_run_dir(tmp_path, consultation_mode="disabled")
    other_scenario = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_two_subnets.v2.yaml").resolve())
    result = await evaluate_checkpoint(
        run_dir, seed_start=1, num_episodes=1, device="cpu", scenario_path=other_scenario
    )
    assert result.scenario == other_scenario


@pytest.mark.asyncio
async def test_evaluate_checkpoint_with_fresh_scaffold_policy_needs_no_checkpoint_file(tmp_path):
    # PLAN_MAKER_ONLY's use case: an untrained scaffold, never loaded from
    # any checkpoint.pt -- construct the run directory's config.yaml only.
    run_dir = make_fake_run_dir(tmp_path, consultation_mode="learned")
    (run_dir / "checkpoint.pt").unlink()  # prove it is genuinely never read
    config = load_run_config(run_dir)
    scaffold = build_policy(config, consultation_enabled=True, device=torch.device("cpu"), checkpoint_path=None, seed=123)

    class _RejectingConsultant:
        cache_hits = 0
        cache_misses = 0

        async def __call__(self, legal_actions, episode_id, step, source_observation_id, observation):
            from marla.learning.rollout import ConsultationResult
            return ConsultationResult(status="schema_rejected", scores=None, request_id="r")

    import marla.evaluation.checkpoint_eval as checkpoint_eval_module
    original = checkpoint_eval_module._build_consult_fn
    checkpoint_eval_module._build_consult_fn = lambda *a, **k: _RejectingConsultant()
    try:
        result = await evaluate_checkpoint(
            run_dir, seed_start=1, num_episodes=1, device="cpu",
            overrides=EvaluationOverrides(query_mode="always", action_selection="plan_maker_argmax"),
            policy_override=scaffold,
        )
    finally:
        checkpoint_eval_module._build_consult_fn = original
    assert len(result.summaries) == 1


def test_evaluation_package_never_imports_torch_optim_or_calls_backward():
    """Structural guarantee (VALIDATION.md check 9): grep every file in
    marla.evaluation for an optimizer import or a gradient-step call."""
    for path in EVALUATION_PACKAGE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "torch.optim" not in alias.name, f"{path.name} imports {alias.name}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "torch.optim" not in node.module, f"{path.name} imports from {node.module}"
            if isinstance(node, ast.Attribute) and node.attr in ("backward", "step"):
                # "step" alone is too broad (RolloutCollector/adapter/etc all
                # have legitimate step() methods) -- only .backward() is an
                # unambiguous gradient-step signal, so only fail on that.
                if node.attr == "backward":
                    pytest.fail(f"{path.name} calls .backward() -- evaluation must never take a gradient step")
