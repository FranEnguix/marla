"""Greedy-evaluation trajectory fixture shared by
``tests/test_evaluation_policy_mode.py`` and the one-off recording of
``tests/data/greedy_eval_golden_v0_10_1.json`` under MARLA v0.10.1.

Uses ONLY APIs that already existed in v0.10.1 (``evaluate_checkpoint``
without ``policy_mode``, ``save_checkpoint``, ``load_config``, ...), so the
exact same code produced the golden file on v0.10.1 and re-runs it on the
current version. A fixed-seed (untrained) policy is saved as a real
``checkpoint.pt`` in a real run directory; MARLA_FULL uses a deterministic
scripted consultant (no model, no network).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

import marla.evaluation.checkpoint_eval as checkpoint_eval
from marla.config.loader import config_hash, load_config, redacted_config_dict
from marla.evaluation.overrides import ALWAYS_QUERY, BETA_ONE, BETA_ZERO, NO_QUERY, NORMAL, PLAN_MAKER_ONLY
from marla.learning.checkpoint import save_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultationResult

INIT_SEED = 20251001
# MARLA_FULL fixture only: an untrained query gate outputs p ~= 0.45 at every
# decision, so greedy evaluation would never consult. Widening its output
# layer and re-centring the bias makes greedy consult on a few decisions
# with a >= 0.2-logit margin from the 0.5 threshold, so advice, BETA_ZERO and
# BETA_ONE actually change the recorded trajectories.
GATE_WEIGHT_SCALE = 100.0
GATE_BIAS = 8.3
MAX_EPISODE_STEPS = 30
SEED_START = 801
NUM_EPISODES = 2
OVERRIDES = {
    "NORMAL": NORMAL,
    "NO_QUERY": NO_QUERY,
    "ALWAYS_QUERY": ALWAYS_QUERY,
    "BETA_ZERO": BETA_ZERO,
    "BETA_ONE": BETA_ONE,
    "PLAN_MAKER_ONLY": PLAN_MAKER_ONLY,
}


def scripted_score(action_id: str) -> float:
    """A fixed, action-dependent Plan Maker score in [0, 1]."""
    return int.from_bytes(hashlib.sha256(action_id.encode("utf-8")).digest()[:2], "little") / 65535.0


class ScriptedConsultant:
    # evaluate_checkpoint reports these off its consultant (DirectConsultant API).
    cache_hits = 0
    cache_misses = 0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, legal_actions, episode_id, step, observation_id, observation, consulted_subnet,
                       global_candidate_action_count):
        self.calls.append({
            "episode_id": episode_id, "step": step, "consulted_subnet": consulted_subnet,
            "action_ids": [a.action_id for a in legal_actions],
            "global_candidate_action_count": global_candidate_action_count,
        })
        return ConsultationResult(
            status="accepted", scores={a.action_id: scripted_score(a.action_id) for a in legal_actions},
            request_id=f"scripted-{episode_id}-{step}",
        )


def make_run_dir(root: Path, repo_root: Path, scenario: str, consultation: bool) -> Path:
    config = load_config(repo_root / "examples" / ("assisted.yaml" if consultation else "baseline.yaml"))
    config = config.model_copy(update={"environment": config.environment.model_copy(
        update={"scenario": scenario, "max_episode_steps": MAX_EPISODE_STEPS})})
    torch.manual_seed(INIT_SEED)
    policy = RecurrentPolicy(config.policy, consultation_enabled=consultation)
    if consultation:
        with torch.no_grad():
            policy.query_gate.mlp[2].weight.mul_(GATE_WEIGHT_SCALE)
            policy.query_gate.mlp[2].bias.fill_(GATE_BIAS)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    run_dir = root / ("run_full" if consultation else "run_ppo")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(redacted_config_dict(config)), encoding="utf-8")
    save_checkpoint(run_dir / "checkpoint.pt", policy, optimizer, update_count=1, environment_steps=512,
                    config_hash=config_hash(config))
    return run_dir


def trajectory(result) -> dict:
    steps = []
    for r in result.records:
        steps.append([
            r.episode_id, r.environment_step, r.legal_action_descriptors[r.selected_action_index].action_id,
            bool(r.sampled_query), r.consulted_subnet, round(float(r.nasimemu_reward), 6), bool(r.terminated),
            bool(r.truncated), bool(r.objective_satisfied_before_action),
            round(float(r.final_logits[r.selected_action_index]), 3),
        ])
    episodes = [[s.seed, s.finish_reason, bool(s.objective_reached), bool(s.successful_finish), s.environment_steps,
                 round(float(s.nasimemu_return), 6)] for s in result.summaries]
    return {"steps": steps, "episodes": episodes}


async def run_greedy_conditions(root: Path, repo_root: Path, scenario: str, **evaluate_kwargs) -> dict:
    """{condition: trajectory} for PPO_ONLY and every MARLA_FULL override,
    all through the public ``evaluate_checkpoint`` (greedy by default)."""
    out = {}
    run_ppo = make_run_dir(root, repo_root, scenario, consultation=False)
    result = await checkpoint_eval.evaluate_checkpoint(
        run_ppo, seed_start=SEED_START, num_episodes=NUM_EPISODES, device="cpu", **evaluate_kwargs)
    out["PPO_ONLY"] = trajectory(result)

    run_full = make_run_dir(root, repo_root, scenario, consultation=True)
    original = checkpoint_eval._build_consult_fn
    checkpoint_eval._build_consult_fn = lambda *a, **k: ScriptedConsultant()
    try:
        for name, overrides in OVERRIDES.items():
            result = await checkpoint_eval.evaluate_checkpoint(
                run_full, seed_start=SEED_START, num_episodes=NUM_EPISODES, overrides=overrides, device="cpu",
                **evaluate_kwargs)
            out[f"MARLA_FULL/{name}"] = trajectory(result)
    finally:
        checkpoint_eval._build_consult_fn = original
    return out
