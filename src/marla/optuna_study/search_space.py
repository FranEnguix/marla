"""The exact PPO hyperparameter search space for this study (spec
sections 28-39) -- deliberately CODE, not a generic YAML-driven DSL: the
ranges below are a specific, reviewed, tested scientific decision for
THIS study, not a reusable general-purpose feature. A future study can
copy and adapt this module rather than configuring one generic sampler
through YAML.

``sample_hyperparameters(trial)`` returns a flat dict of sampled values;
``apply_to_config_dict(base, sampled)`` merges them onto a deep copy of a
base MARLA config dict (from ``research/configs/ppo_only_4x512_target_progress.yaml``),
producing the exact config dict for one trial's agents. Every FIXED
parameter (spec section 28) is left completely untouched by this module
-- it never even looks at those keys.
"""

from __future__ import annotations

import copy
from typing import Any

import optuna

# Spec section 34: fresh paper-oriented study -- no historical conclusion
# (e.g. "epochs=4 is required") is imported into this range.
EPOCHS_CHOICES = (2, 3, 4, 5, 6)
MINIBATCH_SEQUENCES_CHOICES = (4, 8, 16)
SCHEDULER_CHOICES = ("constant", "linear", "cosine")


def sample_hyperparameters(trial: "optuna.trial.Trial") -> dict[str, Any]:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    scheduler_type = trial.suggest_categorical("scheduler_type", SCHEDULER_CHOICES)
    # Spec section 30: only linear/cosine get a final_lr_factor; constant
    # does NOT get an irrelevant conditional parameter asked of Optuna.
    final_lr_factor: float | None = None
    if scheduler_type in ("linear", "cosine"):
        final_lr_factor = trial.suggest_float("scheduler_final_lr_factor", 0.05, 0.5, log=True)

    gamma = trial.suggest_float("gamma", 0.97, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    clip_epsilon = trial.suggest_float("clip_epsilon", 0.10, 0.30)
    epochs = trial.suggest_categorical("epochs", EPOCHS_CHOICES)
    minibatch_sequences = trial.suggest_categorical("minibatch_sequences", MINIBATCH_SEQUENCES_CHOICES)
    value_coefficient = trial.suggest_float("value_coefficient", 0.25, 1.0)
    action_entropy_coefficient = trial.suggest_float("action_entropy_coefficient", 1e-4, 3e-2, log=True)

    return {
        "learning_rate": learning_rate,
        "scheduler_type": scheduler_type,
        "scheduler_final_lr_factor": final_lr_factor,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_epsilon,
        "epochs": epochs,
        "minibatch_sequences": minibatch_sequences,
        "value_coefficient": value_coefficient,
        "action_entropy_coefficient": action_entropy_coefficient,
    }


def _scheduler_block(sampled: dict[str, Any]) -> dict[str, Any]:
    scheduler_type = sampled["scheduler_type"]
    if scheduler_type == "constant":
        return {"type": "constant"}
    if scheduler_type == "linear":
        return {"type": "linear", "end_factor": sampled["scheduler_final_lr_factor"]}
    if scheduler_type == "cosine":
        # eta_min = initial_lr * final_lr_factor -- CosineSchedulerConfig
        # takes an absolute eta_min, not a factor, so it is computed here
        # from the trial's own sampled learning_rate.
        return {"type": "cosine", "eta_min": sampled["learning_rate"] * sampled["scheduler_final_lr_factor"]}
    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}")  # pragma: no cover -- guarded by suggest_categorical


def apply_to_config_dict(
    base_config_dict: dict[str, Any], sampled: dict[str, Any], seed: int, run_id: str, total_environment_steps: int
) -> dict[str, Any]:
    """Deep-copies ``base_config_dict`` and overlays exactly the TUNED
    fields (spec section 28's fixed list is everything else, left
    byte-identical to the base config) plus this agent's own seed/run_id/
    step budget. Never mutates ``base_config_dict`` itself -- one trial's
    two seeds (and every other trial) must each get an independent dict.
    """
    config = copy.deepcopy(base_config_dict)
    config["experiment"]["seed"] = seed
    config["experiment"]["run_id"] = run_id
    ppo = config["policy"]["ppo"]
    ppo["total_environment_steps"] = total_environment_steps
    ppo["gamma"] = sampled["gamma"]
    ppo["gae_lambda"] = sampled["gae_lambda"]
    ppo["clip_epsilon"] = sampled["clip_epsilon"]
    ppo["epochs"] = sampled["epochs"]
    ppo["minibatch_sequences"] = sampled["minibatch_sequences"]
    ppo["value_coefficient"] = sampled["value_coefficient"]
    ppo["action_entropy_coefficient"] = sampled["action_entropy_coefficient"]
    ppo["optimizer"]["learning_rate"] = sampled["learning_rate"]
    ppo["optimizer"]["scheduler"] = _scheduler_block(sampled)
    return config


def hyperparameters_from_params_dict(params: dict[str, Any]) -> dict[str, Any]:
    """The inverse of what a completed Optuna trial's ``.params`` dict
    already looks like -- used when replaying a specific finished trial's
    hyperparameters for finalist confirmation runs (spec section 26),
    without needing the original Trial object.
    """
    sampled = {
        "learning_rate": params["learning_rate"],
        "scheduler_type": params["scheduler_type"],
        "scheduler_final_lr_factor": params.get("scheduler_final_lr_factor"),
        "gamma": params["gamma"],
        "gae_lambda": params["gae_lambda"],
        "clip_epsilon": params["clip_epsilon"],
        "epochs": params["epochs"],
        "minibatch_sequences": params["minibatch_sequences"],
        "value_coefficient": params["value_coefficient"],
        "action_entropy_coefficient": params["action_entropy_coefficient"],
    }
    return sampled
