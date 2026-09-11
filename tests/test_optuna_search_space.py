"""marla.optuna_study.search_space -- sampling, conditional scheduler
params, and mapping onto a MARLA config dict.
"""

from __future__ import annotations

import optuna

from marla.optuna_study.search_space import (
    EPOCHS_CHOICES,
    MINIBATCH_SEQUENCES_CHOICES,
    SCHEDULER_CHOICES,
    apply_to_config_dict,
    hyperparameters_from_params_dict,
    sample_hyperparameters,
)


def _base_config_dict() -> dict:
    return {
        "schema_version": "1.0",
        "experiment": {"name": "x", "seed": 1},
        "execution": {"mode": "local"},
        "device": "cpu",
        "xmpp": {"server": "localhost"},
        "environment": {"mode": "simulation", "scenario": "marla://x.yaml", "max_episode_steps": 100},
        "objective": {"type": "capture_target", "description": "d"},
        "policy": {
            "algorithm": "recurrent_ppo",
            "graph_encoder": {"type": "graphsage", "hidden_size": 128, "layers": 2},
            "action_encoder": {"hidden_size": 128, "action_type_embedding_size": 32},
            "recurrent": {"hidden_size": 128, "sequence_length": 64},
            "ppo": {
                "total_environment_steps": 12288,
                "num_envs": 4,
                "steps_per_env": 512,
                "epochs": 4,
                "minibatch_sequences": 8,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "value_coefficient": 0.5,
                "action_entropy_coefficient": 0.01,
                "query_entropy_coefficient": 0.01,
                "max_grad_norm": 0.5,
                "critic_refinement_epochs": 0,
                "optimizer": {"type": "adam", "learning_rate": 3e-4, "eps": 1e-5, "scheduler": {"type": "linear", "end_factor": 0.0}},
            },
        },
        "consultation": {"mode": "disabled"},
        "rl_orchestrator": {"alias": "rl_orchestrator", "jid": "rl@localhost"},
    }


def _sample_via_study() -> tuple[dict, optuna.trial.Trial]:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    trial = study.ask()
    sampled = sample_hyperparameters(trial)
    return sampled, trial


def test_sampled_values_are_within_the_declared_ranges():
    for _ in range(20):
        sampled, _ = _sample_via_study()
        assert 1e-5 <= sampled["learning_rate"] <= 1e-3
        assert sampled["scheduler_type"] in SCHEDULER_CHOICES
        assert 0.97 <= sampled["gamma"] <= 0.999
        assert 0.90 <= sampled["gae_lambda"] <= 0.99
        assert 0.10 <= sampled["clip_epsilon"] <= 0.30
        assert sampled["epochs"] in EPOCHS_CHOICES
        assert sampled["minibatch_sequences"] in MINIBATCH_SEQUENCES_CHOICES
        assert 0.25 <= sampled["value_coefficient"] <= 1.0
        assert 1e-4 <= sampled["action_entropy_coefficient"] <= 3e-2


def test_constant_scheduler_never_gets_a_final_lr_factor():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    found_constant = False
    for _ in range(30):
        trial = study.ask()
        sampled = sample_hyperparameters(trial)
        study.tell(trial, 0.0)
        if sampled["scheduler_type"] == "constant":
            found_constant = True
            assert sampled["scheduler_final_lr_factor"] is None
            assert "scheduler_final_lr_factor" not in trial.params
    assert found_constant, "expected at least one constant-scheduler draw across 30 samples"


def test_linear_and_cosine_get_a_final_lr_factor_in_range():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2))
    for _ in range(30):
        trial = study.ask()
        sampled = sample_hyperparameters(trial)
        study.tell(trial, 0.0)
        if sampled["scheduler_type"] in ("linear", "cosine"):
            assert sampled["scheduler_final_lr_factor"] is not None
            assert 0.05 <= sampled["scheduler_final_lr_factor"] <= 0.5


def test_apply_to_config_dict_only_changes_the_tuned_fields():
    base = _base_config_dict()
    sampled = {
        "learning_rate": 5e-4, "scheduler_type": "cosine", "scheduler_final_lr_factor": 0.1,
        "gamma": 0.98, "gae_lambda": 0.92, "clip_epsilon": 0.25, "epochs": 3,
        "minibatch_sequences": 16, "value_coefficient": 0.7, "action_entropy_coefficient": 0.005,
    }
    result = apply_to_config_dict(base, sampled, seed=909, run_id="trial0-seed909", total_environment_steps=12288)

    # Fixed parameters (spec section 28) untouched.
    assert result["policy"]["ppo"]["num_envs"] == 4
    assert result["policy"]["ppo"]["steps_per_env"] == 512
    assert result["policy"]["recurrent"]["sequence_length"] == 64
    assert result["policy"]["ppo"]["critic_refinement_epochs"] == 0
    assert result["policy"]["ppo"]["optimizer"]["eps"] == 1e-5
    assert result["policy"]["ppo"]["optimizer"]["type"] == "adam"
    assert result["policy"]["recurrent"]["hidden_size"] == 128
    assert result["environment"]["scenario"] == "marla://x.yaml"

    # Tuned parameters applied.
    assert result["policy"]["ppo"]["optimizer"]["learning_rate"] == 5e-4
    assert result["policy"]["ppo"]["optimizer"]["scheduler"] == {"type": "cosine", "eta_min": 5e-4 * 0.1}
    assert result["policy"]["ppo"]["gamma"] == 0.98
    assert result["policy"]["ppo"]["gae_lambda"] == 0.92
    assert result["policy"]["ppo"]["clip_epsilon"] == 0.25
    assert result["policy"]["ppo"]["epochs"] == 3
    assert result["policy"]["ppo"]["minibatch_sequences"] == 16
    assert result["policy"]["ppo"]["value_coefficient"] == 0.7
    assert result["policy"]["ppo"]["action_entropy_coefficient"] == 0.005

    # Per-agent identity applied.
    assert result["experiment"]["seed"] == 909
    assert result["experiment"]["run_id"] == "trial0-seed909"
    assert result["policy"]["ppo"]["total_environment_steps"] == 12288


def test_apply_to_config_dict_never_mutates_the_base_dict():
    base = _base_config_dict()
    original_learning_rate = base["policy"]["ppo"]["optimizer"]["learning_rate"]
    sampled = {
        "learning_rate": 1e-4, "scheduler_type": "constant", "scheduler_final_lr_factor": None,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2, "epochs": 4,
        "minibatch_sequences": 8, "value_coefficient": 0.5, "action_entropy_coefficient": 0.01,
    }
    apply_to_config_dict(base, sampled, seed=909, run_id="x", total_environment_steps=12288)
    assert base["policy"]["ppo"]["optimizer"]["learning_rate"] == original_learning_rate


def test_two_seeds_of_one_trial_get_independent_dicts_not_aliased():
    base = _base_config_dict()
    sampled = {
        "learning_rate": 1e-4, "scheduler_type": "linear", "scheduler_final_lr_factor": 0.1,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2, "epochs": 4,
        "minibatch_sequences": 8, "value_coefficient": 0.5, "action_entropy_coefficient": 0.01,
    }
    cfg_909 = apply_to_config_dict(base, sampled, seed=909, run_id="a-seed909", total_environment_steps=12288)
    cfg_919 = apply_to_config_dict(base, sampled, seed=919, run_id="a-seed919", total_environment_steps=12288)
    assert cfg_909["experiment"]["seed"] == 909
    assert cfg_919["experiment"]["seed"] == 919
    cfg_909["policy"]["ppo"]["gamma"] = 0.5
    assert cfg_919["policy"]["ppo"]["gamma"] == 0.99  # not aliased


def test_hyperparameters_from_params_dict_round_trips_apply_to_config_dict():
    base = _base_config_dict()
    sampled, trial = _sample_via_study()
    config1 = apply_to_config_dict(base, sampled, seed=909, run_id="a", total_environment_steps=12288)
    replayed = hyperparameters_from_params_dict(trial.params)
    config2 = apply_to_config_dict(base, replayed, seed=909, run_id="a", total_environment_steps=12288)
    assert config1["policy"]["ppo"]["optimizer"] == config2["policy"]["ppo"]["optimizer"]
    assert config1["policy"]["ppo"]["gamma"] == config2["policy"]["ppo"]["gamma"]
