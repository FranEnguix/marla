import pytest

from marla.config.loader import ConfigError, config_hash, load_config, parse_config, redacted_config_dict
from tests.conftest import minimal_config_dict, write_yaml


def test_load_baseline_example(baseline_config_path):
    config = load_config(baseline_config_path)
    assert config.consultation.mode == "disabled"
    assert config.gatekeeper is None
    assert config.agents == []


def test_load_assisted_example(assisted_config_path):
    config = load_config(assisted_config_path)
    assert config.consultation.mode == "learned"
    assert config.gatekeeper is not None
    assert len(config.agents) == 1
    assert config.agents[0].role == "plan_maker"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_extra_field_rejected(tmp_path):
    d = minimal_config_dict("does-not-matter-because-not-yaml-suffix")
    d["not_a_real_field"] = True
    with pytest.raises(ConfigError):
        parse_config(d)


def test_unsupported_schema_version_rejected():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["schema_version"] = "9.9"
    with pytest.raises(ConfigError):
        parse_config(d)


def test_distributed_requires_run_id():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["execution"]["mode"] = "distributed"
    with pytest.raises(ConfigError):
        parse_config(d)


def test_distributed_with_run_id_ok():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["execution"]["mode"] = "distributed"
    d["experiment"]["run_id"] = "run-1"
    config = parse_config(d)
    assert config.experiment.run_id == "run-1"


def test_baseline_forbids_gatekeeper():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["gatekeeper"] = {"alias": "gatekeeper", "jid": "gk@localhost"}
    with pytest.raises(ConfigError):
        parse_config(d)


def test_baseline_forbids_plan_maker_agents():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["agents"] = [
        {
            "alias": "plan_maker_1",
            "jid": "pm@localhost",
            "role": "plan_maker",
            "model": {"backend": "local", "name": "x"},
            "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    with pytest.raises(ConfigError):
        parse_config(d)


def test_learned_requires_gatekeeper():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    with pytest.raises(ConfigError):
        parse_config(d)


def test_learned_requires_plan_maker_agent():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    d["gatekeeper"] = {"alias": "gatekeeper", "jid": "gk@localhost"}
    with pytest.raises(ConfigError):
        parse_config(d)


def test_learned_requires_max_schema_revisions():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["consultation"] = {"mode": "learned", "cost": 0.1}
    d["gatekeeper"] = {"alias": "gatekeeper", "jid": "gk@localhost"}
    d["agents"] = [
        {
            "alias": "plan_maker_1",
            "jid": "pm@localhost",
            "role": "plan_maker",
            "model": {"backend": "local", "name": "x"},
            "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    with pytest.raises(ConfigError):
        parse_config(d)


def test_duplicate_alias_rejected():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    d["gatekeeper"] = {"alias": "rl_orchestrator", "jid": "gk@localhost"}  # duplicate alias
    d["agents"] = [
        {
            "alias": "plan_maker_1",
            "jid": "pm@localhost",
            "role": "plan_maker",
            "model": {"backend": "local", "name": "x"},
            "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    with pytest.raises(ConfigError):
        parse_config(d)


def test_duplicate_jid_rejected():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    d["gatekeeper"] = {"alias": "gatekeeper", "jid": "rl-orchestrator@localhost"}  # duplicate jid
    d["agents"] = [
        {
            "alias": "plan_maker_1",
            "jid": "pm@localhost",
            "role": "plan_maker",
            "model": {"backend": "local", "name": "x"},
            "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    with pytest.raises(ConfigError):
        parse_config(d)


def test_missing_scenario_file_rejected(tmp_path):
    d = minimal_config_dict("no-such-scenario.yaml")
    content_path = tmp_path / "config.yaml"
    import yaml

    content_path.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(content_path)


def test_existing_scenario_file_accepted(tmp_path):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("placeholder: true\n", encoding="utf-8")
    d = minimal_config_dict("scenario.yaml")
    content_path = tmp_path / "config.yaml"
    import yaml

    content_path.write_text(yaml.safe_dump(d), encoding="utf-8")
    config = load_config(content_path)
    assert config.environment.scenario == "scenario.yaml"


def test_config_hash_is_stable_and_sensitive_to_changes(baseline_config_path):
    config = load_config(baseline_config_path)
    h1 = config_hash(config)
    h2 = config_hash(config)
    assert h1 == h2

    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["experiment"]["seed"] = 2
    other_config = parse_config(d)
    assert config_hash(other_config) != h1


def test_redacted_config_dict_hides_password_shaped_keys(baseline_config_path):
    config = load_config(baseline_config_path)
    redacted = redacted_config_dict(config)
    assert redacted["rl_orchestrator"]["password_env"] != "***REDACTED***"
    # password_env only ever stores a variable *name*, never a secret value,
    # so it is intentionally left as-is; this test documents that choice.
    assert redacted["rl_orchestrator"]["password_env"] == "MARLA_RL_ORCHESTRATOR_PASSWORD"


def test_redact_helper_hides_a_raw_secret_shaped_key():
    from marla.config.loader import _redact

    result = _redact({"password": "hunter2", "password_env": "MARLA_X_PASSWORD", "nested": {"api_key": "abc"}})
    assert result["password"] == "***REDACTED***"
    assert result["password_env"] == "MARLA_X_PASSWORD"
    assert result["nested"]["api_key"] == "***REDACTED***"


def test_metrics_eval_episodes_defaults_to_disabled():
    config = parse_config(minimal_config_dict("scenario-name-without-yaml-suffix"))
    assert config.metrics.eval_episodes == 0
    assert config.metrics.eval_every_rollouts == 1


def test_metrics_eval_episodes_rejects_negative_value():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["metrics"] = {"eval_episodes": -1}
    with pytest.raises(ConfigError):
        parse_config(d)


def test_metrics_eval_every_rollouts_rejects_non_positive_value():
    d = minimal_config_dict("scenario-name-without-yaml-suffix")
    d["metrics"] = {"eval_every_rollouts": 0}
    with pytest.raises(ConfigError):
        parse_config(d)
