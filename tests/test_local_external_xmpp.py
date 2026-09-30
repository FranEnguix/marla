"""``execution.embedded_xmpp_server``: local mode's choice between SPADE's
embedded ``pyjabber`` server (default, unchanged behavior) and an external,
already-running XMPP server (e.g. Prosody).

Pure wiring tests -- ``run_local``/``spade.run`` are replaced by recorders,
so no XMPP server (embedded or external) is ever started here. A real
external-server run is an integration concern, not a unit one.
"""

from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from marla.cli import app
from marla.config.loader import config_hash, load_config, parse_config
from tests.conftest import EXAMPLES_DIR, minimal_config_dict

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _config_dict(**execution) -> dict:
    data = minimal_config_dict(SMALL_SCENARIO)
    data["execution"].update(execution)
    return data


def _write_run_config(tmp_path: Path, embedded: bool | None, assisted: bool = False) -> Path:
    data = yaml.safe_load((EXAMPLES_DIR / ("assisted.yaml" if assisted else "baseline.yaml")).read_text())
    if embedded is not None:
        data["execution"]["embedded_xmpp_server"] = embedded
    data["device"] = "cpu"
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["metrics"]["output_directory"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def recorded_run_local(monkeypatch):
    """Replaces run_local (imported by `marla run` at call time) with a
    recorder that fails the run immediately -- enough to observe exactly
    what the CLI passed, without starting SPADE."""
    from marla.runtime.local import LocalRunError
    from marla.scenario.models import ScenarioSolvabilityResult, SolvabilityStatus

    calls = []

    def fake_run_local(config, config_dir, num_rollouts, **kwargs):
        calls.append({"num_rollouts": num_rollouts, **kwargs})
        raise LocalRunError("recorded; not actually run")

    monkeypatch.setattr("marla.runtime.local.run_local", fake_run_local)
    monkeypatch.setattr(
        "marla.cli.preflight_check",
        lambda config, config_dir, scenario_path: ScenarioSolvabilityResult(
            status=SolvabilityStatus.PROVEN_SOLVABLE,
            universally_solvable=True,
            scenario_path=scenario_path,
            scenario_format="v2",
            objective="capture_target",
            randomized=True,
        ),
    )
    return calls


# --- config model -----------------------------------------------------------


def test_local_mode_defaults_to_embedded_server():
    config = parse_config(_config_dict())
    assert config.execution.mode == "local"
    assert config.execution.embedded_xmpp_server is True


@pytest.mark.parametrize("value", [True, False])
def test_local_mode_accepts_explicit_value(value):
    assert parse_config(_config_dict(embedded_xmpp_server=value)).execution.embedded_xmpp_server is value


@pytest.mark.parametrize("name", ["baseline.yaml", "assisted.yaml"])
def test_existing_example_configs_still_validate_with_embedded_default(name):
    config = load_config(EXAMPLES_DIR / name)
    assert config.execution.embedded_xmpp_server is True


def test_distributed_mode_accepts_field_and_is_otherwise_unchanged():
    data = _config_dict(mode="distributed")
    data["experiment"]["run_id"] = "r1"
    assert parse_config(data).execution.mode == "distributed"
    data["execution"]["embedded_xmpp_server"] = False
    assert parse_config(data).execution.embedded_xmpp_server is False


def test_multi_env_validation_unchanged():
    data = _config_dict(mode="distributed", embedded_xmpp_server=False)
    data["experiment"]["run_id"] = "r1"
    data["policy"]["ppo"]["num_envs"] = 4
    with pytest.raises(Exception, match="requires execution.mode == 'local'"):
        parse_config(data)

    local = _config_dict(embedded_xmpp_server=False)
    local["policy"]["ppo"]["num_envs"] = 4
    assert parse_config(local).policy.ppo.num_envs == 4


def _v0_10_0_config_hash(config) -> str:
    """config_hash exactly as released in v0.10.0 (whose model had no
    embedded_xmpp_server field): the full model dump, hashed as-is."""
    import hashlib
    import json

    dumped = config.model_dump(mode="json")
    dumped["execution"].pop("embedded_xmpp_server")  # the only field v0.10.0 didn't have
    return hashlib.sha256(json.dumps(dumped, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_config_hash_absent_field_keeps_v0_10_0_hash():
    config = parse_config(_config_dict())
    assert config_hash(config) == _v0_10_0_config_hash(config)


@pytest.mark.parametrize("name", ["baseline.yaml", "assisted.yaml"])
def test_config_hash_example_configs_keep_v0_10_0_hash(name):
    config = load_config(EXAMPLES_DIR / name)
    assert config_hash(config) == _v0_10_0_config_hash(config)


def test_config_hash_explicit_true_equals_absent_field():
    assert config_hash(parse_config(_config_dict(embedded_xmpp_server=True))) == config_hash(parse_config(_config_dict()))


def test_config_hash_external_differs_from_embedded():
    embedded = config_hash(parse_config(_config_dict()))
    external = config_hash(parse_config(_config_dict(embedded_xmpp_server=False)))
    assert external != embedded
    assert external == config_hash(parse_config(_config_dict(embedded_xmpp_server=False)))  # still deterministic


# --- JID domain vs. xmpp.server (external local mode only) --------------------


def _assisted_dict(server="localhost", domains=("localhost", "localhost", "localhost"), **execution) -> dict:
    data = yaml.safe_load((EXAMPLES_DIR / "assisted.yaml").read_text())
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["execution"] = {"mode": "local", **execution}
    data["xmpp"]["server"] = server
    for identity, domain in zip((data["rl_orchestrator"], data["gatekeeper"], data["agents"][0]), domains):
        identity["jid"] = f"{identity['jid'].split('@')[0]}@{domain}"
    return data


def test_external_mode_accepts_jids_on_the_configured_domain():
    assert parse_config(_assisted_dict(embedded_xmpp_server=False)).execution.embedded_xmpp_server is False
    # case-insensitive, and a JID resource part is not part of the domain
    data = _assisted_dict(server="XMPP.Example.org", domains=("xmpp.example.org",) * 3, embedded_xmpp_server=False)
    data["agents"][0]["jid"] += "/plan-maker-resource"
    parse_config(data)


@pytest.mark.parametrize("position, alias", [(0, "rl_orchestrator"), (1, "gatekeeper"), (2, "plan_maker_1")])
def test_external_mode_rejects_any_agent_jid_on_another_domain(position, alias):
    domains = ["localhost"] * 3
    domains[position] = "other.example.org"
    with pytest.raises(Exception, match=f"JID's domain must equal xmpp.server.*{alias}"):
        parse_config(_assisted_dict(domains=tuple(domains), embedded_xmpp_server=False))


def test_baseline_external_mode_checks_the_orchestrator_jid():
    data = _config_dict(embedded_xmpp_server=False)
    data["rl_orchestrator"]["jid"] = "rl-orchestrator@elsewhere"
    with pytest.raises(Exception, match="rl_orchestrator"):
        parse_config(data)


@pytest.mark.parametrize("server", ["localhost:5222", "[::1]", "xmpp://localhost"])
def test_external_mode_does_not_guess_about_non_plain_server_forms(server):
    parse_config(_assisted_dict(server=server, embedded_xmpp_server=False))


def test_jid_domain_check_does_not_apply_to_embedded_or_distributed_mode():
    mismatched = ("localhost", "other.example.org", "localhost")
    parse_config(_assisted_dict(domains=mismatched))  # embedded (default): unchanged v0.10.0 behavior
    parse_config(_assisted_dict(domains=mismatched, embedded_xmpp_server=True))
    distributed = _assisted_dict(domains=mismatched, embedded_xmpp_server=False)
    distributed["execution"]["mode"] = "distributed"
    parse_config(distributed)


# --- CLI -> run_local wiring ------------------------------------------------


@pytest.mark.parametrize("embedded, expected", [(None, True), (True, True), (False, False)])
def test_marla_run_passes_config_value_to_run_local(tmp_path, recorded_run_local, embedded, expected):
    result = runner.invoke(app, ["run", str(_write_run_config(tmp_path, embedded))])
    assert result.exit_code == 1  # the recorder fails the run on purpose
    assert len(recorded_run_local) == 1
    assert recorded_run_local[0]["embedded_xmpp_server"] is expected
    assert ("XMPP: embedded" if expected else "XMPP: external") in result.output


def test_external_mode_starts_all_configured_agents_locally(tmp_path, recorded_run_local, monkeypatch):
    """External XMPP is still local mode: `marla run` rejects --agent and
    hands the whole config (every agent) to run_local, same as embedded."""
    for var in ("MARLA_RL_ORCHESTRATOR_PASSWORD", "MARLA_GATEKEEPER_PASSWORD", "MARLA_PLAN_MAKER_1_PASSWORD"):
        monkeypatch.setenv(var, "x")
    path = _write_run_config(tmp_path, embedded=False, assisted=True)
    rejected = runner.invoke(app, ["run", str(path), "--agent", "gatekeeper"])
    assert rejected.exit_code == 1 and "local mode does not accept --agent" in rejected.output
    assert recorded_run_local == []
    result = runner.invoke(app, ["run", str(path)])
    assert result.exit_code == 1
    assert recorded_run_local[0]["embedded_xmpp_server"] is False


def test_resume_path_identical_for_embedded_and_external(tmp_path, recorded_run_local):
    from marla.learning.checkpoint import save_checkpoint
    from marla.learning.trainer import build_policy_and_optimizer

    per_mode = {}
    for embedded in (True, False):
        mode_dir = tmp_path / f"embedded_{embedded}"
        mode_dir.mkdir()
        config_path = _write_run_config(mode_dir, embedded)
        config = load_config(config_path)
        policy, optimizer = build_policy_and_optimizer(config, torch.device("cpu"))
        checkpoint = mode_dir / "checkpoint.pt"
        save_checkpoint(checkpoint, policy, optimizer, update_count=3, environment_steps=512, config_hash=config_hash(config))

        result = runner.invoke(app, ["run", str(config_path), "--resume", str(checkpoint)])
        assert result.exit_code == 1  # the recorder fails the run on purpose
        assert "Resuming from" in result.output
        call = recorded_run_local[-1]
        assert call["resume_from"] == checkpoint
        assert call["embedded_xmpp_server"] is embedded
        per_mode[embedded] = (call["num_rollouts"], {k: v for k, v in call.items() if k not in ("resume_from", "run_dir", "embedded_xmpp_server")})

    assert per_mode[True] == per_mode[False]


# --- run_local -> spade.run, and distributed mode ---------------------------


@pytest.mark.parametrize("embedded", [True, False])
def test_run_local_forwards_flag_to_spade_run(monkeypatch, embedded):
    from marla.runtime import local

    seen = {}

    def fake_spade_run(coro, embedded_xmpp_server=False):
        coro.close()  # never started: nothing to await
        seen["embedded_xmpp_server"] = embedded_xmpp_server

    monkeypatch.setattr(local.spade, "run", fake_spade_run)
    config = parse_config(_config_dict(embedded_xmpp_server=embedded))
    local.run_local(config, REPO_ROOT, num_rollouts=1, embedded_xmpp_server=config.execution.embedded_xmpp_server)
    assert seen == {"embedded_xmpp_server": embedded}


@pytest.mark.parametrize("embedded", [True, False])
def test_distributed_mode_never_starts_embedded_server(monkeypatch, embedded):
    from marla.runtime import distributed

    seen = {}

    def fake_spade_run(coro, embedded_xmpp_server=False):
        coro.close()
        seen["embedded_xmpp_server"] = embedded_xmpp_server

    monkeypatch.setattr(distributed.spade, "run", fake_spade_run)
    data = _config_dict(mode="distributed", embedded_xmpp_server=embedded)
    data["experiment"]["run_id"] = "r1"
    distributed.run_distributed(parse_config(data), REPO_ROOT, {"rl_orchestrator"}, num_rollouts=1)
    assert seen == {"embedded_xmpp_server": False}
