from marla.knowledge.retriever import KnowledgeRule
from marla.messaging.schemas import AdvisoryActionDescriptor, AdvisoryObjective
from marla.models.prompt import build_correction_prompt, build_prompt


def _scoped_observation(local_hosts=None, selected_subnet=1):
    return {
        "global_progress": {
            "visible_sensitive_targets_total": 0, "visible_sensitive_targets_with_root": 0,
            "visible_sensitive_targets_remaining": 0, "known_subnets_count": 1,
            "successfully_scanned_subnets_count": 0, "known_unscanned_subnets_count": 1,
            "selected_subnet": selected_subnet,
        },
        "local_hosts": local_hosts or [],
    }


def test_build_prompt_includes_all_sections():
    rules = [KnowledgeRule(id="r1", observation_flags=(), legal_action_types=(), text="Do the thing.")]
    objective = AdvisoryObjective(type="capture_target", description="Get root somewhere.")
    observation = _scoped_observation(local_hosts=[{"target": "host-1-0", "access": "none"}], selected_subnet=1)
    legal_actions = [
        AdvisoryActionDescriptor(action_id="finish", type="finish", target=None, parameters={}),
    ]

    prompt = build_prompt(rules, objective, observation, legal_actions, consulted_subnet=1, global_candidate_action_count=1)

    assert "Do the thing." in prompt
    assert "capture_target" in prompt
    assert "Get root somewhere." in prompt
    assert "host-1-0" in prompt
    assert '"finish"' in prompt
    assert "Return strict JSON" in prompt
    assert "CONSULTED SUBNET 1" in prompt


def test_build_prompt_handles_no_retrieved_rules():
    objective = AdvisoryObjective(type="capture_target", description="d")
    prompt = build_prompt(
        [], objective, _scoped_observation(), [], consulted_subnet=None, global_candidate_action_count=0
    )
    assert "no specific rules retrieved" in prompt


def test_build_correction_prompt_includes_reason_and_action_ids():
    corrected = build_correction_prompt(
        "original prompt text", reason="action_id_coverage_mismatch", detail="missing=['finish']",
        legal_action_ids=["service-scan:host-1-0", "finish"],
    )
    assert "original prompt text" in corrected
    assert "action_id_coverage_mismatch" in corrected
    assert "missing=['finish']" in corrected
    assert "service-scan:host-1-0" in corrected
    assert "finish" in corrected


def test_build_correction_prompt_warns_against_returning_a_bare_array_and_shows_an_example():
    # Regression test: observed in practice that a small model (Qwen2.5-0.5B-
    # Instruct), when the correction prompt ended on a bare JSON array of
    # action IDs, echoed that array straight back instead of producing the
    # required scored mapping -- every correction attempt failed identically.
    corrected = build_correction_prompt(
        "original prompt text", reason="action_id_coverage_mismatch", detail="missing=['finish']",
        legal_action_ids=["service-scan:host-1-0", "finish"],
    )
    assert "Do not return a JSON array of action IDs" in corrected
    assert '{"service-scan:host-1-0": 0.5}' in corrected
    # The actual directive (the scored-mapping instruction) comes after the
    # bare ID list, not before it -- the last thing the model reads should
    # be the instruction, not the array it might otherwise just copy.
    assert corrected.index("Respond with ONLY a JSON object") > corrected.index(
        '["service-scan:host-1-0", "finish"]'
    )


def test_build_correction_prompt_handles_empty_action_list():
    corrected = build_correction_prompt(
        "original prompt text", reason="some_reason", detail="", legal_action_ids=[]
    )
    assert "example_action_id" in corrected


def _finish_prompt(**objective_fields):
    from marla.config.models import ObjectiveConfig

    objective = AdvisoryObjective.from_config(ObjectiveConfig(type="capture_target", description="d", **objective_fields))
    return build_prompt([], objective, _scoped_observation(), [], consulted_subnet=None, global_candidate_action_count=0)


def test_prompt_states_configured_target_dependent_finish_reward():
    """The Plan Maker is told the FINISH reward actually in force (default config)."""
    from marla.models.prompt import PROMPT_VERSION

    prompt = _finish_prompt()
    assert "no further reward" not in prompt
    assert "+50 per sensitive target in the network if every sensitive target already has root access" in prompt
    assert "-100 for each sensitive target still lacking root access" in prompt
    assert "must be chosen explicitly" in prompt
    assert "ends the episode immediately" in prompt
    assert "score \"finish\"\nhighest instead of suggesting further actions" in prompt
    assert PROMPT_VERSION == "nasimemu-plan-maker-v3-finish-reward"


def test_prompt_reflects_explicit_reward_overrides_not_defaults():
    legacy = _finish_prompt(completion_reward=1.0, premature_finish_penalty=-1.0,
                            premature_finish_penalty_per_remaining_target=0.0)
    assert "it pays +1 if every sensitive target already has root access, and otherwise -1 (" in legacy
    assert "+50" not in legacy and "-100" not in legacy
    custom = _finish_prompt(completion_reward_per_target=10.0, premature_finish_penalty=-5.0)
    assert "+10 per sensitive target" in custom and "-5 plus -100 for each sensitive target" in custom


def test_prompt_without_reward_spec_labels_defaults_as_defaults():
    objective = AdvisoryObjective(type="capture_target", description="d")
    prompt = build_prompt([], objective, _scoped_observation(), [], consulted_subnet=None, global_candidate_action_count=0)
    assert "With the default reward configuration it pays +50 per sensitive target" in prompt
