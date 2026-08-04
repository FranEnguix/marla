from marla.knowledge.retriever import KnowledgeRule
from marla.messaging.schemas import AdvisoryActionDescriptor, AdvisoryObjective
from marla.models.prompt import build_correction_prompt, build_prompt


def test_build_prompt_includes_all_sections():
    rules = [KnowledgeRule(id="r1", observation_flags=(), legal_action_types=(), text="Do the thing.")]
    objective = AdvisoryObjective(type="capture_target", description="Get root somewhere.")
    observation = {"hosts": [{"target": "host-1-0", "access": "none"}]}
    legal_actions = [
        AdvisoryActionDescriptor(action_id="finish", type="finish", target=None, parameters={}),
    ]

    prompt = build_prompt(rules, objective, observation, legal_actions)

    assert "Do the thing." in prompt
    assert "capture_target" in prompt
    assert "Get root somewhere." in prompt
    assert "host-1-0" in prompt
    assert '"finish"' in prompt
    assert "Return strict JSON" in prompt


def test_build_prompt_handles_no_retrieved_rules():
    objective = AdvisoryObjective(type="capture_target", description="d")
    prompt = build_prompt([], objective, {}, [])
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
