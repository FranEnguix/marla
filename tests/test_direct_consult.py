"""Unit tests for DirectConsultant: the in-process consult_fn used only for
evaluation. Verifies it reproduces the same accept/correct/reject semantics
as the real Gatekeeper+PlanMaker round trip, without any SPADE agent."""

import pytest

from marla.environment.actions import ActionDescriptor
from marla.evaluation.advisory_cache import AdvisoryCache
from marla.evaluation.direct_consult import DirectConsultant
from marla.knowledge.retriever import KnowledgeBase
from marla.messaging.schemas import AdvisoryObjective
from marla.models.plan_maker_backend import BackendResponse

LEGAL_ACTIONS = [
    ActionDescriptor(action_id="finish", action_type="finish", target_key=None, is_finish=True),
    ActionDescriptor(action_id="scan:1-2:os_scan", action_type="os_scan", target_key="1-2"),
]
OBJECTIVE = AdvisoryObjective(type="capture_target", description="test objective")


class _ScriptedBackend:
    """Returns each response in ``responses`` in order, one per call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        self.call_count += 1
        text = self._responses.pop(0)
        return BackendResponse(raw_text=text, latency_ms=5.0, input_tokens=10, output_tokens=3, total_tokens=13)


def make_consultant(backend, cache=None, max_schema_revisions=3):
    return DirectConsultant(
        backend=backend,
        knowledge_base=KnowledgeBase(version="test-v1", rules=()),
        model_version="test-model",
        objective=OBJECTIVE,
        run_id="test-run",
        max_schema_revisions=max_schema_revisions,
        cache=cache,
    )


@pytest.mark.asyncio
async def test_valid_response_is_accepted_on_the_first_attempt():
    backend = _ScriptedBackend([
        '{"finish": 0.2, "scan:1-2:os_scan": 0.8}',
    ])
    consultant = make_consultant(backend)
    result = await consultant(LEGAL_ACTIONS, episode_id=1, step=0, source_observation_id="obs-1-0", observation={})
    assert result.status == "accepted"
    assert result.scores == {"finish": 0.2, "scan:1-2:os_scan": 0.8}
    assert result.input_tokens == 10 and result.output_tokens == 3 and result.total_tokens == 13
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_malformed_response_is_corrected_then_accepted():
    backend = _ScriptedBackend([
        "not json at all",
        '{"finish": 0.5, "scan:1-2:os_scan": 0.5}',
    ])
    consultant = make_consultant(backend)
    result = await consultant(LEGAL_ACTIONS, episode_id=1, step=0, source_observation_id="obs-1-0", observation={})
    assert result.status == "accepted"
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_persistent_malformed_response_is_rejected_after_max_revisions():
    backend = _ScriptedBackend(["not json"] * 4)  # 1 initial + 3 corrections
    consultant = make_consultant(backend, max_schema_revisions=3)
    result = await consultant(LEGAL_ACTIONS, episode_id=1, step=0, source_observation_id="obs-1-0", observation={})
    assert result.status == "schema_rejected"
    assert result.scores is None
    assert backend.call_count == 4


@pytest.mark.asyncio
async def test_cache_hit_avoids_a_second_backend_call(tmp_path):
    cache = AdvisoryCache(tmp_path / "cache.jsonl")
    backend = _ScriptedBackend(['{"finish": 0.1, "scan:1-2:os_scan": 0.9}'])
    consultant = make_consultant(backend, cache=cache)

    first = await consultant(LEGAL_ACTIONS, episode_id=1, step=0, source_observation_id="obs-1-0", observation={"x": 1})
    second = await consultant(LEGAL_ACTIONS, episode_id=2, step=5, source_observation_id="obs-2-5", observation={"x": 1})

    assert backend.call_count == 1  # second call was a cache hit despite different episode_id/step/observation_id
    assert first.scores == second.scores
    assert consultant.cache_hits == 1
    assert consultant.cache_misses == 1


@pytest.mark.asyncio
async def test_different_observation_is_a_cache_miss(tmp_path):
    cache = AdvisoryCache(tmp_path / "cache.jsonl")
    backend = _ScriptedBackend([
        '{"finish": 0.1, "scan:1-2:os_scan": 0.9}',
        '{"finish": 0.3, "scan:1-2:os_scan": 0.7}',
    ])
    consultant = make_consultant(backend, cache=cache)

    await consultant(LEGAL_ACTIONS, episode_id=1, step=0, source_observation_id="obs-1-0", observation={"x": 1})
    await consultant(LEGAL_ACTIONS, episode_id=1, step=1, source_observation_id="obs-1-1", observation={"x": 2})

    assert backend.call_count == 2
    assert consultant.cache_misses == 2


@pytest.mark.asyncio
async def test_backend_exception_routes_through_correction_not_a_crash():
    class _RaisingBackend:
        async def generate(self, prompt, legal_action_ids):
            raise RuntimeError("simulated backend failure")

    consultant = make_consultant(_RaisingBackend(), max_schema_revisions=0)
    result = await consultant(LEGAL_ACTIONS, episode_id=1, step=0, source_observation_id="obs-1-0", observation={})
    assert result.status == "schema_rejected"
