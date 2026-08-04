"""Unit tests for PlanMakerAgent's --debug query/response file dumping.

Constructs a PlanMakerAgent directly without starting it (no SPADE
connection needed) since _write_debug_files is a plain, synchronous method.
"""

from pathlib import Path

from marla.agents.plan_maker import PlanMakerAgent
from marla.knowledge.retriever import KnowledgeBase
from marla.models.plan_maker_backend import BackendResponse


class _FakeBackend:
    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        return BackendResponse(raw_text="{}", latency_ms=1.0)


def _make_agent(debug_dir: Path | None) -> PlanMakerAgent:
    return PlanMakerAgent(
        jid="plan-maker@localhost",
        password="pass",
        alias="plan_maker_1",
        run_id="test-run",
        gatekeeper_alias="gatekeeper",
        gatekeeper_jid="gatekeeper@localhost",
        backend=_FakeBackend(),
        model_version="test-model",
        knowledge_base=KnowledgeBase(version="v1", rules=()),
        resolved_device="cpu",
        debug_dir=debug_dir,
    )


def test_write_debug_files_is_a_noop_when_debug_dir_is_none():
    agent = _make_agent(debug_dir=None)
    agent._write_debug_files("request-1", "the prompt", "the response")
    # No exception, and nothing else to assert -- there is no directory to check.


def test_write_debug_files_writes_query_and_response(tmp_path):
    agent = _make_agent(debug_dir=tmp_path)
    agent._write_debug_files("request-1", "the prompt text", "the response text")

    query_files = list(tmp_path.glob("*_query.txt"))
    response_files = list(tmp_path.glob("*_response.txt"))
    assert len(query_files) == 1
    assert len(response_files) == 1
    assert query_files[0].name.startswith("0001_request-1")
    assert query_files[0].read_text(encoding="utf-8") == "the prompt text"
    assert response_files[0].read_text(encoding="utf-8") == "the response text"


def test_write_debug_files_numbers_successive_attempts_for_the_same_request(tmp_path):
    agent = _make_agent(debug_dir=tmp_path)
    agent._write_debug_files("request-1", "initial prompt", "bad response")
    agent._write_debug_files("request-1", "correction prompt", "corrected response")

    query_files = sorted(p.name for p in tmp_path.glob("*_query.txt"))
    assert query_files == ["0001_request-1_query.txt", "0002_request-1_query.txt"]
    assert (tmp_path / "0001_request-1_query.txt").read_text(encoding="utf-8") == "initial prompt"
    assert (tmp_path / "0002_request-1_query.txt").read_text(encoding="utf-8") == "correction prompt"


def test_write_debug_files_distinguishes_different_requests(tmp_path):
    agent = _make_agent(debug_dir=tmp_path)
    agent._write_debug_files("request-a", "prompt a", "response a")
    agent._write_debug_files("request-b", "prompt b", "response b")

    query_files = sorted(p.name for p in tmp_path.glob("*_query.txt"))
    assert query_files == ["0001_request-a_query.txt", "0002_request-b_query.txt"]
