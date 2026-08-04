"""Isolated tests of LocalTransformersBackend (no SPADE) using a genuinely
tiny HF model. Downloads a small checkpoint on first run.
"""

import pytest

from marla.models.local_backend import LocalTransformersBackend
from marla.models.plan_maker_backend import BackendResponse


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_backend_generates_text_for_non_chat_model():
    # sshleifer/tiny-gpt2 has no chat_template configured, exercising the
    # plain-encode fallback path rather than apply_chat_template.
    backend = LocalTransformersBackend(model_name="sshleifer/tiny-gpt2", device="cpu", max_new_tokens=8)
    assert backend.resolved_device == "cpu"

    response = await backend.generate("hello world, please respond with json", legal_action_ids=[])
    assert isinstance(response, BackendResponse)
    assert isinstance(response.raw_text, str)
    assert response.latency_ms > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_backend_is_deterministic_with_greedy_decoding():
    backend = LocalTransformersBackend(model_name="sshleifer/tiny-gpt2", device="cpu", max_new_tokens=8)
    first = await backend.generate("the quick brown fox", legal_action_ids=[])
    second = await backend.generate("the quick brown fox", legal_action_ids=[])
    assert first.raw_text == second.raw_text  # do_sample=False -> greedy, deterministic


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_backend_generates_text_for_chat_template_model():
    # Regression test: apply_chat_template(..., return_tensors="pt") returns
    # a BatchEncoding (dict-like: input_ids + attention_mask), not a raw
    # tensor. Passing it as a single positional arg to generate() only fails
    # once generate() reaches into it expecting a tensor -- silently
    # "succeeding" through tokenization and .to(device) first. Every other
    # test here uses a non-chat-template model and never exercised this
    # branch at all.
    backend = LocalTransformersBackend(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM", device="cpu", max_new_tokens=8
    )
    response = await backend.generate("hello world, please respond with json", legal_action_ids=[])
    assert isinstance(response, BackendResponse)
    assert isinstance(response.raw_text, str)
    assert response.latency_ms > 0


def test_estimate_max_new_tokens_grows_with_action_count_and_id_length():
    # Regression test: a fixed max_new_tokens truncated the response mid-JSON
    # once an episode's legal action space grew past what it was tuned for
    # (observed in practice: 11 actions fit in 256 tokens, 31 longer-named
    # actions didn't, and every retry failed identically since retries get
    # the same output budget as the initial attempt).
    backend = LocalTransformersBackend(model_name="sshleifer/tiny-gpt2", device="cpu", max_new_tokens=8)

    empty = backend._estimate_max_new_tokens([])
    assert empty == 8  # falls back to the configured floor with nothing to score

    few_short = backend._estimate_max_new_tokens(["finish"])
    many_long = backend._estimate_max_new_tokens(["exploit:host-2-0:e_elasticsearch"] * 31)

    assert few_short > 8  # exceeds the tiny configured floor even for one action
    assert many_long > few_short * 10  # scales with both action count and ID length


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_uses_the_larger_of_configured_floor_and_dynamic_estimate():
    # A configured floor far below what 20 real action IDs need must not
    # truncate the response -- this is the actual bug: an untruncated
    # response for a modest action count needs more than max_new_tokens=8.
    backend = LocalTransformersBackend(model_name="sshleifer/tiny-gpt2", device="cpu", max_new_tokens=8)
    legal_action_ids = [f"exploit:host-{i}-0:e_service" for i in range(20)]

    response = await backend.generate("score these actions", legal_action_ids=legal_action_ids)

    # tiny-gpt2 generates nonsense, but the token *budget* used must have
    # been the dynamic estimate, not the tiny configured floor -- checked
    # indirectly via how many tokens it was actually allowed to produce.
    generated_tokens = len(backend._tokenizer.encode(response.raw_text, add_special_tokens=False))
    assert generated_tokens > 8
