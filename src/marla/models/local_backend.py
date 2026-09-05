"""Local HuggingFace transformers backend for the Plan Maker (Milestone 9).

Deliberate deviation from spec section 4.1 ("blocking Plan Maker inference
must not block the shared asyncio loop... should run via
asyncio.to_thread"): in this SPADE deployment, ``asyncio.to_thread`` (and an
explicit ``ThreadPoolExecutor`` -- it isn't the default-executor path
specifically) reproducibly hangs *indefinitely* when used to run a real
``transformers`` ``generate()`` call from inside SPADE's multi-agent async
context, which unconditionally runs on ``uvloop`` with no supported way to
disable it. A trivial non-torch call through the same executor (e.g.
``asyncio.to_thread(time.sleep, ...)``) does not hang, so this is specific
to torch/transformers generation off the main thread here, not threading or
pyjabber/presence in general (both were isolated and ruled out).

Given the spec's own "no elapsed timeout" design, a hang here is silent and
permanent for a real training run -- far worse than the brief blocking
delay of running generation directly on the event loop thread, which is
otherwise idle during inference anyway (nothing else can usefully proceed
without the Plan Maker's response). This backend therefore calls
``generate()`` synchronously. Device resolution follows the same
``cpu | gpu | auto`` semantics as every other locally executed model (spec
section 18): a ``gpu`` request that can't actually initialize CUDA raises
before the agent ever loads weights, let alone reports ready.
"""

from __future__ import annotations

import time

from marla.models.plan_maker_backend import BackendResponse
from marla.runtime.device import DeviceRequest, resolve_device


class LocalTransformersBackend:
    """Loads a chat-instruct causal LM and generates via a plain user-turn prompt."""

    # A response is one JSON entry per action ID -- roughly
    # `"<action_id>": 0.85,\n` -- plus the ```json fence/braces wrapping the
    # whole object. Fixed per-call overhead for quotes/colon/score digits/
    # comma/newline/indentation around each entry (the ID itself is counted
    # separately, in real tokens, since IDs vary a lot in length: "finish"
    # vs "exploit:host-2-0:e_elasticsearch").
    _JSON_WRAPPER_TOKENS = 20
    _PER_ACTION_OVERHEAD_TOKENS = 10
    # This is an estimate, not an exact count (JSON escaping, whitespace
    # variance, and small-model formatting quirks all move the real number
    # around) -- padding by 50% costs a bit of generation time on a
    # response that would have fit anyway, but a truncated response is
    # rejected outright and the whole consultation (all max_schema_revisions
    # retries) has to be repeated, which costs far more.
    _SAFETY_MARGIN = 1.5

    def __init__(self, model_name: str, device: DeviceRequest, max_new_tokens: int = 512):
        # Imported lazily: transformers/torch model loading is heavy and only
        # needed when the local backend is actually selected (extra `local-lm`).
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The local Plan Maker backend needs the optional 'local-lm' extra "
                "(transformers, accelerate). Install it with: "
                'pip install -e ".[local-lm]"'
            ) from exc

        resolved = resolve_device(device)
        self.resolved_device = resolved.resolved
        self.requested_device = resolved.requested
        self._max_new_tokens = max_new_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
        self._model.to(self.resolved_device)
        self._model.eval()
        self._torch = torch

    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        # See the module docstring: asyncio.to_thread hangs indefinitely for
        # this exact call in this environment, so this runs synchronously.
        max_new_tokens = max(self._max_new_tokens, self._estimate_max_new_tokens(legal_action_ids))
        return self._generate_sync(prompt, max_new_tokens)

    def _estimate_max_new_tokens(self, legal_action_ids: list[str]) -> int:
        if not legal_action_ids:
            return self._max_new_tokens
        id_tokens = sum(len(self._tokenizer.encode(action_id, add_special_tokens=False)) for action_id in legal_action_ids)
        estimate = self._JSON_WRAPPER_TOKENS + id_tokens + self._PER_ACTION_OVERHEAD_TOKENS * len(legal_action_ids)
        return int(estimate * self._SAFETY_MARGIN)

    def _generate_sync(self, prompt: str, max_new_tokens: int) -> BackendResponse:
        start = time.monotonic()
        # apply_chat_template(..., return_tensors="pt") and a plain tokenizer
        # call both return a BatchEncoding (input_ids + attention_mask), not
        # a raw tensor -- unpacked as kwargs below so generate() gets the
        # attention_mask too, not just input_ids.
        if getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.resolved_device)
        else:
            # Base (non-instruct-tuned) models have no chat template configured.
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.resolved_device)

        input_length = inputs["input_ids"].shape[-1]
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        latency_ms = (time.monotonic() - start) * 1000
        input_tokens = int(input_length)
        output_tokens = int(generated.shape[-1])
        return BackendResponse(
            raw_text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
