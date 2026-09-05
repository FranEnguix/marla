"""In-process ``consult_fn`` for evaluation: the same Plan Maker logic
production uses, called directly instead of over SPADE/XMPP.

No Gatekeeper or Plan Maker *agent* process is started for evaluation --
this module composes the same pure functions those agents call
(``retrieve_rules``, ``build_prompt``/``build_correction_prompt``, the
configured ``PlanMakerBackend.generate``, ``response_parser``,
``validate_advisory_response_body``) directly, in the same order, with the
same correction-retry loop as ``agents/gatekeeper.py``. This is safe to
reuse verbatim because all of those functions are already SPADE-free and
side-effect-free (see ``research/aamas2027/AUDIT.md`` section 5) -- nothing
here reimplements Plan Maker logic, it only removes the message-passing
layer around it.
"""

from __future__ import annotations

import time
from typing import Any

from marla.environment.actions import ActionDescriptor
from marla.evaluation.advisory_cache import AdvisoryCache, CachedAdvisoryResponse, compute_cache_key
from marla.knowledge.retriever import KnowledgeBase, retrieve_rules
from marla.learning.rollout import ConsultationResult
from marla.messaging.advisory_validation import AdvisoryValidationError, validate_advisory_response_body
from marla.messaging.builders import new_id
from marla.messaging.schemas import MESSAGE_SCHEMA_VERSION, AdvisoryActionDescriptor, AdvisoryObjective
from marla.models.plan_maker_backend import BackendResponse, PlanMakerBackend
from marla.models.prompt import PROMPT_VERSION, build_correction_prompt, build_prompt
from marla.models.response_parser import coerce_scores, extract_json_object


class DirectConsultant:
    """Callable matching ``learning.rollout.ConsultFn``'s signature."""

    def __init__(
        self,
        backend: PlanMakerBackend,
        knowledge_base: KnowledgeBase,
        model_version: str,
        objective: AdvisoryObjective,
        run_id: str,
        max_schema_revisions: int,
        cache: AdvisoryCache | None = None,
    ):
        self._backend = backend
        self._knowledge_base = knowledge_base
        self._model_version = model_version
        self._objective = objective
        self._run_id = run_id
        self._max_schema_revisions = max_schema_revisions
        self._cache = cache
        # Counters exposed for VALIDATION.md / manifest bookkeeping: how much
        # the cache actually saved, not just that it exists.
        self.cache_hits = 0
        self.cache_misses = 0

    async def __call__(
        self,
        legal_actions: list[ActionDescriptor],
        episode_id: int,
        step: int,
        source_observation_id: str,
        observation: dict[str, Any],
    ) -> ConsultationResult:
        legal_action_ids = [a.action_id for a in legal_actions]
        cache_key = compute_cache_key(
            observation=observation,
            legal_action_ids=legal_action_ids,
            model_name=self._model_version,
            model_revision=self._model_version,  # see AUDIT.md: no separate HF revision is pinned
            prompt_version=PROMPT_VERSION,
            knowledge_version=self._knowledge_base.version,
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return ConsultationResult(
                    status=cached.status,
                    scores=cached.scores,
                    request_id=new_id("cached-request"),
                    latency_ms=cached.latency_ms,
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                    total_tokens=cached.total_tokens,
                )
        self.cache_misses += 1

        advisory_actions = [
            AdvisoryActionDescriptor(action_id=a.action_id, type=a.action_type, target=a.target_key, parameters=a.parameters)
            for a in legal_actions
        ]
        legal_action_type_set = {a.action_type for a in legal_actions}
        retrieved = retrieve_rules(self._knowledge_base, observation, legal_action_type_set)
        prompt = build_prompt(retrieved, self._objective, observation, advisory_actions)
        expected_action_ids = set(legal_action_ids)

        request_id = new_id("request")
        result = await self._run_correction_loop(prompt, legal_action_ids, expected_action_ids, request_id)

        if self._cache is not None:
            self._cache.put(
                cache_key,
                CachedAdvisoryResponse(
                    status=result.status,
                    scores=result.scores,
                    latency_ms=result.latency_ms or 0.0,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_tokens=result.total_tokens,
                ),
            )
        return result

    async def _run_correction_loop(
        self, prompt: str, legal_action_ids: list[str], expected_action_ids: set[str], request_id: str
    ) -> ConsultationResult:
        """Mirrors agents/gatekeeper.py: 1 initial attempt + up to
        max_schema_revisions corrections before reporting schema_rejected."""
        current_prompt = prompt
        total_latency_ms = 0.0

        for _attempt in range(self._max_schema_revisions + 1):
            start = time.monotonic()
            try:
                response: BackendResponse = await self._backend.generate(current_prompt, legal_action_ids)
            except Exception:
                response = BackendResponse(raw_text="", latency_ms=0.0)
            total_latency_ms += response.latency_ms or (time.monotonic() - start) * 1000

            raw_scores = extract_json_object(response.raw_text) or {}
            scores = coerce_scores(raw_scores)
            body = {
                "schema_version": MESSAGE_SCHEMA_VERSION,
                "run_id": self._run_id,
                "request_id": request_id,
                "scores": scores,
                "model_version": self._model_version,
                "prompt_version": PROMPT_VERSION,
                "knowledge_version": self._knowledge_base.version,
                "inference_latency_ms": response.latency_ms,
                "retrieved_rule_ids": [],
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
            }
            try:
                validated = validate_advisory_response_body(
                    body, self._run_id, request_id, expected_action_ids
                )
                return ConsultationResult(
                    status="accepted",
                    scores=validated.scores,
                    request_id=request_id,
                    latency_ms=total_latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                )
            except AdvisoryValidationError as exc:
                current_prompt = build_correction_prompt(prompt, exc.reason, exc.detail, legal_action_ids)

        return ConsultationResult(
            status="schema_rejected", scores=None, request_id=request_id, latency_ms=total_latency_ms
        )

