#!/usr/bin/env python
"""Profiles representative Plan Maker consultations BEFORE any long MARLA
training run -- per the explicit instruction to check for a fixable
latency bottleneck (CPU fallback, repeated model loading, disabled KV
cache, unnecessary generation past valid JSON) rather than accepting the
pilot's ~73s mean latency as a fixed cost of the experiment.

Builds REAL representative prompts (not synthetic stand-ins) by actually
stepping a real NasimEmuAdapter on the production ID scenario with a
random policy until small/medium/large legal-action-set observations
naturally occur, then runs the real build_prompt()/retrieve_rules() path
used in production. Prints every field requested for the profiling report
and, if --test-early-stop is given, also runs a JSON-completion-aware
early-stopping variant against the same prompts to measure savings and
verify output equivalence before touching any production code.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList  # noqa: E402

from marla.config.loader import load_config  # noqa: E402
from marla.environment.nasimemu_adapter import NasimEmuAdapter  # noqa: E402
from marla.environment.observation_summary import build_observation_summary  # noqa: E402
from marla.knowledge.retriever import load_knowledge_base, retrieve_rules  # noqa: E402
from marla.messaging.schemas import AdvisoryActionDescriptor, AdvisoryObjective  # noqa: E402
from marla.models.prompt import build_prompt  # noqa: E402
from marla.models.response_parser import _find_first_balanced_object_span, coerce_scores, extract_json_object  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SCENARIO = str(REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_two_subnets.v2.yaml")


def collect_representative_observations(targets: list[int], max_steps: int = 400, seed: int = 999) -> dict[int, tuple]:
    """Steps a random policy until it finds an observation whose legal
    action count is >= each target in ``targets`` (closest match kept).
    Returns {target: (legal_actions, observation, legal_action_count)}."""
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(
        scenario=SCENARIO, max_episode_steps=100,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    rng = random.Random(seed)
    best: dict[int, tuple] = {}
    state = adapter.reset(seed=seed)
    for _ in range(max_steps):
        actions = adapter.legal_actions(state)
        count = len(actions)
        for target in targets:
            if target not in best or abs(count - target) < abs(best[target][2] - target):
                observation = build_observation_summary(state)
                best[target] = (actions, observation, count)
        action = rng.choice(actions)
        transition = adapter.step(action)
        if transition.terminated or transition.truncated:
            state = adapter.reset(seed=seed + 1)
            seed += 1
        else:
            state = transition.state
    return best


class BalancedJsonStoppingCriteria(StoppingCriteria):
    """Stops generation the moment the text generated so far contains a
    complete, balanced top-level {...} object -- reuses the EXACT same
    detection logic the response parser already uses to extract the
    answer, so "stopped early" and "stopped at max_new_tokens" are
    guaranteed to parse identically whenever both reach a valid object."""

    def __init__(self, tokenizer, prompt_length: int, check_every: int = 8):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.check_every = check_every
        self._step = 0
        self.stopped_at_token = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        self._step += 1
        if self._step % self.check_every != 0:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        generated = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        done = _find_first_balanced_object_span(text) is not None
        if done and self.stopped_at_token is None:
            self.stopped_at_token = generated.shape[-1]
        return torch.tensor([done], dtype=torch.bool, device=input_ids.device)


def profile_one(model, tokenizer, resolved_device, prompt: str, legal_action_ids: list[str], early_stop: bool) -> dict:
    from marla.models.local_backend import LocalTransformersBackend

    json_wrapper, per_action, margin = (
        LocalTransformersBackend._JSON_WRAPPER_TOKENS,
        LocalTransformersBackend._PER_ACTION_OVERHEAD_TOKENS,
        LocalTransformersBackend._SAFETY_MARGIN,
    )
    id_tokens = sum(len(tokenizer.encode(a, add_special_tokens=False)) for a in legal_action_ids)
    configured_max_new_tokens = int((json_wrapper + id_tokens + per_action * len(legal_action_ids)) * margin)
    configured_max_new_tokens = max(256, configured_max_new_tokens)  # matches assisted.yaml's floor

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(resolved_device)
    input_length = inputs.shape[-1] if not hasattr(inputs, "input_ids") else inputs["input_ids"].shape[-1]
    inputs_dict = {"input_ids": inputs} if not hasattr(inputs, "input_ids") else inputs

    stopping = None
    criteria_obj = None
    if early_stop:
        criteria_obj = BalancedJsonStoppingCriteria(tokenizer, input_length)
        stopping = StoppingCriteriaList([criteria_obj])

    start = time.monotonic()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs_dict,
            max_new_tokens=configured_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    latency_s = time.monotonic() - start

    generated = output_ids[0][input_length:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    output_tokens = int(generated.shape[-1])
    raw_scores = extract_json_object(text) or {}
    scores = coerce_scores(raw_scores)

    return {
        "legal_action_count": len(legal_action_ids),
        "prompt_tokens": int(input_length),
        "configured_max_new_tokens": configured_max_new_tokens,
        "actual_output_tokens": output_tokens,
        "hit_token_cap": output_tokens >= configured_max_new_tokens,
        "stopped_early_at_token": criteria_obj.stopped_at_token if criteria_obj else None,
        "latency_s": round(latency_s, 2),
        "tokens_per_second": round(output_tokens / latency_s, 2) if latency_s > 0 else None,
        "scores_coverage": len(scores),
        "scores_expected": len(legal_action_ids),
        "raw_text_tail": text[-200:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, nargs="+", default=[10, 40, 85])
    parser.add_argument("--test-early-stop", action="store_true")
    args = parser.parse_args()

    print(f"=== Device/model verification ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")
    from marla.runtime.device import resolve_device

    resolved = resolve_device("auto")
    model.to(resolved.resolved)
    model.eval()
    print(f"resolved device: {resolved.resolved} (requested: {resolved.requested})")
    print(f"model param dtype: {next(model.parameters()).dtype}")
    print(f"model.generation_config.use_cache: {model.generation_config.use_cache!r} (None = library default, which is caching-enabled)")
    print(f"model.generation_config.eos_token_id: {model.generation_config.eos_token_id}")
    print(f"model loaded once for this whole profiling session: yes (single model object reused across all prompts below)")
    print()

    print("=== Collecting representative real observations (small/medium/large legal-action-set) ===")
    observations = collect_representative_observations(args.targets)
    knowledge_base = load_knowledge_base("package://marla/knowledge/nasimemu_rules.yaml")
    objective = AdvisoryObjective(type="capture_target", description="Obtain the configured target access")

    for target, (actions, observation, count) in sorted(observations.items()):
        print(f"\n--- Representative observation: target={target}, actual legal_action_count={count} ---")
        advisory_actions = [
            AdvisoryActionDescriptor(action_id=a.action_id, type=a.action_type, target=a.target_key, parameters=a.parameters)
            for a in actions
        ]
        legal_action_types = {a.action_type for a in actions}
        retrieved = retrieve_rules(knowledge_base, observation, legal_action_types)
        prompt = build_prompt(retrieved, objective, observation, advisory_actions)
        legal_action_ids = [a.action_id for a in actions]

        baseline = profile_one(model, tokenizer, resolved.resolved, prompt, legal_action_ids, early_stop=False)
        print("BASELINE (current production code path):")
        for k, v in baseline.items():
            if k != "raw_text_tail":
                print(f"  {k}: {v}")

        if args.test_early_stop:
            early = profile_one(model, tokenizer, resolved.resolved, prompt, legal_action_ids, early_stop=True)
            print("EARLY-STOP (JSON-completion-aware stopping criteria):")
            for k, v in early.items():
                if k != "raw_text_tail":
                    print(f"  {k}: {v}")
            equivalent = baseline["scores_coverage"] == early["scores_coverage"] == len(legal_action_ids)
            print(f"  OUTPUT EQUIVALENT (both cover all {len(legal_action_ids)} action IDs): {equivalent}")
            if baseline["latency_s"] > 0:
                savings = 1 - early["latency_s"] / baseline["latency_s"]
                print(f"  LATENCY SAVINGS: {savings*100:.1f}%")


if __name__ == "__main__":
    main()
