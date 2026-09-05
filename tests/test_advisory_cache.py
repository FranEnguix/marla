from marla.evaluation.advisory_cache import AdvisoryCache, CachedAdvisoryResponse, compute_cache_key


def test_cache_key_is_deterministic_given_identical_inputs():
    key_a = compute_cache_key(
        observation={"hosts": [{"target": "1-2"}]},
        legal_action_ids=["finish", "scan:1-2"],
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="Qwen/Qwen2.5-1.5B-Instruct",
        prompt_version="nasimemu-plan-maker-v1",
        knowledge_version="nasimemu-rules-v1",
    )
    key_b = compute_cache_key(
        observation={"hosts": [{"target": "1-2"}]},
        legal_action_ids=["scan:1-2", "finish"],  # different order, same set
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="Qwen/Qwen2.5-1.5B-Instruct",
        prompt_version="nasimemu-plan-maker-v1",
        knowledge_version="nasimemu-rules-v1",
    )
    assert key_a == key_b  # legal_action_ids order must not matter (sorted internally)


def test_cache_key_changes_with_observation_content():
    base_kwargs = dict(
        legal_action_ids=["finish"],
        model_name="m",
        model_revision="m",
        prompt_version="v1",
        knowledge_version="k1",
    )
    key_a = compute_cache_key(observation={"hosts": []}, **base_kwargs)
    key_b = compute_cache_key(observation={"hosts": [{"target": "1-2"}]}, **base_kwargs)
    assert key_a != key_b


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "advisory_cache.jsonl"
    cache = AdvisoryCache(path)
    response = CachedAdvisoryResponse(
        status="accepted", scores={"finish": 0.9}, latency_ms=123.0,
        input_tokens=10, output_tokens=5, total_tokens=15,
    )
    cache.put("key-1", response)

    reloaded = AdvisoryCache(path)
    assert reloaded.get("key-1") == response
    assert len(reloaded) == 1


def test_put_is_idempotent_and_does_not_duplicate_on_disk(tmp_path):
    path = tmp_path / "advisory_cache.jsonl"
    cache = AdvisoryCache(path)
    response = CachedAdvisoryResponse(
        status="accepted", scores={"finish": 0.9}, latency_ms=1.0,
        input_tokens=1, output_tokens=1, total_tokens=2,
    )
    cache.put("key-1", response)
    cache.put("key-1", response)  # second put with the same key: no-op

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_get_missing_key_returns_none(tmp_path):
    cache = AdvisoryCache(tmp_path / "advisory_cache.jsonl")
    assert cache.get("nonexistent") is None
