from marla.models.response_parser import coerce_scores, extract_json_object


def test_extract_plain_json():
    result = extract_json_object('{"finish": 0.5, "service-scan:host-1-0": 0.8}')
    assert result == {"finish": 0.5, "service-scan:host-1-0": 0.8}


def test_extract_json_from_markdown_fence():
    text = 'Here is the answer:\n```json\n{"finish": 0.5}\n```\nHope that helps.'
    assert extract_json_object(text) == {"finish": 0.5}


def test_extract_json_from_plain_fence_without_json_tag():
    text = "```\n{\"finish\": 0.1}\n```"
    assert extract_json_object(text) == {"finish": 0.1}


def test_extract_json_surrounded_by_prose():
    text = 'Sure, here are the scores: {"finish": 0.3} -- let me know if you need more.'
    assert extract_json_object(text) == {"finish": 0.3}


def test_extract_returns_none_for_garbage():
    assert extract_json_object("I cannot comply with this request.") is None


def test_extract_returns_none_for_invalid_json_braces():
    assert extract_json_object("{not valid json at all}") is None


def test_extract_returns_none_for_json_array_not_object():
    assert extract_json_object("[1, 2, 3]") is None


def test_coerce_scores_passes_through_valid_floats():
    assert coerce_scores({"a": 0.5, "b": 1}) == {"a": 0.5, "b": 1.0}


def test_coerce_scores_parses_numeric_strings():
    assert coerce_scores({"a": "0.7"}) == {"a": 0.7}


def test_coerce_scores_drops_unparseable_values():
    assert coerce_scores({"a": "not a number", "b": 0.5}) == {"b": 0.5}


def test_coerce_scores_drops_boolean_values():
    assert coerce_scores({"a": True, "b": 0.5}) == {"b": 0.5}


def test_coerce_scores_drops_non_string_keys():
    assert coerce_scores({1: 0.5, "b": 0.5}) == {"b": 0.5}
