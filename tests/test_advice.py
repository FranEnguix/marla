import torch

from marla.learning.advice import (
    AdviceScale,
    TrustHead,
    clip_and_logit,
    compute_advice_summary,
    compute_agreement_features,
    normalize_advice,
    summary_to_tensor,
)


def test_clip_and_logit_bounds_extreme_confidences():
    confidence = torch.tensor([0.0, 1.0, 0.5])
    log_odds = clip_and_logit(confidence)
    assert torch.isfinite(log_odds).all()
    assert log_odds[2].item() == 0.0  # logit(0.5) == 0


def test_normalize_advice_zero_mean_unit_ish_variance():
    log_odds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normalized = normalize_advice(log_odds)
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-5)


def test_normalize_advice_constant_vector_is_all_zero():
    log_odds = torch.tensor([2.0, 2.0, 2.0])
    normalized = normalize_advice(log_odds)
    assert torch.equal(normalized, torch.zeros_like(log_odds))


def test_advice_summary_shape_and_finiteness():
    log_odds = torch.tensor([1.0, -1.0, 0.5, 2.0])
    summary = compute_advice_summary(log_odds)
    tensor = summary_to_tensor(summary)
    assert tensor.shape == (5,)
    assert torch.isfinite(tensor).all()


def test_advice_summary_single_action_top1_minus_top2_is_zero():
    log_odds = torch.tensor([1.0])
    summary = compute_advice_summary(log_odds)
    assert summary.top1_minus_top2.item() == 0.0


def test_agreement_features_agree_when_top_actions_match():
    base_logits = torch.tensor([1.0, 5.0, 2.0])
    advice_log_odds = torch.tensor([0.0, 3.0, 1.0])  # same argmax (index 1)
    features = compute_agreement_features(base_logits, advice_log_odds)
    assert features[0].item() == 1.0


def test_agreement_features_disagree_when_top_actions_differ():
    base_logits = torch.tensor([5.0, 1.0, 2.0])
    advice_log_odds = torch.tensor([0.0, 3.0, 1.0])
    features = compute_agreement_features(base_logits, advice_log_odds)
    assert features[0].item() == 0.0


def test_agreement_correlation_zero_when_undefined():
    base_logits = torch.tensor([2.0, 2.0, 2.0])  # zero variance
    advice_log_odds = torch.tensor([1.0, 5.0, 3.0])
    features = compute_agreement_features(base_logits, advice_log_odds)
    assert features[1].item() == 0.0


def test_agreement_correlation_perfectly_positive():
    base_logits = torch.tensor([1.0, 2.0, 3.0])
    advice_log_odds = torch.tensor([10.0, 20.0, 30.0])  # perfectly correlated
    features = compute_agreement_features(base_logits, advice_log_odds)
    assert torch.isclose(features[1], torch.tensor(1.0), atol=1e-4)


def test_trust_head_output_bounded_in_unit_interval():
    head = TrustHead(recurrent_hidden_size=8)
    z = torch.randn(8)
    summary = torch.randn(5)
    agreement = torch.tensor([1.0, 0.5])
    beta = head(z, summary, agreement)
    assert 0.0 <= beta.item() <= 1.0


def test_advice_scale_is_always_positive():
    scale = AdviceScale()
    with torch.no_grad():
        scale.alpha_bar.fill_(-100.0)  # softplus(-100) ~ 0 but > 0
    alpha = scale()
    assert alpha.item() > 0.0
    assert torch.isfinite(alpha)

    with torch.no_grad():
        scale.alpha_bar.fill_(100.0)
    alpha_large = scale()
    assert alpha_large.item() > 0.0
