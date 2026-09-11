"""Anti-leak semantics for the visible-only global objective-progress
feature (FINISH-learnability investigation, spec section 20):
``marla.environment.visible_facts.compute_visible_progress``, fed into
``RecurrentCore``'s ``x_t`` alongside the graph embedding.
"""

from __future__ import annotations

import pytest
from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.visible_facts import (
    VISIBLE_PROGRESS_DIM,
    VisibleHostFacts,
    compute_visible_progress,
)

TARGET_A = "host-2-0"
TARGET_B = "host-2-1"


def _facts(target_key, value, access=AccessLevel.NONE) -> VisibleHostFacts:
    return VisibleHostFacts(
        target_key=target_key, reachable=True, compromised=False, access=access,
        known_os=frozenset(), known_services=frozenset(), known_processes=frozenset(), value=value,
    )


def test_no_sensitive_target_confirmed_yet_gives_the_all_absent_vector():
    facts_by_target = {TARGET_A: _facts(TARGET_A, value=0.0)}  # scanned, but not (yet) known sensitive
    progress = compute_visible_progress(facts_by_target)
    assert progress.shape == (VISIBLE_PROGRESS_DIM,)
    has_target, fraction_with_root, any_without_root = progress.tolist()
    assert has_target == 0.0
    assert fraction_with_root == 0.0
    assert any_without_root == 0.0


def test_undiscovered_sensitive_hosts_do_not_influence_the_feature():
    """A host simply isn't in facts_by_target until NASimEmu's own
    observation includes it at all -- undiscovered hosts can never move
    this feature, by construction (nothing to iterate)."""
    empty = compute_visible_progress({})
    one_non_sensitive = compute_visible_progress({TARGET_A: _facts(TARGET_A, value=0.0)})
    assert empty.tolist() == one_non_sensitive.tolist() == [0.0, 0.0, 0.0]


def test_hidden_sensitive_target_count_is_not_exposed():
    """1 confirmed-sensitive-and-rooted target and 5 confirmed-sensitive-
    and-rooted targets must produce the IDENTICAL vector -- the feature
    never encodes *how many* sensitive targets exist, only fixed-width
    fractions/booleans over what's currently known."""
    one_rooted = compute_visible_progress({TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.ROOT)})
    five_rooted = compute_visible_progress(
        {f"host-2-{i}": _facts(f"host-2-{i}", value=100.0, access=AccessLevel.ROOT) for i in range(5)}
    )
    assert one_rooted.tolist() == five_rooted.tolist() == [1.0, 1.0, 0.0]


def test_visible_root_transition_updates_the_feature():
    before = compute_visible_progress({TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.USER)})
    after = compute_visible_progress({TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.ROOT)})
    assert before.tolist() == [1.0, 0.0, 1.0]  # sensitive, none rooted yet, something still missing root
    assert after.tolist() == [1.0, 1.0, 0.0]  # now rooted, nothing missing


def test_visible_user_access_does_not_count_as_root():
    facts_by_target = {TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.USER)}
    _has_target, fraction_with_root, any_without_root = compute_visible_progress(facts_by_target).tolist()
    assert fraction_with_root == 0.0
    assert any_without_root == 1.0


def test_no_sensitive_target_visible_is_distinguishable_from_all_known_targets_rooted():
    """These must never collapse to the same vector: "nothing confirmed
    sensitive yet" vs. "everything confirmed sensitive is already
    rooted" are very different states for FINISH-appropriateness, even
    though a naive "fraction rooted" alone (0/0 vs 1/1) could conflate
    them if has_visible_sensitive_target didn't exist."""
    nothing_sensitive = compute_visible_progress({TARGET_A: _facts(TARGET_A, value=0.0)})
    all_rooted = compute_visible_progress({TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.ROOT)})
    assert nothing_sensitive.tolist() != all_rooted.tolist()
    assert nothing_sensitive.tolist()[0] == 0.0
    assert all_rooted.tolist()[0] == 1.0


def test_mixed_rooted_and_not_rooted_gives_a_genuine_fraction():
    facts_by_target = {
        TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.ROOT),
        TARGET_B: _facts(TARGET_B, value=50.0, access=AccessLevel.USER),
    }
    has_target, fraction_with_root, any_without_root = compute_visible_progress(facts_by_target).tolist()
    assert has_target == 1.0
    assert fraction_with_root == 0.5
    assert any_without_root == 1.0


def test_plan_maker_and_ppo_agree_on_which_targets_are_confirmed_sensitive():
    """Cross-check against observation_summary.py's own sensitive-target
    accounting (built from the same VisibleHostFacts) -- both consumers'
    notion of "how many confirmed-sensitive targets, how many with root"
    must agree exactly for the identical facts_by_target."""
    facts_by_target = {
        TARGET_A: _facts(TARGET_A, value=100.0, access=AccessLevel.ROOT),
        TARGET_B: _facts(TARGET_B, value=50.0, access=AccessLevel.USER),
    }
    has_target, fraction_with_root, _any_without_root = compute_visible_progress(facts_by_target).tolist()

    # observation_summary.py computes sensitive_hosts_total/
    # sensitive_hosts_with_root_access from is_sensitive_target/.access on
    # this exact same VisibleHostFacts dict -- recomputed here directly
    # (rather than constructing a real EnvironmentState) to check the two
    # consumers' notion of "confirmed sensitive"/"rooted" agree exactly.
    sensitive_total = sum(1 for f in facts_by_target.values() if f.is_sensitive_target)
    sensitive_with_root = sum(
        1 for f in facts_by_target.values() if f.is_sensitive_target and f.access == AccessLevel.ROOT
    )
    assert has_target == (1.0 if sensitive_total > 0 else 0.0)
    assert fraction_with_root == pytest.approx(sensitive_with_root / sensitive_total)
