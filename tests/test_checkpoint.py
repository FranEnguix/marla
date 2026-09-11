from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config
from marla.learning.action_encoder import POLICY_REPRESENTATION_VERSION
from marla.learning.checkpoint import PolicyRepresentationMismatchError, load_checkpoint, save_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_save_and_reload_round_trips_parameters(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # perturb parameters so a naive "load did nothing" bug would be caught
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = {k: v.clone() for k, v in policy.state_dict().items()}

    checkpoint_path = tmp_path / "checkpoints" / "update-1.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=1, environment_steps=128, config_hash="abc123")

    fresh_policy = RecurrentPolicy(config.policy)  # different random init
    fresh_optimizer = torch.optim.Adam(fresh_policy.parameters(), lr=1e-3)
    metadata = load_checkpoint(checkpoint_path, fresh_policy, fresh_optimizer)

    for key, value in before.items():
        assert torch.equal(value, fresh_policy.state_dict()[key])

    assert metadata.update_count == 1
    assert metadata.environment_steps == 128
    assert metadata.config_hash == "abc123"


def test_load_without_optimizer_still_restores_policy(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "update-1.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=5, environment_steps=64, config_hash="xyz")

    fresh_policy = RecurrentPolicy(config.policy)
    metadata = load_checkpoint(checkpoint_path, fresh_policy, optimizer=None)
    assert metadata.update_count == 5
    for key, value in policy.state_dict().items():
        assert torch.equal(value, fresh_policy.state_dict()[key])


def test_save_checkpoint_records_current_policy_representation_version(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "update-1.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=1, environment_steps=1, config_hash="h")

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert data["policy_representation_version"] == POLICY_REPRESENTATION_VERSION


def test_load_checkpoint_rejects_mismatched_representation_version_loudly(tmp_path):
    """Spec sections 44-45: never silently partially load an incompatible
    checkpoint -- raise a clear, MARLA-specific error before ever calling
    load_state_dict, naming both versions.
    """
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "old.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=1, environment_steps=1, config_hash="h")

    # Simulate a pre-versioning (v1) checkpoint by stripping the field, and
    # a checkpoint from some future v3 by overwriting it -- both must fail
    # the same way, loudly, before any state_dict mutation.
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for bad_version in (None, POLICY_REPRESENTATION_VERSION + 1):
        data["policy_representation_version"] = bad_version
        bad_path = tmp_path / f"bad_{bad_version}.pt"
        torch.save(data, bad_path)

        fresh_policy = RecurrentPolicy(config.policy)
        before = {k: v.clone() for k, v in fresh_policy.state_dict().items()}
        with pytest.raises(PolicyRepresentationMismatchError, match=str(POLICY_REPRESENTATION_VERSION)):
            load_checkpoint(bad_path, fresh_policy)
        # Never partially loaded: state_dict must be byte-identical to
        # before the failed load attempt.
        for key, value in before.items():
            assert torch.equal(value, fresh_policy.state_dict()[key])


def test_peek_checkpoint_metadata_reports_representation_version(tmp_path):
    from marla.learning.checkpoint import peek_checkpoint_metadata

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "update-1.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=1, environment_steps=1, config_hash="h")

    metadata = peek_checkpoint_metadata(checkpoint_path)
    assert metadata.policy_representation_version == POLICY_REPRESENTATION_VERSION


def test_critic_refinement_epochs_round_trips_through_checkpoint_metadata(tmp_path):
    """Spec section 53: a checkpoint must record whether its run used
    critic-only refinement, and loading it back (via either
    ``load_checkpoint`` or ``peek_checkpoint_metadata``) must report the
    exact value saved -- not affect ``policy_representation_version``
    (architecture is unchanged either way -- see section 53's own
    "does not necessarily need to affect" -- confirmed here: a mismatched
    ``critic_refinement_epochs`` between the saved checkpoint and the
    loading policy must NOT raise ``PolicyRepresentationMismatchError``,
    unlike a mismatched representation/ablation flag).
    """
    from marla.learning.checkpoint import peek_checkpoint_metadata

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "update-1.pt"
    save_checkpoint(
        checkpoint_path, policy, optimizer, update_count=1, environment_steps=1, config_hash="h",
        critic_refinement_epochs=2,
    )

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert data["critic_refinement_epochs"] == 2

    peeked = peek_checkpoint_metadata(checkpoint_path)
    assert peeked.critic_refinement_epochs == 2

    fresh_policy = RecurrentPolicy(config.policy)  # a policy that "trained" with critic_refinement_epochs=0
    loaded = load_checkpoint(checkpoint_path, fresh_policy)  # must NOT raise despite the mismatch
    assert loaded.critic_refinement_epochs == 2

    # A checkpoint saved before this field existed must still load, with
    # critic_refinement_epochs reported as None (unknown), never a crash.
    del data["critic_refinement_epochs"]
    legacy_path = tmp_path / "legacy.pt"
    torch.save(data, legacy_path)
    legacy_metadata = load_checkpoint(legacy_path, RecurrentPolicy(config.policy))
    assert legacy_metadata.critic_refinement_epochs is None


def test_num_envs_and_steps_per_env_round_trip_through_checkpoint_metadata(tmp_path):
    """A checkpoint must record the num_envs/steps_per_env its run
    collected under (same status as critic_refinement_epochs above -- a
    training-run hyperparameter, not a policy-architecture property, so a
    mismatch must NOT raise ``PolicyRepresentationMismatchError`` and must
    NOT affect ``policy_representation_version``).
    """
    from marla.learning.checkpoint import peek_checkpoint_metadata

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "update-1.pt"
    save_checkpoint(
        checkpoint_path, policy, optimizer, update_count=1, environment_steps=1, config_hash="h",
        num_envs=4, steps_per_env=512,
    )

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert data["num_envs"] == 4
    assert data["steps_per_env"] == 512

    peeked = peek_checkpoint_metadata(checkpoint_path)
    assert peeked.num_envs == 4
    assert peeked.steps_per_env == 512

    fresh_policy = RecurrentPolicy(config.policy)  # a policy that "trained" with num_envs=1
    loaded = load_checkpoint(checkpoint_path, fresh_policy)  # must NOT raise despite the mismatch
    assert loaded.num_envs == 4
    assert loaded.steps_per_env == 512

    # A checkpoint saved before these fields existed must still load, with
    # num_envs/steps_per_env reported as None (unknown), never a crash.
    del data["num_envs"]
    del data["steps_per_env"]
    legacy_path = tmp_path / "legacy-num-envs.pt"
    torch.save(data, legacy_path)
    legacy_metadata = load_checkpoint(legacy_path, RecurrentPolicy(config.policy))
    assert legacy_metadata.num_envs is None
    assert legacy_metadata.steps_per_env is None
