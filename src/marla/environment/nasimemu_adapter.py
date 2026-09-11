"""The single point of contact between MARLA and NASimEmu.

Per spec section 3.4/3.1, the RL Orchestrator is the only component that
interacts directly with NASimEmu, and it does so exclusively through this
adapter. Nothing outside :mod:`marla.environment` should import
``nasimemu`` directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.actions import ActionDescriptor, build_legal_actions, parse_host_target_key, resolve_action_target
from marla.environment.finish import compute_finish_reward
from marla.environment.graph import GraphObservation, build_graph_observation


@dataclass(frozen=True)
class EnvironmentState:
    """Everything the RL Orchestrator needs from the current visible observation."""

    raw_observation: np.ndarray
    host_rows: np.ndarray
    host_addresses: list[tuple[int, int]]
    subnet_graph: set[tuple[int, int]] = field(default_factory=set)
    step_idx: int = 0
    # MARLA-owned episode state (spec: NASimEmu's subnet_graph alone can't
    # distinguish "known subnet never scanned" from "known subnet scanned
    # successfully but discovered nothing new" -- a successful SubnetScan
    # that finds no new subnet leaves subnet_graph unchanged). Populated by
    # NasimEmuAdapter.step()/reset(), never derived from subnet_graph. See
    # marla.environment.visible_facts.VisibleNetworkExploration for the
    # visible-only summary built from this.
    successfully_scanned_subnets: frozenset[int] = frozenset()


@dataclass(frozen=True)
class TransitionResult:
    """Result of a single compound-decision environment advance."""

    state: EnvironmentState | None
    nasimemu_reward: float
    terminated: bool
    truncated: bool
    info: dict


class NasimEmuAdapter:
    """Adapter isolating NASimEmu-specific behavior (spec section 3.4)."""

    def __init__(
        self,
        scenario: str,
        max_episode_steps: int,
        completion_reward: float,
        premature_finish_penalty: float,
        premature_finish_penalty_per_remaining_target: float = 0.0,
    ) -> None:
        self._scenario = scenario
        self._max_episode_steps = max_episode_steps
        self._completion_reward = completion_reward
        self._premature_finish_penalty = premature_finish_penalty
        self._premature_finish_penalty_per_remaining_target = premature_finish_penalty_per_remaining_target
        self._env = NASimEmuEnv(
            scenario_name=scenario,
            emulate=False,
            step_limit=None,  # MARLA tracks truncation itself; see reset()/step().
            random_init=False,
            observation_format="matrix",
            fully_obs=False,
            augment_with_action=False,
        )
        # MARLA-owned, reset every episode (see EnvironmentState.successfully_scanned_subnets).
        self._successfully_scanned_subnets: set[int] = set()

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def completion_reward(self) -> float:
        return self._completion_reward

    @property
    def premature_finish_penalty(self) -> float:
        return self._premature_finish_penalty

    @property
    def premature_finish_penalty_per_remaining_target(self) -> float:
        return self._premature_finish_penalty_per_remaining_target

    def reset(self, seed: int | None = None) -> EnvironmentState:
        """Start a new episode, generating a fresh scenario instance.

        NASimEmu's own ``_generate_env()`` reseeds ``random``/``numpy``
        internally only at env construction time; to get reproducible
        per-episode scenario generation we (re-)seed immediately before
        calling ``reset()``, which is what actually regenerates the scenario.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._successfully_scanned_subnets = set()
        raw_observation = self._env.reset()
        return self._make_state(raw_observation)

    def legal_actions(self, state: EnvironmentState) -> list[ActionDescriptor]:
        return build_legal_actions(self._env, state.host_addresses)

    def to_pyg_data(self, state: EnvironmentState, include_subnet_scan_feature: bool) -> GraphObservation:
        """``include_subnet_scan_feature``: whether the per-subnet-node
        ``subnet_scan_completed`` feature (spec: observable subnet-
        exploration progress) carries its real value or is omitted
        (always 0) -- required, not defaulted, so every call site
        consciously matches the policy's own
        ``visible_subnet_exploration_enabled`` flag. This is what keeps
        the v2/v3-target/v3-full ablation valid: a "v3-target" policy
        must never receive the exploration signal indirectly through
        GraphSAGE either.
        """
        scanned = state.successfully_scanned_subnets if include_subnet_scan_feature else frozenset()
        return build_graph_observation(state.host_rows, state.host_addresses, state.subnet_graph, scanned)

    def objective_satisfied(self) -> bool:
        """All sensitive/value hosts compromised (NASimEmu's native goal).

        A direct query of the underlying environment's own current state,
        not derived from any particular :class:`EnvironmentState` snapshot
        -- there is nothing to pass in.
        """
        return bool(self._env.env.goal_reached())

    def sensitive_target_status(self) -> tuple[int, int]:
        """``(total sensitive/value targets, targets still missing ROOT access)``.

        A direct query of the underlying environment's current state (same
        source as :meth:`objective_satisfied`/``goal_reached()``), not
        derived from any stored :class:`EnvironmentState` snapshot. USER
        access does not count as satisfying a target -- only ROOT does,
        matching NASimEmu's own
        ``Network.all_sensitive_hosts_compromised()`` semantics. Cheap
        (iterates only the scenario's sensitive addresses, not the full
        host set); nothing here retains or persists simulator state.
        """
        network = self._env.env.network
        state = self._env.env.current_state
        addresses = network.sensitive_addresses
        remaining = sum(1 for addr in addresses if not state.host_has_access(addr, AccessLevel.ROOT))
        return len(addresses), remaining

    def step(self, action: ActionDescriptor) -> TransitionResult:
        """Advance the simulation by exactly one compound decision.

        FINISH never touches the underlying simulator (see
        :mod:`marla.environment.finish`); every other action calls
        ``NASimEmuEnv.step`` exactly once.
        """
        if action.is_finish:
            # Reward reflects whether the goal was met *before* finishing;
            # FINISH performs no NASimEmu action, so there is no new state.
            satisfied = self.objective_satisfied()
            total_targets, remaining_targets = self.sensitive_target_status()
            reward = compute_finish_reward(
                satisfied,
                self._completion_reward,
                self._premature_finish_penalty,
                self._premature_finish_penalty_per_remaining_target,
                remaining_targets,
            )
            return TransitionResult(
                state=None,
                nasimemu_reward=reward,
                terminated=True,
                truncated=False,
                info={
                    "finish": True,
                    "objective_satisfied": satisfied,
                    "sensitive_targets_total": total_targets,
                    "sensitive_targets_remaining": remaining_targets,
                },
            )

        target, action_list_index = resolve_action_target(self._env, action.action_id)
        try:
            raw_observation, reward, done, info = self._env.step((target, action_list_index))
        except AssertionError:
            return self._absorb_invalid_action(action, action_list_index)

        # NASimEmuEnv forces done=False for every non-terminal action when no
        # step_limit is configured (see nasimemu.env.NASimEmuEnv.step): the
        # agent must choose FINISH to end an episode. This assertion documents
        # that invariant so a future NASimEmu upgrade can't silently break it.
        assert not done, "unexpected internal termination from a non-FINISH action"

        # Observable subnet-exploration tracking (spec: "attempted scan !=
        # successful scan"; a successful scan that discovers zero new
        # subnets must still count). info["success"] is NASimEmu's own
        # true per-action success signal (nasim.envs.action.ActionResult
        # .info()) -- never inferred from whether subnet_graph changed,
        # which a successful-but-nothing-new scan would leave unchanged.
        # SubnetScan targets a host but the result belongs to that host's
        # *subnet* (spec section 14) -- never the newly-discovered one.
        if action.action_type == "subnet_scan" and info.get("success"):
            assert action.target_key is not None
            origin_subnet, _origin_host = parse_host_target_key(action.target_key)
            self._successfully_scanned_subnets.add(origin_subnet)

        new_state = self._make_state(raw_observation)
        truncated = new_state.step_idx >= self._max_episode_steps

        return TransitionResult(
            state=new_state,
            nasimemu_reward=float(reward),
            terminated=False,
            truncated=truncated,
            info=info,
        )

    def _absorb_invalid_action(self, action: ActionDescriptor, action_list_index: int) -> TransitionResult:
        """Treat a NASimEmu action-space precondition failure as a normal failed attempt.

        MARLA's legal action set deliberately offers every scenario-wide
        exploit/privesc against every visible host and lets the environment
        charge a failure cost for an infeasible one (see
        ``environment/actions.py``'s module docstring) -- that is
        ``NASimEmuEnv.step()``'s normal behavior for e.g. a missing
        required service. But even a ``.yaml`` *static* scenario file can
        still assign each episode's hosts a randomized OS/service
        arrangement under the hood (see ``reset()``'s docstring), so a
        scenario-wide exploit/privesc can turn out not to be a member of
        *this episode's* precomputed action space for a specific host.
        NASimEmu enforces that as a hard precondition
        (``assert a in self.env.action_space.actions`` in
        ``nasimemu.env.NASimEmuEnv._translate_action``) rather than the
        graceful "attempt failed, pay the cost" its own ``step()`` applies
        to every other kind of infeasible attempt -- observed in practice
        after ~11,000 episodes of a real training run. Absorbing it here
        (same cost-only reward, no state change, no crash) keeps that one
        rare, state-dependent combination from taking down an otherwise
        healthy multi-hour run, consistent with the rest of the legal
        action set's "attempt anything, pay for failure" design.
        """
        _, action_params = self._env.action_list[action_list_index]
        cost = action_params.get("cost", 1.0)
        self._env.step_idx += 1
        new_state = self._make_state(self._env.s_raw)
        truncated = new_state.step_idx >= self._max_episode_steps
        return TransitionResult(
            state=new_state,
            nasimemu_reward=-cost / 10.0,  # matches NASimEmuEnv's own reward scaling (r /= 10)
            terminated=False,
            truncated=truncated,
            info={"invalid_action": True, "action_id": action.action_id},
        )

    def _make_state(self, raw_observation: np.ndarray) -> EnvironmentState:
        host_rows = raw_observation[:-1]
        host_addresses = [tuple(int(v) for v in HostVector(row).address) for row in host_rows]
        return EnvironmentState(
            raw_observation=raw_observation,
            host_rows=host_rows,
            host_addresses=host_addresses,
            subnet_graph=set(self._env.subnet_graph),
            step_idx=self._env.step_idx,
            successfully_scanned_subnets=frozenset(self._successfully_scanned_subnets),
        )
