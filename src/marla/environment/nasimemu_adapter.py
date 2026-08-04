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

from marla.environment.actions import ActionDescriptor, build_legal_actions, resolve_action_target
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
    ) -> None:
        self._scenario = scenario
        self._max_episode_steps = max_episode_steps
        self._completion_reward = completion_reward
        self._premature_finish_penalty = premature_finish_penalty
        self._env = NASimEmuEnv(
            scenario_name=scenario,
            emulate=False,
            step_limit=None,  # MARLA tracks truncation itself; see reset()/step().
            random_init=False,
            observation_format="matrix",
            fully_obs=False,
            augment_with_action=False,
        )

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

        raw_observation = self._env.reset()
        return self._make_state(raw_observation)

    def legal_actions(self, state: EnvironmentState) -> list[ActionDescriptor]:
        return build_legal_actions(self._env, state.host_addresses)

    def to_pyg_data(self, state: EnvironmentState) -> GraphObservation:
        return build_graph_observation(state.host_rows, state.host_addresses, state.subnet_graph)

    def objective_satisfied(self, state: EnvironmentState) -> bool:
        """All sensitive/value hosts compromised (NASimEmu's native goal)."""
        return bool(self._env.env.goal_reached())

    def step(self, action: ActionDescriptor) -> TransitionResult:
        """Advance the simulation by exactly one compound decision.

        FINISH never touches the underlying simulator (see
        :mod:`marla.environment.finish`); every other action calls
        ``NASimEmuEnv.step`` exactly once.
        """
        if action.is_finish:
            # Reward reflects whether the goal was met *before* finishing;
            # FINISH performs no NASimEmu action, so there is no new state.
            current_state = self._make_state(None)
            satisfied = self.objective_satisfied(current_state)
            reward = compute_finish_reward(
                satisfied, self._completion_reward, self._premature_finish_penalty
            )
            return TransitionResult(
                state=None,
                nasimemu_reward=reward,
                terminated=True,
                truncated=False,
                info={"finish": True, "objective_satisfied": satisfied},
            )

        target, action_list_index = resolve_action_target(self._env, action.action_id)
        raw_observation, reward, done, info = self._env.step((target, action_list_index))

        # NASimEmuEnv forces done=False for every non-terminal action when no
        # step_limit is configured (see nasimemu.env.NASimEmuEnv.step): the
        # agent must choose FINISH to end an episode. This assertion documents
        # that invariant so a future NASimEmu upgrade can't silently break it.
        assert not done, "unexpected internal termination from a non-FINISH action"

        new_state = self._make_state(raw_observation)
        truncated = new_state.step_idx >= self._max_episode_steps

        return TransitionResult(
            state=new_state,
            nasimemu_reward=float(reward),
            terminated=False,
            truncated=truncated,
            info=info,
        )

    def _make_state(self, raw_observation: np.ndarray | None) -> EnvironmentState:
        if raw_observation is None:
            raw_observation = self._env.s_raw
        host_rows = raw_observation[:-1]
        host_addresses = [tuple(int(v) for v in HostVector(row).address) for row in host_rows]
        return EnvironmentState(
            raw_observation=raw_observation,
            host_rows=host_rows,
            host_addresses=host_addresses,
            subnet_graph=set(self._env.subnet_graph),
            step_idx=self._env.step_idx,
        )
