from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, TypeVar
from uuid import uuid4

import numpy as np

try:
    from openenv_core.client_types import StepResult
    from openenv_core.env_server import Environment
except ImportError:
    try:
        from openenv.core.client_types import StepResult
        from openenv.core.env_server.interfaces import Environment
    except ImportError:
        ObsT = TypeVar("ObsT")

        @dataclass
        class StepResult(Generic[ObsT]):
            observation: ObsT
            reward: Optional[float] = None
            done: bool = False

        ActT = TypeVar("ActT")
        StateT = TypeVar("StateT")

        class Environment(Generic[ActT, ObsT, StateT]):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

try:
    from .physics_engine import DisasterPhysicsEngine, PhysicsConfig
except ImportError:
    from physics_engine import DisasterPhysicsEngine, PhysicsConfig


@dataclass
class DisasterAction:
    """Action chosen by the dispatcher for this timestep."""

    interaction: Literal["dispatch", "suppress", "wait"]
    row: int
    col: int


@dataclass
class DisasterObservation:
    """Partially-observed telemetry grid for the current environment step."""

    masked_telemetry: list[list[int]]
    visible_signals: int
    critical_health_threshold: int
    critical_signals: list[tuple[int, int, int]]


@dataclass
class DisasterState:
    """Episode metadata tracked by the environment."""

    episode_id: str
    step_count: int
    total_victims_saved: int
    total_victims_lost: int
    active_victims: int


class DisasterEnvironment(Environment[DisasterAction, DisasterObservation, DisasterState]):
    """Disaster simulation POMDP environment using OpenEnv's 3-method interface."""

    SUPPORTS_CONCURRENT_SESSIONS = False
    _UNKNOWN_CELL = -1

    def __init__(
        self,
        grid_size: int = 10,
        base_spread_prob: float = 0.1,
        base_decay: int = 2,
        hazard_multiplier: int = 2,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            pass

        if grid_size < 3:
            raise ValueError("grid_size must be >= 3")
        if not (0.0 <= base_spread_prob <= 1.0):
            raise ValueError("base_spread_prob must be between 0 and 1")

        self.grid_size = int(grid_size)
        self.critical_health_threshold = 35
        self._max_steps = self.grid_size * 6
        self._initial_victim_count = 0

        self._physics_config = PhysicsConfig(
            grid_size=self.grid_size,
            base_spread_prob=float(base_spread_prob),
            base_decay=int(base_decay),
            hazard_multiplier=int(hazard_multiplier),
        )
        self._physics = DisasterPhysicsEngine(config=self._physics_config)

        self._hazard_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._victim_health = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self._active_victims = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        self._state = DisasterState(
            episode_id=str(uuid4()),
            step_count=0,
            total_victims_saved=0,
            total_victims_lost=0,
            active_victims=0,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DisasterObservation:
        del kwargs
        self._physics.set_seed(seed)

        self._hazard_grid.fill(0.0)
        self._victim_health.fill(0)
        self._active_victims.fill(False)

        epicenter_row, epicenter_col = self._physics.sample_epicenter()
        self._physics.initialize_hazard_cluster(self._hazard_grid, epicenter_row, epicenter_col)
        self._physics.scatter_victims(
            self._victim_health,
            self._active_victims,
            epicenter_row,
            epicenter_col,
        )

        self._state = DisasterState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            total_victims_saved=0,
            total_victims_lost=0,
            active_victims=int(self._active_victims.sum()),
        )
        self._initial_victim_count = self._state.active_victims

        return self._build_observation()

    def step(
        self,
        action: DisasterAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StepResult[DisasterObservation]:
        del timeout_s, kwargs

        if not isinstance(action, DisasterAction):
            raise TypeError(f"Expected DisasterAction, got {type(action)}")

        self._state.step_count += 1

        reward = self._apply_action(action)
        self._physics.spread_hazards(self._hazard_grid)
        newly_lost = self._physics.apply_health_decay(
            self._hazard_grid,
            self._victim_health,
            self._active_victims,
        )

        if newly_lost:
            self._state.total_victims_lost += newly_lost
            reward -= 8.0 * newly_lost

        active = int(self._active_victims.sum())
        self._state.active_victims = active
        reward -= 0.05 * active

        done = active == 0 or self._state.step_count >= self._max_steps
        if done and self._state.total_victims_saved == self._initial_victim_count:
            reward += 15.0

        observation = self._build_observation()
        return StepResult(observation=observation, reward=float(reward), done=done)

    @property
    def state(self) -> DisasterState:
        return self._state

    def _apply_action(self, action: DisasterAction) -> float:
        row = int(np.clip(action.row, 0, self.grid_size - 1))
        col = int(np.clip(action.col, 0, self.grid_size - 1))

        reward = 0.0

        if action.interaction == "dispatch":
            if self._active_victims[row, col]:
                rescued_health = float(self._victim_health[row, col])
                self._active_victims[row, col] = False
                self._victim_health[row, col] = 0
                self._state.total_victims_saved += 1
                reward += 20.0 + rescued_health / 10.0
            else:
                reward -= 1.0

        elif action.interaction == "suppress":
            row_start = max(0, row - 1)
            row_stop = min(self.grid_size, row + 2)
            col_start = max(0, col - 1)
            col_stop = min(self.grid_size, col + 2)
            self._hazard_grid[row_start:row_stop, col_start:col_stop] *= 0.35
            reward += 0.5

        else:
            reward -= 0.1

        return reward

    def _build_observation(self) -> DisasterObservation:
        telemetry = np.full(
            (self.grid_size, self.grid_size),
            self._UNKNOWN_CELL,
            dtype=np.int16,
        )

        critical_mask = self._active_victims & (
            self._victim_health <= self.critical_health_threshold
        )
        telemetry[critical_mask] = self._victim_health[critical_mask]

        coordinates = np.argwhere(critical_mask)
        critical_signals = [
            (int(row), int(col), int(self._victim_health[row, col]))
            for row, col in coordinates
        ]

        return DisasterObservation(
            masked_telemetry=telemetry.astype(int).tolist(),
            visible_signals=len(critical_signals),
            critical_health_threshold=self.critical_health_threshold,
            critical_signals=critical_signals,
        )


if __name__ == "__main__":
    env = DisasterEnvironment(grid_size=10)
    obs = env.reset(seed=7)
    print("Initial observation:")
    print(obs)

    rng = np.random.default_rng(99)
    interactions = ["dispatch", "suppress", "wait"]

    for step_idx in range(3):
        random_action = DisasterAction(
            interaction=interactions[int(rng.integers(0, len(interactions)))],
            row=int(rng.integers(0, env.grid_size)),
            col=int(rng.integers(0, env.grid_size)),
        )
        result = env.step(random_action)
        print(f"\nStep {step_idx + 1} action={random_action}")
        print(f"reward={result.reward}, done={result.done}")
        print(result.observation)
