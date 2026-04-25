from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.signal import convolve2d
except ImportError:
    convolve2d = None


@dataclass(frozen=True)
class PhysicsConfig:
    grid_size: int = 10
    base_spread_prob: float = 0.1
    base_decay: int = 2
    hazard_multiplier: int = 2
    hazard_cap: float = 5.0
    hazard_growth_factor: float = 0.15
    hazard_decay_factor: float = 0.98
    hazard_cluster_threshold: float = 0.35


class DisasterPhysicsEngine:
    """Vectorized transition rules for hazard and victim dynamics."""

    _NEIGHBOR_KERNEL = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, config: PhysicsConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def sample_epicenter(self) -> tuple[int, int]:
        low = self.config.grid_size // 4
        high = 3 * self.config.grid_size // 4 + 1
        return (
            int(self._rng.integers(low, high)),
            int(self._rng.integers(low, high)),
        )

    def initialize_hazard_cluster(
        self,
        hazard_grid: np.ndarray,
        epicenter_row: int,
        epicenter_col: int,
    ) -> None:
        rows, cols = np.indices((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        sq_dist = (rows - float(epicenter_row)) ** 2 + (cols - float(epicenter_col)) ** 2
        sigma = max(1.0, self.config.grid_size / 6.0)
        gaussian = np.exp(-sq_dist / (2.0 * sigma * sigma))
        hazard_grid[:] = np.where(
            gaussian > self.config.hazard_cluster_threshold,
            gaussian * 3.0,
            0.0,
        ).astype(np.float32)

    def scatter_victims(
        self,
        victim_health: np.ndarray,
        active_victims: np.ndarray,
        epicenter_row: int,
        epicenter_col: int,
        target_victims: Optional[int] = None,
    ) -> None:
        victim_health.fill(0)
        active_victims.fill(False)

        target_count = target_victims or max(6, int(self.config.grid_size * 2.2))
        scale = max(1.0, self.config.grid_size / 4.0)
        placed = 0
        attempts = 0

        while placed < target_count and attempts < 10:
            samples = self._rng.normal(
                loc=np.array([epicenter_row, epicenter_col], dtype=np.float32),
                scale=scale,
                size=(target_count * 2, 2),
            )
            coords = np.rint(samples).astype(np.int32)
            coords = np.clip(coords, 0, self.config.grid_size - 1)
            unique_coords = np.unique(coords, axis=0)

            for row, col in unique_coords:
                if not active_victims[row, col]:
                    active_victims[row, col] = True
                    victim_health[row, col] = int(self._rng.integers(60, 101))
                    placed += 1
                    if placed >= target_count:
                        break
            attempts += 1

    def spread_hazards(self, hazard_grid: np.ndarray) -> None:
        active_hazards = hazard_grid > 0.0
        if not np.any(active_hazards):
            return

        adjacent_hazard_cells = self.neighbor_sum(active_hazards.astype(np.float32))
        spread_prob = 1.0 - np.power(1.0 - self.config.base_spread_prob, adjacent_hazard_cells)
        ignitions = (~active_hazards) & (
            self._rng.random((self.config.grid_size, self.config.grid_size)) < spread_prob
        )

        hazard_grid[ignitions] = np.maximum(hazard_grid[ignitions], 1.0)

        adjacent_intensity = self.neighbor_sum(hazard_grid)
        intensified = np.where(
            hazard_grid > 0.0,
            hazard_grid + self.config.hazard_growth_factor * adjacent_intensity,
            hazard_grid,
        )
        hazard_grid[:] = np.clip(
            intensified * self.config.hazard_decay_factor,
            0.0,
            self.config.hazard_cap,
        ).astype(np.float32)

    def apply_health_decay(
        self,
        hazard_grid: np.ndarray,
        victim_health: np.ndarray,
        active_victims: np.ndarray,
    ) -> int:
        if not np.any(active_victims):
            return 0

        adjacent_hazard_intensity = self.neighbor_sum(hazard_grid)
        hazard_bonus = np.where(
            adjacent_hazard_intensity > 0,
            np.rint(self.config.hazard_multiplier * adjacent_hazard_intensity).astype(np.int16),
            0,
        )
        decay = self.config.base_decay + hazard_bonus

        victim_health[active_victims] = np.clip(
            victim_health[active_victims] - decay[active_victims],
            0,
            100,
        )

        newly_lost_mask = active_victims & (victim_health <= 0)
        newly_lost = int(np.count_nonzero(newly_lost_mask))
        if newly_lost:
            active_victims[newly_lost_mask] = False
            victim_health[newly_lost_mask] = 0

        return newly_lost

    def neighbor_sum(self, grid: np.ndarray) -> np.ndarray:
        if convolve2d is not None:
            return convolve2d(
                grid,
                self._NEIGHBOR_KERNEL,
                mode="same",
                boundary="fill",
                fillvalue=0.0,
            )

        padded = np.pad(grid, 1, mode="constant")
        return (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
