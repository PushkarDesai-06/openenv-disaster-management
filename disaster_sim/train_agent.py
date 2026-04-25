from __future__ import annotations

import argparse
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

try:
    from disaster_sim.envs.disaster_env import DisasterAction, DisasterEnvironment
except ImportError:
    from envs.disaster_env import DisasterAction, DisasterEnvironment


INTERACTIONS = ["dispatch", "suppress", "wait"]


if gym is not None and spaces is not None:

    class DisasterGymAdapter(gym.Env):
        """Gymnasium adapter used only for local trainer interoperability."""

        metadata = {"render_modes": []}

        def __init__(self, grid_size: int = 10) -> None:
            super().__init__()
            self._core = DisasterEnvironment(grid_size=grid_size)
            self.action_space = spaces.MultiDiscrete(
                np.array([len(INTERACTIONS), grid_size, grid_size], dtype=np.int64)
            )
            self.observation_space = spaces.Box(
                low=-1,
                high=100,
                shape=(grid_size, grid_size),
                dtype=np.int16,
            )

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            del options
            observation = self._core.reset(seed=seed)
            obs = np.asarray(observation.masked_telemetry, dtype=np.int16)
            info = {"visible_signals": observation.visible_signals}
            return obs, info

        def step(self, action):
            action_idx, row, col = [int(x) for x in action]
            result = self._core.step(
                DisasterAction(
                    interaction=INTERACTIONS[action_idx],
                    row=row,
                    col=col,
                )
            )

            obs = np.asarray(result.observation.masked_telemetry, dtype=np.int16)
            reward = float(result.reward or 0.0)
            terminated = bool(result.done)
            truncated = False
            info = {
                "visible_signals": result.observation.visible_signals,
                "saved": self._core.state.total_victims_saved,
                "lost": self._core.state.total_victims_lost,
            }
            return obs, reward, terminated, truncated, info

else:

    class DisasterGymAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("gymnasium is required for the SB3 training path.")


def random_policy_rollout(episodes: int = 2, max_steps: int = 50) -> None:
    """Fallback runner when RL dependencies are not installed."""
    rng = np.random.default_rng(123)

    for episode in range(episodes):
        env = DisasterEnvironment(grid_size=10)
        env.reset(seed=int(rng.integers(0, 10000)))
        cumulative_reward = 0.0

        for _ in range(max_steps):
            action = DisasterAction(
                interaction=INTERACTIONS[int(rng.integers(0, len(INTERACTIONS)))],
                row=int(rng.integers(0, env.grid_size)),
                col=int(rng.integers(0, env.grid_size)),
            )
            result = env.step(action)
            cumulative_reward += float(result.reward or 0.0)
            if result.done:
                break

        print(
            f"Episode {episode + 1}: reward={cumulative_reward:.2f}, "
            f"saved={env.state.total_victims_saved}, lost={env.state.total_victims_lost}"
        )


def train_with_sb3(total_timesteps: int, model_path: str) -> None:
    if gym is None:
        print("gymnasium is not installed; running random policy rollout instead.")
        random_policy_rollout()
        return

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 is not installed; running random policy rollout instead.")
        random_policy_rollout()
        return

    env = DisasterGymAdapter(grid_size=10)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=128,
        batch_size=64,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(model_path)
    print(f"Saved PPO model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dispatcher policy for DisasterEnvironment")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--model-path", type=str, default="disaster_dispatcher_ppo")
    args = parser.parse_args()

    train_with_sb3(total_timesteps=args.timesteps, model_path=args.model_path)


if __name__ == "__main__":
    main()
