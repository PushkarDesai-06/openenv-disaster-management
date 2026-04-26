from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError(
        "matplotlib is required for plotting. Install with: pip install matplotlib"
    ) from exc

try:
    from disaster_sim.envs.disaster_env import (
        DisasterAction,
        DisasterEnvironment,
        DisasterObservation,
        DisasterState,
    )
except ImportError:
    from envs.disaster_env import (  # type: ignore
        DisasterAction,
        DisasterEnvironment,
        DisasterObservation,
        DisasterState,
    )


PolicyFn = Callable[
    [DisasterObservation, DisasterState, Optional[tuple[int, int]]],
    tuple[DisasterAction, Optional[tuple[int, int]]],
]


@dataclass
class EpisodeResult:
    policy_name: str
    episode_index: int
    steps: int
    total_reward: float
    saved: int
    lost: int
    active_end: int


def _default_output_dir() -> str:
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/openenv-results/environment-eval"
    if os.path.isdir("/content"):
        return "/content/outputs/environment-eval"
    return "outputs/environment-eval"


def _teacher_policy(
    observation: DisasterObservation,
    state: DisasterState,
    last_signal: Optional[tuple[int, int]],
) -> tuple[DisasterAction, Optional[tuple[int, int]]]:
    if observation.critical_signals:
        row, col, _ = min(observation.critical_signals, key=lambda x: x[2])
        return DisasterAction(interaction="dispatch", row=row, col=col), (row, col)

    if last_signal is not None and state.step_count % 2 == 0:
        return (
            DisasterAction(interaction="suppress", row=last_signal[0], col=last_signal[1]),
            last_signal,
        )

    center = len(observation.masked_telemetry) // 2
    if state.step_count % 3 == 0:
        return DisasterAction(interaction="suppress", row=center, col=center), last_signal

    return DisasterAction(interaction="wait", row=center, col=center), last_signal


def _wait_policy(
    observation: DisasterObservation,
    state: DisasterState,
    last_signal: Optional[tuple[int, int]],
) -> tuple[DisasterAction, Optional[tuple[int, int]]]:
    del state, last_signal
    center = len(observation.masked_telemetry) // 2
    return DisasterAction(interaction="wait", row=center, col=center), None


def _make_random_policy(rng: np.random.Generator, grid_size: int) -> PolicyFn:
    interactions = np.array(["dispatch", "suppress", "wait"], dtype=object)

    def _policy(
        observation: DisasterObservation,
        state: DisasterState,
        last_signal: Optional[tuple[int, int]],
    ) -> tuple[DisasterAction, Optional[tuple[int, int]]]:
        del observation, state, last_signal
        interaction = str(interactions[int(rng.integers(0, len(interactions)))])
        row = int(rng.integers(0, grid_size))
        col = int(rng.integers(0, grid_size))
        return DisasterAction(interaction=interaction, row=row, col=col), None

    return _policy


def _nanmean_std(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    return mean, std


def _rollout(
    env: DisasterEnvironment,
    policy_name: str,
    policy: PolicyFn,
    max_steps: int,
    seed: int,
    episode_index: int,
) -> tuple[EpisodeResult, dict[str, np.ndarray]]:
    observation = env.reset(seed=seed)
    done = False
    step_idx = 0
    total_reward = 0.0
    last_signal: Optional[tuple[int, int]] = None

    cumulative_reward = 0.0
    rewards = np.full(max_steps, np.nan, dtype=np.float64)
    cumulative_rewards = np.full(max_steps, np.nan, dtype=np.float64)
    active_victims = np.full(max_steps, np.nan, dtype=np.float64)
    saved_victims = np.full(max_steps, np.nan, dtype=np.float64)
    lost_victims = np.full(max_steps, np.nan, dtype=np.float64)
    visible_signals = np.full(max_steps, np.nan, dtype=np.float64)

    while not done and step_idx < max_steps:
        action, last_signal = policy(observation, env.state, last_signal)
        result = env.step(action)

        reward = float(result.reward or 0.0)
        cumulative_reward += reward
        total_reward += reward
        observation = result.observation
        done = bool(result.done)

        rewards[step_idx] = reward
        cumulative_rewards[step_idx] = cumulative_reward
        active_victims[step_idx] = float(env.state.active_victims)
        saved_victims[step_idx] = float(env.state.total_victims_saved)
        lost_victims[step_idx] = float(env.state.total_victims_lost)
        visible_signals[step_idx] = float(observation.visible_signals)
        step_idx += 1

    episode_result = EpisodeResult(
        policy_name=policy_name,
        episode_index=episode_index,
        steps=step_idx,
        total_reward=total_reward,
        saved=env.state.total_victims_saved,
        lost=env.state.total_victims_lost,
        active_end=env.state.active_victims,
    )

    traces = {
        "reward": rewards,
        "cumulative_reward": cumulative_rewards,
        "active_victims": active_victims,
        "saved_victims": saved_victims,
        "lost_victims": lost_victims,
        "visible_signals": visible_signals,
    }
    return episode_result, traces


def _plot_policy_comparison(
    summary_rows: list[dict[str, float | str]],
    output_dir: str,
    dpi: int,
) -> None:
    names = [str(row["policy"]) for row in summary_rows]
    mean_reward = np.array([float(row["mean_reward"]) for row in summary_rows], dtype=np.float64)
    std_reward = np.array([float(row["std_reward"]) for row in summary_rows], dtype=np.float64)
    mean_saved = np.array([float(row["mean_saved"]) for row in summary_rows], dtype=np.float64)
    mean_lost = np.array([float(row["mean_lost"]) for row in summary_rows], dtype=np.float64)

    x = np.arange(len(names), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axes[0].bar(x, mean_reward, yerr=std_reward, capsize=6, color=["#4E79A7", "#F28E2B", "#59A14F"])
    axes[0].set_xticks(x, names)
    axes[0].set_title("Average Episode Reward by Policy")
    axes[0].set_xlabel("Policy")
    axes[0].set_ylabel("Reward")

    width = 0.35
    axes[1].bar(x - width / 2.0, mean_saved, width=width, label="Saved", color="#59A14F")
    axes[1].bar(x + width / 2.0, mean_lost, width=width, label="Lost", color="#E15759")
    axes[1].set_xticks(x, names)
    axes[1].set_title("Average Victim Outcomes by Policy")
    axes[1].set_xlabel("Policy")
    axes[1].set_ylabel("Victim Count")
    axes[1].legend()

    fig.suptitle("OpenEnv Disaster Environment: Policy Improvement Summary", fontsize=13)
    fig.savefig(os.path.join(output_dir, "policy_comparison.png"), dpi=dpi)
    plt.close(fig)


def _plot_curves(
    traces_by_policy: dict[str, dict[str, list[np.ndarray]]],
    max_steps: int,
    output_dir: str,
    dpi: int,
) -> None:
    x = np.arange(1, max_steps + 1)
    colors = {
        "Random": "#4E79A7",
        "WaitOnly": "#E15759",
        "TeacherHeuristic": "#59A14F",
    }

    chart_specs = [
        ("cumulative_reward", "Mean Cumulative Reward Over Time", "Cumulative Reward", "reward_over_steps.png"),
        ("active_victims", "Mean Active Victims Over Time", "Active Victims", "active_victims_over_steps.png"),
        ("visible_signals", "Mean Visible Critical Signals Over Time", "Visible Critical Signals", "signals_over_steps.png"),
    ]

    for key, title, ylabel, filename in chart_specs:
        fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
        for policy_name, metric_map in traces_by_policy.items():
            stacked = np.stack(metric_map[key], axis=0)
            mean, std = _nanmean_std(stacked)
            color = colors.get(policy_name, None)
            ax.plot(x, mean, label=policy_name, linewidth=2.0, color=color)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, filename), dpi=dpi)
        plt.close(fig)


def _write_episode_metrics_csv(results: list[EpisodeResult], output_dir: str) -> None:
    file_path = os.path.join(output_dir, "episode_metrics.csv")
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["policy", "episode", "steps", "total_reward", "saved", "lost", "active_end"])
        for row in results:
            writer.writerow(
                [
                    row.policy_name,
                    row.episode_index,
                    row.steps,
                    f"{row.total_reward:.6f}",
                    row.saved,
                    row.lost,
                    row.active_end,
                ]
            )


def _build_summary(results: list[EpisodeResult]) -> tuple[list[dict[str, float | str]], dict[str, dict[str, float]]]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for row in results:
        grouped.setdefault(row.policy_name, []).append(row)

    summary_rows: list[dict[str, float | str]] = []
    summary_map: dict[str, dict[str, float]] = {}
    for policy_name, rows in grouped.items():
        rewards = np.array([r.total_reward for r in rows], dtype=np.float64)
        saved = np.array([r.saved for r in rows], dtype=np.float64)
        lost = np.array([r.lost for r in rows], dtype=np.float64)
        steps = np.array([r.steps for r in rows], dtype=np.float64)

        metrics = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_saved": float(np.mean(saved)),
            "mean_lost": float(np.mean(lost)),
            "mean_steps": float(np.mean(steps)),
        }
        summary_map[policy_name] = metrics
        summary_rows.append({"policy": policy_name, **metrics})

    summary_rows.sort(key=lambda x: str(x["policy"]))
    return summary_rows, summary_map


def _write_summary_json(summary_rows: list[dict[str, float | str]], output_dir: str) -> None:
    payload = {"policies": summary_rows}
    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def _write_report_md(summary_map: dict[str, dict[str, float]], output_dir: str) -> None:
    random_metrics = summary_map.get("Random")
    teacher_metrics = summary_map.get("TeacherHeuristic")

    lines = [
        "# Disaster Environment Evaluation Report",
        "",
        "This report compares baseline policies to demonstrate environment behavior and policy sensitivity.",
        "",
    ]

    if random_metrics is not None and teacher_metrics is not None:
        reward_delta = teacher_metrics["mean_reward"] - random_metrics["mean_reward"]
        saved_delta = teacher_metrics["mean_saved"] - random_metrics["mean_saved"]
        lost_delta = teacher_metrics["mean_lost"] - random_metrics["mean_lost"]
        lines.extend(
            [
                "## Teacher vs Random",
                "",
                f"- Reward improvement: {reward_delta:.2f}",
                f"- Saved victims improvement: {saved_delta:.2f}",
                f"- Lost victims change: {lost_delta:.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Included Artifacts",
            "",
            "- policy_comparison.png",
            "- reward_over_steps.png",
            "- active_victims_over_steps.png",
            "- signals_over_steps.png",
            "- episode_metrics.csv",
            "- summary_metrics.json",
        ]
    )

    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate disaster environment policies and save plots")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=_default_output_dir())
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    rng = np.random.default_rng(args.seed)
    policies: list[tuple[str, PolicyFn]] = [
        ("Random", _make_random_policy(rng, args.grid_size)),
        ("WaitOnly", _wait_policy),
        ("TeacherHeuristic", _teacher_policy),
    ]

    all_episode_results: list[EpisodeResult] = []
    traces_by_policy: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            "reward": [],
            "cumulative_reward": [],
            "active_victims": [],
            "saved_victims": [],
            "lost_victims": [],
            "visible_signals": [],
        }
        for name, _ in policies
    }

    for policy_idx, (policy_name, policy_fn) in enumerate(policies):
        for episode_idx in range(args.episodes):
            env = DisasterEnvironment(grid_size=args.grid_size)
            seed = int(args.seed + policy_idx * 100_000 + episode_idx)
            episode_result, traces = _rollout(
                env=env,
                policy_name=policy_name,
                policy=policy_fn,
                max_steps=args.max_steps,
                seed=seed,
                episode_index=episode_idx,
            )
            all_episode_results.append(episode_result)
            for key, values in traces.items():
                traces_by_policy[policy_name][key].append(values)

    summary_rows, summary_map = _build_summary(all_episode_results)
    _write_episode_metrics_csv(all_episode_results, args.output_dir)
    _write_summary_json(summary_rows, args.output_dir)
    _write_report_md(summary_map, args.output_dir)
    _plot_policy_comparison(summary_rows, args.output_dir, dpi=args.dpi)
    _plot_curves(traces_by_policy, args.max_steps, args.output_dir, dpi=args.dpi)

    print(f"Saved evaluation artifacts to {args.output_dir}")
    for file_name in [
        "policy_comparison.png",
        "reward_over_steps.png",
        "active_victims_over_steps.png",
        "signals_over_steps.png",
        "episode_metrics.csv",
        "summary_metrics.json",
        "report.md",
    ]:
        print(f" - {os.path.join(args.output_dir, file_name)}")


if __name__ == "__main__":
    main()
