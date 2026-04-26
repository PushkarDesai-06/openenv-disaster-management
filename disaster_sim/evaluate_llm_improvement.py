from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
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
    from disaster_sim.train_llm_qlora import _build_prompt
except ImportError:
    from envs.disaster_env import (  # type: ignore
        DisasterAction,
        DisasterEnvironment,
        DisasterObservation,
        DisasterState,
    )
    from train_llm_qlora import _build_prompt  # type: ignore


ActionFn = Callable[
    [DisasterObservation, DisasterState, Optional[tuple[int, int]]],
    tuple[DisasterAction, Optional[tuple[int, int]], bool],
]


@dataclass
class EpisodeMetric:
    policy: str
    episode: int
    steps: int
    total_reward: float
    saved: int
    lost: int
    active_end: int
    parse_success_rate: float


def _default_output_dir() -> str:
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/openenv-results/llm-improvement-eval"
    if os.path.isdir("/content"):
        return "/content/outputs/llm-improvement-eval"
    return "outputs/llm-improvement-eval"


def _teacher_policy(
    observation: DisasterObservation,
    state: DisasterState,
    last_signal: Optional[tuple[int, int]],
) -> tuple[DisasterAction, Optional[tuple[int, int]], bool]:
    if observation.critical_signals:
        row, col, _ = min(observation.critical_signals, key=lambda x: x[2])
        return DisasterAction(interaction="dispatch", row=row, col=col), (row, col), True

    if last_signal is not None and state.step_count % 2 == 0:
        return (
            DisasterAction(interaction="suppress", row=last_signal[0], col=last_signal[1]),
            last_signal,
            True,
        )

    center = len(observation.masked_telemetry) // 2
    if state.step_count % 3 == 0:
        return DisasterAction(interaction="suppress", row=center, col=center), last_signal, True

    return DisasterAction(interaction="wait", row=center, col=center), last_signal, True


def _make_random_policy(rng: np.random.Generator, grid_size: int) -> ActionFn:
    interactions = np.array(["dispatch", "suppress", "wait"], dtype=object)

    def _policy(
        observation: DisasterObservation,
        state: DisasterState,
        last_signal: Optional[tuple[int, int]],
    ) -> tuple[DisasterAction, Optional[tuple[int, int]], bool]:
        del observation, state, last_signal
        interaction = str(interactions[int(rng.integers(0, len(interactions)))])
        row = int(rng.integers(0, grid_size))
        col = int(rng.integers(0, grid_size))
        return DisasterAction(interaction=interaction, row=row, col=col), None, True

    return _policy


def _extract_json_object(text: str) -> Optional[dict[str, object]]:
    # Prefer markdown code block JSON if present.
    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    candidates: list[str] = []
    if fenced_match:
        candidates.append(fenced_match.group(1))

    # Then try any object-like span.
    for match in re.finditer(r"\{[\s\S]*?\}", text):
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_action_payload(
    payload: Optional[dict[str, object]],
    grid_size: int,
) -> tuple[DisasterAction, bool]:
    center = grid_size // 2
    if payload is None:
        return DisasterAction(interaction="wait", row=center, col=center), False

    interaction_raw = payload.get("interaction")
    interaction = str(interaction_raw).strip().lower() if interaction_raw is not None else "wait"
    if interaction not in {"dispatch", "suppress", "wait"}:
        interaction = "wait"

    def _int_or_default(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    row = int(np.clip(_int_or_default(payload.get("row"), center), 0, grid_size - 1))
    col = int(np.clip(_int_or_default(payload.get("col"), center), 0, grid_size - 1))

    action = DisasterAction(interaction=interaction, row=row, col=col)
    parsed_ok = payload.get("interaction") is not None and payload.get("row") is not None and payload.get("col") is not None
    return action, bool(parsed_ok)


def _load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    use_4bit: bool,
) -> tuple[object, object]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM eval dependencies. Install with: pip install -r requirements-llm.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {"device_map": "auto", "low_cpu_mem_usage": True}
    if torch.cuda.is_available() and use_4bit:
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif torch.cuda.is_available():
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16_ok else torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    # Disabling cache reduces peak memory during repeated per-step generation rollouts.
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return model, tokenizer


def _make_llm_policy(
    model: object,
    tokenizer: object,
    grid_size: int,
    max_new_tokens: int,
) -> ActionFn:
    import torch

    def _policy(
        observation: DisasterObservation,
        state: DisasterState,
        last_signal: Optional[tuple[int, int]],
    ) -> tuple[DisasterAction, Optional[tuple[int, int]], bool]:
        del last_signal
        prompt = _build_prompt(observation, state) + "\nAction:\n"

        tokenized = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", torch.device("cpu"))
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.no_grad():
            try:
                generated = model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if (
                    "out of memory" in message
                    or "bitsandbytes" in message
                    or "gemv_4bit" in message
                ):
                    raise RuntimeError(
                        "LLM generation failed due GPU memory or 4-bit kernel limits. "
                        "Retry with --no-4bit --max-new-tokens 32 --episodes 20."
                    ) from exc
                raise

        prompt_tokens = int(tokenized["input_ids"].shape[1])
        generated_tokens = generated[0][prompt_tokens:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        payload = _extract_json_object(text)
        action, parsed_ok = _normalize_action_payload(payload, grid_size)
        return action, None, parsed_ok

    return _policy


def _rollout_policy(
    policy_name: str,
    action_fn: ActionFn,
    seeds: list[int],
    grid_size: int,
    max_steps: int,
) -> tuple[list[EpisodeMetric], dict[str, list[np.ndarray]]]:
    episode_metrics: list[EpisodeMetric] = []
    traces = {
        "total_reward": [],
        "cumulative_reward": [],
        "active_victims": [],
        "saved_victims": [],
        "lost_victims": [],
    }

    for episode_idx, seed in enumerate(seeds):
        env = DisasterEnvironment(grid_size=grid_size)
        observation = env.reset(seed=seed)

        done = False
        step_count = 0
        total_reward = 0.0
        last_signal: Optional[tuple[int, int]] = None
        parse_ok_count = 0

        rewards = np.full(max_steps, np.nan, dtype=np.float64)
        cumulative = np.full(max_steps, np.nan, dtype=np.float64)
        active = np.full(max_steps, np.nan, dtype=np.float64)
        saved = np.full(max_steps, np.nan, dtype=np.float64)
        lost = np.full(max_steps, np.nan, dtype=np.float64)

        while not done and step_count < max_steps:
            action, last_signal, parsed_ok = action_fn(observation, env.state, last_signal)
            if parsed_ok:
                parse_ok_count += 1

            result = env.step(action)
            reward = float(result.reward or 0.0)
            total_reward += reward
            observation = result.observation
            done = bool(result.done)

            rewards[step_count] = reward
            cumulative[step_count] = total_reward
            active[step_count] = float(env.state.active_victims)
            saved[step_count] = float(env.state.total_victims_saved)
            lost[step_count] = float(env.state.total_victims_lost)

            step_count += 1

        parse_success_rate = float(parse_ok_count / max(step_count, 1))
        episode_metrics.append(
            EpisodeMetric(
                policy=policy_name,
                episode=episode_idx,
                steps=step_count,
                total_reward=total_reward,
                saved=env.state.total_victims_saved,
                lost=env.state.total_victims_lost,
                active_end=env.state.active_victims,
                parse_success_rate=parse_success_rate,
            )
        )

        traces["total_reward"].append(rewards)
        traces["cumulative_reward"].append(cumulative)
        traces["active_victims"].append(active)
        traces["saved_victims"].append(saved)
        traces["lost_victims"].append(lost)

    return episode_metrics, traces


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return np.nan_to_num(mean, nan=0.0), np.nan_to_num(std, nan=0.0)


def _build_summary(metrics: list[EpisodeMetric]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[EpisodeMetric]] = {}
    for metric in metrics:
        grouped.setdefault(metric.policy, []).append(metric)

    rows: list[dict[str, float | str]] = []
    for policy_name, policy_rows in grouped.items():
        rewards = np.array([row.total_reward for row in policy_rows], dtype=np.float64)
        saved = np.array([row.saved for row in policy_rows], dtype=np.float64)
        lost = np.array([row.lost for row in policy_rows], dtype=np.float64)
        parse_rate = np.array([row.parse_success_rate for row in policy_rows], dtype=np.float64)

        rows.append(
            {
                "policy": policy_name,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "mean_saved": float(np.mean(saved)),
                "mean_lost": float(np.mean(lost)),
                "mean_parse_success_rate": float(np.mean(parse_rate)),
            }
        )

    rows.sort(key=lambda row: str(row["policy"]))
    return rows


def _save_episode_csv(metrics: list[EpisodeMetric], output_dir: str) -> None:
    out_file = os.path.join(output_dir, "llm_episode_metrics.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "policy",
                "episode",
                "steps",
                "total_reward",
                "saved",
                "lost",
                "active_end",
                "parse_success_rate",
            ]
        )
        for row in metrics:
            writer.writerow(
                [
                    row.policy,
                    row.episode,
                    row.steps,
                    f"{row.total_reward:.6f}",
                    row.saved,
                    row.lost,
                    row.active_end,
                    f"{row.parse_success_rate:.6f}",
                ]
            )


def _save_summary_json(summary: list[dict[str, float | str]], output_dir: str) -> None:
    with open(os.path.join(output_dir, "llm_summary_metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump({"policies": summary}, file_obj, indent=2)


def _plot_reward_comparison(summary: list[dict[str, float | str]], output_dir: str, dpi: int) -> None:
    labels = [str(row["policy"]) for row in summary]
    means = np.array([float(row["mean_reward"]) for row in summary], dtype=np.float64)
    stds = np.array([float(row["std_reward"]) for row in summary], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.bar(np.arange(len(labels)), means, yerr=stds, capsize=6, color="#4E79A7")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_title("LLM Policy Reward Comparison")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Episode Reward")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(output_dir, "llm_policy_reward_comparison.png"), dpi=dpi)
    plt.close(fig)


def _plot_outcomes(summary: list[dict[str, float | str]], output_dir: str, dpi: int) -> None:
    labels = [str(row["policy"]) for row in summary]
    saved = np.array([float(row["mean_saved"]) for row in summary], dtype=np.float64)
    lost = np.array([float(row["mean_lost"]) for row in summary], dtype=np.float64)

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.bar(x - width / 2.0, saved, width=width, label="Saved", color="#59A14F")
    ax.bar(x + width / 2.0, lost, width=width, label="Lost", color="#E15759")
    ax.set_xticks(x, labels)
    ax.set_title("LLM Policy Victim Outcomes")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average Victim Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(output_dir, "llm_policy_outcomes.png"), dpi=dpi)
    plt.close(fig)


def _plot_running_reward(
    all_metrics: list[EpisodeMetric],
    output_dir: str,
    dpi: int,
) -> None:
    grouped: dict[str, list[EpisodeMetric]] = {}
    for metric in all_metrics:
        grouped.setdefault(metric.policy, []).append(metric)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    color_map = {
        "Random": "#E15759",
        "TeacherHeuristic": "#76B7B2",
        "BaseLLM": "#4E79A7",
        "SFTAdapterLLM": "#59A14F",
    }

    for policy_name, rows in grouped.items():
        rows.sort(key=lambda row: row.episode)
        rewards = np.array([row.total_reward for row in rows], dtype=np.float64)
        running_mean = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        ax.plot(
            np.arange(1, len(rows) + 1),
            running_mean,
            label=policy_name,
            linewidth=2.0,
            color=color_map.get(policy_name, None),
        )

    ax.set_title("Running Mean Reward by Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Running Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "llm_policy_running_reward.png"), dpi=dpi)
    plt.close(fig)


def _save_report(summary: list[dict[str, float | str]], output_dir: str) -> None:
    by_name = {str(row["policy"]): row for row in summary}
    base = by_name.get("BaseLLM")
    sft = by_name.get("SFTAdapterLLM")

    lines = [
        "# LLM Improvement Report",
        "",
        "This report measures reward improvement inside the disaster environment.",
        "",
    ]

    if base is not None and sft is not None:
        reward_delta = float(sft["mean_reward"]) - float(base["mean_reward"])
        saved_delta = float(sft["mean_saved"]) - float(base["mean_saved"])
        lost_delta = float(sft["mean_lost"]) - float(base["mean_lost"])
        lines.extend(
            [
                "## SFT Adapter Improvement Over Base LLM",
                "",
                f"- Reward delta: {reward_delta:.2f}",
                f"- Saved victims delta: {saved_delta:.2f}",
                f"- Lost victims delta: {lost_delta:.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            "- llm_policy_reward_comparison.png",
            "- llm_policy_outcomes.png",
            "- llm_policy_running_reward.png",
            "- llm_episode_metrics.csv",
            "- llm_summary_metrics.json",
        ]
    )

    with open(os.path.join(output_dir, "llm_report.md"), "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def _cleanup_model(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM reward improvement on DisasterEnvironment")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=_default_output_dir())
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--run-random-baseline", action="store_true")
    parser.add_argument("--run-teacher-baseline", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_path:
        raise RuntimeError(
            "--adapter-path is required to compare BaseLLM vs SFTAdapterLLM. "
            "Example: --adapter-path /content/drive/MyDrive/openenv-checkpoints/disaster-dispatcher-qlora"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    seed_rng = np.random.default_rng(args.seed)
    episode_seeds = [int(seed_rng.integers(0, 1_000_000)) for _ in range(args.episodes)]

    all_metrics: list[EpisodeMetric] = []

    # Optional lightweight baselines for environment credibility.
    if args.run_random_baseline:
        random_rng = np.random.default_rng(args.seed + 42)
        random_policy = _make_random_policy(random_rng, args.grid_size)
        random_metrics, _ = _rollout_policy(
            policy_name="Random",
            action_fn=random_policy,
            seeds=episode_seeds,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
        )
        all_metrics.extend(random_metrics)

    if args.run_teacher_baseline:
        teacher_metrics, _ = _rollout_policy(
            policy_name="TeacherHeuristic",
            action_fn=_teacher_policy,
            seeds=episode_seeds,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
        )
        all_metrics.extend(teacher_metrics)

    # Base LLM rollout.
    base_model, base_tokenizer = _load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path="",
        use_4bit=not args.no_4bit,
    )
    base_policy = _make_llm_policy(
        model=base_model,
        tokenizer=base_tokenizer,
        grid_size=args.grid_size,
        max_new_tokens=args.max_new_tokens,
    )
    base_metrics, _ = _rollout_policy(
        policy_name="BaseLLM",
        action_fn=base_policy,
        seeds=episode_seeds,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
    )
    all_metrics.extend(base_metrics)
    _cleanup_model(base_model)

    # SFT adapter rollout.
    sft_model, sft_tokenizer = _load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=not args.no_4bit,
    )
    sft_policy = _make_llm_policy(
        model=sft_model,
        tokenizer=sft_tokenizer,
        grid_size=args.grid_size,
        max_new_tokens=args.max_new_tokens,
    )
    sft_metrics, _ = _rollout_policy(
        policy_name="SFTAdapterLLM",
        action_fn=sft_policy,
        seeds=episode_seeds,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
    )
    all_metrics.extend(sft_metrics)
    _cleanup_model(sft_model)

    summary = _build_summary(all_metrics)
    _save_episode_csv(all_metrics, args.output_dir)
    _save_summary_json(summary, args.output_dir)
    _plot_reward_comparison(summary, args.output_dir, dpi=args.dpi)
    _plot_outcomes(summary, args.output_dir, dpi=args.dpi)
    _plot_running_reward(all_metrics, args.output_dir, dpi=args.dpi)
    _save_report(summary, args.output_dir)

    print(f"Saved LLM improvement artifacts to {args.output_dir}")
    for file_name in [
        "llm_policy_reward_comparison.png",
        "llm_policy_outcomes.png",
        "llm_policy_running_reward.png",
        "llm_episode_metrics.csv",
        "llm_summary_metrics.json",
        "llm_report.md",
    ]:
        print(f" - {os.path.join(args.output_dir, file_name)}")


if __name__ == "__main__":
    main()
