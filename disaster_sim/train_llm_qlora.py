from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any, Optional

import numpy as np

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


def _choose_teacher_action(
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


def _build_prompt(observation: DisasterObservation, state: DisasterState) -> str:
    state_payload = {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "active_victims": state.active_victims,
        "total_victims_saved": state.total_victims_saved,
        "total_victims_lost": state.total_victims_lost,
    }
    prompt = (
        "You are a centralized emergency dispatcher for a disaster simulation. "
        "Choose exactly one action that maximizes rescues and reduces risk.\n"
        "Valid interactions: dispatch, suppress, wait.\n"
        "Return only JSON with keys interaction, row, col.\n\n"
        f"State: {json.dumps(state_payload, separators=(',', ':'))}\n"
        f"Critical health threshold: {observation.critical_health_threshold}\n"
        f"Visible distress signals: {observation.visible_signals}\n"
        "Masked telemetry grid (-1 means unknown):\n"
        f"{json.dumps(observation.masked_telemetry, separators=(',', ':'))}\n"
    )
    return prompt


def _build_text_sample(
    observation: DisasterObservation,
    state: DisasterState,
    action: DisasterAction,
) -> dict[str, str]:
    prompt = _build_prompt(observation, state)
    target = {
        "interaction": action.interaction,
        "row": int(action.row),
        "col": int(action.col),
    }
    text = prompt + "\nAction:\n" + json.dumps(target, separators=(",", ":"))
    return {"text": text}


def generate_sft_records(
    episodes: int,
    max_steps_per_episode: int,
    grid_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], float]:
    rng = np.random.default_rng(seed)
    env = DisasterEnvironment(grid_size=grid_size)

    records: list[dict[str, str]] = []
    total_reward = 0.0

    for _ in range(episodes):
        observation = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        steps = 0
        last_signal: Optional[tuple[int, int]] = None

        while not done and steps < max_steps_per_episode:
            action, last_signal = _choose_teacher_action(observation, env.state, last_signal)
            records.append(_build_text_sample(observation, env.state, action))

            result = env.step(action)
            total_reward += float(result.reward or 0.0)
            observation = result.observation
            done = bool(result.done)
            steps += 1

    return records, total_reward


def run_qlora_training(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing LLM training dependencies. Install with: pip install -r requirements-llm.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA training in this script.")

    records, total_reward = generate_sft_records(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        grid_size=args.grid_size,
        seed=args.seed,
    )
    if not records:
        raise RuntimeError("No training samples generated.")

    print(
        f"Generated {len(records)} samples from {args.episodes} episodes "
        f"(teacher total reward={total_reward:.2f})."
    )

    dataset = Dataset.from_list(records)

    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config_signature = inspect.signature(SFTConfig.__init__).parameters

    sft_config_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "bf16": bf16_ok,
        "fp16": not bf16_ok,
        "report_to": "none",
    }

    if "dataset_text_field" in sft_config_signature:
        sft_config_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_config_signature:
        sft_config_kwargs["max_seq_length"] = args.max_seq_len
    elif "max_length" in sft_config_signature:
        sft_config_kwargs["max_length"] = args.max_seq_len
    if "packing" in sft_config_signature:
        sft_config_kwargs["packing"] = args.packing

    sft_config = SFTConfig(**sft_config_kwargs)

    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "peft_config": peft_config,
    }

    # Keep tokenizer wiring compatible across TRL versions.
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    # Older TRL versions expect these fields on SFTTrainer instead of SFTConfig.
    if "dataset_text_field" in trainer_signature and "dataset_text_field" not in sft_config_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_signature and "max_seq_length" not in sft_config_kwargs:
        trainer_kwargs["max_seq_length"] = args.max_seq_len
    elif "max_length" in trainer_signature and "max_length" not in sft_config_kwargs:
        trainer_kwargs["max_length"] = args.max_seq_len
    if "packing" in trainer_signature and "packing" not in sft_config_kwargs:
        trainer_kwargs["packing"] = args.packing

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "base_model": args.base_model,
        "samples": len(records),
        "episodes": args.episodes,
        "grid_size": args.grid_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.hf_adapter_repo:
        print(f"Pushing adapter to {args.hf_adapter_repo}")
        trainer.model.push_to_hub(args.hf_adapter_repo)
        tokenizer.push_to_hub(args.hf_adapter_repo)

    print(f"Saved adapter and tokenizer to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA training for disaster dispatcher LLM")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="outputs/disaster-dispatcher-qlora")
    parser.add_argument("--hf-adapter-repo", type=str, default="")

    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=100)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--packing", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_qlora_training(args)


if __name__ == "__main__":
    main()
