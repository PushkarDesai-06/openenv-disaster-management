---
title: Disaster Simulation API
sdk: docker
app_port: 7860
---

# Disaster Simulation API

This repository contains a disaster response simulation environment and an HTTP API
that can be deployed to Hugging Face Spaces using Docker.

## Project Layout

- disaster_sim/envs/disaster_env.py: OpenEnv-style environment implementation.
- disaster_sim/envs/physics_engine.py: hazard spread and victim decay dynamics.
- disaster_sim/models/telemetry.py: typed telemetry payload schemas.
- disaster_sim/api/server.py: FastAPI service exposing reset, step, and state endpoints.
- disaster_sim/train_agent.py: optional RL training entrypoint.

## API Endpoints

- GET /: service metadata
- GET /health: liveness check
- POST /reset: start a new episode
- POST /step: execute an action
- GET /state: read current episode state
- GET /telemetry/current: current masked telemetry snapshot
- GET /ui: Gradio control panel for reset and step

## Local Run

Install dependencies and run:

```bash
pip install -r requirements.txt
uvicorn disaster_sim.api.server:app --host 0.0.0.0 --port 7860
```

Then open /docs for the interactive OpenAPI UI.
For the Gradio reset/step interface, open /ui.

## Hugging Face Spaces Deployment

1. Create a new Space with SDK set to Docker.
2. Push this repository to the Space.
3. Spaces will build using Dockerfile and expose port 7860.

## LLM QLoRA Training (Colab)

Use the dedicated training script and dependency file for QLoRA runs:

1. Install training dependencies:

	pip install -r requirements-llm.txt

2. Run training:

	python -m disaster_sim.train_llm_qlora \
	  --base-model Qwen/Qwen2.5-3B-Instruct \
	  --output-dir outputs/disaster-dispatcher-qlora \
	  --episodes 250 \
	  --max-steps 45 \
	  --epochs 1 \
	  --batch-size 1 \
	  --grad-accum 8

3. Optional: push adapter directly to Hugging Face Hub:

	python -m disaster_sim.train_llm_qlora \
	  --hf-adapter-repo YOUR_USERNAME/disaster-dispatcher-qlora

## Environment Evaluation Graphs (Colab)

Generate clean, labeled figures to show that the environment responds to policy quality
and save outputs directly to Google Drive.

1. Mount Drive in Colab:

	from google.colab import drive
	drive.mount('/content/drive')

2. Install plotting dependency (if needed):

	pip install matplotlib

3. Run policy benchmark + graph export:

	python -m disaster_sim.evaluate_and_plot \
	  --episodes 100 \
	  --output-dir /content/drive/MyDrive/openenv-results/environment-eval

Artifacts produced:

- policy_comparison.png
- reward_over_steps.png
- active_victims_over_steps.png
- signals_over_steps.png
- episode_metrics.csv
- summary_metrics.json
- report.md

## LLM Reward Improvement Graphs (Base vs SFT)

Use this benchmark to show that your SFT adapter improves reward outcomes in the same
environment compared with the base model.

1. Mount Drive:

	from google.colab import drive
	drive.mount('/content/drive')

2. Run LLM improvement benchmark:

	python -m disaster_sim.evaluate_llm_improvement \
	  --base-model Qwen/Qwen2.5-3B-Instruct \
	  --adapter-path /content/drive/MyDrive/openenv-checkpoints/disaster-dispatcher-qlora \
	  --episodes 30 \
	  --output-dir /content/drive/MyDrive/openenv-results/llm-improvement-eval \
	  --run-random-baseline \
	  --run-teacher-baseline

Artifacts produced:

- llm_policy_reward_comparison.png
- llm_policy_outcomes.png
- llm_policy_running_reward.png
- llm_episode_metrics.csv
- llm_summary_metrics.json
- llm_report.md
