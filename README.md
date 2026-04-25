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

## Local Run

Install dependencies and run:

```bash
pip install -r requirements.txt
uvicorn disaster_sim.api.server:app --host 0.0.0.0 --port 7860
```

Then open /docs for the interactive OpenAPI UI.

## Hugging Face Spaces Deployment

1. Create a new Space with SDK set to Docker.
2. Push this repository to the Space.
3. Spaces will build using Dockerfile and expose port 7860.
