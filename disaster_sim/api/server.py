from __future__ import annotations

import os
from dataclasses import asdict
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from ..envs.disaster_env import (
        DisasterAction,
        DisasterEnvironment,
        DisasterObservation,
        DisasterState,
    )
    from ..models.telemetry import (
        TelemetryGridSnapshot,
        WearableTelemetryEnvelope,
        WearableTelemetrySignal,
    )
except ImportError:
    from disaster_sim.envs.disaster_env import (
        DisasterAction,
        DisasterEnvironment,
        DisasterObservation,
        DisasterState,
    )
    from disaster_sim.models.telemetry import (
        TelemetryGridSnapshot,
        WearableTelemetryEnvelope,
        WearableTelemetrySignal,
    )


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    interaction: Literal["dispatch", "suppress", "wait"]
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


class EnvStepResponse(BaseModel):
    observation: TelemetryGridSnapshot
    reward: Optional[float] = None
    done: bool = False


class StateResponse(BaseModel):
    episode_id: str
    step_count: int
    total_victims_saved: int
    total_victims_lost: int
    active_victims: int


def _to_snapshot(
    observation: DisasterObservation,
    state: DisasterState,
) -> TelemetryGridSnapshot:
    signals = [
        WearableTelemetrySignal(row=row, col=col, health=health)
        for row, col, health in observation.critical_signals
    ]

    return TelemetryGridSnapshot(
        episode_id=state.episode_id,
        step_count=state.step_count,
        critical_health_threshold=observation.critical_health_threshold,
        visible_signals=observation.visible_signals,
        masked_telemetry=observation.masked_telemetry,
        signals=signals,
    )


def create_app() -> FastAPI:
    env = DisasterEnvironment()

    app = FastAPI(
        title="Disaster Simulation API",
        version="0.1.0",
        description="HTTP-native API around an OpenEnv-style disaster environment.",
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "disaster-simulation-api",
            "health": "/health",
            "docs": "/docs",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reset", response_model=EnvStepResponse)
    def reset(payload: ResetRequest) -> EnvStepResponse:
        observation = env.reset(seed=payload.seed, episode_id=payload.episode_id)
        snapshot = _to_snapshot(observation, env.state)
        return EnvStepResponse(observation=snapshot, reward=0.0, done=False)

    @app.post("/step", response_model=EnvStepResponse)
    def step(payload: StepRequest) -> EnvStepResponse:
        result = env.step(
            DisasterAction(
                interaction=payload.interaction,
                row=payload.row,
                col=payload.col,
            )
        )
        snapshot = _to_snapshot(result.observation, env.state)
        return EnvStepResponse(
            observation=snapshot,
            reward=result.reward,
            done=result.done,
        )

    @app.get("/state", response_model=StateResponse)
    def state() -> StateResponse:
        return StateResponse(**asdict(env.state))

    @app.get("/telemetry/current", response_model=WearableTelemetryEnvelope)
    def telemetry_current() -> WearableTelemetryEnvelope:
        observation = env._build_observation()
        snapshot = _to_snapshot(observation, env.state)
        return WearableTelemetryEnvelope(snapshot=snapshot)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("disaster_sim.api.server:app", host="0.0.0.0", port=port, reload=False)
