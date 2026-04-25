from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class WearableTelemetrySignal(BaseModel):
    """Single distress ping emitted by a wearable."""

    source_device: str = Field(default="fall-detector")
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    health: int = Field(..., ge=0, le=100)
    severity: Literal["critical", "severe"] = Field(default="critical")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryGridSnapshot(BaseModel):
    """Snapshot of the masked telemetry plane used by the dispatcher."""

    episode_id: str
    step_count: int = Field(..., ge=0)
    critical_health_threshold: int = Field(..., ge=0, le=100)
    visible_signals: int = Field(..., ge=0)
    masked_telemetry: list[list[int]]
    signals: list[WearableTelemetrySignal] = Field(default_factory=list)


class WearableTelemetryEnvelope(BaseModel):
    """Envelope representing a telemetry event payload on the API stream."""

    channel: Literal["wearable.telemetry"] = "wearable.telemetry"
    snapshot: TelemetryGridSnapshot
