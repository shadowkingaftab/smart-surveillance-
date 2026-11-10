"""Typed metadata exchanged between camera ingestion and later pipeline stages."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DetectionCategory(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    WEAPON = "weapon"
    BENIGN_TOOL = "benign_tool"
    UNATTENDED_BAGGAGE = "unattended_baggage"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectionBox(BaseModel):
    detection_id: str
    label: str
    category: DetectionCategory = DetectionCategory.UNKNOWN
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    local_track_id: int | None = None


class KeypointLandmark(BaseModel):
    landmark_id: int
    name: str
    x: float
    y: float
    z: float = 0
    visibility: float = Field(default=1, ge=0, le=1)


class PoseLandmarks(BaseModel):
    person_track_id: int | None = None
    landmarks: list[KeypointLandmark] = Field(default_factory=list)
    activity_primitive: str | None = None


class FrameMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    edge_node_id: str
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolution: dict[str, int]
    fps: float = Field(ge=0)
    inference_latency_ms: float = Field(default=0, ge=0)
    detections: list[DetectionBox] = Field(default_factory=list)
    poses: list[PoseLandmarks] = Field(default_factory=list)
