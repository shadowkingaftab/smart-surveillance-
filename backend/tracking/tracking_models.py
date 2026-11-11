"""Central tracker output types."""

from dataclasses import dataclass, field

from common.models_schema import CrossCameraTrack


@dataclass(frozen=True)
class TrackAssignment:
    detection_id: str
    global_track_id: str
    camera_id: str
    local_track_id: int | None
    track_age_seconds: float
    speed_normalized_per_second: float
    location: str


@dataclass
class TrackingResult:
    assignments: list[TrackAssignment] = field(default_factory=list)
    active_tracks: list[CrossCameraTrack] = field(default_factory=list)
