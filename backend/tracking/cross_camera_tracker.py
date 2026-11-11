"""Appearance-assisted cross-camera tracker with per-camera Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np

from common.models_schema import CrossCameraTrack, DetectionBox, DetectionCategory, FrameMetadata
from backend.tracking.tracking_models import TrackAssignment, TrackingResult


class _KalmanBox:
    """Constant-velocity filter over normalized center/size coordinates."""

    def __init__(self, measurement: np.ndarray, timestamp: datetime) -> None:
        self.state = np.zeros(8, dtype=np.float64)
        self.state[:4] = measurement
        self.covariance = np.eye(8, dtype=np.float64) * 0.1
        self.timestamp = timestamp
        self._measurement = np.eye(4, 8, dtype=np.float64)
        self._measurement_noise = np.eye(4, dtype=np.float64) * 0.015

    def predict(self, timestamp: datetime) -> None:
        dt = max(0.0, min((timestamp - self.timestamp).total_seconds(), 5.0))
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4) * dt
        process_noise = np.eye(8, dtype=np.float64) * max(dt, 0.01) * 0.01
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.timestamp = timestamp

    def update(self, measurement: np.ndarray) -> None:
        residual = measurement - self._measurement @ self.state
        innovation_covariance = self._measurement @ self.covariance @ self._measurement.T + self._measurement_noise
        gain = self.covariance @ self._measurement.T @ np.linalg.inv(innovation_covariance)
        self.state += gain @ residual
        self.covariance = (np.eye(8) - gain @ self._measurement) @ self.covariance

    def distance(self, measurement: np.ndarray) -> float:
        return float(np.linalg.norm(self.state[:2] - measurement[:2]))


@dataclass
class _Track:
    global_track_id: str
    first_seen: datetime
    last_seen: datetime
    last_camera_id: str
    last_detection: DetectionBox
    embedding: np.ndarray | None
    location: str
    local_track_mappings: dict[str, int] = field(default_factory=dict)
    camera_ids: list[str] = field(default_factory=list)
    filters: dict[str, _KalmanBox] = field(default_factory=dict)


class CrossCameraTracker:
    """Associate per-frame person detections using motion and appearance cues.

    Cross-camera associations require embeddings. Empty transition rules allow
    all camera pairs subject to appearance and travel-time gates.
    """

    def __init__(self, *, max_track_age_seconds: float = 8.0,
                 cross_camera_max_seconds: float = 90.0,
                 appearance_similarity_threshold: float = 0.78,
                 camera_transitions: dict[str, list[str]] | None = None,
                 camera_zones: dict[str, str] | None = None) -> None:
        if max_track_age_seconds <= 0 or cross_camera_max_seconds <= 0:
            raise ValueError("Track age and camera travel window must be positive")
        if not -1 <= appearance_similarity_threshold <= 1:
            raise ValueError("appearance similarity threshold must be between -1 and 1")
        self.max_track_age_seconds = max_track_age_seconds
        self.cross_camera_max_seconds = cross_camera_max_seconds
        self.appearance_similarity_threshold = appearance_similarity_threshold
        self.camera_transitions = camera_transitions or {}
        self.camera_zones = camera_zones or {}
        self._tracks: dict[str, _Track] = {}

    def update(self, metadata: FrameMetadata) -> TrackingResult:
        timestamp = metadata.timestamp
        retention_seconds = max(self.max_track_age_seconds, self.cross_camera_max_seconds)
        width = max(1, metadata.resolution.get("width", 1))
        height = max(1, metadata.resolution.get("height", 1))
        detections = [d for d in metadata.detections if d.category == DetectionCategory.PERSON]
        for track in list(self._tracks.values()):
            if (timestamp - track.last_seen).total_seconds() > retention_seconds:
                del self._tracks[track.global_track_id]
                continue
            kalman = track.filters.get(metadata.camera_id)
            if kalman:
                kalman.predict(timestamp)

        pairs: list[tuple[float, str, int]] = []
        for detection_index, detection in enumerate(detections):
            measurement = _measurement(detection, width, height)
            embedding = _normalized_embedding(detection.appearance_embedding)
            for track_id, track in self._tracks.items():
                score = self._match_score(
                    track, metadata.camera_id, detection.local_track_id, timestamp, measurement, embedding,
                )
                if score is not None:
                    pairs.append((score, track_id, detection_index))
        pairs.sort(reverse=True)
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assigned: dict[int, str] = {}
        for _, track_id, detection_index in pairs:
            if track_id not in matched_tracks and detection_index not in matched_detections:
                assigned[detection_index] = track_id
                matched_tracks.add(track_id)
                matched_detections.add(detection_index)

        result = TrackingResult()
        for index, detection in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self._new_track(detection, metadata, timestamp, width, height)
            else:
                self._update_track(self._tracks[track_id], detection, metadata, timestamp, width, height)
            track = self._tracks[track_id]
            kalman = track.filters[metadata.camera_id]
            if detection.local_track_id is not None:
                track.local_track_mappings[metadata.camera_id] = detection.local_track_id
            result.assignments.append(TrackAssignment(
                detection_id=detection.detection_id,
                global_track_id=track_id,
                camera_id=metadata.camera_id,
                local_track_id=detection.local_track_id,
                track_age_seconds=max(0.0, (timestamp - track.first_seen).total_seconds()),
                speed_normalized_per_second=float(np.linalg.norm(kalman.state[4:6])),
                location=self.camera_zones.get(metadata.camera_id, metadata.camera_id),
            ))
        result.active_tracks = [self._snapshot(track) for track in self._tracks.values()]
        return result

    def _match_score(self, track: _Track, camera_id: str, local_track_id: int | None, timestamp: datetime,
                     measurement: np.ndarray, embedding: np.ndarray | None) -> float | None:
        elapsed = (timestamp - track.last_seen).total_seconds()
        if elapsed < 0:
            return None
        if track.last_camera_id != camera_id:
            if elapsed > self.cross_camera_max_seconds or embedding is None or track.embedding is None:
                return None
            allowed = self.camera_transitions.get(track.last_camera_id)
            if self.camera_transitions and camera_id not in (allowed or []):
                return None
            similarity = _cosine_similarity(track.embedding, embedding)
            return similarity if similarity >= self.appearance_similarity_threshold else None
        if elapsed > self.max_track_age_seconds:
            return None

        mapped_local_id = track.local_track_mappings.get(camera_id)
        if local_track_id is not None and mapped_local_id == local_track_id:
            return 1.0
        kalman = track.filters.get(camera_id)
        if not kalman:
            return None
        distance = kalman.distance(measurement)
        if distance > min(0.35, 0.12 + elapsed * 0.5):
            return None
        motion_score = 1.0 - distance / 0.35
        if embedding is None or track.embedding is None:
            return motion_score * 0.8
        similarity = _cosine_similarity(track.embedding, embedding)
        if similarity < 0.2:
            return None
        return motion_score * 0.55 + max(0.0, similarity) * 0.45

    def _new_track(self, detection: DetectionBox, metadata: FrameMetadata,
                   timestamp: datetime, width: int, height: int) -> str:
        track_id = str(uuid4())
        track = _Track(
            global_track_id=track_id, first_seen=timestamp, last_seen=timestamp,
            last_camera_id=metadata.camera_id, last_detection=detection,
            embedding=_normalized_embedding(detection.appearance_embedding),
            location=self.camera_zones.get(metadata.camera_id, metadata.camera_id),
            camera_ids=[metadata.camera_id],
        )
        track.filters[metadata.camera_id] = _KalmanBox(_measurement(detection, width, height), timestamp)
        self._tracks[track_id] = track
        return track_id

    def _update_track(self, track: _Track, detection: DetectionBox, metadata: FrameMetadata,
                      timestamp: datetime, width: int, height: int) -> None:
        camera_id = metadata.camera_id
        measurement = _measurement(detection, width, height)
        if camera_id not in track.filters:
            # Coordinates from different views must not share a motion state.
            track.filters[camera_id] = _KalmanBox(measurement, timestamp)
        else:
            track.filters[camera_id].update(measurement)
        current_embedding = _normalized_embedding(detection.appearance_embedding)
        if current_embedding is not None:
            if track.embedding is None or track.embedding.shape != current_embedding.shape:
                track.embedding = current_embedding
            else:
                track.embedding = _normalize(track.embedding * 0.8 + current_embedding * 0.2)
        track.last_seen = timestamp
        track.last_camera_id = camera_id
        track.last_detection = detection
        track.location = self.camera_zones.get(camera_id, camera_id)
        if camera_id not in track.camera_ids:
            track.camera_ids.append(camera_id)

    @staticmethod
    def _snapshot(track: _Track) -> CrossCameraTrack:
        return CrossCameraTrack(
            global_track_id=track.global_track_id,
            active_camera_ids=track.camera_ids.copy(),
            local_track_mappings=track.local_track_mappings.copy(),
            feature_embedding=track.embedding.tolist() if track.embedding is not None else None,
            last_known_location=track.location, first_seen=track.first_seen, last_seen=track.last_seen,
        )


def _measurement(detection: DetectionBox, width: int, height: int) -> np.ndarray:
    box = detection.bbox
    x1, y1 = box.x_min / width, box.y_min / height
    x2, y2 = box.x_max / width, box.y_max / height
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


def _normalized_embedding(values: list[float] | None) -> np.ndarray | None:
    if not values:
        return None
    embedding = np.asarray(values, dtype=np.float64)
    if not np.isfinite(embedding).all():
        return None
    return _normalize(embedding)


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 1e-12 else embedding


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return -1.0
    return float(np.clip(np.dot(first, second), -1.0, 1.0))
