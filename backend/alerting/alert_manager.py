"""Temporal confirmation gate that reduces one-frame false alerts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from common.models_schema import (
    BoundingBox, DetectionCategory, FrameMetadata, ThreatAlertPayload,
    ThreatAssessment, ThreatLevel,
)
from backend.tracking.tracking_models import TrackingResult


@dataclass(frozen=True)
class _Observation:
    frame_id: str
    assessment: ThreatAssessment


class AlertManager:
    def __init__(self, *, minimum_confirmations: int = 3,
                 confirmation_window_seconds: float = 5.0,
                 minimum_average_confidence: float = 0.72,
                 minimum_threat_level: ThreatLevel = ThreatLevel.HIGH,
                 alert_cooldown_seconds: float = 60.0) -> None:
        if minimum_confirmations < 2 or confirmation_window_seconds <= 0 or alert_cooldown_seconds < 0:
            raise ValueError("Invalid temporal confirmation settings")
        if not 0 <= minimum_average_confidence <= 1:
            raise ValueError("minimum_average_confidence must be between 0 and 1")
        self.minimum_confirmations = minimum_confirmations
        self.confirmation_window = timedelta(seconds=confirmation_window_seconds)
        self.minimum_average_confidence = minimum_average_confidence
        self.minimum_threat_level = minimum_threat_level
        self.alert_cooldown = timedelta(seconds=alert_cooldown_seconds)
        self._observations: dict[str, deque[_Observation]] = {}
        self._last_alerted: dict[str, datetime] = {}

    def process(self, metadata: FrameMetadata, assessments: list[ThreatAssessment],
                tracking: TrackingResult) -> list[ThreatAlertPayload]:
        alerts: list[ThreatAlertPayload] = []
        expired_cutoff = metadata.timestamp - self.confirmation_window - self.alert_cooldown
        for key in [key for key, items in self._observations.items()
                    if not items or items[-1].assessment.timestamp < expired_cutoff]:
            self._observations.pop(key, None)
        for key in [key for key, sent_at in self._last_alerted.items()
                    if sent_at < metadata.timestamp - timedelta(hours=1)]:
            self._last_alerted.pop(key, None)
        for assessment in assessments:
            key = self._key(metadata, assessment)
            observations = self._observations.setdefault(key, deque())
            cutoff = assessment.timestamp - self.confirmation_window
            while observations and observations[0].assessment.timestamp < cutoff:
                observations.popleft()
            if any(item.frame_id == metadata.frame_id for item in observations):
                continue
            observations.append(_Observation(metadata.frame_id, assessment))
            while len(observations) > self.minimum_confirmations * 3:
                observations.popleft()
            recent = list(observations)[-self.minimum_confirmations:]
            if len(recent) < self.minimum_confirmations:
                continue
            average = sum(item.assessment.confidence_score for item in recent) / len(recent)
            if average < self.minimum_average_confidence:
                continue
            if any(_level_rank(item.assessment.threat_level) < _level_rank(self.minimum_threat_level)
                   for item in recent):
                continue
            previous_alert = self._last_alerted.get(key)
            if previous_alert and assessment.timestamp - previous_alert < self.alert_cooldown:
                continue
            self._last_alerted[key] = assessment.timestamp
            alerts.append(self._payload(metadata, assessment, tracking))
        return alerts

    def _key(self, metadata: FrameMetadata, assessment: ThreatAssessment) -> str:
        if assessment.global_track_id:
            return assessment.global_track_id
        detection = next((item for item in metadata.detections
                          if item.detection_id == assessment.evidence_detection_id), None)
        if detection is None:
            detection = next((item for item in metadata.detections
                              if item.category == DetectionCategory.WEAPON), None)
        if detection:
            center_x = (detection.bbox.x_min + detection.bbox.x_max) / 2
            center_y = (detection.bbox.y_min + detection.bbox.y_max) / 2
            width = max(1, metadata.resolution.get("width", 1))
            height = max(1, metadata.resolution.get("height", 1))
            return f"{metadata.camera_id}:{assessment.behavior_primitive}:{int(center_x / width * 10)}:{int(center_y / height * 10)}"
        return f"{metadata.camera_id}:{assessment.behavior_primitive}"

    @staticmethod
    def _payload(metadata: FrameMetadata, assessment: ThreatAssessment,
                 tracking: TrackingResult) -> ThreatAlertPayload:
        assignment = next((item for item in tracking.assignments
                           if item.global_track_id == assessment.global_track_id
                           and item.camera_id == assessment.camera_id), None)
        detection = next((item for item in metadata.detections
                          if assignment and item.detection_id == assignment.detection_id), None)
        if detection is None and assessment.evidence_detection_id:
            detection = next((item for item in metadata.detections
                              if item.detection_id == assessment.evidence_detection_id), None)
        if detection is None and assessment.global_track_id is None:
            detection = next((item for item in metadata.detections
                              if item.category == DetectionCategory.WEAPON), None)
        return ThreatAlertPayload(
            alert_id=str(uuid4()), timestamp=assessment.timestamp,
            location_name=assignment.location if assignment else assessment.camera_id,
            camera_id=assessment.camera_id, threat_assessment=assessment,
            bounding_box_snapshot=BoundingBox(**detection.bbox.model_dump()) if detection else None,
            false_alarm_mitigated=True,
        )


def _level_rank(level: ThreatLevel) -> int:
    return {
        ThreatLevel.LOW: 0, ThreatLevel.MEDIUM: 1,
        ThreatLevel.HIGH: 2, ThreatLevel.CRITICAL: 3,
    }[level]
