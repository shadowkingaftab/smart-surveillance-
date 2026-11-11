"""Central service boundary for edge metadata, global tracks, and assessments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.models_schema import FrameMetadata, ThreatAlertPayload, ThreatAssessment, ThreatLevel
from backend.alerting.alert_manager import AlertManager
from backend.alerting.notifier import AlertNotifier
from backend.threat_assessment.threat_evaluator import ThreatEvaluator
from backend.tracking.cross_camera_tracker import CrossCameraTracker
from backend.tracking.tracking_models import TrackingResult


@dataclass
class CentralProcessingResult:
    tracking: TrackingResult
    assessments: list[ThreatAssessment]
    alerts: list[ThreatAlertPayload]


class CentralMetadataProcessor:
    def __init__(self, tracker: CrossCameraTracker, evaluator: ThreatEvaluator,
                 alert_manager: AlertManager | None = None) -> None:
        self.tracker = tracker
        self.evaluator = evaluator
        self.alert_manager = alert_manager

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CentralMetadataProcessor:
        tracking = config.get("tracking", {})
        threat = config.get("threat_assessment", {})
        alerting = config.get("alerting", {})
        return cls(
            CrossCameraTracker(
                max_track_age_seconds=tracking.get("max_track_age_seconds", 8.0),
                cross_camera_max_seconds=tracking.get("cross_camera_max_seconds", 90.0),
                appearance_similarity_threshold=tracking.get("appearance_similarity_threshold", 0.78),
                camera_transitions=tracking.get("camera_transitions", {}),
                camera_zones=tracking.get("camera_zones", {}),
            ),
            ThreatEvaluator(
                loiter_seconds=threat.get("loiter_seconds", 180.0),
                normal_waiting_cameras=set(threat.get("normal_waiting_cameras", [])),
                score_decay_seconds=threat.get("score_decay_seconds", 45.0),
            ),
            AlertManager(
                minimum_confirmations=alerting.get("minimum_confirmations", 3),
                confirmation_window_seconds=alerting.get("confirmation_window_seconds", 5.0),
                minimum_average_confidence=alerting.get("minimum_average_confidence", 0.72),
                minimum_threat_level=ThreatLevel(alerting.get("minimum_threat_level", "HIGH")),
                alert_cooldown_seconds=alerting.get("alert_cooldown_seconds", 60.0),
            ),
        )

    def process(self, metadata: FrameMetadata) -> CentralProcessingResult:
        tracking = self.tracker.update(metadata)
        assessments = self.evaluator.evaluate(metadata, tracking)
        alerts = self.alert_manager.process(metadata, assessments, tracking) if self.alert_manager else []
        return CentralProcessingResult(tracking=tracking, assessments=assessments, alerts=alerts)

    async def process_and_dispatch(self, metadata: FrameMetadata,
                                   notifier: AlertNotifier) -> CentralProcessingResult:
        result = self.process(metadata)
        for alert in result.alerts:
            await notifier.dispatch(alert)
        return result
