"""Stateful, explainable threat scoring from detections and track context."""

from __future__ import annotations

from datetime import datetime, timezone
from math import exp
from uuid import uuid4

from common.models_schema import (
    DetectionBox, DetectionCategory, FrameMetadata, KeypointLandmark,
    ThreatAssessment, ThreatLevel,
)
from backend.threat_assessment.behavior_rules import context_scores
from backend.tracking.tracking_models import TrackingResult


class ThreatEvaluator:
    def __init__(self, *, loiter_seconds: float = 180.0,
                 normal_waiting_cameras: set[str] | None = None,
                 score_decay_seconds: float = 45.0) -> None:
        if loiter_seconds <= 0 or score_decay_seconds <= 0:
            raise ValueError("Threat timing thresholds must be positive")
        self.loiter_seconds = loiter_seconds
        self.normal_waiting_cameras = normal_waiting_cameras or set()
        self.score_decay_seconds = score_decay_seconds
        self._scores: dict[str, tuple[float, datetime]] = {}

    def evaluate(self, metadata: FrameMetadata, tracking: TrackingResult) -> list[ThreatAssessment]:
        detections = {detection.detection_id: detection for detection in metadata.detections}
        assignments = {
            assignment.detection_id: assignment for assignment in tracking.assignments
        }
        people = [
            (detections[assignment.detection_id], assignment)
            for assignment in tracking.assignments
            if assignment.detection_id in detections
        ]
        assessments: list[ThreatAssessment] = []
        for person, assignment in people:
            activity = self._pose_activity(person, metadata)
            nearby_weapon = self._nearby_weapon(person, metadata.detections)
            nearby_tool = None if nearby_weapon else self._nearby_tool(person, metadata.detections)
            scoring_category = (
                DetectionCategory.WEAPON if nearby_weapon else
                DetectionCategory.BENIGN_TOOL if nearby_tool else person.category
            )
            scores = context_scores(
                category=scoring_category,
                activity=activity,
                track_age_seconds=assignment.track_age_seconds,
                speed=assignment.speed_normalized_per_second,
                loiter_seconds=self.loiter_seconds,
                normal_waiting_zone=(
                    assignment.camera_id in self.normal_waiting_cameras
                    or assignment.location in self.normal_waiting_cameras
                ),
            )
            evidence_confidence = (
                nearby_weapon.confidence if nearby_weapon else
                nearby_tool.confidence if nearby_tool else person.confidence
            )
            scores = {name: score * evidence_confidence for name, score in scores.items()}
            assessments.append(self._assessment(
                metadata, assignment.global_track_id, assignment.camera_id, scores,
                activity, assignment.location,
                "weapon near tracked person" if nearby_weapon else
                "benign tool near tracked person" if nearby_tool else "tracked person",
            ))

        assigned_weapons: set[str] = set()
        for person, _ in people:
            weapon = self._nearby_weapon(person, metadata.detections)
            if weapon is not None:
                assigned_weapons.add(weapon.detection_id)
        for detection in metadata.detections:
            if detection.category != DetectionCategory.WEAPON or detection.detection_id in assigned_weapons:
                continue
            scores = {
                name: score * detection.confidence
                for name, score in context_scores(category=detection.category, activity=None).items()
            }
            assessments.append(self._assessment(
                metadata, None, metadata.camera_id, scores, None,
                metadata.camera_id, "weapon detected without a nearby tracked person",
                evidence_detection_id=detection.detection_id,
            ))
        return assessments

    def _assessment(self, metadata: FrameMetadata, track_id: str | None, camera_id: str,
                    components: dict[str, float], activity: str | None,
                    location: str, context: str,
                    evidence_detection_id: str | None = None) -> ThreatAssessment:
        instantaneous = min(1.0, sum(components.values()))
        previous, previous_time = self._scores.get(track_id or "", (0.0, metadata.timestamp))
        elapsed = max(0.0, (metadata.timestamp - previous_time).total_seconds())
        decayed = previous * exp(-elapsed / self.score_decay_seconds) if track_id else 0.0
        confidence = instantaneous + decayed * (1.0 - instantaneous)
        if track_id:
            self._scores[track_id] = (confidence, metadata.timestamp)
        level = (
            ThreatLevel.CRITICAL if confidence >= 0.85 else
            ThreatLevel.HIGH if confidence >= 0.60 else
            ThreatLevel.MEDIUM if confidence >= 0.35 else ThreatLevel.LOW
        )
        primitive = max(components, key=components.get) if components else (activity or "routine_activity")
        detail = ", ".join(f"{name}={value:.2f}" for name, value in components.items()) or "no elevated rule matched"
        return ThreatAssessment(
            assessment_id=str(uuid4()), global_track_id=track_id, camera_id=camera_id,
            timestamp=metadata.timestamp or datetime.now(timezone.utc),
            behavior_primitive=primitive, threat_level=level,
            confidence_score=confidence, is_anomaly=confidence >= 0.35,
            context_notes=f"{context}; location={location}; activity={activity or 'unknown'}; evidence: {detail}",
            evidence_detection_id=evidence_detection_id,
        )

    @staticmethod
    def _pose_activity(person: DetectionBox, metadata: FrameMetadata) -> str | None:
        box = person.bbox
        width, height = max(1, metadata.resolution.get("width", 1)), max(1, metadata.resolution.get("height", 1))
        left, top, right, bottom = box.x_min / width, box.y_min / height, box.x_max / width, box.y_max / height
        activities: list[str] = []
        for pose in metadata.poses:
            center = _pose_center(pose.landmarks)
            if center and left - 0.05 <= center[0] <= right + 0.05 and top - 0.05 <= center[1] <= bottom + 0.05:
                if pose.activity_primitive:
                    activities.append(pose.activity_primitive)
        if "arms_raised" in activities:
            return "arms_raised"
        if "moving_fast" in activities:
            return "moving_fast"
        return activities[0] if activities else None

    @staticmethod
    def _nearby_weapon(person: DetectionBox, detections: list[DetectionBox]) -> DetectionBox | None:
        return ThreatEvaluator._nearby_category(person, detections, DetectionCategory.WEAPON)

    @staticmethod
    def _nearby_tool(person: DetectionBox, detections: list[DetectionBox]) -> DetectionBox | None:
        return ThreatEvaluator._nearby_category(person, detections, DetectionCategory.BENIGN_TOOL)

    @staticmethod
    def _nearby_category(person: DetectionBox, detections: list[DetectionBox],
                         category: DetectionCategory) -> DetectionBox | None:
        person_box = person.bbox
        center_x = (person_box.x_min + person_box.x_max) / 2
        center_y = (person_box.y_min + person_box.y_max) / 2
        scale = max(person_box.x_max - person_box.x_min, person_box.y_max - person_box.y_min, 1)
        weapons: list[tuple[float, DetectionBox]] = []
        for detection in detections:
            if detection.category != category:
                continue
            box = detection.bbox
            weapon_x = (box.x_min + box.x_max) / 2
            weapon_y = (box.y_min + box.y_max) / 2
            distance = ((weapon_x - center_x) ** 2 + (weapon_y - center_y) ** 2) ** 0.5 / scale
            if distance <= 0.8:
                weapons.append((distance, detection))
        return min(weapons, key=lambda item: item[0])[1] if weapons else None


def _pose_center(landmarks: list[KeypointLandmark]) -> tuple[float, float] | None:
    by_name = {landmark.name: landmark for landmark in landmarks}
    left, right = by_name.get("left_hip"), by_name.get("right_hip")
    if left and right:
        return (left.x + right.x) / 2, (left.y + right.y) / 2
    left, right = by_name.get("left_shoulder"), by_name.get("right_shoulder")
    if left and right:
        return (left.x + right.x) / 2, (left.y + right.y) / 2
    return None
