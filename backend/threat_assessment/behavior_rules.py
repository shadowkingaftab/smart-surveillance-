"""Auditable rule weights for contextual threat assessment heuristics."""

from common.models_schema import DetectionCategory


def context_scores(*, category: DetectionCategory, activity: str | None,
                   track_age_seconds: float = 0.0, speed: float = 0.0,
                   loiter_seconds: float = 180.0, normal_waiting_zone: bool = False) -> dict[str, float]:
    scores: dict[str, float] = {}
    if category == DetectionCategory.WEAPON:
        scores["weapon_detected"] = 0.55
        if activity == "arms_raised":
            scores["possible_brandishing"] = 0.30
    elif category == DetectionCategory.BENIGN_TOOL:
        scores["tool_context"] = 0.08
        if activity == "arms_raised":
            scores["tool_raised"] = 0.18
    if activity == "arms_raised" and category not in {
        DetectionCategory.WEAPON, DetectionCategory.BENIGN_TOOL,
    }:
        scores["arms_raised"] = 0.18
    if activity == "moving_fast" or speed >= 0.6:
        scores["rapid_movement"] = 0.18
    if category == DetectionCategory.PERSON and track_age_seconds >= loiter_seconds and speed < 0.08:
        scores["prolonged_low_movement"] = 0.55 if not normal_waiting_zone else 0.12
    if normal_waiting_zone and category == DetectionCategory.PERSON:
        scores["normal_waiting_context"] = 0.0
    return scores
