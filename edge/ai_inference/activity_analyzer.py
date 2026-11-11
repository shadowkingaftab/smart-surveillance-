"""MediaPipe PoseLandmarker inference and lightweight pose activity primitives."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from common.models_schema import KeypointLandmark, PoseLandmarks


class ActivityAnalyzer:
    """Extracts MediaPipe pose landmarks and simple per-frame activity cues.

    Track IDs are short-lived hip/shoulder-position associations local to this
    camera, not biometric identities or cross-camera re-identification.
    """

    def __init__(self, model_path: str | Path, *, num_poses: int = 4,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"MediaPipe pose model not found: {model_path}")
        self._timestamp_ms = 0
        self._next_track_id = 1
        self._tracks: dict[int, tuple[float, float, int]] = {}
        self._track_speeds: dict[int, float] = {}
        self._assigned_this_frame: set[int] = set()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def analyze(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[PoseLandmarks]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty HxWx3 BGR image")
        if timestamp_ms is None:
            self._timestamp_ms += 33
            timestamp_ms = self._timestamp_ms
        elif timestamp_ms <= self._timestamp_ms:
            raise ValueError("timestamp_ms must increase for each frame")
        else:
            self._timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        poses: list[PoseLandmarks] = []
        self._assigned_this_frame.clear()
        for landmarks in result.pose_landmarks:
            points = [KeypointLandmark(
                landmark_id=index,
                name=mp_pose_name(index),
                x=float(point.x), y=float(point.y), z=float(point.z),
                visibility=float(point.visibility or 0.0),
            ) for index, point in enumerate(landmarks)]
            track_id = self._assign_track(points, timestamp_ms)
            poses.append(PoseLandmarks(
                person_track_id=track_id, landmarks=points,
                activity_primitive=_activity(points, self._track_speeds.get(track_id, 0.0)),
            ))
        return poses

    def _assign_track(self, points: list[KeypointLandmark], timestamp_ms: int) -> int | None:
        joints = {point.name: point for point in points}
        left, right = joints.get("left_hip"), joints.get("right_hip")
        if not left or not right or min(left.visibility, right.visibility) < 0.3:
            left, right = joints.get("left_shoulder"), joints.get("right_shoulder")
        if not left or not right or min(left.visibility, right.visibility) < 0.3:
            return None
        center = ((left.x + right.x) / 2, (left.y + right.y) / 2)
        matches = [
            (float(np.hypot(center[0] - x, center[1] - y)), track_id,
             min(0.4, 0.12 + max(0, timestamp_ms - last_seen) / 1000 * 0.75))
            for track_id, (x, y, last_seen) in self._tracks.items()
            if timestamp_ms - last_seen <= 2000 and track_id not in self._assigned_this_frame
        ]
        matches = [(distance, track_id) for distance, track_id, limit in matches if distance <= limit]
        if matches:
            _, track_id = min(matches)
        else:
            track_id = self._next_track_id
            self._next_track_id += 1
        previous = self._tracks.get(track_id)
        if previous and timestamp_ms > previous[2]:
            distance = float(np.hypot(center[0] - previous[0], center[1] - previous[1]))
            self._track_speeds[track_id] = distance * 1000 / (timestamp_ms - previous[2])
        else:
            self._track_speeds[track_id] = 0.0
        self._tracks[track_id] = (center[0], center[1], timestamp_ms)
        self._assigned_this_frame.add(track_id)
        self._tracks = {
            key: value for key, value in self._tracks.items()
            if timestamp_ms - value[2] <= 2000
        }
        self._track_speeds = {key: value for key, value in self._track_speeds.items() if key in self._tracks}
        return track_id

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> ActivityAnalyzer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_POSE_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)


def mp_pose_name(index: int) -> str:
    return _POSE_NAMES[index] if index < len(_POSE_NAMES) else f"landmark_{index}"


def _activity(points: list[KeypointLandmark], normalized_speed_per_second: float = 0.0) -> str:
    landmarks = {point.name: point for point in points}
    left_wrist, right_wrist = landmarks.get("left_wrist"), landmarks.get("right_wrist")
    left_shoulder, right_shoulder = landmarks.get("left_shoulder"), landmarks.get("right_shoulder")
    if all(point is not None and point.visibility >= 0.5 for point in
           (left_wrist, right_wrist, left_shoulder, right_shoulder)):
        assert left_wrist and right_wrist and left_shoulder and right_shoulder
        if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
            return "arms_raised"
    if normalized_speed_per_second >= 0.6:
        return "moving_fast"
    return "pose_detected"
