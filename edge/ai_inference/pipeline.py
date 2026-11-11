"""Combine local object detection and pose inference into a frame contract."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from common.models_schema import FrameMetadata
from edge.ai_inference.activity_analyzer import ActivityAnalyzer
from edge.ai_inference.detector import YoloDetector
from edge.ai_inference.reid_encoder import ReIdEncoder


class EdgeInferencePipeline:
    def __init__(self, detector: YoloDetector, activity_analyzer: ActivityAnalyzer,
                 edge_node_id: str, reid_encoder: ReIdEncoder | None = None) -> None:
        self.detector = detector
        self.activity_analyzer = activity_analyzer
        self.edge_node_id = edge_node_id
        self.reid_encoder = reid_encoder
        self._last_timestamp_ms = 0

    def process(self, frame: np.ndarray, camera_id: str, *, fps: float = 0.0,
                timestamp_ms: int | None = None, captured_at: float | None = None) -> FrameMetadata:
        if timestamp_ms is None:
            timestamp_ms = max(self._last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must increase for each frame")
        self._last_timestamp_ms = timestamp_ms
        started = time.perf_counter()
        detections = self.detector.infer(frame)
        if self.reid_encoder:
            detections = [detection.model_copy(update={
                "appearance_embedding": self.reid_encoder.encode(frame, detection),
            }) if detection.category.value == "person" else detection for detection in detections]
        poses = self.activity_analyzer.analyze(frame, timestamp_ms)
        height, width = frame.shape[:2]
        return FrameMetadata(
            frame_id=str(uuid4()), edge_node_id=self.edge_node_id, camera_id=camera_id,
            timestamp=(datetime.fromtimestamp(captured_at, timezone.utc) if captured_at is not None
                       else datetime.now(timezone.utc)),
            resolution={"width": width, "height": height}, fps=fps,
            inference_latency_ms=(time.perf_counter() - started) * 1000,
            detections=detections, poses=poses,
        )

    def close(self) -> None:
        self.activity_analyzer.close()

    def __enter__(self) -> EdgeInferencePipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
