"""Run inference independently for one camera so slow feeds do not block others."""

from __future__ import annotations

import logging
import threading

from edge.ai_inference.pipeline import EdgeInferencePipeline
from edge.camera_ingestion.camera_processor import CameraProcessor

logger = logging.getLogger(__name__)


class CameraInferenceWorker:
    def __init__(self, camera: CameraProcessor, pipeline: EdgeInferencePipeline) -> None:
        self.camera = camera
        self.pipeline = pipeline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_capture: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"inference-{self.camera.camera_id}", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            captured = self.camera.read_latest()
            if captured is None:
                self._stop.wait(0.01)
                continue
            captured_at, frame = captured
            fps = (
                1.0 / (captured_at - self._last_capture)
                if self._last_capture is not None and captured_at > self._last_capture else 0.0
            )
            self._last_capture = captured_at
            try:
                metadata = self.pipeline.process(
                    frame, self.camera.camera_id, fps=fps, captured_at=captured_at,
                )
                logger.info(
                    "camera=%s detections=%d poses=%d inference_ms=%.1f",
                    self.camera.camera_id, len(metadata.detections), len(metadata.poses),
                    metadata.inference_latency_ms,
                )
            except Exception:
                logger.exception("Inference failed for camera %s", self.camera.camera_id)
