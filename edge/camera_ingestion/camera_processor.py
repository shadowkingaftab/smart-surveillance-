"""Background OpenCV camera capture with reconnect and bounded buffering."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2

from edge.camera_ingestion.frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


class CameraProcessor:
    def __init__(self, camera_id: str, source: int | str | list[int | str] | tuple[int | str, ...], buffer_size: int = 2,
                 reconnect_delay: float = 2.0) -> None:
        self.camera_id = camera_id
        self.sources = [source] if isinstance(source, (str, int)) else list(source)
        if not self.sources:
            raise ValueError("At least one camera source is required")
        if reconnect_delay <= 0:
            raise ValueError("reconnect_delay must be greater than zero")
        self.frames = FrameBuffer(buffer_size)
        self.reconnect_delay = reconnect_delay
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()

    def read(self, timeout: float | None = 1.0) -> Any | None:
        return self.frames.get(timeout)

    def read_latest(self) -> Any | None:
        return self.frames.get_latest()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._capture:
            self._capture.release()
        if self._thread:
            self._thread.join(timeout)

    def _capture_loop(self) -> None:
        source_index = 0
        while not self._stop.is_set():
            source = self.sources[source_index]
            capture = cv2.VideoCapture(source)
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                logger.warning("Camera %s source %s unavailable", self.camera_id, source)
                source_index = (source_index + 1) % len(self.sources)
                self._stop.wait(self.reconnect_delay)
                continue
            try:
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        logger.warning("Camera %s source %s failed; switching source", self.camera_id, source)
                        source_index = (source_index + 1) % len(self.sources)
                        break
                    self.frames.put((time.time(), frame))
            finally:
                capture.release()
                self._capture = None
            if not self._stop.is_set() and len(self.sources) == 1:
                self._stop.wait(self.reconnect_delay)
