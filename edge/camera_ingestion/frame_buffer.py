"""Thread-safe, bounded frame buffer that favors recent camera frames."""

from collections import deque
from threading import Condition
from typing import Any


class FrameBuffer:
    def __init__(self, max_size: int = 2) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._frames: deque[Any] = deque(maxlen=max_size)
        self._condition = Condition()

    def put(self, frame: Any) -> None:
        with self._condition:
            self._frames.append(frame)
            self._condition.notify()

    def get(self, timeout: float | None = None) -> Any | None:
        with self._condition:
            if not self._frames:
                self._condition.wait(timeout)
            return self._frames.popleft() if self._frames else None

    def get_latest(self) -> Any | None:
        with self._condition:
            if not self._frames:
                return None
            frame = self._frames[-1]
            self._frames.clear()
            return frame
