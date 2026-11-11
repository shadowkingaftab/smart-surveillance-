"""Thread-safe, bounded frame buffer that favors recent camera frames."""

from collections import deque
from threading import Condition
from typing import Generic, TypeVar

T = TypeVar("T")


class FrameBuffer(Generic[T]):
    def __init__(self, max_size: int = 2) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._frames: deque[T] = deque(maxlen=max_size)
        self._condition = Condition()

    def put(self, frame: T) -> None:
        with self._condition:
            self._frames.append(frame)
            self._condition.notify()

    def get(self, timeout: float | None = None) -> T | None:
        with self._condition:
            self._condition.wait_for(lambda: bool(self._frames), timeout)
            return self._frames.popleft() if self._frames else None

    def get_latest(self) -> T | None:
        with self._condition:
            if not self._frames:
                return None
            frame = self._frames[-1]
            self._frames.clear()
            return frame
