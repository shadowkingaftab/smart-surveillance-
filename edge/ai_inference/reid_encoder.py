"""Optional ONNX person appearance encoder for cross-camera re-identification."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from common.models_schema import DetectionBox, DetectionCategory


class ReIdEncoder:
    """Runs a person ReID ONNX model on person crops and L2-normalizes output."""

    def __init__(self, model_path: str | Path, *, providers: list[str] | None = None,
                 input_height: int = 256, input_width: int = 128) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"ReID ONNX model not found: {model_path}")
        available = ort.get_available_providers()
        selected = providers or ["CPUExecutionProvider"]
        if any(provider not in available for provider in selected):
            raise ValueError(f"Requested unavailable ONNX providers: {selected}; available: {available}")
        self.session = ort.InferenceSession(str(model_path), providers=selected)
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.height = int(shape[2]) if isinstance(shape[2], int) else input_height
        self.width = int(shape[3]) if isinstance(shape[3], int) else input_width
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]

    def encode(self, frame: np.ndarray, detection: DetectionBox) -> list[float] | None:
        if detection.category != DetectionCategory.PERSON:
            return None
        height, width = frame.shape[:2]
        box = detection.bbox
        x1, y1 = max(0, int(box.x_min)), max(0, int(box.y_min))
        x2, y2 = min(width, int(np.ceil(box.x_max))), min(height, int(np.ceil(box.y_max)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop, (self.width, self.height)).astype(np.float32) / 255.0
        tensor = (resized - self.mean) / self.std
        tensor = np.transpose(tensor, (2, 0, 1))[None].astype(np.float32)
        embedding = np.asarray(self.session.run(None, {self.input.name: tensor})[0]).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(embedding).all() or norm <= 1e-12:
            return None
        return (embedding / norm).astype(float).tolist()
