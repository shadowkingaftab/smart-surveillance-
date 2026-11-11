"""YOLO ONNX Runtime inference for edge camera frames.

Supports raw YOLOv5/v8 exports and end-to-end exports with NMS. Export the
chosen weights to ONNX first; custom weights are needed for weapon classes.
"""

from __future__ import annotations

import json
import ast
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime as ort

from common.models_schema import BoundingBox, DetectionBox, DetectionCategory


_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck",
}
_CATEGORY_BY_NAME = {
    "person": DetectionCategory.PERSON,
    "bicycle": DetectionCategory.VEHICLE,
    "car": DetectionCategory.VEHICLE,
    "motorcycle": DetectionCategory.VEHICLE,
    "bus": DetectionCategory.VEHICLE,
    "truck": DetectionCategory.VEHICLE,
    "knife": DetectionCategory.WEAPON,
    "gun": DetectionCategory.WEAPON,
    "pistol": DetectionCategory.WEAPON,
    "rifle": DetectionCategory.WEAPON,
    "weapon": DetectionCategory.WEAPON,
    "tool": DetectionCategory.BENIGN_TOOL,
}


class YoloDetector:
    """Runs a local YOLO ONNX model using CPU or an available ORT provider."""

    def __init__(self, model_path: str | Path, *, input_size: int = 640, confidence_threshold: float = 0.35,
                 iou_threshold: float = 0.45, output_format: str = "auto",
                 class_names: dict[int, str] | None = None,
                 providers: list[str] | None = None) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {model_path}")
        if input_size < 32 or not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("Invalid input size or detection thresholds")
        if output_format not in {"auto", "yolov5", "yolov8", "yolo_nms"}:
            raise ValueError("output_format must be auto, yolov5, yolov8, or yolo_nms")
        available = ort.get_available_providers()
        selected = providers or ["CPUExecutionProvider"]
        if any(provider not in available for provider in selected):
            raise ValueError(f"Requested unavailable ONNX providers: {selected}; available: {available}")
        self.session = ort.InferenceSession(str(model_path), providers=selected)
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else input_size
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else input_size
        self.class_names = class_names or self._model_class_names() or _COCO_NAMES
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.output_format = output_format

    def _model_class_names(self) -> dict[int, str] | None:
        raw_names = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw_names:
            return None
        try:
            names = json.loads(raw_names)
        except (ValueError, TypeError):
            try:
                names = ast.literal_eval(raw_names)
            except (ValueError, SyntaxError):
                return None
        if isinstance(names, list):
            return dict(enumerate(str(name) for name in names))
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return None

    def infer(self, frame: np.ndarray) -> list[DetectionBox]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a non-empty HxWx3 BGR image")
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self.session.run(None, {self.input.name: input_tensor})
        if not outputs:
            return []
        rows = self._decode(np.asarray(outputs[0]))
        candidates: list[tuple[list[float], float, int]] = []
        for row in rows:
            is_nms_output = self._is_nms_output(rows)
            if len(row) < (6 if is_nms_output else 5):
                continue
            if is_nms_output:
                x1, y1, x2, y2, score, class_id = row[:6]
            else:
                is_v5 = self.output_format == "yolov5" or (
                    self.output_format == "auto" and len(row) == 85
                )
                if is_v5:
                    class_scores = row[5:]
                    class_id = int(np.argmax(class_scores))
                    score = float(row[4] * class_scores[class_id])
                else:
                    class_scores = row[4:]
                    class_id = int(np.argmax(class_scores))
                    score = float(class_scores[class_id])
                cx, cy, width, height = row[:4]
                x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
            if score < self.confidence_threshold:
                continue
            x1, x2 = (x1 - pad_x) / scale, (x2 - pad_x) / scale
            y1, y2 = (y1 - pad_y) / scale, (y2 - pad_y) / scale
            h, w = frame.shape[:2]
            box = [max(0.0, min(float(w), float(x1))), max(0.0, min(float(h), float(y1))),
                   max(0.0, min(float(w), float(x2))), max(0.0, min(float(h), float(y2)))]
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((box, float(score), int(class_id)))
        return self._nms(candidates)

    @staticmethod
    def _decode(output: np.ndarray) -> np.ndarray:
        output = np.squeeze(output)
        if output.ndim == 1:
            output = output[None, :]
        if output.ndim != 2:
            raise ValueError(f"Unsupported YOLO output shape: {output.shape}")
        # Raw YOLOv8 exports are [channels, anchors], while v5 are [anchors, channels].
        if output.shape[0] < output.shape[1] and output.shape[0] <= 512:
            output = output.T
        return output

    def _is_nms_output(self, rows: np.ndarray) -> bool:
        return self.output_format == "yolo_nms" or (
            self.output_format == "auto" and rows.shape[1] == 6
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (self.input_width - new_w) // 2, (self.input_height - new_h) // 2
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[None], scale, pad_x, pad_y

    def _nms(self, candidates: list[tuple[list[float], float, int]]) -> list[DetectionBox]:
        detections: list[DetectionBox] = []
        by_class: dict[int, list[tuple[list[float], float, int]]] = {}
        for candidate in candidates:
            by_class.setdefault(candidate[2], []).append(candidate)
        for class_id, items in by_class.items():
            indices = cv2.dnn.NMSBoxes(
                [([x1, y1, x2 - x1, y2 - y1]) for (x1, y1, x2, y2), _, _ in items],
                [score for _, score, _ in items], self.confidence_threshold, self.iou_threshold,
            )
            for index in np.asarray(indices).reshape(-1):
                (x1, y1, x2, y2), score, _ = items[int(index)]
                label = self.class_names.get(class_id, str(class_id))
                normalized = label.lower().replace(" ", "_")
                detections.append(DetectionBox(
                    detection_id=str(uuid4()), label=label,
                    category=_CATEGORY_BY_NAME.get(normalized, DetectionCategory.UNKNOWN),
                    confidence=score,
                    bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                ))
        return detections
