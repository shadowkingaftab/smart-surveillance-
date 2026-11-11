"""Offline ONNX INT8 quantization and optional PyTorch channel pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from collections.abc import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class _ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, model_path: Path, images: Sequence[str | Path | np.ndarray], input_size: int) -> None:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        shape = model_input.shape
        self.input_name = model_input.name
        self.height = int(shape[2]) if isinstance(shape[2], int) else input_size
        self.width = int(shape[3]) if isinstance(shape[3], int) else input_size
        if not images:
            raise ValueError("At least one calibration image is required")
        self._source_images = images
        self._images = iter(images)

    def _prepare(self, image: str | Path | np.ndarray) -> np.ndarray:
        frame = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid calibration image: {image}")
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
        pad_x, pad_y = (self.width - resized.shape[1]) // 2, (self.height - resized.shape[0]) // 2
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[None]

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image = next(self._images)
        except StopIteration:
            return None
        return {self.input_name: self._prepare(image)}

    def rewind(self) -> None:
        self._images = iter(self._source_images)


def quantize_onnx_int8(model_path: str | Path, output_path: str | Path,
                       calibration_images: Sequence[str | Path | np.ndarray],
                       *, input_size: int = 640) -> Path:
    """Create a statically calibrated QDQ INT8 ONNX model for CNN inference."""
    source, destination = Path(model_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = _ImageCalibrationReader(source, calibration_images, input_size)
    quantize_static(
        model_input=str(source), model_output=str(destination), calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, per_channel=True, optimize_model=False,
    )
    return destination


def prune_pytorch_channels(model: object, layer_names: Iterable[str], example_inputs: object,
                           *, amount: float = 0.2) -> object:
    """Physically remove low-magnitude output channels from selected layers.

    Requires ``torch-pruning`` and representative inputs so dependent layers
    (including residual connections) can be updated before ONNX export.
    """
    if not 0 < amount < 1:
        raise ValueError("amount must be between 0 and 1")
    try:
        import torch.nn as nn
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and torch-pruning to prune model channels") from exc
    named_modules = dict(model.named_modules())  # type: ignore[attr-defined]
    for name in layer_names:
        layer = named_modules.get(name)
        if isinstance(layer, nn.Conv2d):
            prune_fn = tp.prune_conv_out_channels
        elif isinstance(layer, nn.Linear):
            prune_fn = tp.prune_linear_out_channels
        else:
            raise ValueError(f"Selected pruning layer is not Conv2d/Linear: {name}")
        channel_norms = layer.weight.detach().flatten(1).norm(p=2, dim=1)
        channel_count = layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features
        prune_count = min(int(channel_count * amount), channel_count - 1)
        if prune_count < 1:
            continue
        prune_indices = channel_norms.argsort()[:prune_count].tolist()
        graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        group = graph.get_pruning_group(layer, prune_fn, idxs=prune_indices)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Unsafe dependency group for pruning layer: {name}")
        group.prune()
    return model
