"""Append-only security audit records and flagged video clip storage."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, Field

from backend.threat_assessment.processor import CentralProcessingResult
from common.models_schema import FrameMetadata


class OperatorActionRequest(BaseModel):
    action: str = Field(min_length=1, max_length=64)
    operator_id: str = Field(min_length=1, max_length=128)
    notes: str | None = Field(default=None, max_length=2000)


class AuditManager:
    def __init__(self, feedback_directory: str | Path = "feedback") -> None:
        self.feedback_directory = Path(feedback_directory)
        self.clip_directory = self.feedback_directory / "flagged_clips"
        self.log_path = self.feedback_directory / "system_flag.log"
        self._lock = threading.Lock()

    def log_processing(self, metadata: FrameMetadata, result: CentralProcessingResult) -> None:
        self._append({
            "record_type": "frame_assessment",
            "timestamp": metadata.timestamp.isoformat(),
            "frame_id": metadata.frame_id,
            "edge_node_id": metadata.edge_node_id,
            "camera_id": metadata.camera_id,
            "bounding_boxes": [item.bbox.model_dump(mode="json") for item in metadata.detections],
            "detections": [item.model_dump(mode="json") for item in metadata.detections],
            "pose_landmarks": [item.model_dump(mode="json") for item in metadata.poses],
            "threat_scores": [item.model_dump(mode="json") for item in result.assessments],
            "operator_action_flag": None,
            "alerts": [item.model_dump(mode="json") for item in result.alerts],
        })

    def log_operator_action(self, alert_id: str, request: OperatorActionRequest) -> None:
        self._append({
            "record_type": "operator_action",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_id": alert_id,
            "operator_action_flag": request.action,
            "operator_id": request.operator_id,
            "notes": request.notes,
        })

    def save_clip(self, alert_id: str, extension: str, source: BinaryIO,
                  max_bytes: int) -> tuple[Path, int]:
        safe_id = "".join(char for char in alert_id if char.isalnum() or char in "-_")
        if not safe_id:
            raise ValueError("Invalid alert ID")
        self.clip_directory.mkdir(parents=True, exist_ok=True)
        target = self.clip_directory / f"{safe_id}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}{extension}"
        size = 0
        try:
            with target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"Clip exceeds the {max_bytes}-byte upload limit")
                    output.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target, size

    def log_clip(self, alert_id: str, path: Path, size: int) -> None:
        self._append({
            "record_type": "flagged_clip",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_id": alert_id,
            "clip_path": str(path),
            "size_bytes": size,
            "operator_action_flag": None,
        })

    def _append(self, record: dict[str, object]) -> None:
        self.feedback_directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.log_path.open("a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(line)
            log_file.flush()
            os.fsync(log_file.fileno())
