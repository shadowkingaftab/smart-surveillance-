"""Authenticated REST and WebSocket interface for the central security service."""

from __future__ import annotations

import hmac
import importlib.util
import os
import time
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from backend.alerting.notifier import AlertNotifier
from backend.feedback_loop.audit_manager import AuditManager, OperatorActionRequest
from backend.threat_assessment.processor import CentralMetadataProcessor
from common.models_schema import FrameMetadata
from config.config_loader import load_config

if TYPE_CHECKING:
    from edge.ai_inference.pipeline import EdgeInferencePipeline

config = load_config()
feedback_config = config.get("feedback_loop", {})
api_token = os.getenv(feedback_config.get("token_environment_variable", "SMARTSURVEILLANCE_API_TOKEN"))
max_clip_bytes = int(feedback_config.get("max_clip_size_mb", 250)) * 1024 * 1024
feedback_directory = Path(feedback_config.get("directory", "feedback"))
if not feedback_directory.is_absolute():
    feedback_directory = Path(__file__).resolve().parents[2] / feedback_directory

processor = CentralMetadataProcessor.from_config(config)
notifier = AlertNotifier.from_config(config.get("alerting", {}))
audit = AuditManager(feedback_directory)
app = FastAPI(title="Smart Surveillance Security Integration", version="1.0.0")
project_root = Path(__file__).resolve().parents[2]
dashboard_path = Path(__file__).with_name("static") / "index.html"
dashboard_inference_lock = asyncio.Lock()
dashboard_pipeline: EdgeInferencePipeline | None = None


def _model_path(key: str) -> Path:
    configured = Path(str(config.get("inference", {}).get(key, "")))
    return configured if configured.is_absolute() else project_root / configured


def _dashboard_status() -> dict[str, object]:
    inference = config.get("inference", {})
    required_files = {
        "YOLO ONNX model": _model_path("yolo_onnx_model"),
        "MediaPipe pose model": _model_path("pose_task_model"),
    }
    missing = [f"{name} not found: {path}" for name, path in required_files.items() if not path.is_file()]
    for module in ("onnxruntime", "mediapipe"):
        if importlib.util.find_spec(module) is None:
            missing.append(f"Python package not installed: {module}")
    return {
        "inference_ready": not missing,
        "missing_requirements": missing,
        "sample_interval_ms": 1000,
        "api_token_configured": bool(api_token),
    }


def _create_dashboard_pipeline() -> EdgeInferencePipeline:
    from edge.ai_inference.activity_analyzer import ActivityAnalyzer
    from edge.ai_inference.detector import YoloDetector
    from edge.ai_inference.pipeline import EdgeInferencePipeline
    from edge.ai_inference.reid_encoder import ReIdEncoder

    inference = config.get("inference", {})
    providers = inference.get("providers")
    detector = YoloDetector(
        _model_path("yolo_onnx_model"),
        input_size=int(inference.get("input_size", 640)),
        confidence_threshold=float(inference.get("confidence_threshold", 0.35)),
        iou_threshold=float(inference.get("iou_threshold", 0.45)),
        output_format=str(inference.get("output_format", "auto")),
        class_names={int(key): value for key, value in inference.get("class_names", {}).items()} or None,
        providers=providers,
    )
    reid_path = inference.get("reid_onnx_model")
    reid = ReIdEncoder(_model_path("reid_onnx_model"), providers=providers) if reid_path else None
    return EdgeInferencePipeline(
        detector, ActivityAnalyzer(_model_path("pose_task_model")),
        str(config.get("edge", {}).get("node_id", "edge-01")), reid,
    )


def _authorize(authorization: str | None) -> None:
    if not api_token:
        raise HTTPException(status_code=503, detail="API token is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, api_token):
        raise HTTPException(status_code=401, detail="Invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard/status")
async def dashboard_status() -> dict[str, object]:
    return _dashboard_status()


@app.post("/dashboard/frame")
async def analyze_dashboard_frame(
    camera_id: str = Form(min_length=1, max_length=128),
    file: UploadFile = File(),
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    global dashboard_pipeline
    _authorize(authorization)
    if not _dashboard_status()["inference_ready"]:
        raise HTTPException(status_code=503, detail="Inference models or runtime packages are not ready")
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Send a JPEG or PNG camera frame")
    content = await file.read(5 * 1024 * 1024 + 1)
    await file.close()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Camera frames must be 5 MB or smaller")
    async with dashboard_inference_lock:
        if dashboard_pipeline is None:
            try:
                dashboard_pipeline = await asyncio.to_thread(_create_dashboard_pipeline)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Could not initialize inference: {exc}") from exc
        frame = await asyncio.to_thread(
            lambda: cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        )
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded camera frame")
        try:
            metadata = await asyncio.to_thread(
                dashboard_pipeline.process, frame, camera_id,
                timestamp_ms=time.monotonic_ns() // 1_000_000,
            )
            result = await asyncio.to_thread(processor.process, metadata)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Frame inference failed: {exc}") from exc
    for alert in result.alerts:
        await notifier.dispatch(alert)
    await asyncio.to_thread(audit.log_processing, metadata, result)
    return {
        "metadata": metadata.model_dump(mode="json"),
        "tracking": result.tracking.model_dump(mode="json"),
        "assessments": [item.model_dump(mode="json") for item in result.assessments],
        "alerts": [item.model_dump(mode="json") for item in result.alerts],
    }


@app.post("/metadata")
async def ingest_metadata(metadata: FrameMetadata,
                          authorization: str | None = Header(default=None)) -> dict[str, object]:
    _authorize(authorization)
    result = await processor.process_and_dispatch(metadata, notifier)
    audit.log_processing(metadata, result)
    return {
        "tracking": result.tracking.model_dump(mode="json"),
        "assessments": [item.model_dump(mode="json") for item in result.assessments],
        "alerts": [item.model_dump(mode="json") for item in result.alerts],
    }


@app.post("/alerts/{alert_id}/action")
async def record_operator_action(alert_id: str, action: OperatorActionRequest,
                                 authorization: str | None = Header(default=None)) -> dict[str, str]:
    _authorize(authorization)
    if not alert_id or len(alert_id) > 128:
        raise HTTPException(status_code=422, detail="Invalid alert ID")
    audit.log_operator_action(alert_id, action)
    return {"status": "recorded", "alert_id": alert_id}


@app.post("/alerts/{alert_id}/clip", status_code=201)
async def save_flagged_clip(alert_id: str, file: UploadFile,
                            authorization: str | None = Header(default=None)) -> dict[str, object]:
    _authorize(authorization)
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload a supported video file")
    try:
        path, size = audit.save_clip(alert_id, extension, file.file, max_clip_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    finally:
        await file.close()
    audit.log_clip(alert_id, path, size)
    return {"status": "saved", "alert_id": alert_id, "clip_path": str(path), "size_bytes": size}


@app.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket) -> None:
    offered_protocols = [item.strip() for item in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    try:
        if "bearer" in offered_protocols and len(offered_protocols) > 1:
            _authorize(f"Bearer {offered_protocols[1]}")
        else:
            _authorize(websocket.headers.get("authorization"))
    except HTTPException as exc:
        await websocket.close(code=4401 if exc.status_code == 401 else 1013, reason=exc.detail)
        return
    await websocket.accept(subprotocol="bearer" if "bearer" in offered_protocols else None)
    try:
        async for event in notifier.dashboard_events():
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
