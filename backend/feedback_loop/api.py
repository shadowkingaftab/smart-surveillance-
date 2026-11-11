"""Authenticated REST and WebSocket interface for the central security service."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from backend.alerting.notifier import AlertNotifier
from backend.feedback_loop.audit_manager import AuditManager, OperatorActionRequest
from backend.threat_assessment.processor import CentralMetadataProcessor
from common.models_schema import FrameMetadata
from config.config_loader import load_config

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


def _authorize(authorization: str | None) -> None:
    if not api_token:
        raise HTTPException(status_code=503, detail="API token is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, api_token):
        raise HTTPException(status_code=401, detail="Invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
    try:
        _authorize(websocket.headers.get("authorization"))
    except HTTPException as exc:
        await websocket.close(code=4401 if exc.status_code == 401 else 1013, reason=exc.detail)
        return
    await websocket.accept()
    try:
        async for event in notifier.dashboard_events():
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
