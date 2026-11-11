"""Start configured edge camera feeds."""

import logging
import signal
import threading

from config.config_loader import load_config
from edge.ai_inference import (
    ActivityAnalyzer, CameraInferenceWorker, EdgeInferencePipeline, ReIdEncoder, YoloDetector,
)
from edge.camera_ingestion import CameraProcessor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config()
    edge_config = config.get("edge", {})
    cameras = [
        CameraProcessor(
            camera_id=camera["id"],
            source=[camera["source"], *camera.get("fallback_sources", [])],
            buffer_size=edge_config.get("frame_buffer_size", 2),
            reconnect_delay=edge_config.get("reconnect_delay_seconds", 2.0),
        )
        for camera in config.get("cameras", [])
    ]
    if not cameras:
        raise SystemExit("No cameras configured. Add camera entries to config/config.yaml.")
    if len({camera.camera_id for camera in cameras}) != len(cameras):
        raise SystemExit("Camera IDs in config/config.yaml must be unique.")

    inference_config = config.get("inference", {})
    detector = YoloDetector(
        inference_config["yolo_onnx_model"],
        input_size=inference_config.get("input_size", 640),
        confidence_threshold=inference_config.get("confidence_threshold", 0.35),
        iou_threshold=inference_config.get("iou_threshold", 0.45),
        output_format=inference_config.get("output_format", "auto"),
        class_names={int(key): value for key, value in inference_config.get("class_names", {}).items()} or None,
        providers=inference_config.get("providers"),
    )
    reid_model = inference_config.get("reid_onnx_model")
    reid_encoder = ReIdEncoder(reid_model, providers=inference_config.get("providers")) if reid_model else None
    pipelines = {
        camera.camera_id: EdgeInferencePipeline(
            detector, ActivityAnalyzer(inference_config["pose_task_model"]),
            edge_config.get("node_id", "edge-01"), reid_encoder,
        )
        for camera in cameras
    }

    shutdown = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    workers = [CameraInferenceWorker(camera, pipelines[camera.camera_id]) for camera in cameras]
    try:
        for camera in cameras:
            camera.start()
            logging.getLogger(__name__).info("Started camera %s", camera.camera_id)
        for worker in workers:
            worker.start()
        while not shutdown.wait(1.0):
            pass
    finally:
        for worker in workers:
            worker.stop(timeout=None)
        for pipeline in pipelines.values():
            pipeline.close()
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
