"""Start configured edge camera feeds."""

import logging
import signal
import threading

from config.config_loader import load_config
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

    shutdown = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    for camera in cameras:
        camera.start()
        logging.getLogger(__name__).info("Started camera %s", camera.camera_id)
    try:
        shutdown.wait()
    finally:
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    main()
