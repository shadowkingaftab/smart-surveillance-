# Stage 1: Camera ingestion

Install Python 3.12 and the dependencies in `requirements.txt`. Configure camera sources in the application that creates `CameraProcessor` (`0` for a local webcam or an RTSP URL). `CameraProcessor.start()` captures in a background thread; `read(timeout)` returns `(capture_timestamp, frame)` and `stop()` releases the camera. A bounded buffer keeps the newest frames when consumers fall behind.

Camera device access and RTSP credentials are environment-specific. Do not commit credentials in `config.yaml`.
