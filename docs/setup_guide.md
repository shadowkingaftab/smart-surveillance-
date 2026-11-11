# Stage 1: Camera ingestion

Install Python 3.12 and the dependencies in `requirements.txt`. Configure camera sources in the application that creates `CameraProcessor` (`0` for a local webcam or an RTSP URL). `CameraProcessor.start()` captures in a background thread; `read(timeout)` returns `(capture_timestamp, frame)` and `stop()` releases the camera. A bounded buffer keeps the newest frames when consumers fall behind.

Camera device access and RTSP credentials are environment-specific. Do not commit credentials in `config.yaml`.

## Stage 2: Edge inference

Place an exported YOLO ONNX model and a MediaPipe Pose Landmarker `.task` model at the paths configured under `inference` in `config/config.yaml`. The model paths are local artifacts and are not downloaded by the application. `python main.py` reads each camera and passes its newest buffered BGR frame through `EdgeInferencePipeline`, returning a validated `FrameMetadata` payload. Pose IDs are short-lived per-camera position tracks; they are not cross-camera identities. Use custom YOLO weights if weapon or benign-tool classes are required. A generic COCO model only covers its trained classes.

`quantize_onnx_int8` needs representative calibration images and writes a separate QDQ INT8 ONNX model. Validate accuracy and speed on the target gateway before replacing the FP32 model. `prune_pytorch_channels` is optional and requires PyTorch plus `torch-pruning`; pass the model, selected layer names, and representative example input before exporting to ONNX. Dependency-aware pruning updates connected layers, but the pruned model still needs fine-tuning and accuracy validation.

## Stage 3: Central tracking and threat assessment

Configure a person-ReID ONNX file as `inference.reid_onnx_model` to attach appearance embeddings to person detections. The encoder expects RGB, ImageNet-normalized NCHW inputs. Without this model, the tracker can associate detections only within the same camera. Set `tracking.camera_transitions` to allowed source-to-destination camera lists and optionally map camera IDs to zones in `tracking.camera_zones`. Call `CrossCameraTracker.update(frame_metadata)` and pass its assignments to `ThreatEvaluator.evaluate(frame_metadata, tracking_result)`. Matches are appearance-based candidates and require human review; they are not proof of identity.

`CentralMetadataProcessor.process(frame_metadata)` runs both components and returns track assignments plus contextual `ThreatAssessment` records. Scores are heuristic review cues: the weapon, tool, raised-arm, rapid-movement, and prolonged-low-movement rules need site-specific calibration. Waiting zones reduce loitering scores. They do not establish intent or confirm a threat.

## Stage 4: Alert confirmation and delivery

`CentralMetadataProcessor.from_config(config)` enables the temporal gate. It requires three distinct frame IDs in a five-second window, a mean confidence of at least `0.72`, and HIGH or CRITICAL threat levels before creating an alert; defaults are configurable in `config/config.yaml`. Call `await processor.process_and_dispatch(metadata, notifier)` from the central async event loop. `AlertNotifier.dashboard_events()` provides an async event stream for a dashboard WebSocket adapter. Set `ALERT_MOBILE_PUSH_URL`, `ALERT_DASHBOARD_WEBHOOK_URL`, and optionally `ALERT_BEARER_TOKEN` in the central server environment to POST alert JSON to your push and dashboard adapters. Delivery is provider-neutral; the receiving adapter must translate the payload for its mobile push service.
