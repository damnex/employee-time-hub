# Vision Service

Production-oriented person detection and tracking service for webcam or RTSP inputs.

## Features

- OpenCV video capture for webcam indices and RTSP URLs
- YOLOv8 person-only detection via Ultralytics
- ByteTrack multi-object tracking with stable track IDs
- Left-to-right and right-to-left line-crossing events
- FastAPI endpoints for latest tracks and service status
- Optional OpenCV debug window

## Files

- `detector.py`: YOLOv8 inference and ByteTrack integration
- `tracker.py`: background processing loop, capture management, and state
- `direction.py`: virtual-line crossing logic
- `api.py`: FastAPI app
- `main.py`: Uvicorn entry point

## Install

```powershell
py -3 -m pip install -r vision_service\requirements.txt
```

Install a Torch build that matches your hardware before starting the service if it is not already present.

## Run

```powershell
py -3 -m vision_service.main
```

The API starts on `http://0.0.0.0:8002` by default.

## Environment Variables

- `VISION_SOURCE`: webcam index like `0` or an RTSP URL
- `VISION_MODEL_PATH`: YOLOv8 weights path, default `yolov8n.pt`
- `VISION_DEVICE`: explicit device such as `cpu`, `0`, or `mps`
- `VISION_IMGSZ`: inference image size, default `640`
- `VISION_CONF`: detection confidence threshold, default `0.25`
- `VISION_IOU`: NMS IoU threshold, default `0.5`
- `VISION_MAX_DET`: maximum detections per frame, default `100`
- `VISION_CAMERA_WIDTH`: requested capture width, default `1280`
- `VISION_CAMERA_HEIGHT`: requested capture height, default `720`
- `VISION_CAMERA_FPS`: requested capture FPS, default `30`
- `VISION_READY_TIMEOUT_MS`: startup ready timeout, default `5000`
- `VISION_RECONNECT_DELAY_MS`: reconnect delay after capture failure, default `500`
- `VISION_LINE_POSITION_FRACTION`: vertical line position, default `0.5`
- `VISION_LINE_DEADBAND_PX`: neutral zone around the line, default `16`
- `VISION_DIRECTION_COOLDOWN_MS`: duplicate event cooldown, default `1500`
- `VISION_STATE_TTL_MS`: inactive-track cleanup window, default `5000`
- `VISION_SHOW_DEBUG`: `true` to enable the OpenCV debug window
- `VISION_DEBUG_WINDOW_NAME`: debug window title
- `VISION_SERVICE_HOST`: API bind host, default `0.0.0.0`
- `VISION_SERVICE_PORT`: API port, default `8002`

## API

- `GET /status`: returns health, FPS, inference timing, frame size, and current error state
- `GET /tracks`: returns the latest timestamp and tracked people with `track_id`, `bbox`, `center`, `confidence`, and optional `direction`
