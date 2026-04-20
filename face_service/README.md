# Face Service

Production-oriented InsightFace service for employee face registration and tracked-person recognition.

## Features

- InsightFace `FaceAnalysis` model loaded once at startup
- Person-bbox face cropping for tracked people
- L2-normalized embeddings stored per employee
- Cosine-similarity matching across multiple embeddings per employee
- Per-track caching to avoid re-running recognition every frame
- FastAPI endpoints for registration, recognition, and database inspection

## Install

```powershell
py -3 -m pip install -r face_service\requirements.txt
```

If you want GPU inference, install a compatible `onnxruntime-gpu` package in place of `onnxruntime`.

## Run

```powershell
py -3 -m face_service.main
```

The API starts on `http://0.0.0.0:8003` by default.

## Endpoints

- `GET /status`
- `GET /faces`
- `POST /register-face`
- `POST /recognize`

## Registration Payload

```json
{
  "id": "EMP001",
  "name": "Dheena",
  "rfid_tag": "E20040D4",
  "images": ["data:image/jpeg;base64,..."]
}
```

## Recognition Payload

```json
{
  "frame": "data:image/jpeg;base64,...",
  "frame_index": 10,
  "stream_id": "gate-1",
  "tracks": [
    {
      "track_id": 101,
      "bbox": [100, 80, 240, 420]
    }
  ]
}
```

## Environment Variables

- `FACE_MODEL_NAME`: default `buffalo_l`
- `FACE_MODEL_ROOT`: optional local InsightFace model root
- `FACE_DET_WIDTH`: default `640`
- `FACE_DET_HEIGHT`: default `640`
- `FACE_SIMILARITY_THRESHOLD`: default `0.7`
- `FACE_CROP_MARGIN_RATIO`: default `0.12`
- `FACE_MIN_BOX_SIZE`: default `40`
- `FACE_MIN_SCORE`: default `0.45`
- `FACE_RECOGNIZE_EVERY_N_FRAMES`: default `5`
- `FACE_DB_PATH`: default `face_service/data/faces.json`
- `FACE_SERVICE_HOST`: default `0.0.0.0`
- `FACE_SERVICE_PORT`: default `8003`
