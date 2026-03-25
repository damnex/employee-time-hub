# Python CCTV Face Training

This folder now has three Python paths:

1. `face_recognition` embeddings for stronger matching when Python 3.11 plus dlib prerequisites are available.
2. OpenCV LBPH fallback for the laptop webcam right now on machines where dlib is hard to install.
3. A new MediaPipe + threaded realtime tracker for low-latency CCTV/webcam streaming with RFID matching.

For your current laptop-first goal, start with the OpenCV LBPH path. Later, when you move to the real CCTV camera, you can keep the same live script and only change `--source` to the RTSP URL.

## Dataset layout

Use one folder per employee:

```text
python-ml/
  dataset/
    EMP001/
      01.jpg
      02.jpg
      03.jpg
    EMP002/
      01.jpg
      02.jpg
      03.jpg
```

## Path A: Laptop-friendly OpenCV LBPH

This is the best path for this machine now because it avoids dlib.

### Install

```powershell
cd python-ml
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-webcam.txt
```

### Train from the dataset

```powershell
py opencv_lbph_train.py --dataset dataset --output output\opencv --metadata metadata.example.csv
```

This writes:

- `output/opencv/lbph-model.yml`
- `output/opencv/lbph-labels.json`
- `output/opencv/lbph-training-report.json`

### Run live attendance on the laptop webcam

```powershell
py opencv_live_attendance.py --model output\opencv\lbph-model.yml --labels output\opencv\lbph-labels.json --source 0
```

What it does:

- opens the laptop webcam with `--source 0`
- keeps scanning continuously
- marks attendance once per employee per session by default
- writes logs to `output/live-attendance-opencv/attendance-events.csv`
- writes a session summary to `output/live-attendance-opencv/session-summary.json`
- shows a live preview window where you can press `Q` to stop

Useful webcam options:

- `--process-every-nth-frame 2`
- `--min-consecutive-detections 2`
- `--repeat-after-seconds 300`
- `--save-snapshots`
- `--distance-threshold 60`

### Move to RTSP later

When you switch from the laptop webcam to the real CCTV stream:

```powershell
py opencv_live_attendance.py --model output\opencv\lbph-model.yml --labels output\opencv\lbph-labels.json --source "rtsp://username:password@camera-ip:554/stream"
```

Useful RTSP options:

- `--no-display`
- `--resize-width 960`
- `--resize-width 1280`

## Path B: face_recognition embeddings

This path is still included for stronger embedding-style matching:

- `train_face_model.py`
- `predict_face.py`
- `verify_multi_face.py`
- `live_attendance.py`

Install for that path:

```powershell
pip install -r requirements.txt
```

Important note for this laptop:

- This machine currently only has Python 3.14.
- `face_recognition` pulls in `dlib`.
- `dlib` failed to build here because `cmake` is missing, and Python 3.14 is a harder target for that stack.
- If you later install Python 3.11 plus the required build tools, you can use that stronger embedding path too.


## Path C: Fast realtime MediaPipe + RFID pipeline

Use this path when you need live stream processing instead of capture-then-process.
It keeps camera I/O on a background thread, resizes frames to 640x480 by default, processes alternate frames, keeps all matching in memory, and reads RFID from stdin or serial without writing images to disk.

### Install

```powershell
cd python-ml
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-realtime.txt
```

### Build the face index once

```powershell
py realtime_rfid_face_tracker.py build-index --dataset dataset --metadata metadata.generated.csv --output output\realtime\face-index.pkl
```

### Run the realtime tracker

```powershell
py realtime_rfid_face_tracker.py run --index output\realtime\face-index.pkl --source 0 --rfid-stdin
```

Type an RFID tag into the terminal and press Enter to simulate a badge scan.
For a real reader, replace `--rfid-stdin` with `--rfid-serial-port COM3` and set `--rfid-baudrate` if needed.

Useful runtime options:

- `--source "rtsp://user:pass@camera-ip:554/stream"`
- `--runtime-width 640 --runtime-height 480`
- `--process-every-nth-frame 2`
- `--mediapipe-model 1`
- `--match-threshold 0.48`
- `--no-display`

Notes:

- This path is optimized for low latency on CPU, but true 200-300ms recognition for 50 visible faces usually needs stronger hardware or a GPU-ready embedding backend.
- `face_recognition` still depends on `dlib`, so use Python 3.11 for the smoothest install.
## Metadata CSV

`metadata.example.csv` is still the format for both paths:

```csv
folder_name,employeeCode,name,department,rfidUid,email,phone,isActive
EMP001,EMP001,Anita Sharma,Operations,04A1BC22,anita@example.com,+91-9000000001,true
```

## Which one should you use now?

Use the OpenCV LBPH path first for the laptop webcam. It is the fastest way to get a working local demo with around 50 employees on this machine.

