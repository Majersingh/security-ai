# Setup & Run Guide

Step-by-step instructions to run the CCTV Operator Monitoring System on a new
machine after cloning the repository.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.12** | `python3 --version` |
| **git** | to clone the repo |
| **Internet (first run)** | pip installs deps; the YOLO weight auto-downloads once |
| **Linux system libs** | `sudo apt install -y libgl1 libglib2.0-0` (OpenCV needs these) |
| **(Optional) NVIDIA GPU + driver** | for real-time speed — see [GPU section](#6-gpu-usage) |

---

## 2. Clone the repository

```bash
git clone <your-repo-url> security-ai
cd security-ai
```

> Videos, the model weight, the virtualenv, and outputs are **not** in the repo
> (they are git-ignored). They are recreated below or supplied by you.

---

## 3. Create a virtual environment

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 4. Install dependencies

```bash
# Web server + core CV stack (FastAPI, uvicorn, PyAV, Ultralytics, …):
pip install -r requirements.txt
```

The YOLOv11 weight (`yolo11n.pt`, ~5 MB) downloads automatically on first run
and is cached in `models/`. If the machine is offline, copy `models/yolo11n.pt`
over manually.

---

## 5. Have a stream URL ready

You'll need a reachable camera stream URL (RTSP / HLS / HTTP), e.g.
`rtsp://user:pass@camera-host:554/stream`. No local video file is needed.

---

## 6. Run

```bash
PYTHONPATH=src python -m uvicorn --app-dir web server:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000**, paste a camera **stream URL** (RTSP / HLS / HTTP)
and click **Add Stream**. Draw a line/zone on a connected stream to add
tripwire / intrusion rules. Per-feed snapshots + `events.csv` land under
`output/<feed_id>/`; the video itself is never stored.

---

## 7. GPU usage

The device is **auto-detected** (config default `device = "auto"`):

- **NVIDIA GPU** present + driver installed + CUDA build of PyTorch → uses `cuda:0`
- **Apple Silicon** → uses `mps`
- **Otherwise** → uses `cpu`

On startup the log prints which device was chosen, e.g.
`Model loaded (device=cuda:0, requested=auto)`.

**To force a device**, edit `device` in `src/config.py` (`"cpu"`, `"0"`, `"mps"`)
if you need to override auto-detection.

### Making sure the GPU is actually available
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
If this prints `False` on a machine with an NVIDIA GPU, install a CUDA build of
PyTorch (pick the command for your CUDA version from https://pytorch.org), e.g.:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Speed guide (YOLOv11n @ imgsz 1280): CPU ≈ 2–5 fps; entry GPU ≈ real-time;
mid/high GPU well above real-time.

---

## 8. Common issues

| Symptom | Fix |
|---------|-----|
| `ImportError: libGL.so.1` (Linux) | `sudo apt install -y libgl1 libglib2.0-0` |
| `could not open stream` | Check the URL/credentials and that the host is reachable; RTSP uses TCP transport with a 5s timeout. |
| Phone never detected | Raise `inference_imgsz` in `config.py`, or set `model_path` to a larger model (`yolo11s.pt`). |
| First run hangs at start | It's downloading the model weight — needs internet once. |
| GPU not used | `torch.cuda.is_available()` is `False` → install a CUDA build of PyTorch (see above). |

---

## Quick reference (copy-paste)

```bash
git clone <your-repo-url> security-ai && cd security-ai
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=src python -m uvicorn --app-dir web server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000, then add a stream URL
```
