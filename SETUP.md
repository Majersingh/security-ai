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
# Web UI (upload + live camera) — includes the core CV stack:
pip install -r requirements-web.txt

# ...or CLI only:
pip install -r requirements.txt
```

The YOLOv11 weight (`yolo11n.pt`, ~5 MB) downloads automatically on first run
and is cached in `models/`. If the machine is offline, copy `models/yolo11n.pt`
over manually.

---

## 5. Add a video (for file mode)

Place a video at:
```
input/operator.mp4
```
Not required if you only use **Live Camera** mode in the web UI.

---

## 6. Run

### Web UI (upload + live camera)
```bash
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --app-dir .
```
Open **http://localhost:8000** — toggle between *Upload Video* and *Live Camera*.

### CLI (batch → annotated.mp4 + events.csv + snapshots)
```bash
python src/main.py                       # uses input/operator.mp4
python src/main.py --input path/to/video.mp4
python src/main.py --device cpu          # force CPU
python src/main.py --no-video            # CSV + snapshots only (faster)
```
Outputs land in `output/` (`annotated.mp4`, `events.csv`, `snapshots/`).

---

## 7. GPU usage

The device is **auto-detected** (config default `device = "auto"`):

- **NVIDIA GPU** present + driver installed + CUDA build of PyTorch → uses `cuda:0`
- **Apple Silicon** → uses `mps`
- **Otherwise** → uses `cpu`

On startup the log prints which device was chosen, e.g.
`Model loaded (device=cuda:0, requested=auto)`.

**To force a device** (CLI): `python src/main.py --device cpu` or `--device 0`.
**For the web UI**, edit `device` in `src/config.py` if you need to override auto.

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
| Camera button does nothing / blocked | Browsers allow camera only on `https://` or `http://localhost`. Use localhost, or serve over HTTPS (e.g. a Cloudflare tunnel). |
| `Input video not found` | Put a file at `input/operator.mp4` or pass `--input`. |
| Phone never detected | High-res footage needs `inference_imgsz=1280` (already the default); try a larger model with `--model yolo11s.pt`. |
| First run hangs at start | It's downloading the model weight — needs internet once. |
| GPU not used | `torch.cuda.is_available()` is `False` → install a CUDA build of PyTorch (see above). |

---

## Quick reference (copy-paste)

```bash
git clone <your-repo-url> security-ai && cd security-ai
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-web.txt
# add input/operator.mp4 (optional if using camera)
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --app-dir .
# open http://localhost:8000
```
