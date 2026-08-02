# Training a PPE model

Everything here is **offline build tooling**. Nothing in `central/`, `module/` or
`streamer/` imports it, and none of it runs in production.

The goal is one fine-tuned model that detects **person, helmet, phone, head** —
all four in a single forward pass, so the GPU cost is unchanged.

> **Run steps 2–4 on the GPU box.** Step 1 is plain video decode and is fine
> anywhere. Pre-labelling and training on a CPU-only machine will take hours.

---

## Where to put things

```
training/
  videos/                     ← YOUR RAW CLIPS GO HERE  (.mp4 .mkv .avi .mov)
  dataset/                    ← YOUR footage, the only source val may come from
    images/train/*.jpg        created by extract_frames.py
    images/val/*.jpg          created by split.py
    labels/train/*.txt        created by prelabel.py, then edited by you
    labels/val/*.txt
  ppe-train/                  ← a DOWNLOADED dataset, after remap_labels.py
    images/  labels/
  phone-data/train/           ← another download, same treatment
    images/  labels/
  ppe.yaml                    4-class scheme + the labelling rules — READ THIS
  ppe7.yaml                   7-class scheme, for the mixed public + own dataset
  extract_frames.py           videos   -> images
  prelabel.py                 images   -> person/phone labels (auto)
  remap_labels.py             a download's class ids -> ours
  split.py                    train    -> train + val
  train.py                    the fine-tune
```

`videos/`, `dataset/` and any downloaded set are gitignored — footage and weights
do not belong in git.

Keep clips out of `uploads/`. That folder belongs to the running app, and
`extract_frames.py` deliberately does not read it by default, so a run can only
ever touch footage you deliberately placed in `training/videos/`.

---

## The order to run things

### 1. Drop your clips in `training/videos/`

Name them after the camera if you can (`cam3_morning.mp4`) — it makes step 4
easier to split cleanly later.

See the shot list at the bottom for **what** to record.

### 2. Videos → images

```bash
python training/extract_frames.py
```

Keeps up to 200 frames per video, skipping near-duplicates (CCTV is mostly the
same picture, and 200 copies of an empty corridor add labelling work while
teaching nothing).

**The three flags that control how much you get:**

| Flag | Default | What it does |
|---|---|---|
| `--every N` | 30 | Look at every Nth frame. At 30 fps, `30` is one frame per second, `60` is one per two seconds. Higher = fewer frames, further apart in time |
| `--min-diff N` | 3.0 | How different a frame must be from the **last one kept** to be worth keeping. Average pixel change, 0–255. Raise it if the output still has near-identical pictures; `0` keeps everything `--every` picked |
| `--max N` | 200 | Ceiling **per video**. Stops one long clip from dominating the dataset — 2,000 frames of one camera and 20 of another teaches the model that one scene |

They apply in that order. A 5-minute clip at 30 fps:

```
9,000 frames
   ↓  --every 30      look at 1 per second
  300 candidates
   ↓  --min-diff 3    drop the ones that look like the last kept frame
 ~120 kept            (an empty corridor drops a lot here)
   ↓  --max 200       ceiling not reached
  120 frames written
```

```bash
python training/extract_frames.py --every 60 --max 300   # further apart, more per clip
python training/extract_frames.py --min-diff 6           # still too similar? raise it
python training/extract_frames.py --src uploads --max 50 # pull from somewhere else
```

Check the count before moving on. Aim for **300–500 of your own frames**.

### 3. Auto-label person and phone

```bash
python training/prelabel.py
```

Runs `yolo26x` over every image and writes the `person` and `phone` boxes, so you
only draw heads and helmets by hand. It **never overwrites an existing label
file** unless you ask it to, so it cannot destroy your work — safe to re-run
after adding more images, and safe to interrupt.

Bigger model on purpose: this runs once, offline, and a missed object is the
expensive mistake (see below). Accuracy beats speed here.

| Flag | Default | What it does |
|---|---|---|
| `--model` | `yolo26x.pt` | Which COCO model does the labelling. Big is right here — it runs once and its misses become your manual work. Downloads automatically (~110 MB) |
| `--imgsz` | 1280 | Detection resolution. Match your deployment so small phones are found |
| `--conf` | 0.25 | Deliberately low. A **missed** object becomes background and teaches the wrong thing; deleting a wrong box in the editor takes a second, spotting a missing one does not |
| `--images` | `dataset/images/train` | Which folder to label — point it at `val` too if you split first |
| `--overwrite` | off | Re-label images that already have a label file. **Leave this off** — it is what stops the tool destroying your hand-drawn helmets |
| `--append` | off | **ADD** person/phone to label files that already exist, keeping every existing line. For a downloaded set that labels helmets but not people — see step 3b |

`--overwrite` and `--append` are opposites and the script refuses both together.

**Progress and interrupting.** It prints a live line — `31/80 (39%) 1.8 img/s
elapsed 17s eta 27s person=31 phone=0` — so a 30k-image run tells you within
seconds whether it is 20 minutes or two hours. Each label file is written before
the next image is read, so `Ctrl-C` is free: everything finished is kept, and
re-running the same command resumes and skips what is done (exit code 130 marks
an interrupted run).

That resume works off the files themselves, not a log. Default mode skips any
image that already has a `.txt`; `--append` skips files that already contain the
ids it writes (`0` person, `2` phone). So adding new images later and re-running
only ever labels the new ones.

On a CPU-only machine, `--model module/models/yolo26n.pt --imgsz 640` finishes in
reasonable time, but expect noticeably more to fix by hand.

### 3b. Using a downloaded dataset

A public dataset never uses your class ids, and ids are just integers — train on
the raw download and the model learns that a person is a hardhat, with no error
anywhere. `remap_labels.py` renumbers to your scheme and **drops every class you
did not map** (dropping is right: unmapped classes become background, which is
what you want for gloves, ladders and cones).

```bash
# always look first — writes nothing
python training/remap_labels.py --labels <download>/labels --map 11:0,3:1 --dry-run
python training/remap_labels.py --labels <download>/labels --map 11:0,3:1
```

`--map` is `SOURCE:DEST`. Read the source ids from the download's own `data.yaml`
and **check them every time** — Roboflow renumbers classes between export
versions. The `--dry-run` counts are the giveaway: the class you expect to
dominate should dominate.

| Flag | What it does |
|---|---|
| `--labels` | labels directory, searched recursively |
| `--map` | `SRC:DST` pairs. Everything unlisted is dropped |
| `--dry-run` | report the counts, write nothing. Always allowed |
| `--force` | remap again despite the marker — you almost never want this |

**It is deliberately not idempotent-safe.** Source and destination ids overlap,
so a second pass re-maps the NEW ids (with `3:1` and `1:4`, every hardhat becomes
a glove). After a successful run it writes a `.remap-applied` marker into the
labels dir and refuses to run again. To genuinely redo it, restore the originals
— which removes the marker with them:

```bash
tar czf labels-orig.tar.gz -C <download> labels     # BEFORE you remap
```

Then fill in the classes the download lacks. A PPE set usually has no people:

```bash
python training/prelabel.py --images <download>/images --append
```

Two rules when mixing sources:

- **Every class present in an image must be labelled in that image.** There is no
  way to tell YOLO "ignore hardhats in this file" — an unlabelled hardhat trains
  as background and works against the ones you do have. So phone photos should
  not contain helmets unless you draw them.
- **Keep each source in its own folder** and list them under `train:`. Then a
  source that turns out to hurt is one line to remove, not a rebuild:

```yaml
train:
  - dataset/images/train        # yours
  - ppe-train/images            # download
  - phone-data/train/images     # download
val: dataset/images/val         # YOURS ONLY, always
```

### 4. Label heads and helmets — the human part

Open `training/dataset/` in a labelling tool (CVAT, Label Studio, Roboflow — all
read this format) with the class list from `ppe.yaml`:

```
0 person   1 helmet   2 phone   3 head
```

**The rules are written at the top of `ppe.yaml`. Read them before starting.**
In short:

- `head` = **any** visible head — covered or bare, front, **back**, side, top-down
- `helmet` = every helmet, **wherever it is** (on a head, in a hand, on the floor)
- A worker wearing a helmet gets **both** boxes, overlapping. That is correct.
- Do not judge compliance while labelling. The rule does that at runtime by
  comparing positions.

Two failure modes to watch for:

- **A missed person.** YOLO treats unlabelled areas as background, so a person
  nobody boxed actively teaches the model that people are not people. Check
  what `prelabel.py` missed.
- **No head box on helmeted workers.** If every `head` box in your data is on a
  bare head, the model learns "head = bare head" and stops detecting heads once
  a helmet goes on — which makes compliant workers invisible to the rule.

### 5. Split off a validation set

```bash
python training/split.py            # moves 15% of the labelled train set to val
```

Two piles:

```
train  →  the model LEARNS from these       (85%)
val    →  the model NEVER sees these        (15%)
```

`val` is the exam. Scoring a model on frames it learned from is like setting an
exam using the practice questions — the mark tells you nothing. `train.py` scores
against `val` automatically, and that per-class recall is the only number that
predicts behaviour on your real cameras.

**Why a script and not drag-and-drop:** every image has a label file in a
separate folder.

```
dataset/images/train/frame_001.jpg
dataset/labels/train/frame_001.txt   ← must move with it
```

Move the image and leave the label behind, and YOLO reads that frame as "nothing
in this picture" — teaching the model to see nothing. The script always moves
both, and warns about any image that has no label yet.

| Flag | Default | What it does |
|---|---|---|
| `--frac N` | 0.15 | Fraction of the labelled train set to move to `val` |
| `--only PREFIX` | — | Only consider files whose name starts with this |
| `--seed N` | 0 | Repeatable split — same seed picks the same frames |
| `--undo` | — | Move everything in `val` back to `train` and start over |

Validation frames must come from **your** cameras. If you also downloaded a
public dataset into `train`, `--only` keeps the split to your own footage —
otherwise the score measures how well you do on internet photos:

```bash
python training/split.py --only cam3_
python training/split.py --undo
```

Run this **after** labelling, never before, or you are splitting images that have
no labels yet.

### 6. Train

```bash
python training/train.py                                    # 100 epochs, ppe.yaml
python training/train.py --data training/ppe7.yaml --name ppe7
python training/train.py --batch 4              # if CUDA runs out of memory
python training/train.py --epochs 200 --model yolo26s.pt
```

| Flag | Default | What it does |
|---|---|---|
| `--data` | `ppe.yaml` | Which class scheme. `ppe7.yaml` for the mixed public + own dataset |
| `--model` | `yolo26n.pt` | Starting weights. Fine-tuning from the COCO checkpoint beats training from scratch on a small dataset. `yolo26s.pt` is more accurate but slower at runtime, on every camera, forever |
| `--epochs` | 100 | Passes over the dataset. Training stops early if it stops improving for 30 (`patience`) |
| `--imgsz` | 0 → 1280 | 0 means "take it from `Config.inference_imgsz`". **Keep training and deployment equal** — train at 640 and deploy at 1280 and the model sees objects at a scale it never learned |
| `--batch` | 8 | Images per step. **Lower this first** if training dies with CUDA out-of-memory at 1280 |
| `--device` | auto | GPU if one is present, else CPU. Force with `0`, `cpu`, `0,1` |
| `--name` | `ppe` | Output goes to `runs/detect/<name>/`. **Ultralytics auto-increments** when the dir exists, so `ppe` may land in `ppe-3` — the final line prints the real path |

**Always run `train.py`, never the raw `yolo` CLI.** The data yamls use a
relative `path:`, and Ultralytics resolves that against its own `datasets_dir`
setting rather than the yaml's location — so `yolo detect train data=…` looks in
the wrong place and dies with "images not found". `train.py` rewrites `path:` to
an absolute path first, and counts the images in each split before starting.

**Think in steps, not epochs.** What matters is `images ÷ batch × epochs`. 211
images for 100 epochs is ~2,600 steps; 32,000 images for 10 epochs is ~40,000.
Run `--epochs 1` on a large set first to get the real per-epoch wall time and a
first per-class table, then decide.

One edge to know if you pick a small number: `close_mosaic` is 10 and mosaic
closes at `epochs - close_mosaic`. At `--epochs 5` that never happens, so mosaic
stays on for the whole run; at exactly `--epochs 10` it closes at epoch 0, so
mosaic is off throughout. Either is fine for a sanity run, but neither gives the
intended schedule — 30 or more does.

Two augmentations are switched off deliberately: horizontal flip and rotation. A
fixed-mount camera never sees the world mirrored or upside down, so those spend
model capacity on poses that cannot occur. Scale and translation stay on, because
people really do walk toward and across the camera.

### 7. Read the per-class numbers

`train.py` prints precision / recall / mAP50 per class. **Watch recall**, not mAP:

| Recall on your own val frames | Verdict |
|---|---|
| below ~0.7 | not enough data, or not enough variety |
| ~0.85+ | ready to pilot |

Add data where it is weak rather than training longer.

### 8. Deploy it

1. Copy `runs/detect/ppe/weights/best.pt` into `module/models/`
2. In `module/core/config.py`:
   ```python
   model_path      = ROOT_DIR / "models" / "best.pt"
   person_class_id = 0
   helmet_class_id = 1
   phone_class_id  = 2
   head_class_id   = 3      # enables helmet/head position matching
   ppe_enabled     = True
   ```
   These ids come from `ppe.yaml`, **not** from COCO. A fine-tuned model numbers
   its classes in your `names:` order, so `phone` is `2` here, never `67`.
3. Re-measure throughput and update `fps_budget`:
   ```bash
   python tests/bench_gpu.py
   ```

---

## Why one model and not two

Fine-tuning **replaces** the detection head. A model trained on helmets alone
forgets person and phone, which silently kills the rest of the pipeline. Running
a second PPE model instead doubles GPU cost on a box already budgeted around
143 fps. One model, four classes, one forward pass.

---

## What to record

Roughly, for ~500 of your own frames:

| Scene | Labels | Share |
|---|---|---|
| Worker **wearing** helmet — front, back, side, top-down | person + head + helmet | 35% |
| Worker **bare-headed** — same angles | person + head | 25% |
| Worker **carrying** helmet (hand, under arm, shoulder) | person + head + helmet | 10% |
| Phone in hand, with and without helmet | person + head + helmet/– + phone | 15% |
| Several people together, mixed compliance | all of the above | 7% |
| Helmet alone on the floor or a bench | helmet | 3% |
| Empty scene, nobody present | none (empty label file) | 5% |

Vary **distance** (mostly far — that is your real case), **angle** (back and
top-down matter most; a ceiling camera rarely sees faces), **lighting**,
**occlusion**, **motion blur**, and **helmet colour**.

Two things you must arrange deliberately, because a normal working day will not
produce them:

- **Bare heads.** On a compliant site nobody is bare-headed, so the model never
  learns the thing you need it to spot. Stage it.
- **Carried helmets.** Without these the model has never seen a helmet away from
  a head, which is exactly the distinction the rule depends on.

The efficient way is one 20–30 minute session with two or three people, cameras
already running: walk the area wearing helmets, then bare-headed, then carrying
them, then a short pass using phones, then leave it empty for a few minutes.
Repeat per camera position if your views differ a lot.

Public hard-hat datasets (Roboflow Universe) are worth adding for volume, but
they are photos taken at eye level in good light. Your own frames are what close
the gap to a ceiling camera, and they are the reason the val set must be yours.
