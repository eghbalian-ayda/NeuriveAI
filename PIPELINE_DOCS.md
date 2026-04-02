# NeurivAI v2 — Head Impact Pipeline Documentation

## Overview

NeurivAI v2 detects head impacts in multi-person video footage and computes
clinically-grounded brain injury metrics (BrIC_R, KLC, DAMAGE) for every
flagged athlete. It uses **head keypoints** instead of helmet bounding boxes,
making it scale-invariant and applicable to unprotected athletes.

**Key design principle — Two-pass architecture:**
- **Pass 1 (fast):** YOLOv8x-pose + 3 lightweight detection stages scan every
  frame to find impact timestamps. Runs at near real-time speed.
- **Pass 2 (deep):** HybrIK pose estimator runs retrospectively *only* on the
  short window of frames around each confirmed impact. Produces SMPL rotation
  matrices → angular velocity → brain injury scores.

---

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════╗
║                    PASS 1  — fast scan                          ║
╚══════════════════════════════════════════════════════════════════╝

Video File
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0 — HeadKeypointTracker           head_tracker.py        │
│                                                                   │
│  Model  : YOLOv8x-pose (COCO 17-keypoint)                        │
│  Tracker: ByteTrack (Ultralytics, persist=True)                  │
│                                                                   │
│  Per person extracts 5 head keypoints:                           │
│    0=nose  1=left_eye  2=right_eye  3=left_ear  4=right_ear     │
│                                                                   │
│  Computes:                                                        │
│    centroid  = mean of visible keypoints (conf > 0.3)            │
│    radius_px = ear span / 2  (fallback: box_width / 8)           │
│                                                                   │
│  Maintains:                                                       │
│    frame_buffer  — rolling deque of raw BGR frames               │
│    kp_history    — dict[track_id → list[HeadState]]              │
│                                                                   │
│  IN  : video path                                                 │
│  OUT : (frame_idx, frame_bgr, list[HeadState]) per frame         │
└──────────────────────────┬──────────────────────────────────────┘
                           │  list[HeadState]
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
┌─────────────────┐ ┌────────────────┐ ┌──────────────────────┐
│  Stage 1        │ │  Stage 2       │ │  Stage 3              │
│  Proximity      │ │  Velocity      │ │  Skull Rotation       │
│  Detector       │ │  Detector      │ │  Detector             │
│                 │ │                │ │                        │
│ proximity_      │ │ velocity_      │ │ skull_rotation_        │
│ detector.py     │ │ detector.py    │ │ detector.py            │
│                 │ │                │ │                        │
│ Pairwise norm.  │ │ Z-score on     │ │ Ear-to-ear angle       │
│ head distance   │ │ centroid disp. │ │ angular velocity       │
│ (head radii)    │ │ [px/frame]     │ │ [rad/s]                │
└────────┬────────┘ └───────┬────────┘ └──────────┬───────────┘
         │                  │                      │
         └──────────────────┴──────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ImpactBuffer                             impact_buffer.py       │
│                                                                   │
│  Merges signals. Requires ≥ 2 stages to fire.                    │
│  Confidence = prox×0.40 + vel×0.35 + skull×0.25                 │
│  Cooldown: 45 frames per pair after each event                   │
│                                                                   │
│  OUT : list[ImpactEvent]  (frame_idx, track_ids, conf, stages)   │
└──────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════╗
║                    PASS 2  — deep analysis                      ║
╚══════════════════════════════════════════════════════════════════╝

For each ImpactEvent  →  ±15 frames around event_frame
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — HybrIKRetrospective      hybrik_retrospective.py     │
│                                                                   │
│  Model  : HybrIK HRNet-W48 (SMPL, 24 joints)                    │
│  Weights: models/hybrik/hybrik_hrnet.pth  (309 MB)               │
│                                                                   │
│  For each frame in window, for each flagged track_id:            │
│    1. Crop person using body_box from HeadState                  │
│    2. Resize to 256×256, normalise (ImageNet stats)              │
│    3. Run HybrIK → pred_theta_mats (1, 24×9)                    │
│    4. Reshape → (24, 3, 3) rotation matrices                     │
│    5. Extract joint 15 (head) → R_head(t)  (3×3)                │
│                                                                   │
│  OUT : dict[frame_idx → R_head (3×3 numpy array)]               │
└──────────────────────────┬──────────────────────────────────────┘
                           │  rotation matrix time series
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BrainInjuryProfiler               brain_injury_profiler.py     │
│                                                                   │
│  Step 1 — Angular velocity                                       │
│    dR(t) = R(t+1) @ R(t).T          (relative rotation)         │
│    ω(t)  = log(dR(t)) / Δt          (axis-angle / dt, rad/s)    │
│    Smoothed with Savitzky-Golay (window=7, poly=2)               │
│                                                                   │
│  Step 2 — Brain injury metrics                                   │
│    BrIC_R = max(‖ω‖) / 53.0         (Takhounts 2013)            │
│    KLC    = max(‖ω‖)   [rad/s]      (Kleiven 2007)              │
│    DAMAGE = peak of spring-mass      (Gabler 2019)               │
│             convolution with ω(t)                                │
│             ωn=30.1 rad/s, ζ=0.746                               │
│                                                                   │
│  Risk thresholds:                                                 │
│    BrIC_R : LOW < 0.25 ≤ ELEVATED < 0.50 ≤ HIGH                 │
│    KLC    : LOW < 15.0 ≤ ELEVATED < 30.0 ≤ HIGH  [rad/s]       │
│    DAMAGE : LOW < 0.10 ≤ ELEVATED < 0.20 ≤ HIGH                 │
│                                                                   │
│  OUT : risk report dict per track per event                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
            track_video.py  →  annotated video
                               impact_report.json
```

---

## Modules

---

### `head_tracker.py` — HeadKeypointTracker  *(Stage 0)*

**Purpose:** Detects all people in each frame, extracts 5 head keypoints per
person, and assigns persistent track IDs via ByteTrack.

**Model:** `yolov8x-pose.pt` — largest YOLOv8 pose model, best accuracy in
crowded scenes (133 MB, auto-downloaded on first run).

**HeadState dataclass** — one instance per person per frame:
```python
@dataclass
class HeadState:
    track_id:   int           # persistent ByteTrack ID
    frame_idx:  int           # absolute frame number
    centroid:   np.ndarray    # (2,)  [cx, cy] in pixels
    radius_px:  float         # head radius proxy in pixels
    keypoints:  np.ndarray    # (5, 3)  [x, y, conf]
    body_box:   np.ndarray    # (4,)  [x1,y1,x2,y2] full body, for HybrIK
```

**Head radius estimation:**
| Condition | Formula |
|-----------|---------|
| Both ears visible | `span(left_ear, right_ear) / 2` |
| Ears not visible | `body_box_width / 8` (fallback) |

**Buffers maintained:**
| Buffer | Type | Purpose |
|--------|------|---------|
| `frame_buffer` | `deque(maxlen=N)` | Raw BGR frames for HybrIK pass 2 |
| `kp_history` | `dict[int → list[HeadState]]` | Per-track history for pass 2 crop lookup |

**Output per frame (generator):**
`(frame_idx: int, frame_bgr: np.ndarray, head_states: list[HeadState])`

---

### `proximity_detector.py` — ProximityDetector  *(Stage 1)*

**Purpose:** Detects when two heads are close enough to have made contact.
Scale-invariant — threshold is in head-radius units, not pixels.

**Algorithm:**
```
dist_norm = euclidean(centroid_A, centroid_B) / mean(radius_A, radius_B)
score     = max(0,  1 − dist_norm / threshold)
fires if  dist_norm < 2.5
```

A `dist_norm` of 2.0 means the two head centres are 2 head-radii apart —
approximately touching. Setting threshold to 2.5 catches imminent contact
and glancing blows.

**Input:** `list[HeadState]`

**Output:**
```python
[{
    "id_a":      int,    # track ID of first person
    "id_b":      int,    # track ID of second person
    "dist_norm": float,  # distance in head-radius units
    "score":     float,  # 0–1, higher = closer
}]   # sorted by dist_norm ascending
```

**Parameters:**
| Name | Default | Effect of increasing |
|------|---------|----------------------|
| `threshold` | `2.5` | Require heads to be closer before firing |

---

### `velocity_detector.py` — KeypointVelocityDetector  *(Stage 2)*

**Purpose:** Detects sudden head displacement spikes using a z-score test on
the rolling centroid velocity. Uses the **keypoint centroid** (anatomically
stable) rather than a bounding box centroid.

**Algorithm:**
```
v(t)  = ‖centroid(t) − centroid(t−1)‖   [pixels/frame]
z(t)  = |v(t) − mean(v_window)| / std(v_window)
fires if z > 3.5 AND ≥ 6 history frames available
```

**Input:** `list[HeadState]`

**Output:**
```python
[{
    "id":       int,    # track ID
    "velocity": float,  # px/frame
    "z_score":  float,  # standard deviations above baseline
}]   # only anomalous tracks
```

**Parameters:**
| Name | Default | Description |
|------|---------|-------------|
| `window` | `10` | Rolling baseline window (frames) |
| `z_thresh` | `3.5` | Minimum z-score to fire |
| `min_history` | `6` | Frames required before firing |

---

### `skull_rotation_detector.py` — SkullRotationDetector  *(Stage 3)*

**Purpose:** Detects sudden head *rotation* — the defining physical signature
of a head impact. Computes the 2D angular velocity of the head orientation
vector directly from keypoints. No ML required, runs in microseconds.

**Orientation vector priority:**
| Priority | Keypoints used | Notes |
|----------|---------------|-------|
| 1 | left_ear → right_ear | Most stable; direct head-width vector |
| 2 | left_eye → right_eye | Used when ears not visible |
| 3 | midpoint(eyes) → nose | Fallback for partial occlusion |

**Algorithm:**
```
θ(t)  = arctan2(dy, dx)  of best available vector
ω(t)  = |unwrap(θ)(t) − unwrap(θ)(t−1)| × fps   [rad/s]
score = min(ω / 15.0,  1.0)
fires if ω > 5.0 rad/s   (≈ 9.5°/frame at 30fps)
```

**Input:** `list[HeadState]`

**Output:**
```python
[{
    "id":          int,    # track ID
    "omega_rad_s": float,  # angular velocity [rad/s]
    "score":       float,  # 0–1, saturates at 15 rad/s
}]
```

**Parameters:**
| Name | Default | Description |
|------|---------|-------------|
| `fps` | `30.0` | Video frame rate (updated from video metadata) |
| `omega_thresh` | `5.0 rad/s` | Minimum angular velocity to fire |
| `min_history` | `3` | Frames required before firing |

---

### `impact_buffer.py` — ImpactBuffer

**Purpose:** Merges signals from all three stages into confirmed impact events.
Enforces a minimum stage count and a per-pair cooldown to suppress duplicates.

**Confidence formula:**

| Stage | Weight | Score input |
|-------|--------|-------------|
| Proximity | 40% | `proximity_score` (0–1) |
| Velocity | 35% | `min(z_score / 10, 1.0)` |
| Skull rotation | 25% | `skull_score` (0–1) |

Maximum possible confidence: **1.0** (all three stages at full score).

**Rules:**
- At least **2 stages** must fire (any single stage is treated as noise)
- Minimum confidence of **0.25** to emit an event
- **45-frame cooldown** per track pair after each event (prevents duplicate
  events for the same collision)

**ImpactEvent dataclass:**
```python
@dataclass
class ImpactEvent:
    frame_idx:  int          # frame where event was confirmed
    track_ids:  list[int]    # track IDs of both people involved
    confidence: float        # combined score (0–1)
    stages:     list[str]    # e.g. ["proximity", "velocity"]
    details:    dict         # per-stage values for debugging
```

---

### `hybrik_retrospective.py` — HybrIKRetrospective  *(Stage 4)*

**Purpose:** Runs HybrIK on the short buffered window around each confirmed
impact. This is the only compute-heavy step — it runs on at most
`num_events × 2 × HALF_WINDOW` frames total, regardless of video length.

**Model:** HybrIK HRNet-W48, SMPL 24-joint skeleton, trained on Human3.6M
+ 3DPW + MPI-INF-3DHP. Uses hybrid analytical-neural inverse kinematics to
produce accurate per-joint rotation matrices.

**Preprocessing per crop:**
1. Extract body bounding box from `HeadState.body_box`
2. Pad by 15% on all sides
3. Resize to **256×256**, convert BGR→RGB
4. Normalise with ImageNet mean/std

**Output extraction:**
```
model(inp) → output.pred_theta_mats   shape: (1, 216)
reshape to (24, 3, 3)
extract joint 15 (head) → R_head   shape: (3, 3)
```

**SMPL joint 15** is the head joint, representing rotation of the head
relative to the neck in the SMPL kinematic chain.

**Output:** `dict[frame_idx → R_head (3×3 np.ndarray)]`

---

### `brain_injury_profiler.py` — BrainInjuryProfiler

**Purpose:** Converts head rotation matrices to angular velocity and computes
three validated brain injury metrics.

**Why rotation matrices instead of position differentiation:**

| Method | Formula | Problem |
|--------|---------|---------|
| Position-based | `ω ≈ cross(d(t), d(t+1)) / dt` | Amplifies detection noise |
| Rotation-based | `dR = R(t+1) @ R(t).T` then `ω = log(dR) / dt` | Exact, no noise amplification |

**Angular velocity computation:**
```
dR(t)  = R(t+1) @ R(t).T          relative rotation between frames
ω(t)   = log_map(dR(t)) / Δt      axis-angle vector / Δt  [rad/s]
ω_mag  = ‖ω(t)‖                   resultant magnitude
```
Smoothed with Savitzky-Golay filter (window=7, polynomial order=2).

**BrIC_R** — Brain Rotational Injury Criterion, resultant (Takhounts 2013):
```
BrIC_R = max(‖ω(t)‖) / 53.0
```
Value ≥ 1.0 corresponds to 50% probability of AIS2+ concussion.
Uses the resultant (direction-independent) because HybrIK outputs in
camera space, not world space.

**KLC** — Kleiven's Linear Combination, rotation component (Kleiven 2007):
```
KLC_rot = max(‖ω(t)‖)   [rad/s]
```
Values > 30 rad/s broadly associated with concussion risk in literature.

**DAMAGE** — spring-mass convolution model (Gabler 2019):
```
System:  ẍ + 2ζωn ẋ + ωn²x = ωn² × ‖ω(t)‖
         ωn = 30.1 rad/s,  ζ = 0.746
DAMAGE  = max|x(t)|
```
Models the brain as a damped oscillator driven by angular velocity.
Captures both the peak and duration of the impact pulse.
Values > 0.2 indicate elevated Maximum Principal Strain risk.

**Risk thresholds:**

| Metric | LOW | ELEVATED | HIGH |
|--------|-----|----------|------|
| BrIC_R | < 0.25 | 0.25–0.50 | ≥ 0.50 |
| KLC [rad/s] | < 15 | 15–30 | ≥ 30 |
| DAMAGE | < 0.10 | 0.10–0.20 | ≥ 0.20 |

Overall risk = highest of the three individual risk labels.

**Output report dict:**
```python
{
    "track_id":         int,
    "event_frame":      int,
    "n_frames":         int,          # frames in the rotation window
    "frame_indices":    list[int],    # frame numbers used
    "omega_xyz":        list[list],   # (T, 3) angular velocity [rad/s]
    "omega_peak_rad_s": float,
    "bric_r":           float,
    "bric_r_risk":      str,          # "LOW" / "ELEVATED" / "HIGH"
    "klc_rot_rad_s":    float,
    "klc_risk":         str,
    "damage":           float,
    "damage_risk":      str,
    "risk_summary":     str,          # overall worst-case label
}
```

---

### `track_video.py` — Pipeline Orchestrator

**Purpose:** Runs both passes, renders the annotated output video, and saves
the JSON impact report.

**Video output annotations:**

| Element | Condition | Description |
|---------|-----------|-------------|
| Green circle | All frames | Head circle (radius = `radius_px`) |
| Coloured keypoints | Visible kps | Nose=white, eyes=cyan, ears=yellow |
| Track ID label | All frames | `T{id}` above head circle |
| Red vignette border | Impact ±2 frames | Semi-transparent red overlay |
| Red circle + label | Impact ±2 frames | Impacted heads highlighted in red |
| Brain injury badge | Impact frames | `ω`, `BrIC_R`, risk label under head |
| Impact banner | Impact frame | Top-of-frame banner with tracks + confidence |
| 2-second freeze | Exact impact frame | Frame held for `fps × 2` extra frames |

**Output files:**
| File | Description |
|------|-------------|
| `{name}_annotated.mp4` | Annotated video with freeze on impact |
| `{name}.impact_report.json` | All events + brain injury profiles |

**CLI:**
```bash
python track_video.py --video path/to/video.mp4 [--window 15] [--show]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Input video path |
| `--window` | `15` | Half-window for HybrIK (frames before/after impact) |
| `--show` | off | Display live OpenCV window during Pass 1 |

---

## Tuning Reference

| Parameter | File | Default | Effect of increasing |
|-----------|------|---------|----------------------|
| `threshold` | `ProximityDetector` | `2.5` | Require heads to be closer (fewer triggers) |
| `z_thresh` | `KeypointVelocityDetector` | `3.5` | Require sharper velocity spike |
| `min_history` | `KeypointVelocityDetector` | `6` | Longer warmup before velocity fires |
| `omega_thresh` | `SkullRotationDetector` | `5.0 rad/s` | Require faster head snap |
| `CONFIDENCE_THRESHOLD` | `ImpactBuffer` | `0.25` | Only emit higher-confidence events |
| `MIN_STAGES_REQUIRED` | `ImpactBuffer` | `2` | Require more stage agreement |
| `COOLDOWN_FRAMES` | `ImpactBuffer` | `45` | Longer suppression after each event |
| `HALF_WINDOW` | `track_video.py` | `15` | More frames fed to HybrIK per event |

---

## File Structure

```
competition/
├── track_video.py              ← pipeline entry point + video renderer
├── head_tracker.py             ← Stage 0: YOLOv8x-pose + ByteTrack
├── proximity_detector.py       ← Stage 1: normalised head distance
├── velocity_detector.py        ← Stage 2: keypoint centroid z-score
├── skull_rotation_detector.py  ← Stage 3: ear-vector angular velocity
├── impact_buffer.py            ← signal merger + event queue
├── hybrik_retrospective.py     ← Stage 4: on-demand HybrIK
├── brain_injury_profiler.py    ← BrIC_R / KLC / DAMAGE from rot mats
├── PIPELINE_DOCS.md            ← this file
└── models/
    └── hybrik/
        └── hybrik_hrnet.pth    ← HybrIK weights (309 MB)

../HybrIK/                      ← cloned repo (pip install -e .)
    └── model_files/
        ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
        ├── J_regressor_h36m.npy
        └── h36m_mean_beta.npy
```

---

## References

| Metric | Paper |
|--------|-------|
| BrIC | Takhounts et al., *Stapp Car Crash Journal* 57, 2013 |
| KLC | Kleiven, *Stapp Car Crash Journal* 51, 2007 |
| DAMAGE | Gabler et al., *J. Neurotrauma* 36(4), 2019 |
| HybrIK | Li et al., *CVPR* 2021 |
