# NeurivAI — Impact Detection Pipeline Documentation

## Overview

This pipeline detects head impacts in video footage of helmeted individuals.
It operates in two main stages: **helmet tracking** (Stage 0) and **impact detection** (Stages 1–3).
All three impact detection stages run on **every frame** — impacts do not necessarily cause visible overlap.

---

## Architecture Diagram

```
Video File
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 0 — HelmetTracker          helmet_tracker.py     │
│                                                          │
│  YOLOv8m (keremberke/yolov8m-hard-hat-detection)        │
│  + ByteTrack multi-object tracker                        │
│                                                          │
│  IN  : video path                                        │
│  OUT : frame (BGR numpy), detections[]                   │
└───────────────────────┬─────────────────────────────────┘
                        │  frame + detections[]
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
┌──────────────┐ ┌────────────┐ ┌─────────────────────┐
│  Stage 1     │ │  Stage 2   │ │  Stage 3             │
│  IoU         │ │  HOT Model │ │  Velocity Anomaly    │
│  Heuristic   │ │            │ │  Detector            │
│              │ │            │ │                      │
│ iou_         │ │ hot_       │ │ velocity_            │
│ detector.py  │ │ detector.py│ │ detector.py          │
└──────┬───────┘ └─────┬──────┘ └──────────┬───────────┘
       │               │                   │
       └───────────────┴───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  ImpactDetector                   impact_detector.py    │
│                                                          │
│  Combines signals from all 3 stages.                     │
│  Emits impact events with confidence score.              │
│                                                          │
│  OUT : impact events[], filtered by confidence ≥ 0.18   │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
                  track_video.py
                  (console output)
```

---

## Modules

---

### `helmet_tracker.py` — HelmetTracker

**Purpose:** Detects and tracks helmeted heads across video frames. Serves as the
input source for all downstream stages.

**Model:** YOLOv8m fine-tuned on hard-hat detection
(`keremberke/yolov8m-hard-hat-detection`, downloaded from HuggingFace).
Tracking is handled by **ByteTrack** (built into Ultralytics), which assigns
persistent IDs across frames using a Kalman filter + Hungarian assignment.

**Input:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `str` | Path to video file |
| `show` | `bool` | Display window while processing |
| `save` | `bool` | Save annotated output video |

**Output (per frame, yielded):**
| Field | Type | Description |
|-------|------|-------------|
| `frame` | `np.ndarray` (H×W×3, BGR) | Raw video frame |
| `detections` | `list[dict]` | Per-detection results |

Each detection dict:
```python
{
    "id":    int,        # persistent track ID (ByteTrack)
    "label": str,        # "Hardhat" or "NO-Hardhat"
    "box":   [x1,y1,x2,y2],  # bounding box in pixel coords
    "conf":  float,      # detection confidence (0–1)
}
```

**Classes detected:** `Hardhat`, `NO-Hardhat`

---

### `hot_detector.py` — HOTDetector

**Purpose:** Determines whether a detected helmet region shows signs of physical
contact, and identifies which body parts are involved. Acts as the secondary
confirmation stage after IoU.

**Model architecture:**

```
Input frame crop (256×256 RGB)
        │
        ▼
┌──────────────────────────────┐
│  _HOTEncoder (ResNet-50)     │
│                               │
│  Deep stem:                   │
│  Conv(3→64, 3×3, s=2)        │
│  Conv(64→64, 3×3, s=1)       │
│  Conv(64→128, 3×3, s=1)      │
│  MaxPool(3×3, s=2)           │
│                               │
│  Layer1: 3× Bottleneck       │  128 → 256
│  Layer2: 4× Bottleneck       │  256 → 512
│  Layer3: 6× Bottleneck       │  512 → 1024
│  Layer4: 3× Bottleneck       │  1024 → 2048
└──────────────┬───────────────┘
               │  [B, 2048, H/32, W/32]
               ▼
┌──────────────────────────────┐
│  _HOTDecoder                 │
│                               │
│  cbr:      2048→512 (main)   │
│  cbr_part: 2048→512 (parts)  │
│                               │
│  _PartBranch (18 body parts) │
│  ├─ 18× CBR(512→64)          │
│  ├─ conv_last_part: 512→18   │  part segmentation
│  └─ conv_last_cont: 1152→18  │  part contact map
│                               │
│  conv_last:   512→18         │  per-part contact heatmap
│  conv_binary: 512→2          │  contact / no-contact
└──────────────────────────────┘
```

**18 body parts tracked:**
`head`, `neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`,
`left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`,
`left_ankle`, `right_ankle`, `left_hand`, `right_hand`, `left_foot`, `right_foot`

**Weights:** `models/hot-c1/encoder_epoch_14.pth`, `models/hot-c1/decoder_epoch_14.pth`

**Input:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `frame_bgr` | `np.ndarray` | Full video frame (BGR) |
| `box` | `[x1,y1,x2,y2]` | Bounding box to analyse (padded 20%) |

**Output:**
| Field | Type | Description |
|-------|------|-------------|
| `contact_prob` | `float` | Mean spatial probability of contact (0–1) |
| `top_parts` | `list[str]` | Top-3 body parts with highest contact activation |

**Filtering applied in ImpactDetector:**
- Only fires if `contact_prob ≥ 0.60`
- Only fires if at least one of the top parts is in `{head, neck, left_shoulder, right_shoulder}`

---

### `iou_detector.py` — IoUDetector

**Purpose:** Primary real-time trigger. Computes pairwise Intersection over Union
between all helmet bounding boxes in the frame. A high IoU means two helmets are
overlapping — strong geometric evidence of a collision.

**How it works:**
For every pair of tracked boxes `(A, B)`:
1. Compute intersection area
2. Compute union area
3. `IoU = intersection / union`
4. Also compute normalised centre-to-centre distance (proximity)
5. Emit a hit if `IoU ≥ threshold`

**Input:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `detections` | `list[dict]` | Helmet detections from HelmetTracker |

**Output:**
```python
[
    {
        "id_a":      int,    # track ID of first helmet
        "id_b":      int,    # track ID of second helmet
        "iou":       float,  # intersection-over-union (0–1)
        "proximity": float,  # normalised centre distance
    },
    ...  # sorted by iou descending
]
```

**Default threshold:** `0.05` (low, to catch glancing touches)

---

### `velocity_detector.py` — VelocityAnomalyDetector

**Purpose:** Physics-based impact confirmation. Tracks the centroid of each helmet
box over time and flags sudden changes in speed using a z-score test. Catches
impacts that cause no visible overlap (e.g. glancing blows, head snap-back).

**How it works:**
For each tracked ID per frame:
1. Compute centroid `(cx, cy)` from box
2. Append to rolling history window (default: 8 frames)
3. Compute frame-to-frame Euclidean displacement (velocity)
4. Compare the latest velocity to the window baseline:
   - `z = |v_current − mean(v_baseline)| / std(v_baseline)`
5. Flag as anomaly if `z > z_thresh`

A minimum history of 5 frames is required before any anomaly is reported,
preventing false positives when a new track ID first appears.

**Input:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `detections` | `list[dict]` | Helmet detections from HelmetTracker |

**Output:**
```python
[
    {
        "id":       int,    # track ID
        "velocity": float,  # latest displacement in px/frame
        "z_score":  float,  # how many std deviations above baseline
    },
    ...  # only anomalous tracks
]
```

**Parameters:**
| Name | Default | Description |
|------|---------|-------------|
| `window` | `8` | Rolling history length (frames) |
| `z_thresh` | `3.0` | Minimum z-score to flag |
| `min_history` | `5` | Frames required before detection starts |

---

### `impact_detector.py` — ImpactDetector

**Purpose:** Orchestrates all three stages. Runs them every frame, merges their
signals into unified impact events, and scores each event.

**Confidence scoring:**

Each stage contributes a weighted portion to the final confidence score:

| Stage | Weight | Score calculation |
|-------|--------|-------------------|
| IoU | 30% | `iou_value × 0.30` |
| HOT | 50% | `contact_prob × 0.50` |
| Velocity | 20% | `min(z_score / 10, 1.0) × 0.20` |

Scores from multiple stages are **additive** — an event confirmed by all three
stages can reach a confidence of 1.0.

**Event merging logic:**
- IoU events are keyed as `"id_a-id_b"` (pair of IDs)
- HOT and velocity events are keyed as `"id"` (single track)
- If a single-ID event matches one ID in an IoU pair, stages are merged

**Input:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `frame_bgr` | `np.ndarray` | Current video frame (BGR) |
| `detections` | `list[dict]` | Output from HelmetTracker |

**Output:**
```python
[
    {
        "id":         str or int,  # track ID or "A-B" pair
        "confidence": float,       # combined score (0–1)
        "stages":     list[str],   # which stages fired, e.g. ["iou", "velocity"]
        "parts":      list[str],   # contacted body parts (from HOT, if fired)
        "details": {
            "iou":      float,     # present if IoU stage fired
            "hot_prob": float,     # present if HOT stage fired
            "velocity": float,     # present if velocity stage fired
            "z_score":  float,     # present if velocity stage fired
        }
    },
    ...  # sorted by confidence descending
]
```

**Parameters:**
| Name | Default | Description |
|------|---------|-------------|
| `iou_threshold` | `0.05` | Min IoU to trigger IoU stage |
| `hot_threshold` | `0.60` | Min contact prob to trigger HOT stage |
| `vel_z_thresh` | `3.0` | Min z-score to trigger velocity stage |

---

### `track_video.py` — Pipeline Entry Point

**Purpose:** Connects HelmetTracker to ImpactDetector and prints results.

**Flow:**
1. Initialise `HelmetTracker` and `ImpactDetector`
2. Stream video frame by frame via `tracker.track()`
3. Pass each `(frame, detections)` pair to `impact.detect()`
4. Filter events with `confidence < 0.18` (observed noise floor)
5. Print structured output per event

---

## Tuning Reference

| Parameter | Location | Current Value | Effect of increasing |
|-----------|----------|---------------|----------------------|
| `iou_threshold` | `ImpactDetector` | `0.05` | Fewer IoU triggers (require more overlap) |
| `hot_threshold` | `ImpactDetector` | `0.60` | Fewer HOT triggers (require stronger contact signal) |
| `vel_z_thresh` | `ImpactDetector` + `VelocityAnomalyDetector` | `3.0` | Fewer velocity triggers (require sharper speed spike) |
| `min_history` | `VelocityAnomalyDetector` | `5` | Longer warmup before velocity can fire |
| `window` | `VelocityAnomalyDetector` | `8` | Longer baseline for z-score (more stable, slower to react) |
| Confidence filter | `track_video.py` | `0.18` | Only show higher-confidence events |

---

## File Structure

```
competition/
├── track_video.py          ← pipeline entry point
├── helmet_tracker.py       ← Stage 0: YOLOv8m + ByteTrack
├── impact_detector.py      ← stage orchestrator + confidence scoring
├── hot_detector.py         ← Stage 2: HOT encoder-decoder
├── iou_detector.py         ← Stage 1: pairwise IoU heuristic
├── velocity_detector.py    ← Stage 3: centroid velocity anomaly
└── models/
    └── hot-c1/
        ├── encoder_epoch_14.pth
        └── decoder_epoch_14.pth
```
