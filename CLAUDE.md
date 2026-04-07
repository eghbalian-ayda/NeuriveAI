# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeurivAI is a **two-pass head impact detection and brain injury analysis pipeline** for football/multi-person video. It detects head impacts using existing cameras (no special equipment) and computes clinically-validated brain injury scores per impact event.

**Key design principle:** Head *keypoints* (5 COCO points) are used instead of bounding boxes, making detection scale-invariant.

## GPU

Always use GPU ID 2 for this project:

```bash
export CUDA_VISIBLE_DEVICES=2
```

Prefix any command with `CUDA_VISIBLE_DEVICES=2` or set it in your shell before running.

## Setup

```bash
# Install core dependencies
pip install -r requirements.txt

# Install HybrIK — must be cloned at /workspace/storage/HybrIK (sibling of this repo)
git clone https://github.com/Jeff-sjtu/HybrIK.git /workspace/storage/HybrIK
# Wire it into the conda env via .pth file (setup.py develop is broken on newer setuptools)
echo "/workspace/storage/HybrIK" > $CONDA_PREFIX/lib/python3.10/site-packages/hybrik-src.pth

# Place model weights (not in repo):
# - YOLOv8x-pose: auto-downloaded on first run
# - HybrIK HRNet-W48 weights → models/hybrik/hybrik_hrnet.pth (309 MB)
# - SMPL model files → HybrIK/common/utils/smplpytorch/native/models/
#     basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
#     J_regressor_h36m.npy, h36m_mean_beta.npy
```

## Running the Pipeline

```bash
# Main pipeline (Pass 1 + Pass 2)
python track_video.py --video path/to/video.mp4

# With custom HybrIK retrospective window (default: 15 frames)
python track_video.py --video path/to/video.mp4 --window 20

# With live OpenCV display during Pass 1 (slows processing)
python track_video.py --video path/to/video.mp4 --show

# Docker (requires NVIDIA GPU + CUDA 12.1)
docker-compose up
```

**Outputs:**
- `{name}_annotated.mp4` — annotated video with impact overlays and 2-second freeze at impact frames
- `{name}.impact_report.json` — all detected events + brain injury profiles (BrIC_R, KLC, DAMAGE)

## Visualization Tools

```bash
python plot_profiles.py --report test.impact_report.json [--save]
python impact_moment_viz.py --annotated out_annotated.mp4 --source test.mp4 --report test.impact_report.json
python impact_frame_viz.py --video test.mp4 --report test.impact_report.json
```

## Architecture: Two-Pass Pipeline

### Pass 1 — Real-time scan (every frame)

| Stage | File | Signal |
|-------|------|--------|
| 0 — Tracking | `head_tracker.py` | YOLOv8x-pose + ByteTrack → `HeadState` per person per frame |
| 1 — Proximity | `proximity_detector.py` | Pairwise normalized head distance (< 2.5 head-radii fires) |
| 2 — Velocity | `velocity_detector.py` | Z-score spike on keypoint centroid velocity (z > 3.5 fires) |
| 3 — Rotation | `skull_rotation_detector.py` | Ear-vector angular velocity (> 5.0 rad/s fires) |
| Merger | `impact_buffer.py` | Requires ≥2 stages + confidence > 0.25 → `ImpactEvent` |

`ImpactBuffer` confidence formula:
```
confidence = proximity_score × 0.40 + min(z_score/10, 1) × 0.35 + skull_score × 0.25
```

45-frame cooldown per track pair after each confirmed event.

### Pass 2 — Deep analysis (only on ±window frames around confirmed impacts)

1. **`hybrik_retrospective.py`** — Runs HybrIK HRNet-W48 on cropped body regions to extract SMPL head rotation matrices `R_head (3×3)` for each frame in the window. Only joint 15 (head) is used.
2. **`brain_injury_profiler.py`** — Converts rotation matrices → angular velocity via matrix logarithm (`dR = R(t+1) @ R(t).T`), then computes three clinical metrics:
   - **BrIC_R** (Takhounts 2013): `max(‖ω‖) / 53.0` — ≥ 1.0 = 50% concussion probability
   - **KLC** (Kleiven 2007): peak `‖ω‖` in rad/s
   - **DAMAGE** (Gabler 2019): spring-mass oscillator (`ωn=30.1`, `ζ=0.746`) driven by ‖ω(t)‖

### Data flow

```
HeadKeypointTracker → {ProximityDetector, VelocityDetector, SkullRotationDetector}
                    → ImpactBuffer (merge + filter)
                    → ImpactEvent list
                    → HybrIKRetrospective (buffered frames from Pass 1)
                    → BrainInjuryProfiler
                    → JSON report + annotated video
```

`HeadKeypointTracker` maintains two rolling buffers during Pass 1:
- `frame_buffer` (deque) — raw frames for HybrIK crops
- `kp_history` (dict: track_id → list[HeadState]) — keypoint history for retrospective lookup

## Key Tunable Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `threshold` | `ProximityDetector.__init__` | 2.5 | Min head-radius distance to fire |
| `z_thresh` | `KeypointVelocityDetector.__init__` | 3.5 | Z-score required to fire |
| `omega_thresh` | `SkullRotationDetector.__init__` | 5.0 rad/s | Angular velocity to fire |
| `CONFIDENCE_THRESHOLD` | `impact_buffer.py` | 0.25 | Min confidence to emit event |
| `MIN_STAGES_REQUIRED` | `impact_buffer.py` | 2 | Min stages that must fire |
| `COOLDOWN_FRAMES` | `impact_buffer.py` | 45 | Frames suppressed after event |
| `HALF_WINDOW` | `track_video.py` | 15 | HybrIK window radius (frames) |

## Test Artifacts

`test.impact_report.json` and `test.impact_report_profiles_frame00077.png` are sample outputs from a real run and can be used to test visualization scripts without re-running the full pipeline. There is no automated test suite.

## Detailed Documentation

`PIPELINE_DOCS.md` contains in-depth technical documentation on each module, signal logic, and clinical metric derivations.
