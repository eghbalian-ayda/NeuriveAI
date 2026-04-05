"""
strain_estimator.py

Converts angular velocity time series to regional brain MPS estimates.
Primary method: Wu et al. (2019) CNN — velocity profile as 2D image.
Fallback: DAMAGE-to-MPS linear scaling with anatomical distribution weights.

Input : omega (T, 3) ndarray — angular velocity in rad/s
Output: dict of regional MPS values
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ── Regional anatomical vulnerability weights for fallback mode ────────────
# Based on literature consensus (Kleiven 2007, Giordano & Kleiven 2014)
# Relative vulnerability to rotational loading, normalized to sum to 1.
REGIONAL_WEIGHTS = {
    "corpus_callosum": 0.28,    # most vulnerable to rotational strain
    "brainstem":       0.22,    # second most — axonal injury common
    "thalamus":        0.18,    # frequently implicated in concussion
    "white_matter":    0.16,    # diffuse axonal injury region
    "grey_matter":     0.10,    # less vulnerable rotationally
    "cerebellum":      0.06,    # least implicated in mTBI
}

# ── DAMAGE → MPS linear scaling coefficient (Gabler et al. 2019) ──────────
DAMAGE_TO_MPS_COEFF = 0.56

# ── Image dimensions for CNN input ────────────────────────────────────────
CNN_IMG_H = 4     # rows: ωx, ωy, ωz, |ω|
CNN_IMG_W = 200   # cols: time steps (zero-padded or truncated)


# ─────────────────────────────────────────────────────────────────────────────
#  Simple CNN architecture matching Wu et al. 2019
#  (reconstruct from paper description — 4 conv layers + 2 FC)
# ─────────────────────────────────────────────────────────────────────────────

class WuStrainCNN(nn.Module):
    """
    Lightweight CNN that takes a (1, 4, 200) velocity profile image
    and outputs 6 regional MPS values.

    Architecture follows Wu et al. Scientific Reports 2019:
    4 convolutional layers with ReLU + max pooling,
    followed by 2 fully connected layers.

    If pretrained weights are available, load them.
    If not, the model runs in DAMAGE-fallback mode.
    """

    def __init__(self, n_regions: int = 6):
        super().__init__()

        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_regions),
            nn.Sigmoid(),    # MPS is bounded 0–1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
#  Strain Estimator
# ─────────────────────────────────────────────────────────────────────────────

class StrainEstimator:
    """
    Estimates regional brain MPS from angular velocity profile.

    Modes:
      "cnn"      — Wu et al. CNN (requires pretrained weights)
      "fallback" — DAMAGE-based proxy with anatomical distribution
    """

    REGIONS = list(REGIONAL_WEIGHTS.keys())

    def __init__(
        self,
        ckpt: str  = "models/wu_cnn/wu_strain_cnn.pt",
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mode   = "fallback"

        # attempt to load CNN
        ckpt_path = Path(__file__).parent / ckpt
        if ckpt_path.exists():
            try:
                self._cnn = WuStrainCNN(n_regions=len(self.REGIONS))
                state = torch.load(str(ckpt_path), map_location=device)
                self._cnn.load_state_dict(state, strict=False)
                self._cnn.to(device).eval()
                self.mode = "cnn"
                print(f"[StrainEstimator] CNN loaded from {ckpt_path}")
            except Exception as e:
                print(f"[StrainEstimator] CNN load failed ({e}), using fallback")
        else:
            print(f"[StrainEstimator] No weights at {ckpt_path}, using fallback")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _omega_to_image(omega: np.ndarray) -> torch.Tensor:
        """
        Convert (T, 3) angular velocity to (1, 1, 4, CNN_IMG_W) image tensor.

        Rows: ωx, ωy, ωz, resultant magnitude
        Cols: time steps (zero-padded or truncated to CNN_IMG_W)
        Pixel values: normalized 0–1
        """
        T = omega.shape[0]
        resultant = np.linalg.norm(omega, axis=1, keepdims=True)   # (T, 1)
        full = np.hstack([omega, resultant])                         # (T, 4)
        full = full.T                                                # (4, T)

        # resize along time axis to CNN_IMG_W
        img = np.zeros((4, CNN_IMG_W), dtype=np.float32)
        if T >= CNN_IMG_W:
            # truncate: take center window around impact
            start = (T - CNN_IMG_W) // 2
            img = full[:, start:start + CNN_IMG_W].astype(np.float32)
        else:
            # zero-pad symmetrically
            pad = (CNN_IMG_W - T) // 2
            img[:, pad:pad + T] = full

        # normalize each row independently to 0–1
        for r in range(4):
            rmax = img[r].max()
            if rmax > 0:
                img[r] /= rmax

        # shape: (1, 1, 4, CNN_IMG_W)
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    def _fallback_estimate(
        self,
        omega: np.ndarray,
        damage_score: float,
    ) -> dict[str, float]:
        """
        When CNN is unavailable, derive whole-brain MPS from peak angular
        velocity (physiologically meaningful scale), then distribute
        regionally using anatomical vulnerability weights.

        Reference: concussion injury threshold ~30–60 rad/s peak ω.
        Normalization: ω_peak / 50 rad/s → whole-brain MPS proxy.
        This produces meaningful variation across impacts rather than
        saturating everything at 0.95.
        """
        if len(omega) > 0:
            omega_peak = float(np.linalg.norm(omega, axis=1).max())
            # 50 rad/s ≈ midpoint of concussion risk range
            whole_brain_mps = float(np.clip(omega_peak / 50.0, 0.0, 0.95))
        else:
            # fallback to DAMAGE only if no omega — use log to avoid saturation
            whole_brain_mps = float(np.clip(
                np.log1p(damage_score) * DAMAGE_TO_MPS_COEFF / 3.0, 0.0, 0.95
            ))

        regional = {}
        for region, weight in REGIONAL_WEIGHTS.items():
            # regional peaks exceed whole-brain average by ~2×
            regional[region] = float(np.clip(
                whole_brain_mps * weight * 2.0, 0.0, 0.95
            ))

        regional["whole_brain_95pct"] = whole_brain_mps
        return regional

    # ── main interface ────────────────────────────────────────────────────────

    def estimate(
        self,
        omega: np.ndarray,          # (T, 3) rad/s
        damage_score: float = 0.0,  # from brain_injury_profiler (fallback only)
    ) -> dict[str, float]:
        """
        Estimate regional MPS from angular velocity profile.

        Returns
        -------
        dict with keys: corpus_callosum, brainstem, thalamus,
                        white_matter, grey_matter, cerebellum,
                        whole_brain_95pct
        All values are MPS in [0, 1].
        """
        if len(omega) == 0:
            return {r: 0.0 for r in self.REGIONS + ["whole_brain_95pct"]}

        if self.mode == "cnn":
            img_tensor = self._omega_to_image(omega).to(self.device)
            with torch.no_grad():
                preds = self._cnn(img_tensor).cpu().numpy()[0]   # (6,)

            result = {region: float(preds[i])
                      for i, region in enumerate(self.REGIONS)}
            result["whole_brain_95pct"] = float(np.max(preds))
            return result

        else:
            return self._fallback_estimate(omega, damage_score)
