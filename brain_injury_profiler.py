"""
brain_injury_profiler.py

Input : dict[frame_idx → (3,3) rotation matrix] from HybrIKRetrospective
Output: full kinematic profile + BrIC_R, KLC, DAMAGE scores

Computed quantities
-------------------
  omega_xyz          (T, 3)   angular velocity components  [rad/s]
  omega_magnitudes   (T,)     resultant ‖ω(t)‖             [rad/s]
  omega_unit_vectors (T, 3)   unit direction of ω(t)       (dimensionless)
  alpha_xyz          (T, 3)   angular acceleration dω/dt   [rad/s²]
  alpha_magnitudes   (T,)     resultant ‖α(t)‖             [rad/s²]
  omega_peak_rad_s            max(‖ω‖)
  alpha_peak_rad_s2           max(‖α‖)
  delta_omega                 peak ‖ω‖ − pre-impact baseline ‖ω‖
  pulse_duration_s            FWHM of the ‖ω‖ pulse        [s]
  BrIC_R, KLC, DAMAGE         brain-injury metrics

Key advantage of rotation matrices over position differentiation:
  dR = R(t+1) @ R(t).T    ← exact relative rotation
  ω(t) = log(dR) / dt     ← angular velocity, no noise amplification
"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import logm
from scipy.special import expit as _sigmoid
from strain_estimator import StrainEstimator

# ── Regional TBI risk curves (logistic model) ─────────────────────────────
# P(injury) = sigmoid(β0 + β1 × MPS); calibrated at 50%-risk MPS thresholds
# from Kleiven 2007 / Giordano & Kleiven 2014 / Patton 2015.
_RISK_CURVES: dict[str, tuple[float, float]] = {
    "corpus_callosum": (-5.0,  25.0),   # 50% at MPS=0.20
    "brainstem":       (-6.0,  25.0),   # 50% at MPS=0.24
    "thalamus":        (-5.5,  25.0),   # 50% at MPS=0.22
    "white_matter":    (-6.75, 25.0),   # 50% at MPS=0.27
    "grey_matter":     (-8.0,  25.0),   # 50% at MPS=0.32
    "cerebellum":      (-7.5,  25.0),   # 50% at MPS=0.30
}
_REGION_W: dict[str, float] = {
    "corpus_callosum": 0.30,
    "brainstem":       0.25,
    "thalamus":        0.20,
    "white_matter":    0.12,
    "grey_matter":     0.08,
    "cerebellum":      0.05,
}
_strain_estimator = StrainEstimator()   # CNN weights optional; falls back gracefully

# BrIC_R critical value — resultant, direction-independent (Takhounts 2013)
BRIC_OMEGA_CRIT_R = 53.0   # rad/s  → BrIC_R = 1.0 at 50% AIS2+ risk

# DAMAGE model parameters (Gabler 2019)
DAMAGE_OMEGA_N = 30.1      # rad/s  natural frequency
DAMAGE_ZETA    = 0.746     # damping ratio


# ── rotation helpers ──────────────────────────────────────────────────────────

def _rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → axis-angle vector (3,)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_R = logm(R)
    log_R = np.array(log_R).real
    rvec = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]], dtype=np.float64)
    return rvec.real


# ── kinematic computations ────────────────────────────────────────────────────

def compute_omega(
    rot_dict: dict[int, np.ndarray],
    fps: float,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert frame-indexed rotation matrices to angular velocity time series.

    Returns
    -------
    frame_indices : (T-1,)
    omega         : (T-1, 3)  rad/s
    """
    frames_sorted = sorted(rot_dict.keys())
    if len(frames_sorted) < 2:
        return np.array([]), np.zeros((0, 3))

    omega = []
    fidxs = []

    for i in range(len(frames_sorted) - 1):
        f1, f2    = frames_sorted[i], frames_sorted[i + 1]
        actual_dt = (f2 - f1) / fps
        if actual_dt <= 0:
            continue
        dR   = rot_dict[f2] @ rot_dict[f1].T
        rvec = _rot_to_rotvec(dR)
        omega.append(rvec / actual_dt)
        fidxs.append(f1)

    if not omega:
        return np.array([]), np.zeros((0, 3))

    omega = np.array(omega, dtype=np.float64)   # (T, 3)
    fidxs = np.array(fidxs)

    if smooth and len(omega) >= 7:
        for ax in range(3):
            omega[:, ax] = savgol_filter(omega[:, ax], 7, 2)

    return fidxs, omega


def compute_angular_acceleration(omega: np.ndarray, fps: float) -> np.ndarray:
    """
    α(t) = dω/dt using central finite differences.
    Returns (T, 3) in rad/s² — same length as omega via np.gradient.
    """
    if len(omega) < 2:
        return np.zeros_like(omega)
    dt = 1.0 / fps
    return np.gradient(omega, dt, axis=0)


def compute_delta_omega(omega_mag: np.ndarray, impact_local_idx: int) -> float:
    """
    Δω = peak ‖ω‖ − mean pre-impact baseline ‖ω‖.

    Parameters
    ----------
    omega_mag        : (T,)  resultant angular velocity magnitude
    impact_local_idx : index within omega_mag that corresponds to the impact frame
    """
    pre_slice = omega_mag[:max(impact_local_idx, 1)]
    baseline  = float(pre_slice.mean()) if len(pre_slice) > 0 else 0.0
    return round(float(omega_mag.max()) - baseline, 4)


def compute_pulse_duration(omega_mag: np.ndarray, fps: float,
                           threshold_frac: float = 0.5) -> float:
    """
    Full Width at Half Maximum (FWHM) of the ‖ω‖ pulse in seconds.
    Returns 0.0 if the signal never exceeds threshold_frac × peak.
    """
    if len(omega_mag) == 0:
        return 0.0
    peak      = omega_mag.max()
    threshold = peak * threshold_frac
    if threshold == 0:
        return 0.0

    peak_idx = int(np.argmax(omega_mag))
    left     = peak_idx
    while left > 0 and omega_mag[left - 1] >= threshold:
        left -= 1
    right = peak_idx
    while right < len(omega_mag) - 1 and omega_mag[right + 1] >= threshold:
        right += 1

    return round(float((right - left) / fps), 4)


def compute_unit_vectors(omega: np.ndarray) -> np.ndarray:
    """
    Unit direction vectors of ω(t) — (T, 3).
    Zero vector where ‖ω‖ ≈ 0.
    """
    mags = np.linalg.norm(omega, axis=1, keepdims=True)
    safe = np.where(mags > 1e-9, mags, 1.0)
    return np.where(mags > 1e-9, omega / safe, 0.0)


# ── injury metrics ────────────────────────────────────────────────────────────

def compute_bric_r(omega: np.ndarray) -> float:
    if len(omega) == 0:
        return 0.0
    return float(np.linalg.norm(omega, axis=1).max()) / BRIC_OMEGA_CRIT_R


def compute_klc_rotation(omega: np.ndarray) -> float:
    if len(omega) == 0:
        return 0.0
    return float(np.linalg.norm(omega, axis=1).max())


def compute_damage(omega: np.ndarray, fps: float) -> float:
    if len(omega) == 0:
        return 0.0
    dt        = 1.0 / fps
    T         = len(omega)
    t_arr     = np.arange(T) * dt
    wn, zeta  = DAMAGE_OMEGA_N, DAMAGE_ZETA
    wd        = wn * np.sqrt(max(1.0 - zeta**2, 1e-9))
    h         = np.exp(-zeta * wn * t_arr) * np.sin(wd * t_arr) / wd
    omega_mag = np.linalg.norm(omega, axis=1)
    x_t       = wn**2 * np.convolve(omega_mag, h * dt)[:T]
    return float(np.abs(x_t).max())


def _risk_label(val: float, elev_thr: float, high_thr: float) -> str:
    if val < elev_thr:  return "LOW"
    if val < high_thr:  return "ELEVATED"
    return "HIGH"


# ── main profiler ─────────────────────────────────────────────────────────────

class BrainInjuryProfiler:
    """
    Called once per confirmed impact event, after HybrIK retrospective pass.
    Computes the full kinematic + injury profile and returns it as a dict
    that is JSON-serializable (all numpy arrays converted to lists).
    """

    def profile(
        self,
        rot_dict:    dict[int, np.ndarray],
        fps:         float,
        track_id:    int,
        event_frame: int,
    ) -> dict:

        fidxs, omega = compute_omega(rot_dict, fps)

        if len(omega) == 0:
            return {"track_id": track_id, "error": "insufficient rotation data"}

        # ── kinematic quantities ───────────────────────────────────────────
        omega_mag      = np.linalg.norm(omega, axis=1)          # (T,)
        omega_units    = compute_unit_vectors(omega)              # (T, 3)
        alpha          = compute_angular_acceleration(omega, fps) # (T, 3)
        alpha_mag      = np.linalg.norm(alpha, axis=1)           # (T,)

        omega_peak     = float(omega_mag.max())
        alpha_peak     = float(alpha_mag.max())

        # impact local index (frame in the omega array closest to event_frame)
        fidx_list      = fidxs.tolist()
        impact_local   = int(np.argmin(np.abs(fidxs - event_frame)))

        delta_omega    = compute_delta_omega(omega_mag, impact_local)
        pulse_duration = compute_pulse_duration(omega_mag, fps)

        # ── injury metrics ─────────────────────────────────────────────────
        bric_r     = compute_bric_r(omega)
        klc_rot    = compute_klc_rotation(omega)
        damage     = compute_damage(omega, fps)

        bric_r_risk = _risk_label(bric_r,    0.25, 0.50)
        klc_risk    = _risk_label(klc_rot,  15.0,  30.0)
        damage_risk = _risk_label(damage,    0.10,  0.20)

        risks   = [bric_r_risk, klc_risk, damage_risk]
        overall = "HIGH" if "HIGH" in risks else (
                  "ELEVATED" if "ELEVATED" in risks else "LOW")

        return {
            # ── identification ─────────────────────────────────────────────
            "track_id":              track_id,
            "event_frame":           event_frame,
            "n_frames":              len(rot_dict),
            "frame_indices":         fidx_list,

            # ── angular velocity (3-axis vectors + magnitude + direction) ──
            "omega_vectors":         omega.tolist(),          # (T,3) [rad/s]
            "omega_magnitudes":      omega_mag.tolist(),      # (T,)  [rad/s]
            "omega_unit_vectors":    omega_units.tolist(),    # (T,3) unit
            # legacy alias kept for backward compat
            "omega_xyz":             omega.tolist(),

            # ── peak resultant ─────────────────────────────────────────────
            "omega_peak_rad_s":      round(omega_peak, 3),

            # ── change in angular velocity across the pulse ────────────────
            "delta_omega_rad_s":     delta_omega,             # Δω [rad/s]

            # ── pulse width ────────────────────────────────────────────────
            "pulse_duration_s":      pulse_duration,          # FWHM [s]

            # ── angular acceleration ───────────────────────────────────────
            "alpha_vectors":         alpha.tolist(),          # (T,3) [rad/s²]
            "alpha_magnitudes":      alpha_mag.tolist(),      # (T,)  [rad/s²]
            "alpha_peak_rad_s2":     round(alpha_peak, 3),

            # ── injury metrics ─────────────────────────────────────────────
            "bric_r":                round(bric_r,   4),
            "bric_r_risk":           bric_r_risk,
            "klc_rot_rad_s":         round(klc_rot,  3),
            "klc_risk":              klc_risk,
            "damage":                round(damage,   4),
            "damage_risk":           damage_risk,
            "risk_summary":          overall,

            # ── regional brain strain & TBI probabilities ──────────────────
            **self._regional_brain_probs(omega, damage),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _regional_brain_probs(
        self,
        omega:  np.ndarray,   # (T, 3) rad/s
        damage: float,
    ) -> dict:
        """
        Estimate per-region TBI probabilities using StrainEstimator (fallback
        mode is always available) + logistic risk curves from literature.
        Returned keys are JSON-serializable floats.
        """
        regional_mps = _strain_estimator.estimate(omega, damage_score=damage)

        regional_tbi: dict[str, float] = {}
        for region, (b0, b1) in _RISK_CURVES.items():
            mps  = regional_mps.get(region, 0.0)
            prob = float(_sigmoid(b0 + b1 * mps)) * 100.0
            regional_tbi[region] = round(prob, 1)

        overall_tbi = sum(
            regional_tbi[r] * w for r, w in _REGION_W.items()
        )
        return {
            "regional_tbi_probs":  regional_tbi,
            "tbi_probability_pct": round(overall_tbi, 1),
        }
