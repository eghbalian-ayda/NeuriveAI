"""
brain_injury_profiler.py

Input : dict[frame_idx → (3,3) rotation matrix] from HybrIKRetrospective
Output: angular velocity time series + BrIC_R, KLC, DAMAGE scores

Key advantage of using rotation matrices (vs position differentiation):
  dR = R(t+1) @ R(t).T         ← exact relative rotation, no noise amplification
  ω(t) = log(dR) / dt          ← axis-angle / Δt = angular velocity [rad/s]
"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import logm

# BrIC_R critical value — resultant, direction-independent (Takhounts 2013)
BRIC_OMEGA_CRIT_R = 53.0   # rad/s  → BrIC_R = 1.0 at 50% AIS2+ risk

# DAMAGE model parameters (Gabler 2019)
DAMAGE_OMEGA_N = 30.1      # rad/s  natural frequency
DAMAGE_ZETA    = 0.746     # damping ratio


def _rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → axis-angle vector (3,)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_R = logm(R)
    rvec = np.array([log_R[2,1], log_R[0,2], log_R[1,0]], dtype=np.float64)
    return rvec.real


def compute_omega(
    rot_dict: dict[int, np.ndarray],
    fps: float,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert frame-indexed rotation matrices to angular velocity time series.

    Returns
    -------
    frame_indices : np.ndarray (T-1,)
    omega         : np.ndarray (T-1, 3)  rad/s
    """
    frames_sorted = sorted(rot_dict.keys())
    if len(frames_sorted) < 2:
        return np.array([]), np.zeros((0, 3))

    omega = []
    fidxs = []

    for i in range(len(frames_sorted) - 1):
        f1 = frames_sorted[i]
        f2 = frames_sorted[i + 1]
        actual_dt = (f2 - f1) / fps
        if actual_dt <= 0:
            continue

        R1   = rot_dict[f1]
        R2   = rot_dict[f2]
        dR   = R2 @ R1.T
        rvec = _rot_to_rotvec(dR)
        omega.append(rvec / actual_dt)
        fidxs.append(f1)

    if not omega:
        return np.array([]), np.zeros((0, 3))

    omega = np.array(omega)   # (T-1, 3)
    fidxs = np.array(fidxs)

    if smooth and len(omega) >= 7:
        for ax in range(3):
            omega[:, ax] = savgol_filter(omega[:, ax], 7, 2)

    return fidxs, omega


def compute_bric_r(omega: np.ndarray) -> float:
    if len(omega) == 0:
        return 0.0
    return float(np.linalg.norm(omega, axis=1).max()) / BRIC_OMEGA_CRIT_R


def compute_klc_rotation(omega: np.ndarray) -> float:
    """KLC rotation component = peak resultant ω [rad/s]."""
    if len(omega) == 0:
        return 0.0
    return float(np.linalg.norm(omega, axis=1).max())


def compute_damage(omega: np.ndarray, fps: float) -> float:
    """
    DAMAGE: spring-mass convolution driven by ω_magnitude(t).
    Returns peak relative displacement. >0.2 → elevated MPS risk.
    """
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


def _risk_label(val, elev_thr, high_thr) -> str:
    if val < elev_thr:
        return "LOW"
    elif val < high_thr:
        return "ELEVATED"
    return "HIGH"


class BrainInjuryProfiler:
    """
    Called once per confirmed impact event, after HybrIK retrospective pass.
    """

    def profile(
        self,
        rot_dict: dict[int, np.ndarray],  # frame_idx → (3,3) head rotation matrix
        fps: float,
        track_id: int,
        event_frame: int,
    ) -> dict:
        """
        Returns a full brain injury report for one person in one impact event.
        """
        fidxs, omega = compute_omega(rot_dict, fps)

        if len(omega) == 0:
            return {"track_id": track_id, "error": "insufficient rotation data"}

        omega_peak = float(np.linalg.norm(omega, axis=1).max())
        bric_r     = compute_bric_r(omega)
        klc_rot    = compute_klc_rotation(omega)
        damage     = compute_damage(omega, fps)

        bric_r_risk = _risk_label(bric_r,  0.25, 0.50)
        klc_risk    = _risk_label(klc_rot, 15.0, 30.0)
        damage_risk = _risk_label(damage,  0.10, 0.20)

        risks   = [bric_r_risk, klc_risk, damage_risk]
        overall = "HIGH" if "HIGH" in risks else (
                  "ELEVATED" if "ELEVATED" in risks else "LOW")

        return {
            "track_id":         track_id,
            "event_frame":      event_frame,
            "n_frames":         len(rot_dict),
            "frame_indices":    fidxs.tolist(),
            "omega_xyz":        omega.tolist(),
            "omega_peak_rad_s": round(omega_peak, 3),
            "bric_r":           round(bric_r,     4),
            "bric_r_risk":      bric_r_risk,
            "klc_rot_rad_s":    round(klc_rot,    3),
            "klc_risk":         klc_risk,
            "damage":           round(damage,      4),
            "damage_risk":      damage_risk,
            "risk_summary":     overall,
        }
