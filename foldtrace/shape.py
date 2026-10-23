"""Shape-only trajectory condition: resample a trajectory uniformly by arc length, dropping the driving angle.

The target is then a curve, not a schedule: a design that traces the same curve faster or slower in the driving
angle gets the same condition. Points are linearly interpolated on the polyline of the simulated (5 deg) samples.
"""

import numpy as np


def resample_arclength(traj: np.ndarray, n_steps, num_points: int = 32) -> np.ndarray:
    """traj [B, A, 3] (rows valid up to n_steps[b]), n_steps [B] -> [B, num_points, 3] equally spaced in arc length.

    Accepts a single trajectory [A, 3] with an int n_steps as well."""
    single = traj.ndim == 2
    traj = np.asarray(traj, dtype=np.float64)
    if single:
        traj, n_steps = traj[None], np.array([n_steps])
    n = np.asarray(n_steps, dtype=np.int64)
    B, A, _ = traj.shape
    valid_seg = np.arange(A - 1)[None] < (n[:, None] - 1)
    seg = np.where(valid_seg, np.linalg.norm(np.diff(traj, axis=1), axis=-1), 0.0)
    cum = np.concatenate([np.zeros((B, 1)), np.cumsum(seg, axis=1)], axis=1)  # [B, A]
    total = cum[np.arange(B), n - 1]
    t = np.linspace(0.0, 1.0, num_points)[None] * total[:, None]  # [B, N]
    # segment j with cum[j] <= t < cum[j + 1], restricted to the valid segments
    j = (cum[:, None, :] <= t[:, :, None]).sum(-1) - 1
    j = np.clip(j, 0, np.maximum(n - 2, 0)[:, None])
    b = np.arange(B)[:, None]
    c0, c1 = cum[b, j], cum[b, j + 1]
    w = np.where(c1 > c0, (t - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0).clip(0.0, 1.0)
    out = traj[b, j] * (1 - w[..., None]) + traj[b, np.minimum(j + 1, A - 1)] * w[..., None]
    return out[0] if single else out
