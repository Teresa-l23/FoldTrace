"""Single-branch designs as per-vertex tokens: correction and rebuild into a flat crease pattern, curve matching.

A design is {"geom": [d, 7] (4 sectors in rad ccw from the incoming crease, 3 lengths), "next_slot": [d]
(last entry ignored), "r": [d] (rigid-body mode of each chain vertex), "end_slot": int (child slot of the last
vertex whose trajectory is compared)}. See scripts/build_tokens.py for the token definition.

Folding a design needs a rigid-origami simulator, which is not part of this repository. best_prefix_match scores a
simulated end-vertex curve against a target the way the reported numbers are computed.
"""

import math
from typing import List

import numpy as np

from foldtrace.shape import resample_arclength

TWO_PI = 2.0 * math.pi


def correct_geometry(geom: np.ndarray, min_sector_deg: float = 10.0, min_length: float = 0.05):
    """Clamp sectors to >= min and rescale their excess so they sum to 2 pi; clamp lengths.

    Returns the corrected geometry and the largest sector change (rad) made by the correction."""
    geom = np.asarray(geom, dtype=np.float64).copy()
    lo = math.radians(min_sector_deg)
    s = np.maximum(geom[:, :4], lo)
    excess = s - lo
    total = excess.sum(1, keepdims=True)
    s = lo + excess * (TWO_PI - 4 * lo) / np.maximum(total, 1e-9)
    change = float(np.abs(s - geom[:, :4]).max()) if len(geom) else 0.0
    geom[:, :4] = s
    geom[:, 4:] = np.maximum(geom[:, 4:], min_length)
    return geom, change


def rebuild(geom: np.ndarray, next_slot, merge_tol: float = 0.05):
    """Crease pattern of a design. Leaves closer than merge_tol to an existing leaf are merged (shared leaf).

    Returns dict with xy [N, 2], edges [E, 2], chain (vertex ids), kids [d, 3] and the number of merges."""
    xy = [np.zeros(2), np.array([1.0, 0.0])]
    leaves: List[int] = []
    edges, chain, kids = [(0, 1)], [1], []
    parent, merges, merge_dist = 0, 0, 0.0
    depth = len(geom)
    for k in range(depth):
        v = chain[-1]
        inc = xy[parent] - xy[v]
        base = math.atan2(inc[1], inc[0])
        cum, row = 0.0, []
        for s in range(3):
            cum += geom[k, s]
            p = xy[v] + geom[k, 4 + s] * np.array([math.cos(base + cum), math.sin(base + cum)])
            near = [(float(np.linalg.norm(xy[u] - p)), u) for u in leaves if u not in row]
            near = [t for t in near if t[0] < merge_tol]
            if near:
                dist, c = min(near)
                merges += 1
                merge_dist = max(merge_dist, dist)
            else:
                c = len(xy)
                xy.append(p)
                leaves.append(c)
            edges.append((v, c))
            row.append(c)
        kids.append(row)
        if k + 1 < depth:
            nxt = row[int(next_slot[k])]
            if nxt in leaves:
                leaves.remove(nxt)
            parent = v
            chain.append(nxt)
    return {"xy": np.array(xy), "edges": np.array(edges), "chain": chain, "kids": kids,
            "merges": merges, "merge_dist": merge_dist}


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Frechet distance between point sequences a [N, 3] and b [M, 3]."""
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    ca = np.empty_like(d)
    for i in range(len(a)):
        for j in range(len(b)):
            prev = (0.0 if i == 0 and j == 0 else
                    ca[i, j - 1] if i == 0 else ca[i - 1, j] if j == 0 else
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]))
            ca[i, j] = max(prev, d[i, j])
    return float(ca[-1, -1])


ALIGNS = ("rot", "scale", "rotscale")  # pose left free on top of the start: rotation about z, scale, both


def align_curves(c: np.ndarray, t: np.ndarray, align: str) -> np.ndarray:
    """Curves c [B, N, 3] and target t [N, 3], both with the start at the origin -> c with the least-squares rotation
    about z ("rot" in align) and/or scale ("scale" in align) about the origin applied."""
    if "rot" in align:
        th = np.arctan2((c[..., 0] * t[:, 1] - c[..., 1] * t[:, 0]).sum(-1),
                        (c[..., 0] * t[:, 0] + c[..., 1] * t[:, 1]).sum(-1))[:, None]
        c = np.stack([np.cos(th) * c[..., 0] - np.sin(th) * c[..., 1],
                      np.sin(th) * c[..., 0] + np.cos(th) * c[..., 1], c[..., 2]], -1)
    if "scale" in align:
        sc = (c * t).sum((-1, -2)) / np.maximum((c ** 2).sum((-1, -2)), 1e-12)
        c = c * np.clip(sc, 1e-6, None)[:, None, None]
    return c


def best_prefix_match(curve: np.ndarray, target_pts: np.ndarray, cuts=None, rel=False, align=""):
    """Prefix of curve [k, 3] whose arc-length resampling best matches target_pts [N, 3] (RMSE over the N points).

    cuts: candidate prefix lengths (default every sample, >= 2). rel: translation-free, the curve is moved so its
    start sits on the target's start. align (one of ALIGNS, implies rel): per prefix, also apply the least-squares
    rotation about the vertical axis through the start and/or scale about the start, i.e. the pose a "pose"
    model was not given. Returns (rmse, prefix length, resampled prefix)."""
    k, N = len(curve), len(target_pts)
    cuts = np.arange(2, k + 1) if cuts is None else np.asarray([c for c in cuts if 2 <= c <= k])
    if len(cuts) == 0:
        return None, 0, None
    if rel or align:
        curve = curve - curve[:1] + target_pts[:1]
    res = resample_arclength(np.broadcast_to(curve, (len(cuts), k, 3)), cuts, N)
    if align:
        res = align_curves(res - target_pts[:1], target_pts - target_pts[:1], align) + target_pts[:1]
    rmse = np.sqrt((np.linalg.norm(res - target_pts[None], axis=-1) ** 2).mean(-1))
    j = int(np.argmin(rmse))
    return float(rmse[j]), int(cuts[j]), res[j]
