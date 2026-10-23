"""Per-vertex token dataset from single-branch crops.

The input is a directory of crops_*.npz shards from the crop + simulation step, which is not part of this repository
(it needs a rigid-folding simulator). Each shard holds, per crop: the flat pattern (vertices, edges, orig_ids), the
chain (chain, depth), its three possible end vertices (ends, ends_orig_leaf), and per rigid-body mode the simulated
end-vertex trajectories over the driving angles (traj [M, 3, A, 3], final_idx, jump, n_combo).

A crop is the chain 1 = c_0 -> c_1 -> ... -> c_{d-1} = p (driving crease 0 -> 1 is fixed). Chain vertex c_k is one
token, described in its own frame: its incoming crease points from c_k to its parent (vertex 0 for c_0), and its
three children are ordered counter-clockwise from that crease.

  sectors[4]  the four sector angles around c_k, counter-clockwise from the incoming crease (sum 2 pi)
  lengths[3]  crease lengths to the three children, in slot order
  next_slot   slot of c_{k+1} (the child that continues); the last token has none (-1)
  r           rigid-body mode of c_k itself (bit k of the sample's mode)
  end_slot    last token only: slot of the end vertex whose trajectory is the condition

Inputs that are derived from the tokens (not generated): xy of c_k and of its children (dataset frame: vertex 0 at
the origin, driving crease 0-1 along +x with length 1).

Samples are (crop, mode, end); kept when the end is a leaf of the original pattern, the crop folds to
>= --min-final-angle and the end moves at least --min-amplitude. Splits are by original pattern.

Usage: python scripts/build_tokens.py --run CROPS_DIR --out OUT_DIR [--max-shards N]
"""

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def ccw_angle(base: np.ndarray, vec: np.ndarray) -> float:
    a = math.atan2(vec[1], vec[0]) - math.atan2(base[1], base[0])
    return a % (2.0 * math.pi)


def chain_tokens(xy: np.ndarray, edges: np.ndarray, chain: list):
    """Tokens of one crop (local vertex ids). Returns geometry [d, 7], next_slot [d], kids [d, 3]."""
    out = {}
    for i, j in edges:
        out.setdefault(int(i), []).append(int(j))
    geom, next_slot, kids_sorted = [], [], []
    for k, v in enumerate(chain):
        parent = 0 if k == 0 else chain[k - 1]
        inc = xy[parent] - xy[v]
        kids = out[v]
        if len(kids) != 3:
            raise ValueError(f"vertex {v} has {len(kids)} children")
        turn = sorted((ccw_angle(inc, xy[c] - xy[v]), c) for c in kids)
        a = [t for t, _ in turn]
        sectors = [a[0], a[1] - a[0], a[2] - a[1], 2.0 * math.pi - a[2]]
        order = [c for _, c in turn]
        lengths = [float(np.linalg.norm(xy[c] - xy[v])) for c in order]
        geom.append(sectors + lengths)
        next_slot.append(order.index(chain[k + 1]) if k + 1 < len(chain) else -1)
        kids_sorted.append(order)
    return np.asarray(geom, np.float32), np.asarray(next_slot, np.int8), np.asarray(kids_sorted, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory of crops_*.npz shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-final-angle", type=float, default=90.0)
    ap.add_argument("--min-amplitude", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--max-chain", type=int, default=8)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{args.run}/crops_*.npz"))
    if args.max_shards:
        files = files[: args.max_shards]
    split_u = np.random.default_rng(args.seed).random(8_000_000)  # per original pattern (group * 20000 + key)
    D = args.max_chain

    crop = {k: [] for k in ("group", "key", "depth", "shard", "local", "geom", "next_slot", "xy", "kids_xy",
                            "end_slot", "chain")}
    samp = {s: {k: [] for k in ("crop", "mode", "end", "end_slot", "n_steps", "final_angle", "amplitude", "jump",
                                "traj")} for s in SPLITS}
    stats = {"crops_seen": 0, "crops_kept": 0, "token_errors": 0, "candidates": 0}
    t0 = time.perf_counter()
    angles = None
    for si, fn in enumerate(files):
        d = np.load(fn)
        angles = d["angles"]
        min_idx = int(np.searchsorted(angles, args.min_final_angle - 1e-6))
        co = np.concatenate([[0], np.cumsum(d["n_combo"])])
        ch = np.concatenate([[0], np.cumsum(d["depth"])])
        vo = np.concatenate([[0], np.cumsum(d["n_vert"])])
        eo = np.concatenate([[0], np.cumsum(d["n_edge"])])
        traj_all, final_all, jump_all = d["traj"], d["final_idx"], d["jump"]
        leaf_all = d["ends_orig_leaf"].reshape(-1, 3)
        ends_all = d["ends"].reshape(-1, 3)
        for i in range(len(d["depth"])):
            stats["crops_seen"] += 1
            t = traj_all[co[i]:co[i + 1]]  # [M, 3, A, 3]
            f = final_all[co[i]:co[i + 1]].astype(int)
            stats["candidates"] += int(leaf_all[i].sum()) * len(f)
            amp = np.nan_to_num(np.nanmax(np.linalg.norm(t - t[:, :, :1], axis=-1), axis=-1))  # [M, 3]
            keep = leaf_all[i][None, :] & (f[:, None] >= min_idx) & (amp >= args.min_amplitude)
            if not keep.any():
                continue
            old = d["orig_ids"][vo[i]:vo[i + 1]].tolist()
            chain_orig = d["chain"][ch[i]:ch[i + 1]].tolist()
            chain = [old.index(c) for c in chain_orig]
            xy = d["vertices"][vo[i]:vo[i + 1]].astype(np.float64)
            try:
                geom, nxt, kids = chain_tokens(xy, d["edges"][eo[i]:eo[i + 1]], chain)
            except Exception:
                stats["token_errors"] += 1
                continue
            end_slot = [kids[-1].tolist().index(int(e)) for e in ends_all[i]]
            dep = len(chain)
            c_id = len(crop["group"])
            g, key = int(d["group"][i]), int(d["key"][i])
            pad = lambda a, fill=0: np.concatenate([a, np.full((D - len(a), *a.shape[1:]), fill, a.dtype)])
            crop["group"].append(g)
            crop["key"].append(key)
            crop["depth"].append(dep)
            crop["shard"].append(si)
            crop["local"].append(i)
            crop["geom"].append(pad(geom))
            crop["next_slot"].append(pad(nxt, -1))
            crop["xy"].append(pad(xy[chain].astype(np.float32)))
            crop["kids_xy"].append(pad(xy[kids].astype(np.float32)))
            crop["end_slot"].append(np.asarray(end_slot, np.int8))
            crop["chain"].append(pad(np.asarray(chain_orig, np.int16), -1))
            u = split_u[g * 20000 + key]
            split = "val" if u < args.val_frac else "test" if u < args.val_frac + args.test_frac else "train"
            ms, js = np.nonzero(keep)
            s = samp[split]
            s["crop"].append(np.full(len(ms), c_id, np.int32))
            s["mode"].append(ms.astype(np.int16))
            s["end"].append(js.astype(np.int8))
            s["end_slot"].append(np.asarray(end_slot, np.int8)[js])
            s["n_steps"].append((f[ms] + 1).astype(np.int8))
            s["final_angle"].append(angles[f[ms]].astype(np.float32))
            s["amplitude"].append(amp[ms, js].astype(np.float32))
            s["jump"].append(jump_all[co[i] + ms, js].astype(np.float16))
            s["traj"].append(np.nan_to_num(t[ms, js]).astype(np.float16))
        print(f"shard {si + 1}/{len(files)}: {len(crop['group'])} crops, "
              f"{sum(len(x) for x in samp['train']['crop'])} train samples, {time.perf_counter() - t0:.0f}s", flush=True)

    stats["crops_kept"] = len(crop["group"])
    for k, val in crop.items():
        np.save(out / f"crop_{k}.npy", np.stack(val) if val and np.ndim(val[0]) else np.asarray(val, np.int32))
    counts = {}
    for split, s in samp.items():
        for k in ("crop", "mode", "end", "end_slot", "n_steps", "final_angle", "amplitude", "jump"):
            np.save(out / f"{split}_{k}.npy", np.concatenate(s[k]) if s[k] else np.zeros(0))
            s[k] = None
        total = sum(len(x) for x in s["traj"])
        arr = np.lib.format.open_memmap(out / f"{split}_traj.npy", mode="w+", dtype=np.float16,
                                        shape=(total, len(angles), 3))
        pos = 0
        while s["traj"]:
            x = s["traj"].pop(0)
            arr[pos:pos + len(x)] = x
            pos += len(x)
        arr.flush()
        counts[split] = total

    meta = {
        "format": "single_branch_vertex_tokens",
        "max_chain": D,
        "angles": [float(a) for a in angles],
        "num_crops": len(crop["group"]),
        "num_samples": counts,
        "token_fields": {
            "geom": "[sector_1..4 (rad, ccw from incoming crease, sum 2pi), length_1..3 (slot order)]",
            "next_slot": "slot of the next chain vertex, -1 on the last token",
            "r": "bit k of the sample's mode = rigid-body mode of chain vertex k",
            "end_slot": "slot (on the last token) of the end vertex of the sample",
        },
        "inputs": {"xy": "chain vertex position", "kids_xy": "children positions in slot order"},
        "coordinate_frame": "dataset frame: vertex 0 at origin, driving crease 0-1 along +x with length 1",
        "trajectory_condition": "end-vertex position per driving angle, same frame, start not subtracted",
        "args": vars(args),
        "stats": stats,
        "seconds": time.perf_counter() - t0,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("num_crops", "num_samples", "stats", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
