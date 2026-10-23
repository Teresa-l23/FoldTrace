"""Generate single-branch crease patterns for a target 3D curve and draw them flat.

The target is a polyline [N, 3] in the driving-crease frame (driving crease from (0, 0, 0) to (1, 0, 0), the flat
sheet in the z = 0 plane); it is resampled to the model's 32 points by arc length. Each of the K samples is corrected
(sectors >= 10 deg, summing to 2 pi), rebuilt into a crease pattern and written to designs.json with its tokens.
Samples are listed with crossing creases last; whether a design actually folds, and how closely its end vertex
follows the target, needs a rigid-origami simulator (not included, see README).

Usage:
  python scripts/sample.py --target wideS --k 16 --out out/wideS            # an example from examples/targets.json
  python scripts/sample.py --target my_curve.npy --ckpt foldtrace_abs.pt    # [N, 3] .npy / .csv / .json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foldtrace.design import correct_geometry, rebuild  # noqa: E402
from foldtrace.model import SingleBranchGenerator  # noqa: E402
from foldtrace.shape import resample_arclength  # noqa: E402

HF_REPO = "jiayin-l23/FoldTrace"


def load_model(ckpt, device):
    if ckpt in ("abs", "rel"):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(HF_REPO, f"foldtrace_{ckpt}.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = SingleBranchGenerator(**state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def load_target(spec):
    p = Path(spec)
    if p.suffix == ".npy":
        pts = np.load(p)
    elif p.suffix == ".csv":
        pts = np.loadtxt(p, delimiter=",")
    elif p.suffix == ".json":
        pts = np.asarray(json.loads(p.read_text()))
    else:
        pts = json.loads((ROOT / "examples" / "targets.json").read_text())["curves"][spec]
    pts = np.asarray(pts, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 3, f"target must be [N, 3], got {pts.shape}"
    return pts


def segments_cross(xy, edges, eps=1e-6):
    """True if two creases that share no vertex intersect in the flat sheet."""
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if set(edges[i]) & set(edges[j]):
                continue
            p, q = xy[edges[i][0]], xy[edges[i][1]]
            a, b = xy[edges[j][0]], xy[edges[j][1]]
            d1, d2 = q - p, b - a
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-12:
                continue
            e = a - p
            t = (e[0] * d2[1] - e[1] * d2[0]) / den
            s = (e[0] * d1[1] - e[1] * d1[0]) / den
            if eps < t < 1 - eps and eps < s < 1 - eps:
                return True
    return False


def draw(target, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results) + 1
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(3.2 * cols, 3.2 * rows))
    ax = fig.add_subplot(rows, cols, 1, projection="3d")
    ax.plot(*target.T, "k--", lw=1.5)
    ax.plot([0, 1], [0, 0], [0, 0], color="k", lw=3)
    ax.set_title("target", fontsize=9)
    for i, r in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 2)
        xy, chain = np.asarray(r["xy"]), r["chain"]
        on_chain = {tuple(sorted(e)) for e in zip([0] + chain[:-1], chain)}
        for a, b in r["edges"]:
            key = tuple(sorted((a, b)))
            color, lw = ("k", 3) if key == (0, 1) else ("#d6404e", 1.6) if key in on_chain else ("#4a7fc1", 1.0)
            ax.plot(*xy[[a, b]].T, color=color, lw=lw)
        end = r["kids"][-1][r["end_slot"]]
        ax.plot(*xy[end], "o", color="#f28c28", ms=6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"#{r['sample']}  {len(chain)} units" + ("  (creases cross)" if r["crossing"] else ""),
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="example name (examples/targets.json) or a .npy/.csv/.json [N, 3]")
    ap.add_argument("--ckpt", default="abs", help="checkpoint path, or abs / rel to download it from Hugging Face")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps per token")
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance scale (1 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    model = load_model(args.ckpt, args.device)
    target = load_target(args.target)
    cond = resample_arclength(target, len(target), model.cond_points)
    designs = model.generate(cond.astype(np.float32), model.cond_grid(None), args.k, args.steps,
                             guidance=args.guidance)
    results = []
    for i, d in enumerate(designs):
        geom, _ = correct_geometry(d["geom"])
        rb = rebuild(geom, d["next_slot"])
        results.append({"sample": i, "geom": geom.tolist(), "next_slot": [int(s) for s in d["next_slot"]],
                        "r": [int(r) for r in d["r"]], "end_slot": d["end_slot"], "xy": rb["xy"].tolist(),
                        "edges": rb["edges"].tolist(), "chain": rb["chain"], "kids": rb["kids"],
                        "crossing": segments_cross(rb["xy"], rb["edges"].tolist())})
    results.sort(key=lambda r: r["crossing"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "designs.json").write_text(json.dumps({"target": cond.tolist(), "designs": results}))
    draw(cond, results, out / "designs.png")
    ok = sum(not r["crossing"] for r in results)
    print(f"{len(results)} designs ({ok} without crossing creases) -> {out / 'designs.json'}, {out / 'designs.png'}")


if __name__ == "__main__":
    main()
