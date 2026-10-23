"""Train SingleBranchGenerator (foldtrace/model.py) on per-vertex token data (scripts/build_tokens.py).

Step-based loop with warmup + cosine LR, periodic validation loss, checkpoints (step_*.pt, last.pt) and --resume.
--cond shape conditions on the end trajectory resampled by arc length (no driving-angle schedule), see
foldtrace/shape.py; --shape-frame rel / pose present it without the start point / also without scale
and heading as separate droppable tokens (foldtrace/model.py).

Usage: python scripts/train.py --data TOKENS_DIR --variant joint --cond shape --out RUNS_DIR
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foldtrace.model import SingleBranchGenerator  # noqa: E402
from foldtrace.shape import resample_arclength  # noqa: E402


class TokenSamples(Dataset):
    """Samples (crop, mode, end) of one split; __getitem__ takes a list of indices and returns a batch."""

    def __init__(self, data_dir, split, max_samples=0, seed=0, in_memory=True, cond="angle", cond_points=32):
        d = Path(data_dir)
        self.cond, self.cond_points = cond, cond_points
        self.meta = json.loads((d / "metadata.json").read_text())
        L = lambda k, mmap=None: np.load(d / f"{k}.npy", mmap_mode=mmap)
        self.geom, self.next_slot, self.depth = L("crop_geom"), L("crop_next_slot"), L("crop_depth")
        self.crop, self.mode = L(f"{split}_crop"), L(f"{split}_mode")
        self.end_slot, self.n_steps = L(f"{split}_end_slot"), L(f"{split}_n_steps")
        self.traj = L(f"{split}_traj", None if in_memory else "r")
        self.index = np.arange(len(self.crop))
        if max_samples and max_samples < len(self.index):
            self.index = np.sort(np.random.default_rng(seed).choice(len(self.index), max_samples, replace=False))
        self.D = self.meta["max_chain"]
        self.A = len(self.meta["angles"])
        print(f"TokenSamples[{split}]: {len(self.index)} samples", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i = self.index[np.asarray(idx)]
        c = self.crop[i]
        depth = self.depth[c].astype(np.int64)
        k = np.arange(self.D)[None]
        tok_mask = k < depth[:, None]
        last = k == depth[:, None] - 1
        r = (self.mode[i].astype(np.int64)[:, None] >> k) & 1
        slot = np.where(last, self.end_slot[i].astype(np.int64)[:, None] + 3, self.next_slot[c].astype(np.int64))
        cls = np.where(tok_mask, 2 * slot + r, 0)
        geom = np.where(tok_mask[..., None], self.geom[c], self.geom[c][:, :1])  # pad with a valid token
        n = self.n_steps[i].astype(np.int64)
        if self.cond == "shape":
            traj = resample_arclength(self.traj[i], n, self.cond_points).astype(np.float32)
            traj_mask = np.ones(traj.shape[:2], dtype=bool)
        else:
            traj_mask = np.arange(self.A)[None] < n[:, None]
            traj = np.where(traj_mask[..., None], self.traj[i].astype(np.float32), 0.0)
        return {"geom": torch.from_numpy(geom.astype(np.float32)), "cls": torch.from_numpy(cls),
                "tok_mask": torch.from_numpy(tok_mask), "traj": torch.from_numpy(traj),
                "traj_mask": torch.from_numpy(traj_mask)}


def geom_stats(ds, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(len(ds), min(n, len(ds)), replace=False)
    c = ds.crop[ds.index[i]]
    g = ds.geom[c][np.arange(ds.D)[None] < ds.depth[c][:, None]]
    return g.mean(0).tolist(), g.std(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", choices=["factorized", "joint"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", choices=["angle", "shape"], default="angle")
    ap.add_argument("--cond-points", type=int, default=32, help="arc-length points of the shape condition")
    ap.add_argument("--shape-frame", choices=["abs", "rel", "pose"], default="abs")
    ap.add_argument("--pose-drop", type=float, default=0.3, help="per-token drop of scale / heading (frame pose)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=3)
    ap.add_argument("--dec-layers", type=int, default=6)
    ap.add_argument("--head-width", type=int, default=512)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--noise-mul", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=300_000)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--val-samples", type=int, default=50_000)
    ap.add_argument("--save-every", type=int, default=25_000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

    state = torch.load(args.resume, map_location="cpu") if args.resume else None
    cond = state["model_config"].get("cond", "angle") if state else args.cond
    cond_points = state["model_config"].get("cond_points", 32) if state else args.cond_points
    train = TokenSamples(args.data, "train", args.max_train_samples, args.seed, cond=cond, cond_points=cond_points)
    val = TokenSamples(args.data, "val", args.val_samples, args.seed, cond=cond, cond_points=cond_points)

    if state:
        cfg = state["model_config"]
    else:
        mean, std = geom_stats(train)
        cfg = dict(variant=args.variant, d_model=args.d_model, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                   max_chain=train.D, head_width=args.head_width, cond_drop=args.cond_drop,
                   geom_mean=mean, geom_std=std, cond=cond, cond_points=cond_points, shape_frame=args.shape_frame,
                   pose_drop=args.pose_drop)
    model = SingleBranchGenerator(**cfg).to(dev)
    angles = model.cond_grid(train.meta["angles"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        out = Path(args.resume).resolve().parent
    else:
        tag = args.variant if cond == "angle" else f"{args.variant}_shape{cond_points}"
        tag += "" if args.shape_frame == "abs" else f"_{args.shape_frame}"
        out = Path(args.out) / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"args": vars(args), "model_config": cfg}, indent=1))
    print(f"params {sum(p.numel() for p in model.parameters()):,}  out {out}", flush=True)
    writer = SummaryWriter(out / "tensorboard")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(args.max_steps - args.warmup, 1)
        return 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    def loader(ds, shuffle):
        sampler = RandomSampler(ds, replacement=True, num_samples=10 ** 10) if shuffle else SequentialSampler(ds)
        return DataLoader(ds, sampler=BatchSampler(sampler, args.batch_size, drop_last=shuffle), batch_size=None,
                          num_workers=args.num_workers if shuffle else 2, pin_memory=dev.type == "cuda",
                          persistent_workers=shuffle and args.num_workers > 0)

    @torch.no_grad()
    def validate():
        model.eval()
        torch.manual_seed(1234)
        acc, n = {}, 0
        for b in loader(val, False):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            o = model.loss(b, angles, noise_mul=1)
            for k, v in o.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n += 1
        model.train()
        return {k: v / n for k, v in acc.items()}

    model.train()
    t0, run = time.perf_counter(), {}
    log = open(out / "metrics.jsonl", "a")
    for batch in loader(train, True):
        if step >= args.max_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        o = model.loss(batch, angles, args.noise_mul)
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in o.items():
            run[k] = run.get(k, 0.0) + float(v)
        run["grad_norm"] = run.get("grad_norm", 0.0) + float(gn)
        if step % args.log_every == 0:
            m = {k: v / args.log_every for k, v in run.items()}
            run = {}
            rate = args.log_every * args.batch_size / (time.perf_counter() - t0)
            t0 = time.perf_counter()
            print(f"step {step} lr {lr_at(step):.2e} " + " ".join(f"{k} {v:.4f}" for k, v in m.items())
                  + f" | {rate:.0f} samples/s", flush=True)
            for k, v in m.items():
                writer.add_scalar(f"train/{k}", v, step)
            log.write(json.dumps({"step": step, "train": m}) + "\n")
        if step % args.val_every == 0 or step == args.max_steps:
            v = validate()
            print(f"VAL step {step} " + " ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
            for k, x in v.items():
                writer.add_scalar(f"val/{k}", x, step)
            log.write(json.dumps({"step": step, "val": v}) + "\n")
            log.flush()
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "model_config": cfg}
            torch.save(ckpt, out / "last.pt")
            if step % args.save_every == 0 or step == args.max_steps:
                torch.save(ckpt, out / f"step_{step}.pt")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
