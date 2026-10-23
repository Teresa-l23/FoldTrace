"""Trajectory-conditioned generation of single-branch designs, one token per chain vertex.

Token k (chain vertex c_k): geometry g_k = [4 sectors (rad, ccw from the incoming crease), 3 lengths] and a discrete
class y_k = 2 * a_k + r_k with a_k in {continue via slot 0/1/2, end at slot 0/1/2} and r_k the mode of c_k itself.

Condition ("cond"): "angle" = end-vertex position per driving angle (5 deg grid, the angle is a per-point input);
"shape" = the curve resampled to `cond_points` points equally spaced in arc length (foldtrace/shape.py),
the arc-length fraction replacing the angle, so the driving-angle schedule is not part of the condition.
"shape_frame" sets how the shape curve is presented (see shape_frame_transform): "abs" = driving-crease frame,
"rel" = start moved to the origin (the start point is not a condition), "pose" = start at the origin, unit arc length,
turned about z to a canonical heading, with scale and heading as two extra condition tokens that are dropped
independently with probability `pose_drop` in training, so generation can leave either one free.

Backbone: a transformer encoder over the target trajectory points and a causal transformer decoder over chain
vertices with cross-attention to the trajectory. The decoder input at position k only uses what is fixed before
token k is generated: the previous token (g_{k-1}, y_{k-1}) and 2D features derived from all previous tokens
(position of c_k, direction of its incoming crease, positions of c_{k-1}'s children), computed by `chain_xy`
in training and in generation alike.

Heads per token, given the decoder state h_k:
- "factorized": p(y_k | h_k) by a 12-way softmax, then g_k ~ DDPM conditioned on (h_k, y_k);
- "joint": y_k (absorbing-mask discrete diffusion) and g_k (Gaussian diffusion) denoised together by one network
  with a shared timestep.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 12  # (continue slot 0/1/2, end slot 0/1/2) x r
MASK_CLASS = NUM_CLASSES
GEOM_DIM = 7


def chain_xy(geom: torch.Tensor, cls: torch.Tensor):
    """2D layout of chains. geom [B, D, 7] (rad, lengths), cls [B, D] (classes; slot = (cls // 2) % 3).

    Returns pos [B, D, 2] (c_k), parent [B, D, 2], kids [B, D, 3, 2]; entries after the end are garbage."""
    B, D, _ = geom.shape
    pos = geom.new_zeros(B, 2)
    pos[:, 0] = 1.0
    parent = geom.new_zeros(B, 2)
    out_pos, out_par, out_kids = [], [], []
    for k in range(D):
        inc = parent - pos
        base = torch.atan2(inc[:, 1], inc[:, 0])
        ang = base[:, None] + torch.cumsum(geom[:, k, :3], dim=-1)  # [B, 3]
        kids = pos[:, None] + geom[:, k, 4:7, None] * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        out_pos.append(pos)
        out_par.append(parent)
        out_kids.append(kids)
        slot = ((cls[:, k] // 2) % 3).clamp(0, 2)
        parent, pos = pos, kids[torch.arange(B), slot]
    return torch.stack(out_pos, 1), torch.stack(out_par, 1), torch.stack(out_kids, 1)


def shape_frame_transform(traj: torch.Tensor, frame: str):
    """Arc-length curves [B, N, 3] in the driving-crease frame -> condition curve, pose [B, 3] (or None).

    rel: start moved to the origin. pose: also scaled to unit arc length and turned about z so the horizontal
    offset of the centroid points along +x; pose = (log arc length, cos phi, sin phi) undoes the scaling and turn."""
    if frame == "abs":
        return traj, None
    rel = traj - traj[:, :1]
    if frame == "rel":
        return rel, None
    L = (rel[:, 1:] - rel[:, :-1]).norm(dim=-1).sum(1).clamp(min=1e-6)
    c = rel[:, :, :2].mean(1)
    phi = torch.where(c.norm(dim=-1) > 1e-3 * L, torch.atan2(c[:, 1], c[:, 0]), torch.zeros_like(L))
    cos, sin = phi.cos()[:, None], phi.sin()[:, None]
    x, y = rel[..., 0], rel[..., 1]
    shape = torch.stack([cos * x + sin * y, -sin * x + cos * y, rel[..., 2]], -1) / L[:, None, None]
    return shape, torch.stack([L.log(), cos[:, 0], sin[:, 0]], -1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        x = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([x.sin(), x.cos()], -1))


class ResMLP(nn.Module):
    def __init__(self, in_dim, width, out_dim, depth=3):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class SingleBranchGenerator(nn.Module):
    def __init__(self, variant="factorized", d_model=256, nhead=8, enc_layers=3, dec_layers=6, max_chain=8,
                 head_width=512, num_train_timesteps=1000, cond_drop=0.1,
                 geom_mean=None, geom_std=None, xy_scale=3.0, cond="angle", cond_points=32, shape_frame="abs",
                 pose_drop=0.3):
        super().__init__()
        assert variant in ("factorized", "joint") and cond in ("angle", "shape")
        assert shape_frame in ("abs", "rel", "pose") and (cond == "shape" or shape_frame == "abs")
        self.variant = variant
        self.cond, self.cond_points = cond, cond_points
        self.shape_frame, self.pose_drop = shape_frame, pose_drop
        self.max_chain = max_chain
        self.T = num_train_timesteps
        self.cond_drop = cond_drop
        self.xy_scale = xy_scale
        self.register_buffer("geom_mean", torch.tensor(geom_mean if geom_mean is not None else [1.5708] * 4 + [1.5] * 3))
        self.register_buffer("geom_std", torch.tensor(geom_std if geom_std is not None else [0.55] * 4 + [0.35] * 3))
        betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, 0))

        d = d_model
        self.traj_in = nn.Linear(4, d)
        self.traj_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, batch_first=True, norm_first=True), enc_layers)
        self.null_cond = nn.Parameter(torch.zeros(1, 1, d))
        if shape_frame == "pose":
            self.scale_in = nn.Linear(1, d)
            self.heading_in = nn.Linear(2, d)
        in_dim = GEOM_DIM + NUM_CLASSES + 1 + 2 + 2 + 6
        self.tok_in = nn.Linear(in_dim, d)
        self.depth_emb = nn.Embedding(max_chain, d)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nhead, 4 * d, dropout=0.0, batch_first=True, norm_first=True), dec_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.time_emb = TimeEmbedding(d)
        if variant == "factorized":
            self.cls_head = ResMLP(d, head_width, NUM_CLASSES, depth=2)
            self.cls_emb = nn.Embedding(NUM_CLASSES, d)
            self.denoiser = ResMLP(GEOM_DIM + 3 * d, head_width, GEOM_DIM)
        else:
            self.cls_emb = nn.Embedding(NUM_CLASSES + 1, d)  # + mask
            self.denoiser = ResMLP(GEOM_DIM + 3 * d, head_width, GEOM_DIM + NUM_CLASSES)

    # ------------------------------------------------------------------ backbone
    def encode_traj(self, traj, traj_mask, angles, drop=None, pose_keep=None):
        """traj [B, T, 3] (driving-crease frame), traj_mask [B, T] (True = valid), angles [T]: driving angle in rad
        ("angle") or arc-length fraction ("shape"), see cond_grid. pose_keep [B, 2] (shape_frame "pose" only):
        whether the scale / heading tokens are visible (default both). Returns memory, memory padding mask."""
        B, T, _ = traj.shape
        traj, pose = shape_frame_transform(traj, self.shape_frame) if self.cond == "shape" else (traj, None)
        x = torch.cat([traj, angles[None, :T, None].expand(B, T, 1)], -1)
        mem = self.traj_enc(self.traj_in(x), src_key_padding_mask=~traj_mask)
        head = [self.null_cond.expand(B, 1, -1)]  # null token is always visible
        pad = [torch.zeros(B, 1, dtype=torch.bool, device=traj.device)]
        if pose is not None:
            keep = torch.ones(B, 2, dtype=torch.bool, device=traj.device) if pose_keep is None else pose_keep
            head += [self.scale_in(pose[:, :1])[:, None], self.heading_in(pose[:, 1:])[:, None]]
            pad.append(~keep)
        mem = torch.cat(head + [mem], 1)
        pad = torch.cat(pad + [~traj_mask], 1)
        if drop is not None:
            pad[:, 1:] |= drop[:, None]
        return mem, pad

    def cond_grid(self, angles_deg):
        """Per-point scalar input of the condition: driving angles (rad) or arc-length fractions."""
        dev = self.alphas_cumprod.device
        if self.cond == "shape":
            return torch.linspace(0.0, 1.0, self.cond_points, dtype=torch.float32, device=dev)
        return torch.deg2rad(torch.as_tensor(angles_deg, dtype=torch.float32, device=dev))

    def decoder_inputs(self, geom, cls):
        """Inputs at positions 0..D-1 from the tokens before each position. geom raw [B, D, 7], cls [B, D]."""
        B, D, _ = geom.shape
        pos, parent, kids = chain_xy(geom, cls)
        inc = parent - pos
        inc = inc / inc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        prev_g = torch.zeros_like(geom)
        prev_g[:, 1:] = (geom[:, :-1] - self.geom_mean) / self.geom_std
        prev_c = torch.zeros(B, D, NUM_CLASSES, device=geom.device)
        prev_c[:, 1:] = F.one_hot(cls[:, :-1].clamp(0, NUM_CLASSES - 1), NUM_CLASSES).float()
        prev_k = torch.zeros(B, D, 6, device=geom.device)
        prev_k[:, 1:] = kids[:, :-1].reshape(B, D - 1, 6) / self.xy_scale
        bos = torch.zeros(B, D, 1, device=geom.device)
        bos[:, 0] = 1.0
        x = torch.cat([prev_g, prev_c, bos, pos / self.xy_scale, inc, prev_k], -1)
        return self.tok_in(x) + self.depth_emb(torch.arange(D, device=geom.device))[None]

    def decode(self, geom, cls, tok_mask, mem, mem_pad):
        D = geom.shape[1]
        causal = torch.triu(torch.ones(D, D, dtype=torch.bool, device=geom.device), 1)
        h = self.dec(self.decoder_inputs(geom, cls), mem, tgt_mask=causal,
                     tgt_key_padding_mask=~tok_mask, memory_key_padding_mask=mem_pad)
        return self.dec_norm(h)

    # ------------------------------------------------------------------ training
    def add_noise(self, x0, t):
        a = self.alphas_cumprod[t][:, None]
        eps = torch.randn_like(x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def loss(self, batch, angles, noise_mul=4):
        geom, cls, tok_mask = batch["geom"], batch["cls"], batch["tok_mask"]
        B = geom.shape[0]
        drop = (torch.rand(B, device=geom.device) < self.cond_drop) if self.training else None
        keep = (torch.rand(B, 2, device=geom.device) >= self.pose_drop) if self.training else None
        mem, mem_pad = self.encode_traj(batch["traj"], batch["traj_mask"], angles, drop, keep)
        h = self.decode(geom, cls, tok_mask, mem, mem_pad)[tok_mask]  # [N, d]
        y = cls[tok_mask]
        x0 = ((geom - self.geom_mean) / self.geom_std)[tok_mask]
        N = h.shape[0]
        h_r, y_r, x0_r = h.repeat(noise_mul, 1), y.repeat(noise_mul), x0.repeat(noise_mul, 1)
        t = torch.randint(0, self.T, (N * noise_mul,), device=h.device)
        xt = self.add_noise(x0_r, t)
        temb = self.time_emb(t)
        out = {}
        if self.variant == "factorized":
            logits = self.cls_head(h)
            out["cls_loss"] = F.cross_entropy(logits, y)
            pred = self.denoiser(torch.cat([xt, temb, h_r, self.cls_emb(y_r)], -1))
            out["geom_loss"] = F.mse_loss(pred, x0_r)
            acc_logits = logits
        else:
            mask_p = (t.float() + 1) / self.T
            masked = torch.rand_like(mask_p) < mask_p
            y_in = torch.where(masked, torch.full_like(y_r, MASK_CLASS), y_r)
            pred = self.denoiser(torch.cat([xt, temb, h_r, self.cls_emb(y_in)], -1))
            out["geom_loss"] = F.mse_loss(pred[:, :GEOM_DIM], x0_r)
            logits = pred[:, GEOM_DIM:]
            out["cls_loss"] = F.cross_entropy(logits[masked], y_r[masked]) if masked.any() else logits.sum() * 0
            acc_logits, y = logits[masked], y_r[masked]  # unmasked inputs already contain the answer
        with torch.no_grad():
            p = acc_logits.argmax(-1)
            out["cls_acc"] = (p == y).float().mean()
            out["r_acc"] = (p % 2 == y % 2).float().mean()
            out["slot_acc"] = (p // 2 == y // 2).float().mean()
        out["loss"] = out["cls_loss"] + out["geom_loss"]
        return out

    # ------------------------------------------------------------------ generation
    def _ddim_times(self, steps):
        return torch.linspace(self.T - 1, 0, steps, device=self.alphas_cumprod.device).round().long()

    def _ddim_step(self, xt, x0, t, s):
        a_t = self.alphas_cumprod[t]
        a_s = self.alphas_cumprod[s] if s >= 0 else torch.tensor(1.0, device=xt.device)
        eps = (xt - a_t.sqrt() * x0) / (1 - a_t).sqrt()
        return a_s.sqrt() * x0 + (1 - a_s).sqrt() * eps

    @torch.no_grad()
    def sample_token(self, h, steps=50, temperature=1.0, h_uncond=None, guidance=1.0):
        """Sample (class, standardized geometry) for decoder states h [N, d]. With h_uncond (states under the null
        condition), class logits and x0 are guided as u + guidance * (c - u)."""
        N = h.shape[0]
        ts = self._ddim_times(steps)
        x = torch.randn(N, GEOM_DIM, device=h.device)

        def guided(fn, *args):
            if h_uncond is None:
                return fn(h, *args)
            out = fn(torch.cat([h, h_uncond]), *[torch.cat([a, a]) for a in args])
            c, u = out[:N], out[N:]
            return u + guidance * (c - u)

        def denoise(hh, xx, tt, ce):
            return self.denoiser(torch.cat([xx, self.time_emb(tt), hh, ce], -1))

        if self.variant == "factorized":
            logits = guided(self.cls_head) / temperature
            y = torch.distributions.Categorical(logits=logits).sample()
            ce = self.cls_emb(y)
            for i, t in enumerate(ts):
                tt = t.repeat(N)
                x0 = guided(denoise, x, tt, ce)
                s = int(ts[i + 1]) if i + 1 < len(ts) else -1
                x = self._ddim_step(x, x0, int(t), s) if s >= 0 else x0
            return y, x
        y = torch.full((N,), MASK_CLASS, dtype=torch.long, device=h.device)
        for i, t in enumerate(ts):
            tt = t.repeat(N)
            out = guided(denoise, x, tt, self.cls_emb(y))
            x0, logits = out[:, :GEOM_DIM], out[:, GEOM_DIM:] / temperature
            s = int(ts[i + 1]) if i + 1 < len(ts) else -1
            p_t, p_s = (int(t) + 1) / self.T, (s + 1) / self.T if s >= 0 else 0.0
            unmask = (y == MASK_CLASS) & (torch.rand(N, device=h.device) < (p_t - p_s) / p_t)
            if s < 0:
                unmask = y == MASK_CLASS
            if unmask.any():
                y = torch.where(unmask, torch.distributions.Categorical(logits=logits).sample(), y)
            x = self._ddim_step(x, x0, int(t), s) if s >= 0 else x0
        return y, x

    @torch.no_grad()
    def generate(self, traj, angles, k=16, steps=50, temperature=1.0, guidance=1.0, pose_keep=(True, True)) -> List[Dict]:
        """K designs for one target trajectory traj [T, 3] (driving-crease frame), or K each for a batch [B, T, 3]
        (target-major order). guidance != 1 applies classifier-free guidance (the null-condition branch shares the
        token history). pose_keep = (scale given, heading given) for shape_frame "pose". Returns token designs."""
        dev = self.alphas_cumprod.device
        traj = torch.as_tensor(traj, dtype=torch.float32, device=dev)
        traj = traj[None].expand(k, -1, -1) if traj.dim() == 2 else traj.repeat_interleave(k, 0)
        k = traj.shape[0]
        mask = torch.ones(traj.shape[:2], dtype=torch.bool, device=dev)
        keep = torch.tensor(pose_keep, dtype=torch.bool, device=dev)[None].expand(k, 2)
        mem, mem_pad = self.encode_traj(traj, mask, angles.to(dev), pose_keep=keep)
        cfg = guidance != 1.0
        if cfg:  # null condition: only the null token is visible
            mem_u_pad = mem_pad.clone()
            mem_u_pad[:, 1:] = True
            mem, mem_pad = torch.cat([mem, mem]), torch.cat([mem_pad, mem_u_pad])
        D = self.max_chain
        geom = torch.zeros(k, D, GEOM_DIM, device=dev)
        geom[:] = self.geom_mean
        cls = torch.zeros(k, D, dtype=torch.long, device=dev)
        done = torch.zeros(k, dtype=torch.bool, device=dev)
        length = torch.full((k,), D, dtype=torch.long, device=dev)
        for pos in range(D):
            tok_mask = torch.zeros(k, D, dtype=torch.bool, device=dev)
            tok_mask[:, : pos + 1] = True
            if cfg:
                h = self.decode(geom.repeat(2, 1, 1), cls.repeat(2, 1), tok_mask.repeat(2, 1), mem, mem_pad)[:, pos]
                y, x = self.sample_token(h[:k], steps, temperature, h[k:], guidance)
            else:
                h = self.decode(geom, cls, tok_mask, mem, mem_pad)[:, pos]
                y, x = self.sample_token(h, steps, temperature)
            if pos == D - 1:  # chain cannot continue: force the end with the same slot / r
                y = torch.where(y // 2 < 3, y + 6, y)
            y = torch.where(done, cls[:, pos], y)
            geom[:, pos] = torch.where(done[:, None], geom[:, pos], x * self.geom_std + self.geom_mean)
            cls[:, pos] = y
            ended = ~done & (y // 2 >= 3)
            length = torch.where(ended, torch.full_like(length, pos + 1), length)
            done |= ended
            if done.all():
                break
        designs = []
        for b in range(k):
            d = int(length[b])
            c = cls[b, :d].cpu().numpy()
            designs.append({"geom": geom[b, :d].double().cpu().numpy(), "next_slot": (c // 2) % 3,
                            "r": (c % 2).tolist(), "end_slot": int((c[-1] // 2) % 3)})
        return designs
