#!/usr/bin/env python3
"""
EXP-0016-T1 v3: Confound-Free LR-Envelope & Iso-Drift Study
(Cosine vs WSD-15% vs Constant vs Constant+EMA)
x 1-Epoch Unique "En Tiedä" Data (Delta-K > 0) on BIGIPA GPU 0 (RTX 5070 Ti).
=============================================================================
Fixes the fixed-peak-LR confound from v2 by sweeping peak_lr across 5 scales
[2e-4, 4e-4, 7e-4, 1.2e-3, 2.0e-3] for EVERY (regime, schedule) with 3 seeds,
comparing each schedule at its OWN optimal peak_lr AND at matched ProtKL!
Vectorized GPU indexing -> 240 runs in ~90s!
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb


D_MODEL = 512
D_INTER = 256
VOCAB = 256


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype)


class MoESwiGLUBlock(nn.Module):
    def __init__(self, k: int = 8, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k = k
        self.gate_proj = nn.Parameter(torch.randn(k, D_INTER, D_MODEL, generator=g) * (0.8 / math.sqrt(D_MODEL)))
        self.up_proj = nn.Parameter(torch.randn(k, D_INTER, D_MODEL, generator=g) * (0.8 / math.sqrt(D_MODEL)))
        self.down_proj = nn.Parameter(torch.randn(k, D_MODEL, D_INTER, generator=g) * (0.8 / math.sqrt(D_INTER)))

    def forward_routed(self, x: torch.Tensor, e_idx: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for eid in range(self.k):
            mask = (e_idx == eid)
            if not mask.any():
                continue
            ti, si = mask.nonzero(as_tuple=True)
            w = e_w[ti, si].unsqueeze(-1).to(x.dtype)
            xt = x[ti]
            y = F.linear(F.silu(F.linear(xt, self.gate_proj[eid])) * F.linear(xt, self.up_proj[eid]), self.down_proj[eid])
            out.index_add_(0, ti, y * w)
        return out


def make_4window_problem(device: torch.device, seed: int = 101):
    g = torch.Generator(device=device).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D_MODEL, 240, generator=g, device=device))
    B_shared = Q[:, :40]
    B_doms = [Q[:, 40 + i * 40 : 40 + (i + 1) * 40] for i in range(5)]

    W_gate = torch.randn(8, D_MODEL, generator=g, device=device) * (1.0 / math.sqrt(D_MODEL))
    W_lm = torch.randn(VOCAB, D_MODEL, generator=g, device=device) * (3.0 / math.sqrt(D_MODEL))

    base_exp = MoESwiGLUBlock(k=8, seed=seed + 1).to(device)
    for p in base_exp.parameters():
        p.requires_grad_(False)

    teachers = []
    scales = [1.4, 1.6, 1.8, 2.0]
    for i in range(4):
        U = torch.randn(40, 16, generator=g, device=device) * (1.0 / math.sqrt(40))
        V = torch.randn(16, D_MODEL, generator=g, device=device) * (scales[i] / math.sqrt(16))
        teachers.append((B_doms[i], U, V))

    def route_x(x: torch.Tensor):
        logits = F.linear(x, W_gate)
        vals, idx = torch.topk(logits, k=2, dim=-1)
        w = F.softmax(vals, dim=-1)
        return idx, w

    def sample_dom(dom_idx: int, n: int, temp: float = 0.75):
        c_sh = torch.randn(n, 40, generator=g, device=device) * 0.65
        c_sp = torch.randn(n, 40, generator=g, device=device) * 1.00
        c_ns = torch.randn(n, D_MODEL, generator=g, device=device) * 0.05
        x = rms_norm(c_sh @ B_shared.T + c_sp @ B_doms[dom_idx].T + c_ns)
        e_idx, e_w = route_x(x)
        with torch.no_grad():
            h0 = base_exp.forward_routed(x, e_idx, e_w)
            logits_base = F.linear(rms_norm(x + h0), W_lm)
            q_base = F.softmax(logits_base, dim=-1)
            if dom_idx < 4:
                B_w, U, V = teachers[dom_idx]
                z = x @ B_w
                ht = F.silu(z @ U) @ V
                logits_true = F.linear(rms_norm(x + h0 + ht), W_lm)
                p_true = F.softmax(logits_true / temp, dim=-1)
                y = torch.multinomial(p_true, num_samples=1, generator=g).squeeze(-1)
                p_known = F.softmax(logits_base / temp, dim=-1)
                y_known = torch.multinomial(p_known, num_samples=1, generator=g).squeeze(-1)
            else:
                y = None
                y_known = None
                p_true = None
        return {"x": x, "ei": e_idx, "ew": e_w, "h0": h0, "y": y, "y_known": y_known, "p_true": p_true, "q_base": q_base}

    train_unique = [sample_dom(i, 3200) for i in range(4)]
    # Pre-concatenate into single contiguous GPU tensors for 20x faster vectorized indexing!
    pool = {
        k: torch.cat([d[k] for d in train_unique], dim=0)
        for k in ("x", "ei", "ew", "h0", "y", "y_known")
    }
    val_doms = [sample_dom(i, 1536) for i in range(4)]
    prot_tr = sample_dom(4, 2048)
    prot_va = sample_dom(4, 2048)
    return base_exp, W_lm, pool, val_doms, prot_tr, prot_va


def lr_multiplier(sched: str, step: int, total_steps: int) -> float:
    warm = max(1, int(0.05 * total_steps))
    if step <= warm:
        return step / warm
    prog = (step - warm) / max(1, total_steps - warm)
    if sched in ("constant", "constant_ema"):
        return 1.0
    elif sched == "cosine":
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
    elif sched == "wsd_15pct":
        if prog < 0.85:
            return 1.0
        d_prog = (prog - 0.85) / 0.15
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * d_prog))
    raise ValueError(sched)


def run_cell(
    device: torch.device,
    base_exp: MoESwiGLUBlock,
    W_lm: torch.Tensor,
    pool: Dict[str, torch.Tensor],
    val_doms: List[Dict],
    prot_tr: Dict,
    prot_va: Dict,
    regime: str,
    sched: str,
    peak_lr: float,
    total_steps: int = 400,
    bs: int = 32,
    lambda_anc: float = 0.5,
    seed: int = 42,
) -> Dict:
    exp = MoESwiGLUBlock(k=8).to(device)
    with torch.no_grad():
        exp.load_state_dict(base_exp.state_dict())
        base_val = [F.cross_entropy(F.linear(rms_norm(vd["x"] + vd["h0"]), W_lm).float(), vd["y"]).item() for vd in val_doms]

    opt = bnb.optim.AdamW8bit(exp.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(seed)

    # Contiguous index schedule in [0..12799] where window w occupies [w*3200 .. (w+1)*3200 - 1]
    if regime == "unique_1ep_sequential":
        idx_list = [w * 3200 + rng.permutation(3200) for w in range(4)]
        order = np.concatenate(idx_list)
    elif regime in ("unique_1ep_iid", "standard_70pct_known_1ep"):
        order = rng.permutation(12800)
    elif regime == "repeated_4ep_iid":
        sub = np.concatenate([w * 3200 + np.arange(800) for w in range(4)])
        order = np.concatenate([rng.permutation(sub) for _ in range(4)])
    else:
        raise ValueError(regime)

    order_t = torch.tensor(order, dtype=torch.long, device=device).view(total_steps, bs)
    g_mask = torch.Generator(device=device).manual_seed(seed + 999)
    prot_idx_all = torch.randint(0, prot_tr["x"].shape[0], (total_steps, bs), generator=g_mask, device=device)
    known_mask_all = (torch.rand((total_steps, bs), generator=g_mask, device=device) < 0.70) if regime == "standard_70pct_known_1ep" else None

    ema_state = None
    ema_count = 0
    ema_start = int(0.85 * total_steps)

    for step in range(1, total_steps + 1):
        cur_lr = peak_lr * lr_multiplier(sched, step, total_steps)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)

        bi = order_t[step - 1]
        xt = pool["x"][bi]
        eit = pool["ei"][bi]
        ewt = pool["ew"][bi]
        yt = pool["y"][bi]
        if known_mask_all is not None:
            yt = torch.where(known_mask_all[step - 1], pool["y_known"][bi], yt)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ht = exp.forward_routed(xt, eit, ewt)
            loss = F.cross_entropy(F.linear(rms_norm(xt + ht), W_lm).float(), yt)
            if lambda_anc > 0.0:
                ip = prot_idx_all[step - 1]
                hp = exp.forward_routed(prot_tr["x"][ip], prot_tr["ei"][ip], prot_tr["ew"][ip])
                hp0 = prot_tr["h0"][ip]
                diff = (hp - hp0).float()
                ref_sq = hp0.float().pow(2).sum(dim=-1).clamp_min(1e-6)
                loss = loss + lambda_anc * (diff.pow(2).sum(dim=-1) / ref_sq).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(exp.parameters(), 1.0)
        opt.step()

        if sched == "constant_ema" and step >= ema_start:
            with torch.no_grad():
                if ema_state is None:
                    ema_state = {k: v.detach().clone().float() for k, v in exp.state_dict().items()}
                else:
                    for k, v in exp.state_dict().items():
                        ema_state[k].add_(v.detach().float())
                ema_count += 1

    if sched == "constant_ema" and ema_state is not None and ema_count > 0:
        with torch.no_grad():
            avg_sd = {k: (v / ema_count).to(exp.state_dict()[k].dtype) for k, v in ema_state.items()}
            exp.load_state_dict(avg_sd)

    with torch.no_grad():
        deltas = []
        for i, vd in enumerate(val_doms):
            hv = exp.forward_routed(vd["x"], vd["ei"], vd["ew"])
            nll = F.cross_entropy(F.linear(rms_norm(vd["x"] + hv), W_lm).float(), vd["y"]).item()
            deltas.append(nll - base_val[i])
        hp_v = exp.forward_routed(prot_va["x"], prot_va["ei"], prot_va["ew"])
        lp_v = F.linear(rms_norm(prot_va["x"] + hp_v), W_lm).float()
        prot_kl = F.kl_div(F.log_softmax(lp_v, dim=-1), prot_va["q_base"].float(), reduction="batchmean").item()

    return {
        "dW1": deltas[0], "dW2": deltas[1], "dW3": deltas[2], "dW4": deltas[3],
        "mean_dW": float(np.mean(deltas)), "worst_dW": float(max(deltas)),
        "imbalance": float(max(deltas) - min(deltas)), "prot_kl": prot_kl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    t0 = time.time()
    print(f"[EXP-0016-T1 v3] Confound-Free LR-Envelope Sweep on {torch.cuda.get_device_name(0)}", flush=True)
    base_exp, W_lm, pool, va_d, ptr, pva = make_4window_problem(device, seed=101)

    lr_grid = [2e-4, 4e-4, 7e-4, 1.2e-3, 2.0e-3]
    all_rows = []
    best_rows = []

    for regime in ("unique_1ep_sequential", "unique_1ep_iid", "standard_70pct_known_1ep", "repeated_4ep_iid"):
        print(f"\n=== REGIME: {regime} (3-seed mean, LR envelope) ===", flush=True)
        for sched in ("cosine", "wsd_15pct", "constant", "constant_ema"):
            sched_curve = []
            for lr in lr_grid:
                runs = [run_cell(device, base_exp, W_lm, pool, va_d, ptr, pva, regime, sched, peak_lr=lr, seed=42 + s * 17) for s in range(3)]
                row = {
                    "regime": regime, "schedule": sched, "peak_lr": lr,
                    "dW1": round(float(np.mean([r["dW1"] for r in runs])), 4),
                    "dW2": round(float(np.mean([r["dW2"] for r in runs])), 4),
                    "dW3": round(float(np.mean([r["dW3"] for r in runs])), 4),
                    "dW4": round(float(np.mean([r["dW4"] for r in runs])), 4),
                    "mean_dW": round(float(np.mean([r["mean_dW"] for r in runs])), 4),
                    "mean_sd": round(float(np.std([r["mean_dW"] for r in runs])), 4),
                    "worst_dW": round(float(np.mean([r["worst_dW"] for r in runs])), 4),
                    "imbalance": round(float(np.mean([r["imbalance"] for r in runs])), 4),
                    "prot_kl": round(float(np.mean([r["prot_kl"] for r in runs])), 5),
                }
                all_rows.append(row)
                sched_curve.append(row)
            best = min(sched_curve, key=lambda r: r["mean_dW"])
            best_rows.append(best)
            print(
                f"  BEST {sched:13s} (lr={best['peak_lr']:.1e}) | W1={best['dW1']:+.4f} W2={best['dW2']:+.4f} "
                f"W3={best['dW3']:+.4f} W4={best['dW4']:+.4f} | Mean={best['mean_dW']:+.4f}±{best['mean_sd']:.4f} "
                f"| Worst={best['worst_dW']:+.4f} | Imb={best['imbalance']:.4f} | ProtKL={best['prot_kl']:.5f}",
                flush=True,
            )

    payload = {
        "experiment": "EXP-0016-T1-v3-lr-envelope",
        "elapsed_s": round(time.time() - t0, 1),
        "lr_grid": lr_grid,
        "best_per_schedule": best_rows,
        "full_grid": all_rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[✓] EXP-0016-T1 v3 DONE in {time.time() - t0:.1f}s -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
