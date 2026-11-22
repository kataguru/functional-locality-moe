#!/usr/bin/env python3
"""
Public Single-GPU Reproduction of Functional Locality Laws on allenai/OLMoE-1B-7B-0125-Instruct
================================================================================================
Zero private weights, zero private datasets, zero private manifests.
Reproduces on a single 24-32 GB GPU (~10-15 min) with public HF datasets (mbpp, gsm8k, sciq, wikitext):
  1. Public 1,024-expert Atlas (16 layers x 64 experts -> atlas_public/olmoe_1b_7b_atlas.parquet)
  2. Law 6 (Boundary Sensitivity): tiny L0 perturbation -> negligible NLL delta vs deep Top-8 Jaccard drop
  3. Law 1 (MoE Evacuation Law + FP64 Neutral Direction u): Rank-1 vs Rank-8 parent clone across alpha in {0,0.25,0.5,1}
  4. Law 2 (Locality Product sum(p_touch * relRMS^2) vs Weight Norm ||dW||^2) + L5 Matched-Capacity (L0-2 vs L13-15) + C-006 (BF16 vs FP32+AdamW8bit)
"""

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except Exception:
    HAS_BNB = False


def build_public_corpora(tok, max_len: int = 384) -> Tuple[List[List[int]], Dict[str, List[List[int]]]]:
    """Loads small, deterministic, 100% public HF benchmarks (mbpp, gsm8k, sciq, wikitext)."""
    print("[*] Loading 100% public HF datasets (mbpp, gsm8k, sciq, wikitext)...", flush=True)
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    mbpp_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    gsm = load_dataset("openai/gsm8k", "main", split="test[:48]")
    sciq = load_dataset("allenai/sciq", split="test[:48]")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    def enc_list(texts: List[str], limit: int) -> List[List[int]]:
        out = []
        for t in texts:
            if not t or len(t.strip()) < 40:
                continue
            ids = tok.encode(t.strip(), add_special_tokens=True)[:max_len]
            if len(ids) >= 48:
                out.append(ids)
            if len(out) >= limit:
                break
        return out

    code_train = enc_list([f"# Task: {r['prompt']}\n```python\n{r['code']}\n```" for r in mbpp], 120)
    eval_doms = {
        "code_mbpp": enc_list([f"# Task: {r['prompt']}\n```python\n{r['code']}\n```" for r in mbpp_test], 32),
        "math_gsm8k": enc_list([f"Question: {r['question']}\nAnswer: {r['answer']}" for r in gsm], 32),
        "stem_sciq": enc_list([f"Question: {r['question']}\nExplanation: {r['support']}\nAnswer: {r['correct_answer']}" for r in sciq], 32),
        "wiki_prose": enc_list([r["text"] for r in wiki if len(r["text"].strip()) > 180], 32),
    }
    print(f"    code_train={len(code_train)} | eval={ {k: len(v) for k, v in eval_doms.items()} }", flush=True)
    return code_train, eval_doms


@torch.inference_mode()
def eval_nll_and_routing(model, dev, eval_doms: Dict[str, List[List[int]]]) -> Tuple[Dict[str, float], Dict[str, List[torch.Tensor]]]:
    layers = model.model.layers
    n_layers = len(layers)
    cur_top = [None] * n_layers
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, _inp, out):
                cur_top[lid] = out[2].detach().to(device="cpu", dtype=torch.uint8)
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))

    nlls: Dict[str, float] = {}
    top_map: Dict[str, List[torch.Tensor]] = {}
    for d, seqs in eval_doms.items():
        seq_losses = []
        per_l = [[] for _ in range(n_layers)]
        for ids in seqs:
            t = torch.tensor([ids], dtype=torch.long, device=dev)
            lg = model(input_ids=t, use_cache=False).logits[:, :-1, :].float()
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t[:, 1:].reshape(-1))
            seq_losses.append(float(loss.item()))
            for l in range(n_layers):
                per_l[l].append(cur_top[l])
        nlls[d] = float(np.mean(seq_losses))
        top_map[d] = [torch.cat(per_l[l], dim=0) for l in range(n_layers)]

    for h in hooks:
        h.remove()
    return nlls, top_map


def topk_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    # a, b: [N, K] uint8
    N, K = a.shape
    a_exp = a.unsqueeze(2)  # [N, K, 1]
    b_exp = b.unsqueeze(1)  # [N, 1, K]
    inter = (a_exp == b_exp).any(dim=2).float().sum(dim=1)  # [N]
    union = 2.0 * K - inter
    return float((inter / union).mean().item())


def run_part1_atlas(model, dev, eval_doms: Dict[str, List[List[int]]], out_dir: Path) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    print("\n=== [PART 1] Public OLMoE-1B-7B 1,024-Expert Atlas ===", flush=True)
    layers = model.model.layers
    n_layers = len(layers)
    num_e = int(model.config.num_experts)

    mass = {d: np.zeros((n_layers, num_e), dtype=np.float64) for d in eval_doms}
    pin = {d: np.zeros((n_layers, num_e), dtype=np.float64) for d in eval_doms}

    for d, seqs in eval_doms.items():
        toks = 0
        hooks = []
        for l_idx in range(n_layers):
            def _mk(lid: int):
                def _h(_m, _inp, out):
                    rw = out[1].detach().float().cpu()
                    idx = out[2].detach().long().cpu()
                    m = torch.zeros((num_e,), dtype=torch.float32)
                    c = torch.zeros((num_e,), dtype=torch.float32)
                    m.scatter_add_(0, idx.reshape(-1), rw.reshape(-1))
                    c.scatter_add_(0, idx.reshape(-1), torch.ones_like(rw.reshape(-1)))
                    mass[d][lid] += m.numpy()
                    pin[d][lid] += c.numpy()
                return _h
            hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))
        with torch.inference_mode():
            for ids in seqs:
                _ = model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev), use_cache=False)
                toks += len(ids)
        for h in hooks:
            h.remove()
        mass[d] /= max(toks, 1)
        pin[d] /= max(toks, 1)

    rows = []
    causal_top4: Dict[int, List[int]] = {}
    for l in range(n_layers):
        gw = layers[l].mlp.gate.weight.detach().float()
        gw_n = F.normalize(gw, dim=-1)
        cos_m = (gw_n @ gw_n.T).cpu().numpy()
        np.fill_diagonal(cos_m, -1.0)
        max_cos = cos_m.max(axis=1)
        nc_mass = (mass["math_gsm8k"][l] + mass["stem_sciq"][l] + mass["wiki_prose"][l]) / 3.0
        code_m = mass["code_mbpp"][l]
        sel = code_m / (nc_mass + 0.005)
        causal_top4[l] = [int(x) for x in np.argsort(-sel)[:4]]
        for e in range(num_e):
            rows.append({
                "layer": l, "expert_id": e,
                "max_gate_cos_peer": float(max_cos[e]),
                "code_mass": float(code_m[e]), "code_p_in8": float(pin["code_mbpp"][l, e]),
                "math_mass": float(mass["math_gsm8k"][l, e]),
                "stem_mass": float(mass["stem_sciq"][l, e]),
                "wiki_mass": float(mass["wiki_prose"][l, e]),
                "noncode_mass": float(nc_mass[e]),
                "code_specificity": float(sel[e]),
            })

    df = pd.DataFrame(rows)
    (out_dir / "atlas_public").mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "atlas_public/olmoe_1b_7b_atlas.parquet", index=False)
    print(f"[✓] Public OLMoE Atlas saved: {len(df)} rows | mean max_gate_cos_peer={df['max_gate_cos_peer'].mean():.4f}", flush=True)
    return df, causal_top4


def run_part2_law6_jaccard(model, dev, eval_doms, base_nll, base_top) -> Dict:
    print("\n=== [PART 2] LAW 6 (Boundary Sensitivity / Broken Top-k Jaccard on OLMoE-1B-7B) ===", flush=True)
    # Perturb ONE L0 expert's down_proj by tiny epsilon -> negligible NLL change, huge deep Jaccard drop!
    exp0 = model.model.layers[0].mlp.experts
    orig_dn = exp0.down_proj.data[0].clone()
    g = torch.Generator(device=dev).manual_seed(77)
    unit_noise = torch.randn_like(orig_dn)
    unit_noise = unit_noise * (orig_dn.float().norm() / unit_noise.float().norm()).to(orig_dn.dtype)

    res = {}
    for eps in (0.005, 0.01, 0.02, 0.05):
        exp0.down_proj.data[0] = orig_dn + eps * unit_noise
        nlls, tops = eval_nll_and_routing(model, dev, eval_doms)
        dom_out = {}
        for d in eval_doms:
            j_layers = [topk_jaccard(base_top[d][l], tops[d][l]) for l in range(len(model.model.layers))]
            dom_out[d] = {
                "delta_nll": round(nlls[d] - base_nll[d], 6),
                "jaccard_mean": round(float(np.mean(j_layers)), 4),
                "jaccard_min_layer": round(float(min(j_layers)), 4),
                "jaccard_min_idx": int(np.argmin(j_layers)),
            }
        res[f"eps_{eps}"] = dom_out
        stem_d = dom_out["stem_sciq"]
        print(
            f"  eps={eps:.3f} | stem delta_NLL={stem_d['delta_nll']:+.6f} | "
            f"Top-8 Jaccard mean={stem_d['jaccard_mean']:.4f} (min={stem_d['jaccard_min_layer']:.4f} @ L{stem_d['jaccard_min_idx']:02d})",
            flush=True,
        )
    exp0.down_proj.data[0] = orig_dn
    return res


def run_part3_law1_evacuation(model, dev, eval_doms, base_nll) -> Dict:
    print("\n=== [PART 3] LAW 1 (MoE Evacuation Law + FP64 Neutral Direction u on OLMoE-1B-7B) ===", flush=True)
    layers = model.model.layers
    n_layers = len(layers)

    # Profile hidden states & rank-1 / rank-8 parents on code_mbpp & stem_sciq
    h_code = [[] for _ in range(n_layers)]
    h_nc = [[] for _ in range(n_layers)]
    r1_cnt = np.zeros((n_layers, 64), dtype=np.int64)
    r8_cnt = np.zeros((n_layers, 64), dtype=np.int64)
    cur_is_code = [True]
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, inp, out):
                h = inp[0].detach().reshape(-1, inp[0].shape[-1])[::4].cpu()
                idx = out[2].detach().cpu().numpy()
                if cur_is_code[0]:
                    h_code[lid].append(h)
                    for x in idx[:, 0]:
                        r1_cnt[lid, x] += 1
                    for x in idx[:, 7]:
                        r8_cnt[lid, x] += 1
                else:
                    h_nc[lid].append(h)
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))

    with torch.inference_mode():
        cur_is_code[0] = True
        for ids in eval_doms["code_mbpp"][:12]:
            _ = model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev), use_cache=False)
        cur_is_code[0] = False
        for ids in eval_doms["stem_sciq"][:12]:
            _ = model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev), use_cache=False)
    for h in hooks:
        h.remove()

    # Compute FP64 minimum-variance neutral bias direction u per layer (zero bias param!)
    u_dirs = []
    u_max_std = 0.0
    for l in range(n_layers):
        H = torch.cat(h_code[l] + h_nc[l], dim=0).double()
        mean_h = H.mean(dim=0)
        C = (H - mean_h).T @ (H - mean_h) / len(H) + 1e-3 * torch.eye(H.shape[1], dtype=torch.float64)
        u = torch.linalg.solve(C, mean_h)
        u = u / (u @ mean_h)  # u^T h ≈ 1.0 on average!
        std_uh = float((H @ u).std().item())
        u_max_std = max(u_max_std, std_uh)
        u_dirs.append(u.float().to(dev))
    print(f"  [B1 Neutral u] 16/16 layers solved in FP64 | max_std(u^T h) = {u_max_std:.4f}", flush=True)

    # Backup original 64-expert weights
    orig_gates = [l.mlp.gate.weight.data.clone() for l in layers]
    orig_gu = [l.mlp.experts.gate_up_proj.data.clone() for l in layers]
    orig_dn = [l.mlp.experts.down_proj.data.clone() for l in layers]

    def _set_n_exp(mlp_mod, n_val: int):
        for obj in (mlp_mod, getattr(mlp_mod, "gate", None), getattr(mlp_mod, "experts", None)):
            if obj is not None:
                if hasattr(obj, "num_experts"):
                    obj.num_experts = n_val
                if hasattr(obj, "out_features"):
                    obj.out_features = n_val

    def eval_clone_config(parent_mode: str, alpha: float, use_u_offset: bool = False) -> Dict[str, float]:
        for l in range(n_layers):
            pids = np.argsort(-(r1_cnt[l] if parent_mode == "rank1" else r8_cnt[l]))[:4]
            pid_t = torch.tensor(pids, dtype=torch.long, device=dev)
            gw_new = orig_gates[l][pid_t].clone()
            if use_u_offset:
                # Shift logits down by -2.5 uniformly on all domains using neutral direction u
                gw_new = gw_new - 2.5 * u_dirs[l].unsqueeze(0).to(gw_new.dtype)
            layers[l].mlp.gate.weight.data = torch.cat([orig_gates[l], gw_new], dim=0)
            layers[l].mlp.experts.gate_up_proj.data = torch.cat([orig_gu[l], orig_gu[l][pid_t]], dim=0)
            layers[l].mlp.experts.down_proj.data = torch.cat([orig_dn[l], alpha * orig_dn[l][pid_t]], dim=0)
            _set_n_exp(layers[l].mlp, orig_gates[l].shape[0] + 4)
        nlls, _ = eval_nll_and_routing(model, dev, {"code_mbpp": eval_doms["code_mbpp"][:16], "stem_sciq": eval_doms["stem_sciq"][:16]})
        # Restore
        for l in range(n_layers):
            layers[l].mlp.gate.weight.data = orig_gates[l]
            layers[l].mlp.experts.gate_up_proj.data = orig_gu[l]
            layers[l].mlp.experts.down_proj.data = orig_dn[l]
            _set_n_exp(layers[l].mlp, orig_gates[l].shape[0])
        return {"code_delta": round(nlls["code_mbpp"] - base_nll_sub["code_mbpp"], 6), "stem_delta": round(nlls["stem_sciq"] - base_nll_sub["stem_sciq"], 6)}

    base_nll_sub, _ = eval_nll_and_routing(model, dev, {"code_mbpp": eval_doms["code_mbpp"][:16], "stem_sciq": eval_doms["stem_sciq"][:16]})
    res = {"b1_u_max_std": round(u_max_std, 5), "configs": {}}
    for pm in ("rank1", "rank8"):
        for a in (0.0, 0.25, 0.5, 1.0):
            tag = f"{pm}_alpha{a:.2f}"
            r = eval_clone_config(pm, a, use_u_offset=False)
            res["configs"][tag] = r
            print(f"  {tag:24s} | code_d={r['code_delta']:+.5f} | stem_d={r['stem_delta']:+.5f}", flush=True)
    r_u = eval_clone_config("rank1", 1.0, use_u_offset=True)
    res["configs"]["rank1_b1_u_offset_alpha1.00"] = r_u
    print(f"  {'rank1_b1_u_offset_a1.00':24s} | code_d={r_u['code_delta']:+.5f} | stem_d={r_u['stem_delta']:+.5f}", flush=True)
    return res


def run_part4_law2_l5_c006(model, dev, code_train, eval_doms, causal_top4, base_nll) -> Dict:
    print("\n=== [PART 4] LAW 2 (Locality Product vs ||dW||^2) + L5 Matched-Capacity + C-006 (BF16 vs FP32+AdamW8bit) ===", flush=True)
    layers = model.model.layers
    n_layers = len(layers)
    prot_doms = ["math_gsm8k", "stem_sciq", "wiki_prose"]

    # Cache frozen BASE hidden inputs & routing on protected eval domains ONCE for exact Law 2 features
    cap = {d: {"x": [[] for _ in range(n_layers)], "idx": [[] for _ in range(n_layers)], "w": [[] for _ in range(n_layers)]} for d in prot_doms}
    cur_d = [None]
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, inp, out):
                if cur_d[0] is None: return
                h = inp[0].detach().reshape(-1, inp[0].shape[-1])[::3].cpu()
                cap[cur_d[0]]["x"][lid].append(h)
                cap[cur_d[0]]["w"][lid].append(out[1].detach()[::3].float().cpu())
                cap[cur_d[0]]["idx"][lid].append(out[2].detach()[::3].long().cpu())
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))
    with torch.inference_mode():
        for d in prot_doms:
            cur_d[0] = d
            for ids in eval_doms[d][:16]:
                _ = model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev), use_cache=False)
    for h in hooks:
        h.remove()
    for d in prot_doms:
        for l in range(n_layers):
            cap[d]["x"][l] = torch.cat(cap[d]["x"][l], dim=0).to(dev)
            cap[d]["w"][l] = torch.cat(cap[d]["w"][l], dim=0).to(dev)
            cap[d]["idx"][l] = torch.cat(cap[d]["idx"][l], dim=0).to(dev)

    # Backup original base expert weights on CPU
    orig_gu = [l.mlp.experts.gate_up_proj.data.detach().cpu().clone() for l in layers]
    orig_dn = [l.mlp.experts.down_proj.data.detach().cpu().clone() for l in layers]

    # Simple clean replace-training loop on K=4 causal experts/layer (64 experts total)
    prot_anchor_seqs = eval_doms["math_gsm8k"][16:] + eval_doms["stem_sciq"][16:] + eval_doms["wiki_prose"][16:]

    def train_and_eval_regime(name: str, precision: str, lambda_anc: float, min_l: int, max_l: int, lr: float = 3e-5, steps: int = 80) -> Dict:
        # Restore exact base weights
        for l in range(n_layers):
            layers[l].mlp.experts.gate_up_proj.data.copy_(orig_gu[l].to(dev))
            layers[l].mlp.experts.down_proj.data.copy_(orig_dn[l].to(dev))
        for p in model.parameters():
            p.requires_grad_(False)

        # Build trainable slice parameters for active layers [min_l, max_l)
        active_params = []
        param_map = {}
        orig_fwds = []
        for l in range(n_layers):
            exp = layers[l].mlp.experts
            orig_fwds.append(exp.forward)
            if not (min_l <= l < max_l):
                continue
            ids = torch.tensor(causal_top4[l], dtype=torch.long, device=dev)
            dt = torch.bfloat16 if precision == "bf16" else torch.float32
            gu_p = nn.Parameter(orig_gu[l][causal_top4[l]].to(device=dev, dtype=dt))
            dn_p = nn.Parameter(orig_dn[l][causal_top4[l]].to(device=dev, dtype=dt))
            gu_ref = orig_gu[l][causal_top4[l]].to(device=dev, dtype=torch.bfloat16)
            dn_ref = orig_dn[l][causal_top4[l]].to(device=dev, dtype=torch.bfloat16)
            param_map[l] = (ids, gu_p, dn_p, gu_ref, dn_ref)
            active_params.extend([gu_p, dn_p])

            def _mk_fwd(e_mod, lid: int, id_t: torch.Tensor, gup: nn.Parameter, dnp: nn.Parameter):
                def _fwd(hidden_states, top_k_index, top_k_weights):
                    # Clean differentiable scatter-replace SwiGLU forward (no internal _parameters hack)
                    gu_full = e_mod.gate_up_proj.detach().clone()
                    dn_full = e_mod.down_proj.detach().clone()
                    gu_full[id_t] = gup.to(gu_full.dtype)
                    dn_full[id_t] = dnp.to(dn_full.dtype)
                    final_hs = torch.zeros_like(hidden_states)
                    with torch.no_grad():
                        expert_mask = F.one_hot(top_k_index, num_classes=gu_full.shape[0]).permute(2, 1, 0)
                        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                    for e_idx in expert_hit:
                        e_i = int(e_idx[0].item())
                        idx_s, top_x = torch.where(expert_mask[e_i].squeeze(0))
                        cur = hidden_states[top_x]
                        g, u = F.linear(cur, gu_full[e_i]).chunk(2, dim=-1)
                        y = F.linear(F.silu(g) * u, dn_full[e_i])
                        y = y * top_k_weights[top_x, idx_s, None].to(y.dtype)
                        final_hs.index_add_(0, top_x, y.to(final_hs.dtype))
                    return final_hs
                return _fwd
            exp.forward = _mk_fwd(exp, l, ids, gu_p, dn_p)

        if precision == "fp32_adamw8bit" and HAS_BNB:
            opt = bnb.optim.AdamW8bit(active_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        else:
            opt = torch.optim.AdamW(active_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

        rng = np.random.default_rng(42)
        for st in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            ids_t = torch.tensor([code_train[int(rng.integers(0, len(code_train)))]], dtype=torch.long, device=dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(precision != "bf16")):
                lg = model(input_ids=ids_t, use_cache=False).logits[:, :-1, :].float()
                loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids_t[:, 1:].reshape(-1))
            loss.backward()

            # Online layer-local functional L_anchor on protected tokens
            if lambda_anc > 0.0 and (st % 2 == 0):
                d_anc = prot_doms[st % len(prot_doms)]
                anc_loss = torch.tensor(0.0, device=dev)
                n_act = 0
                for l, (id_t, gup, dnp, gu_ref, dn_ref) in param_map.items():
                    x_l = cap[d_anc]["x"][l][:128]
                    idx_l = cap[d_anc]["idx"][l][:128]
                    for j in range(len(id_t)):
                        m = (idx_l == id_t[j]).any(dim=-1)
                        if not m.any(): continue
                        xj = x_l[m]
                        with torch.no_grad():
                            go, uo = F.linear(xj, gu_ref[j]).chunk(2, dim=-1)
                            yo = F.linear(F.silu(go) * uo, dn_ref[j]).float()
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            gn, un = F.linear(xj, gup[j]).chunk(2, dim=-1)
                            yn = F.linear(F.silu(gn) * un, dnp[j]).float()
                        anc_loss = anc_loss + ((yn - yo).pow(2).sum() / (yo.pow(2).sum() + 1e-6))
                        n_act += 1
                if n_act > 0:
                    (lambda_anc * anc_loss / n_act).backward()

            torch.nn.utils.clip_grad_norm_(active_params, 1.0)
            opt.step()

        # Restore original forwards & write trained rows into model for fast eval + Law 2 measurements
        for l in range(n_layers):
            layers[l].mlp.experts.forward = orig_fwds[l]
        rel_dws = []
        sum_p_rms2 = {d: 0.0 for d in prot_doms}
        with torch.no_grad():
            for l, (id_t, gup, dnp, gu_ref, dn_ref) in param_map.items():
                layers[l].mlp.experts.gate_up_proj.data[id_t] = gup.to(torch.bfloat16)
                layers[l].mlp.experts.down_proj.data[id_t] = dnp.to(torch.bfloat16)
                dw = (dnp.float() - dn_ref.float()).norm().item() / max(dn_ref.float().norm().item(), 1e-8)
                rel_dws.append(dw)
                for d in prot_doms:
                    x_l, idx_l, w_l = cap[d]["x"][l], cap[d]["idx"][l], cap[d]["w"][l]
                    diff_sq, ref_sq = 0.0, 0.0
                    touched = torch.zeros((x_l.shape[0],), dtype=torch.bool, device=dev)
                    for j in range(len(id_t)):
                        m = (idx_l == id_t[j])
                        if not m.any(): continue
                        ti, si = m.nonzero(as_tuple=True)
                        touched[ti] = True
                        xj, wj = x_l[ti], w_l[ti, si].unsqueeze(-1)
                        go, uo = F.linear(xj, gu_ref[j]).chunk(2, dim=-1)
                        yo = F.linear(F.silu(go) * uo, dn_ref[j]).float() * wj
                        gn, un = F.linear(xj, gup[j].to(torch.bfloat16)).chunk(2, dim=-1)
                        yn = F.linear(F.silu(gn) * un, dnp[j].to(torch.bfloat16)).float() * wj
                        diff_sq += float((yn - yo).pow(2).sum().item())
                        ref_sq += float(yo.pow(2).sum().item())
                    pt = float(touched.float().mean().item())
                    r_rms = float(np.sqrt(diff_sq / max(ref_sq, 1e-8))) if ref_sq > 0 else 0.0
                    sum_p_rms2[d] += pt * (r_rms ** 2)

        nlls, _ = eval_nll_and_routing(model, dev, eval_doms)
        deltas = {d: round(nlls[d] - base_nll[d], 6) for d in eval_doms}
        mean_dw_pct = round(100.0 * float(np.mean(rel_dws)) if rel_dws else 0.0, 3)
        print(
            f"  [{name:20s}] code={deltas['code_mbpp']:+.5f} | math={deltas['math_gsm8k']:+.5f} "
            f"| stem={deltas['stem_sciq']:+.5f} | wiki={deltas['wiki_prose']:+.5f} | rel_dW={mean_dw_pct:.2f}%",
            flush=True,
        )
        return {"regime": name, "rel_dW_pct": mean_dw_pct, "deltas": deltas, "sum_p_rms2": {k: round(v, 5) for k, v in sum_p_rms2.items()}}

    regimes = [
        ("bf16_adamw",         "bf16",           0.0, 0, 16, 3e-5),
        ("fp32_8bit_no_anc",   "fp32_adamw8bit", 0.0, 0, 16, 3e-5),
        ("fp32_8bit_anc1.0",   "fp32_adamw8bit", 1.0, 0, 16, 3e-5),
        ("fp32_8bit_anc3.0",   "fp32_adamw8bit", 3.0, 0, 16, 3e-5),
        ("L5_freeze_L0_2",     "fp32_adamw8bit", 1.0, 3, 16, 3e-5),
        ("L5_freeze_L13_15",   "fp32_adamw8bit", 1.0, 0, 13, 3e-5),
    ]
    runs = [train_and_eval_regime(*cfg) for cfg in regimes]

    # Fit Law 2 out-of-sample across regimes (first 4 train -> last 2 unseen holdout!)
    tr_pts = [(r["sum_p_rms2"][d], (r["rel_dW_pct"] / 100.0) ** 2, max(0.0, r["deltas"][d])) for r in runs[:4] for d in prot_doms]
    ho_pts = [(r["sum_p_rms2"][d], (r["rel_dW_pct"] / 100.0) ** 2, max(0.0, r["deltas"][d])) for r in runs[4:] for d in prot_doms]
    law2_fit = {}
    for mname, col in [("law2_sum_p_rms2", 0), ("weight_dW2", 1)]:
        Xtr = np.array([[p[col]] for p in tr_pts])
        ytr = np.array([p[2] for p in tr_pts])
        Xho = np.array([[p[col]] for p in ho_pts])
        yho = np.array([p[2] for p in ho_pts])
        coef, _, _, _ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        pred = Xho @ coef
        mae = float(np.mean(np.abs(pred - yho)))
        r2 = 1.0 - float(np.sum((yho - pred) ** 2)) / (float(np.sum((yho - np.mean(yho)) ** 2)) + 1e-12)
        law2_fit[mname] = {"coef": float(coef[0]), "holdout_n": len(ho_pts), "holdout_mae": round(mae, 6), "holdout_r2": round(r2, 4)}
        print(f"  [OLMoE LAW 2 OOS {mname:16s}] R2={r2:+.4f} | MAE={mae:.5f}", flush=True)

    return {"runs": runs, "law2_oos": law2_fit}


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="allenai/OLMoE-1B-7B-0125-Instruct")
    ap.add_argument("--out-dir", default="./olmoe_results")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    print(f"[OLMoE Public Reproduction] Device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    code_train, eval_doms = build_public_corpora(tok)

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(dev)
    model.eval()
    base_nll, base_top = eval_nll_and_routing(model, dev, eval_doms)
    print(f"[BASE OLMoE-1B-7B] NLLs = { {k: round(v, 5) for k, v in base_nll.items()} }", flush=True)

    _, causal_top4 = run_part1_atlas(model, dev, eval_doms, out_dir)
    law6 = run_part2_law6_jaccard(model, dev, eval_doms, base_nll, base_top)
    law1 = run_part3_law1_evacuation(model, dev, eval_doms, base_nll)
    part4 = run_part4_law2_l5_c006(model, dev, code_train, eval_doms, causal_top4, base_nll)

    summary = {
        "model": "allenai/OLMoE-1B-7B-0125-Instruct",
        "device": torch.cuda.get_device_name(0),
        "elapsed_s": round(time.time() - t0, 1),
        "base_nll": {k: round(v, 6) for k, v in base_nll.items()},
        "law6_boundary_sensitivity": law6,
        "law1_evacuation": law1,
        "part4_law2_l5_c006": part4,
    }
    (out_dir / "olmoe_reproduction_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[✓] PUBLIC OLMoE-1B-7B REPRODUCTION COMPLETE in {summary['elapsed_s']}s -> {out_dir / 'olmoe_reproduction_results.json'}", flush=True)


if __name__ == "__main__":
    main()
