#!/usr/bin/env python3
"""
Public Multi-GPU Bonus Verification: L1 (Evacuation + Neutral u) & L6 (Boundary Sensitivity / Broken Jaccard)
on untouched public Qwen/Qwen3.5-35B-A3B-Base (256 experts, top_k=8, 40 layers, Apache 2.0).
Zero private weights, zero private datasets (uses public HF mbpp, gsm8k, sciq, wikitext).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer


def balanced_dm_4gpu() -> Dict[str, int]:
    dm = {
        "model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.rotary_emb": 0,
        "model.language_model.norm": 3, "lm_head": 3, "mtp": 3,
    }
    for l_i in range(40):
        dm[f"model.language_model.layers.{l_i}"] = 0 if l_i <= 12 else (1 if l_i <= 27 else (2 if l_i <= 34 else 3))
    return dm


def build_public_eval(tok, max_len: int = 384) -> Dict[str, List[List[int]]]:
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    gsm = load_dataset("openai/gsm8k", "main", split="test[:24]")
    sciq = load_dataset("allenai/sciq", split="test[:24]")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    def enc(texts, n):
        out = []
        for t in texts:
            if not t or len(t.strip()) < 40: continue
            ids = tok.encode(t.strip(), add_special_tokens=True)[:max_len]
            if len(ids) >= 48: out.append(ids)
            if len(out) >= n: break
        return out

    return {
        "code_mbpp": enc([f"# Task: {r['prompt']}\n```python\n{r['code']}\n```" for r in mbpp], 16),
        "math_gsm8k": enc([f"Question: {r['question']}\nAnswer: {r['answer']}" for r in gsm], 16),
        "stem_sciq": enc([f"Question: {r['question']}\nExplanation: {r['support']}\nAnswer: {r['correct_answer']}" for r in sciq], 16),
        "wiki_prose": enc([r["text"] for r in wiki if len(r["text"].strip()) > 180], 16),
    }


@torch.inference_mode()
def eval_nll_and_routing(model, first_dev, eval_doms: Dict[str, List[List[int]]]):
    layers = model.model.language_model.layers
    n_layers = len(layers)
    cur_top = [None] * n_layers
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, _inp, out):
                cur_top[lid] = out[2].detach().to(device="cpu", dtype=torch.int16)
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))

    nlls, top_map = {}, {}
    for d, seqs in eval_doms.items():
        losses = []
        per_l = [[] for _ in range(n_layers)]
        for ids in seqs:
            t = torch.tensor([ids], dtype=torch.long, device=first_dev)
            lg = model(input_ids=t, use_cache=False).logits[:, :-1, :].float()
            losses.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t[:, 1:].reshape(-1)).item()))
            for l in range(n_layers):
                per_l[l].append(cur_top[l])
        nlls[d] = float(np.mean(losses))
        top_map[d] = [torch.cat(per_l[l], dim=0) for l in range(n_layers)]
    for h in hooks:
        h.remove()
    return nlls, top_map


def topk_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    K = a.shape[1]
    inter = (a.unsqueeze(2) == b.unsqueeze(1)).any(dim=2).float().sum(dim=1)
    return float((inter / (2.0 * K - inter)).mean().item())


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen/Qwen3.5-35B-A3B-Base")
    ap.add_argument("--out-json", default="./qwen35_base_l1_l6_results.json")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eval_doms = build_public_eval(tok)
    print(f"[Public Qwen3.5-35B-A3B-Base] Loading across 4 GPUs...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=balanced_dm_4gpu(),
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    first_dev = next(model.parameters()).device
    layers = model.model.language_model.layers
    n_layers = len(layers)
    num_e = int(model.config.text_config.num_experts)
    top_k = int(model.config.text_config.num_experts_per_tok)
    print(f"  Loaded Qwen3.5-35B-A3B-Base: layers={n_layers}, num_experts={num_e}, top_k={top_k}", flush=True)

    base_nll, base_top = eval_nll_and_routing(model, first_dev, eval_doms)
    print(f"  [BASE] NLLs = { {k: round(v, 5) for k, v in base_nll.items()} }", flush=True)

    # --- PART A: LAW 6 (Boundary Sensitivity / Broken Top-k Jaccard) ---
    print("\n=== [PART A] LAW 6 (Boundary Sensitivity on public Qwen3.5-35B-A3B-Base 256E top_k=8) ===", flush=True)
    exp0 = layers[0].mlp.experts
    orig_dn0 = exp0.down_proj.data[0].clone()
    g = torch.Generator(device=orig_dn0.device).manual_seed(77)
    noise = torch.randn(orig_dn0.shape, generator=g, device=orig_dn0.device, dtype=torch.float32)
    noise = (noise * (orig_dn0.float().norm() / noise.norm())).to(orig_dn0.dtype)
    law6 = {}
    for eps in (0.005, 0.01, 0.02, 0.05):
        exp0.down_proj.data[0] = orig_dn0 + eps * noise
        nlls, tops = eval_nll_and_routing(model, first_dev, eval_doms)
        dom_out = {}
        for d in eval_doms:
            js = [topk_jaccard(base_top[d][l], tops[d][l]) for l in range(n_layers)]
            dom_out[d] = {
                "delta_nll": round(nlls[d] - base_nll[d], 6),
                "jaccard_mean": round(float(np.mean(js)), 4),
                "jaccard_min_layer": round(float(min(js)), 4),
                "jaccard_min_idx": int(np.argmin(js)),
            }
        law6[f"eps_{eps}"] = dom_out
        s = dom_out["stem_sciq"]
        print(f"  eps={eps:.3f} | stem delta_NLL={s['delta_nll']:+.6f} | Top-{top_k} Jaccard mean={s['jaccard_mean']:.4f} (min={s['jaccard_min_layer']:.4f} @ L{s['jaccard_min_idx']:02d})", flush=True)
    exp0.down_proj.data[0] = orig_dn0

    # --- PART B: LAW 1 (MoE Evacuation Law + FP64 Neutral Direction u) ---
    print("\n=== [PART B] LAW 1 (Evacuation Law + FP64 Neutral u on public Qwen3.5-35B-A3B-Base 256E) ===", flush=True)
    h_all = [[] for _ in range(n_layers)]
    r1_cnt = np.zeros((n_layers, num_e), dtype=np.int64)
    rk_cnt = np.zeros((n_layers, num_e), dtype=np.int64)
    cur_c = [True]
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, inp, out):
                h = inp[0].detach().reshape(-1, inp[0].shape[-1])[::4].cpu()
                h_all[lid].append(h)
                if cur_c[0]:
                    idx = out[2].detach().cpu().numpy()
                    for x in idx[:, 0]: r1_cnt[lid, x] += 1
                    for x in idx[:, top_k - 1]: rk_cnt[lid, x] += 1
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))
    with torch.inference_mode():
        cur_c[0] = True
        for ids in eval_doms["code_mbpp"]:
            _ = model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=first_dev), use_cache=False)
        cur_c[0] = False
        for ids in eval_doms["stem_sciq"]:
            _ = model.model(input_ids=torch.tensor([ids], dtype=torch.long, device=first_dev), use_cache=False)
    for h in hooks:
        h.remove()

    u_dirs = []
    u_max_std = 0.0
    for l in range(n_layers):
        H = torch.cat(h_all[l], dim=0).double()
        mu = H.mean(dim=0)
        C = (H - mu).T @ (H - mu) / len(H) + 1e-3 * torch.eye(H.shape[1], dtype=torch.float64)
        u = torch.linalg.solve(C, mu)
        u = u / (u @ mu)
        u_max_std = max(u_max_std, float((H @ u).std().item()))
        u_dirs.append(u.float())
    print(f"  [B1 Neutral u] 40/40 layers solved in FP64 | max_std(u^T h) = {u_max_std:.4f}", flush=True)

    # Backup original weights on CPU for clean restore
    orig_g = [l.mlp.gate.weight.data.detach().cpu().clone() for l in layers]
    orig_gu = [l.mlp.experts.gate_up_proj.data.detach().cpu().clone() for l in layers]
    orig_dn = [l.mlp.experts.down_proj.data.detach().cpu().clone() for l in layers]

    sub_eval = {"code_mbpp": eval_doms["code_mbpp"], "stem_sciq": eval_doms["stem_sciq"]}
    law1 = {"b1_u_max_std": round(u_max_std, 5), "configs": {}}

    def _set_n_exp(mlp_mod, n_val: int):
        for obj in (mlp_mod, getattr(mlp_mod, "gate", None), getattr(mlp_mod, "experts", None)):
            if obj is not None:
                if hasattr(obj, "num_experts"):
                    obj.num_experts = n_val
                if hasattr(obj, "out_features"):
                    obj.out_features = n_val

    def test_clone(pm: str, alpha: float, use_u: bool = False):
        for l in range(n_layers):
            dev_l = layers[l].mlp.gate.weight.device
            pids = np.argsort(-(r1_cnt[l] if pm == "rank1" else rk_cnt[l]))[:4]
            pid_t = torch.tensor(pids, dtype=torch.long)
            gw_new = orig_g[l][pid_t].clone()
            if use_u:
                gw_new = gw_new - 2.5 * u_dirs[l].unsqueeze(0).to(gw_new.dtype)
            layers[l].mlp.gate.weight.data = torch.cat([orig_g[l], gw_new], dim=0).to(dev_l)
            layers[l].mlp.experts.gate_up_proj.data = torch.cat([orig_gu[l], orig_gu[l][pid_t]], dim=0).to(dev_l)
            layers[l].mlp.experts.down_proj.data = torch.cat([orig_dn[l], alpha * orig_dn[l][pid_t]], dim=0).to(dev_l)
            _set_n_exp(layers[l].mlp, orig_g[l].shape[0] + 4)
        nlls, _ = eval_nll_and_routing(model, first_dev, sub_eval)
        for l in range(n_layers):
            dev_l = layers[l].mlp.gate.weight.device
            layers[l].mlp.gate.weight.data = orig_g[l].to(dev_l)
            layers[l].mlp.experts.gate_up_proj.data = orig_gu[l].to(dev_l)
            layers[l].mlp.experts.down_proj.data = orig_dn[l].to(dev_l)
            _set_n_exp(layers[l].mlp, orig_g[l].shape[0])
        return {"code_delta": round(nlls["code_mbpp"] - base_nll["code_mbpp"], 6), "stem_delta": round(nlls["stem_sciq"] - base_nll["stem_sciq"], 6)}

    for pm in ("rank1", "rank8"):
        for a in (0.0, 0.25, 0.5, 1.0):
            tag = f"{pm}_alpha{a:.2f}"
            r = test_clone(pm, a, False)
            law1["configs"][tag] = r
            print(f"  {tag:24s} | code_d={r['code_delta']:+.5f} | stem_d={r['stem_delta']:+.5f}", flush=True)
    ru = test_clone("rank1", 1.0, True)
    law1["configs"]["rank1_b1_u_offset_alpha1.00"] = ru
    print(f"  {'rank1_b1_u_offset_a1.00':24s} | code_d={ru['code_delta']:+.5f} | stem_d={ru['stem_delta']:+.5f}", flush=True)

    out = {
        "model": "Qwen/Qwen3.5-35B-A3B-Base",
        "num_experts": num_e,
        "top_k": top_k,
        "elapsed_s": round(time.time() - t0, 1),
        "base_nll": {k: round(v, 6) for k, v in base_nll.items()},
        "law6_boundary_sensitivity": law6,
        "law1_evacuation": law1,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[✓] PUBLIC Qwen3.5-35B-A3B-Base L1+L6 COMPLETE in {out['elapsed_s']}s -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
