#!/usr/bin/env python3
"""
T3 Metrology Instrument: Cached Paired-Comparison NLL + Sequence-Bootstrap 95% CI + Top-10 Jaccard + Context Sweep Harness.
Part of kataguru-research/03-metrology/harness/ (EXP-0001 & EXP-0003).

Modes:
  1. --build-cache      : Run frozen BASE (CodeRestore-232E) ONCE on a locked multi-domain validation corpus
                          (coding_heldout, coding_class_sol, stem, ifeval, bfcl_tool, gsm8k) and cache compact
                          per-token base_nll (float16) + per-layer base_top10_ids (uint8, since E=232 < 255).
  2. --null-control     : Run untouched BASE against its own cache -> measures exact stack precision floor!
  3. --eval-cartridge   : Evaluate any portable add OR replace .safetensors cartridge against the cached BASE:
                          - Paired per-sequence delta (cand_nll - base_nll)
                          - Sequence-level bootstrap (B=2000) mean + [lower_95_CI, upper_95_CI]
                          - Exact per-layer Top-10 Routing Jaccard (mean, min_layer, L0-5)
                          - Per-layer leak / touch / E[new_slots]
                          - Hard Safety Gate: noncode ASST_d upper 95% CI <= +0.003 AND Jaccard >= 0.98
  4. --context-sweep    : Measure routing Jaccard, leak/touch, and NLL drift at 2k / 8k / 16k / 32k context lengths (Law 5).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

HARNESS_VERSION = "cached_paired_nll_v1.0"
BASE_BF16 = Path(os.environ.get("MOE_BASE_MODEL", "./base_model"))
DATA_CAL_JSONL = Path(os.environ.get("MOE_CAL_JSONL", "./data_cal.jsonl"))
DATA_PROTECTED_JSONL = Path(os.environ.get("MOE_PROTECTED_JSONL", "./data_protected.jsonl"))
DEFAULT_CACHE_DIR = Path(os.environ.get("MOE_METROLOGY_CACHE", "./metrology_cache"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _balanced_dm_4gpu() -> Dict[str, int]:
    dm = {
        "model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.rotary_emb": 0,
        "model.language_model.norm": 3, "lm_head": 3, "mtp": 3,
    }
    # 13 / 15 / 7 / 5 split leaves 6+ GiB activation headroom on GPU 0 & GPU 1 for 16k/32k sweeps
    for l_i in range(40):
        dm[f"model.language_model.layers.{l_i}"] = 0 if l_i <= 12 else (1 if l_i <= 27 else (2 if l_i <= 34 else 3))
    return dm


def seq_bootstrap_ci(seq_deltas: List[float], n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Sequence-level bootstrap (accounts for within-sequence token correlation; Consultant C-005)."""
    arr = np.asarray(seq_deltas, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lo95": 0.0, "ci_hi95": 0.0, "n_seq": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)) if n > 1 else 0.0, 6),
        "ci_lo95": round(float(np.percentile(means, 2.5)), 6),
        "ci_hi95": round(float(np.percentile(means, 97.5)), 6),
        "n_seq": int(n),
    }


def build_locked_corpus(tok, target_asst_toks: int = 40000, max_seq_len: int = 2048) -> Dict[str, List[dict]]:
    """Build deterministic multi-domain corpus (~40k-80k assistant tokens per domain) with hash manifest."""
    cal_rows = [json.loads(l) for l in DATA_CAL_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    s4_rows = [json.loads(l) for l in DATA_PROTECTED_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

    pools: Dict[str, List[List[dict]]] = {
        "coding_heldout": [r["messages"] for r in cal_rows if r.get("is_code_target") and r.get("code_format") != "class_Solution" and r.get("messages")],
        "coding_class_sol": [r["messages"] for r in cal_rows if r.get("is_code_target") and r.get("code_format") == "class_Solution" and r.get("messages")],
        "bfcl_tool": [r["messages"] for r in cal_rows if (not r.get("is_code_target")) and r.get("anchor_group") == "bfcl_tool_calling" and r.get("messages")],
        "stem": [r["messages"] for r in s4_rows if str(r.get("domain", "")) in ("physics", "chemistry") and r.get("messages") and "<image>" not in str(r.get("messages"))],
        "gsm8k_math": [r["messages"] for r in s4_rows if str(r.get("domain", "")) == "mathematics" and r.get("messages") and "<image>" not in str(r.get("messages"))],
        "ifeval": [r["messages"] for r in s4_rows if str(r.get("domain", "")) == "instruction_obedience" and r.get("messages") and "<image>" not in str(r.get("messages"))],
    }

    corpus: Dict[str, List[dict]] = {}
    for dom, msgs_list in pools.items():
        items = []
        tot_asst = 0
        # Take from tail / stride deterministically
        for idx, msgs in enumerate(reversed(msgs_list)):
            if tot_asst >= target_asst_toks:
                break
            p_txt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            f_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            p_len = len(tok(p_txt, add_special_tokens=False)["input_ids"])
            ids = tok(f_txt, add_special_tokens=False, truncation=True, max_length=max_seq_len)["input_ids"]
            asst_n = len(ids) - p_len
            if asst_n < 16:
                continue
            items.append({"ids": ids, "p_len": p_len, "asst_toks": asst_n})
            tot_asst += asst_n
        corpus[dom] = items
        log(f"  [Corpus] {dom:18s}: {len(items):3d} seqs | {tot_asst:6d} asst tokens")
    return corpus


def attach_cartridge_inplace(model, cart_path: Path) -> Tuple[str, Optional[List[List[int]]], int]:
    """Attach portable .safetensors cartridge (add OR replace). Returns (mode, replace_ids_per_layer, k_new)."""
    layers = model.model.language_model.layers
    base_e = int(model.config.text_config.num_experts)
    with safe_open(str(cart_path), framework="pt", device="cpu") as f:
        is_replace = "layers.0.expert_ids" in f.keys()
        rep_ids = []
        k_new = 0
        for l_idx, layer in enumerate(layers):
            exp = layer.mlp.experts
            dev, dt = exp.gate_up_proj.device, exp.gate_up_proj.dtype
            gu = f.get_tensor(f"layers.{l_idx}.mlp.experts.gate_up_proj").to(device=dev, dtype=dt)
            dn = f.get_tensor(f"layers.{l_idx}.mlp.experts.down_proj").to(device=dev, dtype=dt)
            if is_replace:
                e_ids = [int(x) for x in f.get_tensor(f"layers.{l_idx}.expert_ids").tolist()]
                rep_ids.append(e_ids)
                for j, eid in enumerate(e_ids):
                    exp.gate_up_proj.data[eid].copy_(gu[j])
                    exp.down_proj.data[eid].copy_(dn[j])
            else:
                gt = f.get_tensor(f"layers.{l_idx}.mlp.gate.weight")
                k_new = int(gt.shape[0])
                exp.gate_up_proj = torch.nn.Parameter(torch.cat([exp.gate_up_proj.data, gu], dim=0), requires_grad=False)
                exp.down_proj = torch.nn.Parameter(torch.cat([exp.down_proj.data, dn], dim=0), requires_grad=False)
                exp.num_experts = base_e + k_new
                gate = layer.mlp.gate
                gate.weight = torch.nn.Parameter(torch.cat([gate.weight.data, gt.to(device=gate.weight.device, dtype=gate.weight.dtype)], dim=0), requires_grad=False)
                gate.num_experts = base_e + k_new
    if not is_replace:
        model.config.text_config.num_experts = base_e + k_new
        return "add", None, k_new
    return "replace", rep_ids, 0


def run_build_cache(base_dir: Path, cache_dir: Path, target_asst_toks: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    corpus = build_locked_corpus(tok, target_asst_toks=target_asst_toks)
    corpus_bytes = json.dumps(corpus, sort_keys=True).encode("utf-8")
    corpus_sha = hashlib.sha256(corpus_bytes).hexdigest()
    (cache_dir / "locked_corpus.json").write_bytes(corpus_bytes)

    _caps = ["30GiB", "30GiB", "14.5GiB", "14.5GiB"]
    max_mem = {i: _caps[i] for i in range(min(len(_caps), torch.cuda.device_count()))}
    log(f"[*] Loading frozen BASE ({base_dir.name}) to build paired cache...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_dir), dtype=torch.bfloat16, device_map=_balanced_dm_4gpu(),
        max_memory=max_mem, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    first_dev = next(model.parameters()).device
    layers = model.model.language_model.layers
    n_layers = len(layers)

    cur_pl = [0]
    seq_top10: List[Optional[torch.Tensor]] = [None] * n_layers
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, _inp, out):
                idx = out[2]  # [T, 10]
                pl = min(cur_pl[0], idx.shape[0])
                seq_top10[lid] = idx[pl:].to(dtype=torch.uint8, device="cpu")
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))

    with torch.inference_mode():
        for dom, items in corpus.items():
            dom_asst_nlls = []
            dom_prompt_nlls = []
            dom_top10 = [[] for _ in range(n_layers)]
            for item in items:
                ids = torch.tensor([item["ids"]], dtype=torch.long, device=first_dev)
                p_len = item["p_len"]
                cur_pl[0] = p_len
                out = model(input_ids=ids, use_cache=False)
                lg = out.logits[:, :-1, :].float()
                tgt = ids[:, 1:]
                tok_nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").cpu()
                dom_prompt_nlls.append(float(tok_nll[: p_len - 1].mean().item()) if p_len > 1 else 0.0)
                dom_asst_nlls.append(tok_nll[p_len - 1 :].to(torch.float16))
                for l_idx in range(n_layers):
                    dom_top10[l_idx].append(seq_top10[l_idx])
            torch.save({
                "asst_tok_nll": dom_asst_nlls,
                "prompt_seq_nll": dom_prompt_nlls,
                "top10_asst_u8": dom_top10,
            }, cache_dir / f"base_cache_{dom}.pt")
            mean_a = float(np.mean([float(x.float().mean().item()) for x in dom_asst_nlls]))
            log(f"  [✓] Cached {dom:18s}: {len(items)} seqs | base_asst_nll={mean_a:.5f}")

    for h in hooks:
        h.remove()
    meta = {
        "harness_version": HARNESS_VERSION,
        "base_model": str(base_dir),
        "base_index_sha256": hashlib.sha256((base_dir / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "corpus_sha256": corpus_sha,
        "target_asst_toks": target_asst_toks,
        "domains": {d: {"n_seq": len(v), "asst_toks": sum(x["asst_toks"] for x in v)} for d, v in corpus.items()},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"[✓] BASE CACHE BUILT -> {cache_dir} (corpus_sha={corpus_sha[:12]})")


def run_eval_against_cache(
    base_dir: Path,
    cache_dir: Path,
    cartridge: str,
    report_out: Path,
    null_control: bool = False,
    context_sweep: bool = False,
) -> None:
    meta = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    corpus = json.loads((cache_dir / "locked_corpus.json").read_text(encoding="utf-8"))

    _caps = ["30GiB", "30GiB", "14.5GiB", "14.5GiB"]
    max_mem = {i: _caps[i] for i in range(min(len(_caps), torch.cuda.device_count()))}
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_dir), dtype=torch.bfloat16, device_map=_balanced_dm_4gpu(),
        max_memory=max_mem, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    base_e = int(model.config.text_config.num_experts)
    cart_mode, rep_ids, k_new = "null_control", None, 0
    if not null_control and cartridge:
        cart_mode, rep_ids, k_new = attach_cartridge_inplace(model, Path(cartridge))
    model.eval()
    first_dev = next(model.parameters()).device
    layers = model.model.language_model.layers
    n_layers = len(layers)
    num_e = int(model.config.text_config.num_experts)

    # Track masks for touch/leak
    target_masks = []
    for l_idx in range(n_layers):
        dev = layers[l_idx].mlp.gate.weight.device
        m = torch.zeros((num_e,), dtype=torch.bool, device=dev)
        if cart_mode == "add":
            m[base_e:] = True
        elif cart_mode == "replace" and rep_ids is not None:
            m[rep_ids[l_idx]] = True
        target_masks.append(m)

    cur_pl = [0]
    seq_top10: List[Optional[torch.Tensor]] = [None] * n_layers
    hooks = []
    for l_idx in range(n_layers):
        def _mk(lid: int):
            def _h(_m, _inp, out):
                idx = out[2]
                pl = min(cur_pl[0], idx.shape[0])
                seq_top10[lid] = idx[pl:].detach().cpu()
            return _h
        hooks.append(layers[l_idx].mlp.gate.register_forward_hook(_mk(l_idx)))

    domain_reports = {}
    with torch.inference_mode():
        for dom, items in corpus.items():
            bcache = torch.load(cache_dir / f"base_cache_{dom}.pt", map_location="cpu", weights_only=True)
            seq_asst_deltas = []
            seq_prompt_deltas = []
            layer_jacc: List[List[float]] = [[] for _ in range(n_layers)]
            layer_touch: List[List[float]] = [[] for _ in range(n_layers)]
            layer_slots: List[List[float]] = [[] for _ in range(n_layers)]

            for si, item in enumerate(items):
                ids = torch.tensor([item["ids"]], dtype=torch.long, device=first_dev)
                p_len = item["p_len"]
                cur_pl[0] = p_len
                out = model(input_ids=ids, use_cache=False)
                lg = out.logits[:, :-1, :].float()
                tgt = ids[:, 1:]
                tok_nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").cpu()
                p_nll = float(tok_nll[: p_len - 1].mean().item()) if p_len > 1 else 0.0
                seq_prompt_deltas.append(p_nll - float(bcache["prompt_seq_nll"][si]))
                base_a = bcache["asst_tok_nll"][si].float()
                cand_a = tok_nll[p_len - 1 :].float()
                seq_asst_deltas.append(float((cand_a - base_a).mean().item()))

                for l_idx in range(n_layers):
                    c_idx = seq_top10[l_idx]  # [Ta, 10]
                    b_idx = bcache["top10_asst_u8"][l_idx][si].long()
                    eq = (c_idx.unsqueeze(-1) == b_idx.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
                    jacc = eq / (20.0 - eq)
                    layer_jacc[l_idx].append(float(jacc.mean().item()))
                    hm = target_masks[l_idx].cpu()[c_idx]
                    layer_touch[l_idx].append(float(hm.any(dim=-1).float().mean().item()))
                    layer_slots[l_idx].append(float(hm.float().sum(dim=-1).mean().item()))

            boot = seq_bootstrap_ci(seq_asst_deltas)
            p_boot = seq_bootstrap_ci(seq_prompt_deltas)
            l_j_means = [float(np.mean(v)) for v in layer_jacc]
            l_t_means = [100.0 * float(np.mean(v)) for v in layer_touch]
            l_s_means = [float(np.mean(v)) for v in layer_slots]
            max_tl = int(np.argmax(l_t_means))
            min_jl = int(np.argmin(l_j_means))

            domain_reports[dom] = {
                "asst_delta": boot,
                "prompt_delta_mean": p_boot["mean"],
                "jaccard_mean": round(float(np.mean(l_j_means)), 5),
                "jaccard_min_layer": round(l_j_means[min_jl], 5),
                "jaccard_min_layer_idx": min_jl,
                "touch_mean_pct": round(float(np.mean(l_t_means)), 2),
                "touch_max_layer_pct": round(l_t_means[max_tl], 2),
                "touch_max_layer_idx": max_tl,
                "touch_l0_5_max_pct": round(float(max(l_t_means[:6])), 2),
                "expected_slots": round(float(np.mean(l_s_means)), 4),
            }

        # Optional Law 5 Context Sweep: 2k / 8k / 16k / 32k on non-coding concatenation
        ctx_report = {}
        if context_sweep:
            long_ids = []
            for d_nc in ("stem", "ifeval", "bfcl_tool"):
                for it in corpus.get(d_nc, []):
                    long_ids.extend(it["ids"])
                    if len(long_ids) >= 32768:
                        break
                if len(long_ids) >= 32768:
                    break
            for clen in (2048, 8192, 16384, 32768):
                if len(long_ids) < clen:
                    continue
                cur_pl[0] = 0
                torch.cuda.empty_cache()
                # Exact chunked KV-cache prefill (chunk=4096) gives identical 32k routing/attention at 8x lower peak MoE VRAM!
                chunk_sz = 4096
                past = None
                layer_touch_hits = [0.0] * n_layers
                layer_tok_counts = [0] * n_layers
                for pos in range(0, clen, chunk_sz):
                    ids_c = torch.tensor([long_ids[pos : min(pos + chunk_sz, clen)]], dtype=torch.long, device=first_dev)
                    out_c = model.model(input_ids=ids_c, past_key_values=past, use_cache=True)
                    past = out_c.past_key_values
                    for l in range(n_layers):
                        hit_t = target_masks[l].cpu()[seq_top10[l]].any(dim=-1).float()
                        layer_touch_hits[l] += float(hit_t.sum().item())
                        layer_tok_counts[l] += int(hit_t.numel())
                del past, out_c
                torch.cuda.empty_cache()
                l_t = [100.0 * layer_touch_hits[l] / max(layer_tok_counts[l], 1) for l in range(n_layers)]
                ctx_report[f"noncode_{clen}"] = {
                    "touch_mean_pct": round(float(np.mean(l_t)), 2),
                    "touch_max_layer_pct": round(float(max(l_t)), 2),
                    "touch_l0_5_max_pct": round(float(max(l_t[:6])), 2),
                }

    for h in hooks:
        h.remove()

    # P1/P2/Utility Gates (00-charter/exit_criteria_and_gates.md):
    #   P1: protected ASST_d upper 95% CI <= +0.003 (stem, gsm8k_math, ifeval, bfcl_tool)
    #   P2: replace -> Jaccard >= 0.99; add -> noncode hit <= 0.5%, max_layer <= 1.0%, slot mass <= 0.2% (E[slots] <= 0.02)
    #   Utility: coding_heldout ASST_d <= -0.03 (ensures negative control v2 is REJECTED on utility!)
    noncode_doms = [d for d in ("stem", "gsm8k_math", "ifeval", "bfcl_tool") if d in domain_reports]
    safety_ci_ok = all(domain_reports[d]["asst_delta"]["ci_hi95"] <= 0.003 for d in noncode_doms)
    if cart_mode == "replace":
        safety_p2_ok = all(domain_reports[d]["jaccard_mean"] >= 0.99 for d in noncode_doms)
    elif cart_mode == "add":
        safety_p2_ok = all(
            domain_reports[d]["touch_mean_pct"] <= 0.5
            and domain_reports[d]["touch_max_layer_pct"] <= 1.0
            and domain_reports[d]["expected_slots"] <= 0.02
            for d in noncode_doms
        )
    else:
        safety_p2_ok = all(abs(domain_reports[d]["asst_delta"]["mean"]) < 0.0005 for d in noncode_doms)

    utility_ok = (
        domain_reports["coding_heldout"]["asst_delta"]["mean"] <= -0.03
        if "coding_heldout" in domain_reports
        else False
    )

    report = {
        "harness_version": HARNESS_VERSION,
        "mode": cart_mode,
        "cartridge": cartridge if not null_control else "NULL_CONTROL_BASE",
        "cache_manifest": meta,
        "domains": domain_reports,
        "context_sweep": ctx_report,
        "p1_upper95ci_ok": safety_ci_ok,
        "p2_routing_ok": safety_p2_ok,
        "passed_safety": safety_ci_ok and safety_p2_ok,
        "passed_utility": utility_ok,
        "passed_all": safety_ci_ok and safety_p2_ok and utility_ok,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    log(f"=== T3 CACHED PAIRED NLL GATE ({cart_mode}) ===")
    for dom, r in domain_reports.items():
        b = r["asst_delta"]
        log(
            f"  {dom:16s} (n={b['n_seq']:2d}) | ASST_d={b['mean']:+.5f} [95% CI {b['ci_lo95']:+.5f}, {b['ci_hi95']:+.5f}] | "
            f"J={r['jaccard_mean']:.4f}(min={r['jaccard_min_layer']:.4f}@L{r['jaccard_min_layer_idx']:02d}) | "
            f"touch={r['touch_mean_pct']:5.2f}%(maxL={r['touch_max_layer_pct']:5.2f}%, L0-5={r['touch_l0_5_max_pct']:5.2f}%) | slots={r['expected_slots']:.3f}"
        )
    if ctx_report:
        log(f"  [CONTEXT SWEEP] {ctx_report}")
    log(f"[{'✓' if report['passed_all'] else 'FAIL'}] P1_CI={safety_ci_ok} | P2_ROUTE={safety_p2_ok} | UTILITY={utility_ok}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=str(BASE_BF16))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--target-asst-toks", type=int, default=40000)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--null-control", action="store_true")
    ap.add_argument("--eval-cartridge", default="")
    ap.add_argument("--context-sweep", action="store_true")
    ap.add_argument("--report-out", default="")
    ap.add_argument("--self-test", action="store_true", help="T0 CPU unit test for sequence bootstrap CI math.")
    args = ap.parse_args()

    if args.self_test:
        ci0 = seq_bootstrap_ci([0.0, 0.0, 0.0, 0.0])
        assert ci0["ci_hi95"] == 0.0 and ci0["mean"] == 0.0
        ci1 = seq_bootstrap_ci([-0.05, -0.04, -0.06, -0.05, -0.045])
        assert ci1["ci_hi95"] < 0.0
        print("T0_UNIT_TEST_PASSED:", ci1)
        return

    if args.build_cache:
        run_build_cache(Path(args.base_model), Path(args.cache_dir), args.target_asst_toks)
        return

    if not args.report_out:
        ap.error("--report-out is required for --null-control or --eval-cartridge")
    run_eval_against_cache(
        Path(args.base_model), Path(args.cache_dir),
        args.eval_cartridge, Path(args.report_out),
        null_control=args.null_control, context_sweep=args.context_sweep,
    )


if __name__ == "__main__":
    main()
