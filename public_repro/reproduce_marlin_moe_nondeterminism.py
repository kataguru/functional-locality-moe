#!/usr/bin/env python3
"""
Standalone sm_120 Verification Micro-Reproducer for vLLM Marlin MoE Non-Determinism
===================================================================================
Independently verifies on consumer Blackwell (RTX 5090, sm_120) the kernel mechanism
documented in vllm-project/vllm#52525 (PRs #52532, #48032):
  1. `moe_align_block_size` uses CUDA atomicAdd (for E > 64 || numel >= 1024) ->
     non-deterministic `sorted_token_ids` permutation across runs.
  2. `ops.moe_wna16_marlin_gemm` (GEMM1) and full `_fused_marlin_moe` (GEMM1 -> SwiGLU
     -> GEMM2 -> top-k weighted sum = full MoE layer output) are NOT invariant to
     within-expert M-block token placement under live `moe_align_block_size`, even with
     VLLM_MARLIN_USE_ATOMIC_ADD=False.
  3. Fixing `sorted_token_ids` deterministically eliminates 100% of both GEMM1 and
     full-layer-output variance (0/repeats, max_abs = 0.0).
  4. `Triton fused_experts` (BF16) is bit-exact even under live `moe_align_block_size`.

License: Apache-2.0 (see LICENSE-CODE)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import vllm
import vllm.envs as vllm_envs
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import marlin_moe_permute_scales
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import _fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="./results/marlin_moe_kernel_microrepro.json")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    # Exact Qwen3-30B-A3B MoE layer dimensions (E=128, K=2048, N=768, top_k=8)
    E, M, K, N, top_k = 128, 256, 2048, 768, 8
    w1_q = ops.gptq_marlin_moe_repack(
        torch.randint(0, 2**31 - 1, (E, K // 8, 2 * N), dtype=torch.int32, device=dev),
        torch.empty((E, 0), dtype=torch.int32, device=dev), K, 2 * N, 4,
    )
    w2_q = ops.gptq_marlin_moe_repack(
        torch.randint(0, 2**31 - 1, (E, N // 8, K), dtype=torch.int32, device=dev),
        torch.empty((E, 0), dtype=torch.int32, device=dev), N, K, 4,
    )
    w1_s = marlin_moe_permute_scales(torch.randn((E, K // 128, 2 * N), dtype=torch.bfloat16, device=dev) * 0.01, K, 2 * N, 128)
    w2_s = marlin_moe_permute_scales(torch.randn((E, N // 128, K), dtype=torch.bfloat16, device=dev) * 0.01, N, K, 128)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
    logits = torch.randn((M, E), device=dev) + torch.linspace(0, 3.0, E, device=dev)
    tw, tid = torch.topk(logits, top_k, dim=-1)
    tw = torch.softmax(tw, dim=-1).float()
    tid = tid.int()
    ws = marlin_make_workspace_new(dev, 4)

    def run_gemm1(s_ids, e_ids, n_tok, bs_m: int):
        return ops.moe_wna16_marlin_gemm(
            input=x, output=None, b_qweight=w1_q, b_bias=None, b_scales=w1_s,
            a_scales=None, global_scale=None, b_qzeros=None, g_idx=None, perm=None,
            workspace=ws, sorted_token_ids=s_ids, expert_ids=e_ids,
            num_tokens_past_padded=n_tok, topk_weights=tw,
            moe_block_size=bs_m, top_k=top_k, mul_topk_weights=False,
            b_q_type=scalar_types.uint4b8, size_m=M, size_n=2 * N, size_k=K,
            is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )

    # 1. moe_align_block_size alone (bs_m=16 reference)
    s0, e0, n0 = moe_align_block_size(tid, 16, E)
    align_diffs = sum(
        1 for _ in range(args.repeats)
        if (moe_align_block_size(tid, 16, E)[0] != s0).any()
    )

    def bf16_ulp_at_scale(val: float) -> float:
        # Exact BF16 ULP (7 mantissa bits) at magnitude `val`: 2^(floor(log2(|val|)) - 7)
        import math
        return 2.0 ** (math.floor(math.log2(abs(val))) - 7) if val != 0.0 else 0.0

    # 2. Marlin MoE across block_size_m in (8, 16, 32, 64):
    #    Separately record sorted_token_ids diffs, GEMM1 diffs, FULL layer-output diffs,
    #    and exact reference row magnitude scales + row-scale BF16 ULP.
    marlin_res = {}
    for bs_m in (8, 16, 32, 64):
        sf, ef, nf = moe_align_block_size(tid, bs_m, E)
        ref_g1 = run_gemm1(sf, ef, nf, bs_m)
        ref_layer = _fused_marlin_moe(
            x, w1_q, w2_q, None, None, w1_s, w2_s, tw, top_k, scalar_types.uint4b8,
            False, None, bs_m, sf, ef, nf, topk_ids=tid, workspace=ws,
        ).view(-1, top_k, K).sum(dim=1)
        torch.cuda.synchronize()
        ref_g1_abs_max = float(ref_g1.float().abs().max().item())
        ref_g1_abs_med = float(ref_g1.float().abs().median().item())
        ref_layer_abs_max = float(ref_layer.float().abs().max().item())
        ref_layer_abs_med = float(ref_layer.float().abs().median().item())
        g1_row_ulp = bf16_ulp_at_scale(ref_g1_abs_max)
        layer_row_ulp = bf16_ulp_at_scale(ref_layer_abs_max)

        fixed_g1_diffs, fixed_g1_max = 0, 0.0
        fixed_layer_diffs, fixed_layer_max = 0, 0.0
        for _ in range(args.repeats):
            og1 = run_gemm1(sf, ef, nf, bs_m)
            ol = _fused_marlin_moe(
                x, w1_q, w2_q, None, None, w1_s, w2_s, tw, top_k, scalar_types.uint4b8,
                False, None, bs_m, sf, ef, nf, topk_ids=tid, workspace=ws,
            ).view(-1, top_k, K).sum(dim=1)
            torch.cuda.synchronize()
            if (og1 != ref_g1).any():
                fixed_g1_diffs += 1
                fixed_g1_max = max(fixed_g1_max, float((og1.float() - ref_g1.float()).abs().max().item()))
            if (ol != ref_layer).any():
                fixed_layer_diffs += 1
                fixed_layer_max = max(fixed_layer_max, float((ol.float() - ref_layer.float()).abs().max().item()))

        live_sid_diffs = 0
        live_g1_diffs, live_g1_max = 0, 0.0
        live_layer_diffs, live_layer_max = 0, 0.0
        for _ in range(args.repeats):
            si, ei, ni = moe_align_block_size(tid, bs_m, E)
            if (si != sf).any():
                live_sid_diffs += 1
            og1 = run_gemm1(si, ei, ni, bs_m)
            ol = _fused_marlin_moe(
                x, w1_q, w2_q, None, None, w1_s, w2_s, tw, top_k, scalar_types.uint4b8,
                False, None, bs_m, si, ei, ni, topk_ids=tid, workspace=ws,
            ).view(-1, top_k, K).sum(dim=1)
            torch.cuda.synchronize()
            if (og1 != ref_g1).any():
                live_g1_diffs += 1
                live_g1_max = max(live_g1_max, float((og1.float() - ref_g1.float()).abs().max().item()))
            if (ol != ref_layer).any():
                live_layer_diffs += 1
                live_layer_max = max(live_layer_max, float((ol.float() - ref_layer.float()).abs().max().item()))

        marlin_res[f"bs_m_{bs_m}"] = {
            "sorted_token_ids_diff_runs": live_sid_diffs,
            "ref_gemm1_abs_max": round(ref_g1_abs_max, 6),
            "ref_gemm1_abs_median": round(ref_g1_abs_med, 6),
            "ref_gemm1_bf16_ulp_at_max": g1_row_ulp,
            "ref_layer_abs_max": round(ref_layer_abs_max, 6),
            "ref_layer_abs_median": round(ref_layer_abs_med, 6),
            "ref_layer_bf16_ulp_at_max": layer_row_ulp,
            "fixed_align_gemm1_diff_runs": fixed_g1_diffs,
            "fixed_align_gemm1_max_abs": round(fixed_g1_max, 6),
            "fixed_align_diff_runs": fixed_layer_diffs,
            "fixed_align_max_abs": round(fixed_layer_max, 6),
            "live_align_gemm1_diff_runs": live_g1_diffs,
            "live_align_gemm1_max_abs": round(live_g1_max, 6),
            "live_align_gemm1_ulps_at_row_max": round(live_g1_max / g1_row_ulp, 3),
            "live_align_diff_runs": live_layer_diffs,
            "live_align_max_abs": round(live_layer_max, 6),
            "live_align_layer_ulps_at_row_max": round(live_layer_max / layer_row_ulp, 3),
            "repeats": args.repeats,
        }
        print(
            f"  [Marlin MoE bs_m={bs_m:2d}] sorted_ids={live_sid_diffs:2d}/{args.repeats} | "
            f"GEMM1 LIVE={live_g1_diffs:2d}/{args.repeats} (max={live_g1_max:.3e}, ref_max={ref_g1_abs_max:.2f}, row_ulp={g1_row_ulp:.4f}) FIXED={fixed_g1_diffs:2d} | "
            f"LAYER_OUT LIVE={live_layer_diffs:2d}/{args.repeats} (max={live_layer_max:.3e}, ref_max={ref_layer_abs_max:.2f}, row_ulp={layer_row_ulp:.4f}) FIXED={fixed_layer_diffs:2d}",
            flush=True,
        )

    # 3. Triton BF16 fused_experts under LIVE moe_align_block_size
    w1_bf = torch.randn((E, 2 * N, K), dtype=torch.bfloat16, device=dev) * 0.02
    w2_bf = torch.randn((E, K, N), dtype=torch.bfloat16, device=dev) * 0.02
    ref_t = fused_experts(x, w1_bf, w2_bf, tw, tid)
    torch.cuda.synchronize()
    triton_diffs, triton_max = 0, 0.0
    for _ in range(args.repeats):
        o = fused_experts(x, w1_bf, w2_bf, tw, tid)
        torch.cuda.synchronize()
        if (o != ref_t).any():
            triton_diffs += 1
            triton_max = max(triton_max, float((o.float() - ref_t.float()).abs().max().item()))
    print(f"  [Triton BF16 LIVE align] layer_out_diff={triton_diffs:2d}/{args.repeats} (max={triton_max:.3e})", flush=True)

    cc = torch.cuda.get_device_capability(0)
    payload = {
        "environment": {
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{cc[0]}{cc[1]}",
            "VLLM_MARLIN_USE_ATOMIC_ADD": bool(vllm_envs.VLLM_MARLIN_USE_ATOMIC_ADD),
            "VLLM_BATCH_INVARIANT": bool(getattr(vllm_envs, "VLLM_BATCH_INVARIANT", False)),
        },
        "shape": {"E": E, "M": M, "K": K, "N": N, "top_k": top_k},
        "moe_align_block_size_diff_runs": align_diffs,
        "repeats": args.repeats,
        "marlin_moe": marlin_res,
        "triton_bf16_live_align": {"diff_runs": triton_diffs, "max_abs": round(triton_max, 6), "repeats": args.repeats},
        "elapsed_s": round(time.time() - t0, 2),
    }
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2))
    print(f"[✓] Micro-repro saved -> {out_p} ({payload['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
