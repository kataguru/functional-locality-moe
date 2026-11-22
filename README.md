# Functional Locality in Sparse Mixture-of-Experts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24445840.svg)](https://doi.org/10.5281/zenodo.24445840)

**Empirical Laws, Self-Falsified Predictors, Fail-Closed Certificates, and Open Multi-Architecture Reproduction (`v1.0.1`)**

**Author:** Jari Katajisto (Kataguru Research) · **Release:** `v1.0.1` (2026-11-22) · **DOI:** [10.5281/zenodo.24445840](https://doi.org/10.5281/zenodo.24445840) · **Companion Systems Release:** [kataguru/moe-longcot-nondeterminism](https://github.com/kataguru/moe-longcot-nondeterminism) (DOI: [10.5281/zenodo.24445305](https://doi.org/10.5281/zenodo.24445305))

> **Release Scope (`v1.0.0`, Open Confirmatory Gates Explicitly Marked):** Contains **100% open code, public-model reproduction scripts, `0.68 s` `sm_120` Marlin MoE verifier (confirming upstream `vllm#52525`), public OLMoE routing atlas, machine-enforced JSON Schema Draft 2020-12 (`v1.4`) fail-closed certificates (7 mutation tests), negative results, and aggregated tables**. Proprietary weights, cartridges, expert-ID manifests, data recipes, and private atlases are **strictly excluded** ([NOTICE-weights.md](NOTICE-weights.md)). Open confirmatory gates (`n_bad = 2` for L2, `L7` candidate across 3 domains, `R4 OPEN 2/3`, no behavioral benchmark surviving 12-test FWER/FDR, `LCB-Medium` post-hoc primary pending fresh holdout `EXP-0017`) are explicitly disclosed in the abstract and scoreboard.

## Four Headline Results

1. **Necessary ≠ Sufficient + Candidate Law 7 (*Disjoint Parameter Support ≠ Disjoint Functional Support*):** Disjoint expert IDs (`M[i][j] = 0%`), early-layer freeze (`L00–L05`), `FP32` master weights (`C-006`), and functional anchoring (`L_anchor`) are **necessary** but **not sufficient** (`2/4` cartridges failed under identical conditions). Across all **`6/6` ordered pairs** of `Union-3` (`L06–L39`), cross-cartridge dominance $\max_{j \neq i} \rho(j \to i)$ separates composition failure (**`Testing-QA`: `1.8723× > 1.0` → `61.46%` FAIL**) from success (**`Coder`: `0.1611×` → `98.11%` PASS**; **`Cyber`: `0.6014×` → `110.72%` PASS**). *Candidate condition across `3` domains (`6/6` ordered pairs); hidden-state Grassmann overlap was falsified (`0.2015 < 0.2292`).*
2. **Self-Falsification of Unweighted Law 2, Binary Indicator Ablation & Exact Permutations (Blocker C + C1/C2):** Our own T1 predictor $\sum p_{\text{touch}}\cdot\text{relRMS}^2$ (`T1 R² = 0.982`) **failed out-of-sample on real 32B models** (`LOCO-11 R² = -0.1003`, `FN = 4/5`), and a binary early-layer flag **`1[L00–L05 open]` ALSO FAILED (`R² = -0.1231`, `TP = 0/5`)**. Continuous early-layer locality exposure **`L05_continuous_only`** achieves **`LOCO-11 R² = +0.5787`, `MAE = 0.00150`, `TP = 5/5`, `FN = 0/5`** (exact finite-population permutations: `5/990 = 0.00505` continuous outcome vectors, `1/110 = 0.00909` ordered bad-pair, `1/55 = 0.01818` unordered binary $\binom{11}{2}$; unconstrained OLS `w_L639 = -0.00792` clamped / `-0.03891` raw → NNLS `w_L639 = 0` is a constraint clamp; safe-pair bias `+0.00094` vs `0` / `+0.00336` vs raw; scope: `n_bad = 2`, single failure mode).
3. **Companion Systems Finding — `sm_120` Reproduction of `vllm#52525` & Long-CoT Behavioral Impact (`0.68 s` Verifier):** We independently reproduce the known `vLLM` Marlin MoE token-order non-determinism (upstream **`vllm#52525`**; fix PRs `#52532`, `#48032`; `VLLM_MARLIN_USE_ATOMIC_ADD=False`) on **`sm_120` / RTX 5090** with public `AWQ W4A16` and private `NVFP4`-Marlin MoE (`GPTQ` known upstream; `MXFP4` untested here). While upstream GSM8K/IFEval stayed within noise, long-CoT GPQA flipped `44/198` (`22.2%`) answers under identical weights (`w=16`), `15/30` (`50%`) at `TP=1, batch=1, eager`, and `7/20` (`35%`, `0/20` exact) on public `Qwen3-Coder-30B-A3B-Instruct-AWQ`. On the **identical AWQ checkpoint**, both **`VLLM_BATCH_INVARIANT=1`** (`TritonWNA16Experts`) and open **PR `#52532`** (`MarlinExperts` retained) restore **`20/20` (`100%`) bit-exact GPQA completions** (`0/20` flips).
4. **Open Three-Architecture Verification (`./reproduce.sh` — ~16 min on 1× RTX 5090 BF16):**

| Structural Law | `OLMoE-1B-7B` (`64E, top-8, 16L`) | `Qwen3.5-35B-A3B-Base` (`256E, top-8, 40L`) | Private 32B Validation (`232E, top-10, 40L`) |
|---|---|---|---|
| **L6: Broken Raw Top-k Jaccard** | `eps=0.01` (`+0.0023`) → **L15 `J = 0.9792`** | `eps=0.005` (`+0.00012`) → **L35 `J = 0.9050`** | `0.08%`-touch no-op (`+0.00015`) → **L21 `J = 0.9301`** |
| **L1: Evacuation (`rank1`, `α = 0.00`, zero clone output)** | `code +0.03374`, **`stem +0.03348`** | `code +0.00533`, `stem +0.00070` | `code +0.00379`, **`noncode +0.00871`** (vs `rank10`: `-0.00169`) |
| **L1: Clone Double-Count (`rank1`, `α = 1.00`)** | `code +0.05656`, `stem +0.04098` | `code +0.02588`, `stem +0.00931` | `code +0.01872`, **`noncode +0.03347`** (`rank10`: `+0.00521`) |
| **L1: FP64 Neutral `u` (`α = 1.00`)** | `code +0.00685` (`8.3×`), **`stem +0.00375` (`10.9×`)\*** | **`code +0.000139` (`186×`)**, `stem +0.00215` (`4.3×`) | **`noncode +0.00086` (`39×`)** |
| **L1: Post-Cast `u` (`BF16` vs `FP32`)** | `BF16(-2.5) bias_err = 0.00281` (`93.7%` budget → **FP32 gate rows `1.3 MB`**) | `FP64 max_std = 0.00084` | `40/40` solved |
| **C-006: `BF16` vs `FP32+8bit`** | `bf16` stalled (`1.07%`) vs `fp32_8bit` (`1.80%`) | *(inference-only)* | 743-step T1: `fp32_8bit == fp32` (6 dec.), `bf16` rel `val_mse −3.56%` |

\* Note: On `OLMoE-1B-7B`, neutral `u` cuts `stem` shock `10.9×` (`+0.04098 → +0.00375`), but `+0.00375` still exceeds `+0.003`. On pooled stable-item behavioral benchmarks (12 comparisons), no test survives family-wise correction (`p_min = 0.03125 > 0.00417`); `LCB-Medium (Union-3)` is declared post-hoc primary pending fresh-split confirmatory replication.

## Quick Verification & Open Reproduction
```bash
# 1. Fast CPU self-check (JSON Schema Draft 2020-12 + 7 fail-closed malformed-cert tests + McNemar/CP unit tests + py_compile)
./reproduce.sh --verify

# 2. 0.68s Standalone sm_120 Marlin MoE verifier (sorted_token_ids + GEMM1 + full layer output)
python3 public_repro/reproduce_marlin_moe_nondeterminism.py

# 3. Single-GPU public OLMoE reproduction (~16 min on 1x NVIDIA RTX 5090 32GB BF16, determinism-pinned)
./reproduce.sh

# 4. Optional multi-GPU Qwen3.5-35B-A3B-Base L1+L6 bonus row
MODEL_QWEN35=Qwen/Qwen3.5-35B-A3B-Base ./reproduce.sh
```

## Repository Layout & Licensing
- `reproduce.sh` — Single-forward BF16 NLL reproduction (`runtime: hf_transformers_single_forward`, explicit `pins_effective` vs `pins_inert`, `flip_rate: null`) + `--verify` self-test.
- `certificates/certificate.schema.json` + `validate_certificate.py` — Valid JSON Schema Draft 2020-12 (`v1.4`: mechanically enforces `carries_neutral_u: true` ⇒ `engine.router_dtype == "float32"` and symmetric `measured_on`/`flip_rate` invariants; 7 fail-closed mutation tests; `P2` demoted to `diagnostics`).
- `report/TECHNICAL_REPORT.md` — Complete Methods report (`v1.0.0`, open confirmatory gates explicitly marked; plain-Finnish summary: [report/EXECUTIVE_SUMMARY_FI.md](report/EXECUTIVE_SUMMARY_FI.md)).
- `public_repro/`, `harness/`, `atlas_public/olmoe_1b_7b_atlas.parquet`, `results/`, `ledger/`.
- **Licenses:** Code Apache-2.0 ([LICENSE-CODE](LICENSE-CODE)); Report CC-BY-4.0 ([LICENSE-REPORT](LICENSE-REPORT)); Proprietary weights/manifests strictly excluded ([NOTICE-weights.md](NOTICE-weights.md)).
