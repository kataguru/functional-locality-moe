# Functional Locality of Post-Hoc Expert Modification in Sparse Mixture-of-Experts

**Kataguru Modular MoE Research Program — Methods Technical Report (`v1.0.0`, Open Confirmatory Gates Explicitly Marked)**
- **Author:** Jari Katajisto (Kataguru Research)
- **Date:** 2026-11-22
- **Repository:** `https://github.com/kataguru/functional-locality-moe`
- **Companion Systems Publication:** [kataguru/moe-longcot-nondeterminism](https://github.com/kataguru/moe-longcot-nondeterminism) (DOI: [10.5281/zenodo.24445305](https://doi.org/10.5281/zenodo.24445305); upstream comment [vllm#52525](https://github.com/vllm-project/vllm/issues/52525#issuecomment-7082387518)) — 100% public `0.68 s` `sm_120` kernel verification + public `AWQ` 3-way A/B/C (`VLLM_BATCH_INVARIANT=1` & PR `#52532` both `20/20` bit-exact) + public `OLMoE`/`Qwen3.5-Base` Law 6, with clearly marked private-substrate `2×2`/`Ctrl D` corroboration. This report (`v1.0.0`) presents the Methods findings with full statistical disclosures and symmetric mechanical certificate enforcement (`schema v1.4`, 7 fail-closed mutation tests).
- **Open Public Reproduction Substrates (Apache 2.0):** `allenai/OLMoE-1B-7B-0125-Instruct` (`16L × 64E`, `top_k=8`, ~16 min on 1× RTX 5090 BF16), `Qwen/Qwen3.5-35B-A3B-Base` (`40L × 256E`, `top_k=8`), and `dark-side-of-the-code/Qwen3-Coder-30B-A3B-Instruct-AWQ` (`48L × 128E`, `top_k=8`, Marlin WNA16 control)
- **Private 32B Validation Substrate:** `32B-A4B-MoE` (`40L × 232E`, `top_k=10`, `BF16 index sha256: 991ce1997dc22af34b3a5d027e122dbed413d1e1a92eacf8ccb14d54922dcec1`; `vLLM 0.29.0`, `PyTorch 2.12.1+cu130`, `CUDA 13.0`, `NVFP4`)
- **Archive:** [CONFOUNDS.md](../ledger/CONFOUNDS.md) (`C-001..C-008`), [GLOSSARY.md](../ledger/GLOSSARY.md)

---

## Abstract & Four Core Findings

1. **Necessary ≠ Sufficient Conditions for Modular MoE Locality — and Candidate Law 7 (`F-050`):** Disjoint parameter support (`M[i][j] = 0%`), freezing early hub layers (`L00–L05`), strict `FP32` master weights (`autocast(bf16)` + `AdamW8bit`), and functional anchoring (`L_anchor`) are **necessary conditions** for functionally local post-hoc expert editing in a frozen Sparse MoE. **Sufficiency is NOT established:** `2 / 4` cartridges (`Testing-QA-4` `61.46%` union retention; `Estonian-4` `+0.0086` overfit) failed under the exact same four conditions. Measuring the **complete `3 × 3` (`6/6` ordered pairs) token-level cross-cartridge router mass matrix** (`L06–L39`) exposes **Candidate Law 7 (*Disjoint Parameter Support ≠ Disjoint Functional Support*)**: using $\max_{j \neq i} \rho(j \to i)$ where $\rho(j \to i) = \text{mass}(S_j \text{ on } D_i)/\text{mass}(S_i \text{ on } D_i)$, `Coder-12` ($\max_j \rho = 0.1611\times \to 98.11\%$ PASS) and `Cyber-4` ($\max_j \rho = 0.6014\times \to 110.72\%$ PASS) satisfy $\max_j \rho < 1.0$, whereas `Testing-QA-4` is dominated by `Coder-12` (**$\max_j \rho = 1.8723\times > 1.0 \to 61.46\%$ FAIL P6**). *L7 ($\max_j \rho < 1.0$) is a candidate condition supported by `3` domains (`6/6` ordered pairs), not yet a validated universal law; the hidden-state Grassmann subspace alternative was empirically falsified (`0.2015 < 0.2292`).*
2. **Self-Falsification of Unweighted Law 2, Binary Indicator Ablation, Unconstrained OLS & Exact Permutation Floors (`F-045`, Blocker C + C1/C2 Closed):** Our own synthetic-T1 Locality Product $\sum_l p_{\text{touch},l}\cdot\text{relRMS}_l^2$ (`T1 R² = 0.982–0.993`) **failed out-of-sample on real 32B cartridges** (`LOCO-11`, `44` pairs: **`R² = -0.1003`**, `FN = 4/5`; naive `||ΔW||²`: `R² = -0.1347`, `FN = 5/5`). Crucially, a pure binary early-layer flag **`1[L00–L05 open]` ALSO FAILED (`LOCO-11 R² = -0.1231`, `TP = 0/5`, `FN = 5/5`, `perm_p = 0.3978`)** because 7/11 cartridges had `L00–L05` open yet 5/7 stayed safe. Continuous early-layer locality exposure **`L05_continuous_only` ($\sum_{l=0}^5 p_{\text{touch},l}\cdot\text{relRMS}_l^2$)** achieves **`LOCO-11 R² = +0.5787`, `MAE = 0.00150 nats`, `TP = 5/5`, `FN = 0/5`, `TN = 36/39`**, beating the binary flag by `+0.7018 R²`. Exact finite-population permutation enumerations give **`p = 5/990 = 0.00505`** (continuous 4-domain outcome vectors across 11 cartridges, floor `1/990`), **`p = 1/110 = 0.00909`** (ordered bad-pair, floor `1/110`), and **`p = 1/55 = 0.01818`** (unordered binary bad-pair $\binom{11}{2}$, exact minimum possible). Unconstrained OLS shows `w_L639` is negative (`-0.00792`, `95% CI [-0.01653, +0.00069]` clamped; `-0.03891`, `95% CI [-0.04917, -0.02865]` raw), so NNLS `w_L639 = 0.0` is a non-negativity constraint clamp rather than an unconstrained zero. Explicit caveats: all `5` positive drift pairs come from **`n_bad = 2` early-layer `Cyber-4` cartridges** (single failure mode), and safe-cartridge overprediction bias (`+0.00094` vs clamped `0`, `+0.00336` vs raw) is reported explicitly.
   - *Note on External Falsifications:* Across Audits #1–#4, three independently proposed alternative hypotheses were empirically falsified on hardware: (i) hidden-state Grassmann subspace overlap for L7 (`0.2015 < 0.2292`), (ii) binary early-layer indicator `1[L00–L05 open]` for L2 (`R² = -0.1231`, `TP = 0/5`), and (iii) continuous batching / `TP=2` AllReduce as the sole source of greedy flips (`TP=1 eager` still flipped `15/30`).
3. **Companion Systems Finding — `sm_120` Reproduction of `vllm#52525` & Long-CoT Behavioral Amplification (`F-048 / F-049 / F-051`):** We independently reproduce the known `vLLM` Marlin MoE token-order non-determinism (upstream **`vllm#52525`**; fix PRs **`#52532`**, **`#48032`**; previously reproduced on `sm_80 / sm_90 / sm_121`) on **`sm_120` / RTX 5090** (`vLLM 0.29.0`, `PyTorch 2.12.1+cu130`, `VLLM_MARLIN_USE_ATOMIC_ADD=False`) with public `AWQ W4A16` and private `NVFP4`-Marlin MoE serving (`GPTQ int4` known upstream; `MXFP4`-Marlin not tested here). **Our contribution is not the kernel root cause, but its task-dependent behavioral amplification:** while upstream short-output benchmarks (`GSM8K`, `IFEval`) remained within noise, long-CoT `GPQA-Diamond` flipped `44/198` (`22.2%`) answers under identical weights (`w=16`), `15/30` (`50%`) at `TP=1, batch=1, --enforce-eager` (maximal-instability subset), and `7/20` (`35%`, `0/20` exact) on public `Qwen3-Coder-30B-A3B-Instruct-AWQ`. Enabling **`VLLM_BATCH_INVARIANT=1`** on the **identical public AWQ checkpoint** auto-switches to **`TritonWNA16Experts`** and restores **`20/20` (`100%`) bit-exact GPQA completions** (`0/20` flips). We connect this amplification to **Law 6** (`#k/#k+1` router boundary sensitivity, 3/3 architectures) and derive a paired evaluation protocol.
4. **Pooled Stable-Item Behavioral Signal & Multiple-Comparison Disclosure (`F-041` Revised, Blocker B Closed):** Restricting paired McNemar to items where `Base Seed-1 == Base Seed-2` removes single-seed comparator selection bias — and **weakens the headline `LCB-Medium` p-value from `0.00781` to `0.03125`** (`b=6, c=0`, `45/52` stable). Across the full `6 benchmarks × 2 candidates = 12` comparisons, **no benchmark survives family-wise multiple-testing correction** (`p_min = 0.03125 > Bonferroni α/12 = 0.00417`; fails Bonferroni, Holm, and BH-FDR `q=0.05`). Accordingly, `LCB-Medium (Union-3, stable-item)` is designated as a **post-hoc primary endpoint** (all other 11 tests exploratory); **confirmatory replication on a fresh problem split is required** before claiming a confirmatory coding gain.

---

## 1. Three-Architecture Open Cross-Validation Table

*Parameter definition (`α`):* In Law 1 (`EXP-L1`), **`α` (`alpha`) is the cloned expert's `down_proj` output scale (`W_down_clone = α · W_down_parent`)**. Setting `α = 0.00` zeroes the clone's output to isolate **pure `e_k` slot eviction**; setting `α = 1.00` measures **eviction + full clone double-counting**.

| Structural Law / Test | **`allenai/OLMoE-1B-7B`** (Public Apache-2.0, `64E, top-8, 16L`) | **`Qwen/Qwen3.5-35B-A3B-Base`** (Public Apache-2.0, `256E, top-8, 40L`) | **`Private 32B-A4B`** (`232E, top-10, 40L`) | Status |
|---|---|---|---|---|
| **L6: Boundary Sensitivity** (tiny L0 perturb → deep `J` drop) | `eps=0.01`, `stem +0.0023` → **L15 `J = 0.9792`** (`< 0.99`) | `eps=0.005`, `code +0.00012` → **L35 `J = 0.9050`** (`mean 0.9397`) | `0.08%`-touch no-op (`+0.00015`) → **L21 `J = 0.9301`** (`mean 0.9492`) | **3/3 CONFIRMED** |
| **L1: Pure Evacuation Intercept (`rank1`, `α = 0.00`)**\* | `code +0.03374`, **`stem +0.03348`** | `code +0.00533`, `stem +0.00070` | `code +0.00379`, **`noncode +0.00871`** (vs `rank10 @ α=0`: `-0.00169`)\* | **3/3 CONFIRMED**\* |
| **L1: Clone Double-Counting (`rank1`, `α = 1.00`)** | `code +0.05656`, `stem +0.04098` | `code +0.02588`, `stem +0.00931` | `code +0.01872`, **`noncode +0.03347`** (`rank10`: `+0.00521`, `6.4×` lower) | **3/3 CONFIRMED** |
| **L1: FP64 Neutral `u` Cure (`rank1`, `α = 1.00`, zero bias)** | `code +0.00685` (`8.3×` cut), **`stem +0.00375` (`10.9×` cut)**† | **`code +0.000139` (`186×` cut)**, `stem +0.00215` (`4.3×` cut) | `code +0.00820`, **`noncode +0.00086` (`39×` cut vs `+0.03347`)** | **3/3 CONFIRMED**† |
| **L1: Post-Cast `u` Precision (`FP64 → FP32 → BF16`)**‡ | `max_std`: `0.00336` → `0.00336` → `0.00369`; **`BF16(-2.5)` `bias_err = 0.00281` (`93.7%` of `+0.003`!)** | `FP64 max_std = 0.00084` | `FP64` solved `40/40` layers | **FP32 GATE ROWS MANDATED**‡ |
| **C-006: `BF16` Trap vs `FP32+AdamW8bit`** | `bf16` stalled (`rel_dW 1.07%`, `code -0.740`) vs `fp32_8bit` (`1.80%`, `-0.785`) | *(inference-only)* | 743-step T1: `fp32_8bit == fp32` (6 dec., `cos >= 0.9998`), `bf16` **relative `val_mse` degraded `-3.56%`** (`cos 0.15–0.35`)§ | **2/2 CONFIRMED** |
| **L5: Early-Layer Hub (`L00–L05`, matched `34/40`)** | *(113-row MBPP: 0 positive drift — non-discriminative)* | *(inference-only)* | `L00–L33` FAIL (`kl 0.57`, `bfcl +0.0220`, `stem +0.0183`) vs `L06–L39` PASS (`kl 0.008`, `-0.0019`) | **32B CAUSAL (DOMAIN-SCOPED)** |
| **L4: Live Fused NVFP4 Kernel Parity** | *(N/A)* | *(N/A)* | Live `vLLM 0.29.0 TP=2 NVFP4` matched BF16 within `0.00011–0.00120 nats` (6/6 improved) | **32B KERNEL CLOSED** |

- \* **L1 Rank-Dependent Evacuation Intercept (Audit #3 Correction):** On the 32B substrate (`EXP-L1`), the pure evacuation intercept (`α = 0.00`, zero clone output) is **rank-dependent**: **`noncode +0.00871` (`code +0.00379`)** for a `rank1` parent (which evicts `e_10`), vs **`noncode -0.00169 ≈ 0` (`code +0.00151`)** for a `rank10` parent (no `e_10` eviction). This **confirms the two-term evacuation + double-counting decomposition** (with double-counting dominating at `α = 1.00`: `+0.03347` vs `+0.00871`, and `rank10 @ α = 1.00` `6.4×` smaller at `+0.00521`). *(Our earlier pre-registered `claim_1` only missed its arbitrary numeric cutoff `coding_asst_d > +0.01` at `rank1 @ α=0.00`, observing `+0.00379`; labeling that as falsifying the evacuation intercept was overly aggressive.)*
- † **Explicit OLMoE Residual Limitation:** On `OLMoE-1B-7B` (`64E, top_k=8`), neutral `u` reduces `stem` init shock `10.9×` (`+0.04098 → +0.00375`), **but residual `+0.00375` still exceeds our `+0.003` safety gate** — `u` mitigates init shock without certifying standalone zero-shot safety on denser `64E/top-8` MoEs.
- ‡ **Mandatory FP32 Precision Policy for `u`-Bearing Gate Rows (`1.31 MB`, Mechanically Enforced):** Casting a `-2.5`-shifted neutral-`u` gate row to `BF16` introduces `bias_err = 0.00281`, consuming **`93.7%` (`0.00281 / 0.003`)** of the entire `P1` safety budget before training even starts. **Therefore, `u`-bearing router gate rows MUST be kept in `FP32` at serve time.** Memory cost is `40 layers × 4 experts × 2048 dim × 4 B = 1,310,720 B (1.31 MB)`. This requirement is **mechanically enforced** in `certificate.schema.json` (`v1.4` `allOf` conditional: `carries_neutral_u: true` ⇒ `engine.router_dtype == "float32"`) and `validate_certificate.py` (semantic invariant + fail-closed negative mutation test #5).
- § **Unit Disambiguation:** `-3.56 pp` refers exclusively to `v1` `Bench 15` percentage-point drop (`NEG-001`), whereas `-3.56%` in `C-006` refers to relative validation MSE degradation (`bf16_adamw` vs base at step `743` in synthetic T1 parity).

---

## 2. Audit #1 Complete Resolution: `sm_120` Verification of `vllm#52525`, Long-CoT Impact & `VLLM_BATCH_INVARIANT=1` (`F-049 / F-051`)

*(Full standalone Systems release & `#52525` verification: [kataguru/moe-longcot-nondeterminism](https://github.com/kataguru/moe-longcot-nondeterminism), DOI [10.5281/zenodo.24445305](https://doi.org/10.5281/zenodo.24445305). Kernel root cause attributed to upstream `vllm#52525` / PRs `#52532` & `#48032`; tested here on `sm_120` with `AWQ W4A16` & `NVFP4`-Marlin MoE; `GPTQ int4` known upstream; `MXFP4`-Marlin not tested here; `VLLM_MARLIN_USE_ATOMIC_ADD=False`.)*

**Explicit Selection-Bias Disclosure:** The 30-item (`2×2` factorial) and 20-item (Controls D/E2/I/K/L/M) GPQA-Diamond subsets were drawn from the `44` items that flipped under `w = 16`. Flip rates below reflect this **maximal-instability subset**, NOT population-wide rates. Selection-bias-free verification is provided independently by the fresh-prompt prefill A/B/C test and the `0.68 s` synthetic `sm_120` kernel verification.

### 2.1 Seven-Configuration Cross-Quantization, `VLLM_BATCH_INVARIANT=1` & PR `#52532` Matrix (`TP=1, batch=1, temp=0.0, --enforce-eager, --no-enable-prefix-caching`, `vLLM 0.29.0`, `PyTorch 2.12.1+cu130`)

| Control | Model | Arch | Quant | `BATCH_INV` | Backend | Exact Match | Letter Flips | Correct Flips | First Div. Word (Med [Min–Max]) |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| **Ctrl F** | `Dense-27B-NVFP4` | Dense | `NVFP4` | `0` | `FlashInferCutlass` | **`13/13` completed pairs**\* | `0` | `0` | *none* (≤26,203 chars) |
| **Ctrl E2** | `allenai/OLMoE-1B-7B` (public) | MoE (64E) | `BF16` | `0` | `TritonExperts` | **`20/20` (`100%`)** | `0` | `0` | *none* (mean 2,732 chars) |
| **Ctrl I** | `allenai/OLMoE-1B-7B` (public) | MoE (64E) | `FP8` | `0` | `TritonExperts` | **`20/20` (`100%`)** | `0` | `0` | *none* (mean 2,580 chars) |
| **Ctrl L** | **`Qwen3-Coder-30B-A3B-AWQ`** (public) | MoE (128E) | **`AWQ W4A16`** | **`1`** | **`TritonWNA16Experts`**† | **`20/20` (`100%`)** | **`0/20` (`0%`)** | **`0/20` (`0%`)** | *none* (mean 2,932, ≤7,067 chars) |
| **Ctrl M** | **`Qwen3-Coder-30B-A3B-AWQ`** (public) | MoE (128E) | **`AWQ W4A16`** | `0` | **`MarlinExperts` + PR `#52532`**‡ | **`20/20` (`100%`)** | **`0/20` (`0%`)** | **`0/20` (`0%`)** | *none* (mean 3,059, ≤6,079 chars) |
| **Ctrl K** | **`Qwen3-Coder-30B-A3B-AWQ`** (public) | MoE (128E) | **`AWQ W4A16`** | `0` | **`MarlinExperts`** | **`0/20` (`0.0%`)** | **`7/20` (`35%`)** | **`4/20` (`20%`)** | **`34.0` `[1–138]`** (char `234.5`) |
| **Ctrl D** | `32B-A4B-MoE-NVFP4` | MoE (232E) | **`NVFP4`** | `0` | **`MarlinExperts`** | **`2/20` (`10.0%`)** | **`9/20` (`45%`)** | **`6/20` (`30%`)** | **`31.5` `[0–122]`** (char `225.0`) |

- \* **Ctrl F:** `13/13` completed pairs (stopped at 13/20 due to unrelated concurrent host RAM exhaustion; all 13 completed pairs, >190k chars, were 100% bit-exact). The earlier `2×2` factorial (`n=30`) ruled out continuous batching, `TP=2` AllReduce, and CUDAGraphs (`TP=2 CG: 2/30`, `TP=2 Eager: 0/30`, `TP=1 CG: 2/30`, `TP=1 Eager: 1/30` exact).
- † **Ctrl L (`VLLM_BATCH_INVARIANT=1` A/B Cure):** On the identical public AWQ checkpoint, `VLLM_BATCH_INVARIANT=1` auto-switches `MarlinExperts → TritonWNA16Experts`, restoring `20/20` bit-exact GPQA completions (`0/20` flips, `+10.2%` latency at `batch=1`).
- ‡ **Ctrl M (Open PR `#52532` A/C Cure):** Applying PR `#52532`'s `_canonicalize_marlin_moe_token_order` keeps `MarlinExperts` active (`VLLM_BATCH_INVARIANT=0`) and restores `20/20` bit-exact GPQA completions (`0/20` flips, `+10.8%` latency at `batch=1`; `0/20` kernel diffs across `bs_m ∈ {8,16,32,64}`, `+63.6–97.3 µs/layer` on `M=256` prefill, `0 µs` on `M=1` decode).

### 2.2 Selection-Bias-Free Fresh-Prompt Prefill A/B/C & `0.68 s` `sm_120` Kernel Verification ([reproduce_marlin_moe_nondeterminism.py](../public_repro/reproduce_marlin_moe_nondeterminism.py))

1. **Fresh Chemistry Prompt (`Qwen3-Coder-30B-A3B-Instruct-AWQ`, `--generation-config vllm, top_k=-1`, 5 identical prefills):**
   - Under default `MarlinExperts` (`VLLM_BATCH_INVARIANT=0`): **greedy argmax at Token #0 (`'I'`, logprob `-5.0e-5..-1.31e-4`) did NOT change (`0/4`)**, while all 5 top-5 logprob vectors differed (rank-2 `'Let'` showed a `1.000`-nat logprob swing `-10.00005` to `-9.000132`). Downstream greedy divergence first appeared at **char 288 (word 45)** on Run 2 and **char 341 (word 53)** on Run 3 (emitting `"5d⁵"` instead of `"3d⁵"`), consistent with multi-layer amplification across Law 6 `#k/#k+1` boundaries.
   - Under **`VLLM_BATCH_INVARIANT=1` (`TritonWNA16Experts`)**: **`5/5` top-5 Token #0 logprobs bit-identical** (`'I' = -9.6e-5`, `'Let' = -9.250096`) and **`3/3` 512-token completions bit-identical** (`1,317` chars).
   - Under **PR `#52532` (`MarlinExperts`, `VLLM_BATCH_INVARIANT=0`)**: **`5/5` top-5 Token #0 logprobs bit-identical** (`'I' = -7.5e-5`, `'Let' = -9.500075`) and **`3/3` 512-token completions bit-identical** (`1,419` chars).
2. **`sm_120` 3-Level Kernel Verification (`0.68 s`, `E=128, M=256, K=2048, N=768, top_k=8`, `VLLM_MARLIN_USE_ATOMIC_ADD=False`, `20` repeats):**
   - `sorted_token_ids` (`moe_align_block_size`): **`20/20`** non-deterministic across all `bs_m ∈ {8, 16, 32, 64}`.
   - Standalone **`GEMM1`** (`ops.moe_wna16_marlin_gemm`): **`LIVE = 20/20`** (`ref_max = 12.75`, row-max BF16 ULP `0.0625`, `max_abs = 0.03125` at `bs_m=8,16,32`, `0.015625` at `bs_m=64`) vs **`FIXED = 0/20` (`0.0`)**.
   - **Full MoE `LAYER_OUT`** (`_fused_marlin_moe().sum(dim=1)`): **`LIVE = 20/20`** (`ref_max = 11.50`, row-max BF16 ULP `0.0625`, `max_abs = 0.03125` at `bs_m=8,16`, `0.015625` at `bs_m=32,64`) vs **`FIXED = 0/20` (`0.0`)** vs `Triton BF16 LIVE = 0/20` (`0.0`).

By contrast, our single-forward BF16 PyTorch T3 metrology harness achieves **`0.00000 nats` (`CI <= +0.00001`), `J = 1.0000`** on `null_control`.

---

## 3. Candidate Law 7 (Empirical): Full `3 × 3` (`6/6` Ordered Pairs) Dominance Matrix (`F-050`)

All 3 cartridges (`Coder-12`, `Cyber-4 L6–39`, `Testing-QA-4 L6–39`) have **strictly disjoint expert IDs** (`M[i][j] = 0.00%`, `order_variance = 0.00%`, `recal = 0 s`). Measuring token-level cross-cartridge router mass across **all `6/6` ordered pairs** on unseen holdout assistant tokens (`L06–L39`, [l7_functional_overlap_and_ucast.json](../results/l7_functional_overlap_and_ucast.json)) resolves the `Union-3` retention paradox:

| Target Domain $D_i$ (Own Cartridge) | Own Fire % / Mass % | `Coder-12` Mass % ($\rho$) | `Cyber-4` Mass % ($\rho$) | `Testing-QA-4` Mass % ($\rho$) | **$\max_{j \neq i} \rho(j \to i)$** | Grassmann ($k=16$) vs `Coder` | **Observed `Union-3` Retention** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `coding` (`Coder-12`) | `83.22%` / `25.26%` | *(own `1.00×`)* | `2.24%` (`0.0887×`) | `4.07%` (`0.1611×`) | **`0.1611×`** | `1.0000` | **`98.11%` (PASS)**\* |
| `cyber` (`Cyber-4`) | `62.43%` / `13.02%` | `7.83%` (`0.6014×`) | *(own `1.00×`)* | `2.37%` (`0.1820×`) | **`0.6014×`** | `0.2292` | **`110.72%` (PASS)** |
| `testing_qa` (`Testing-QA-4`) | `54.20%` / `8.46%` | **`15.84%` (`1.8723×`)** | `4.08%` (`0.4823×`) | *(own `1.00×`)* | **`1.8723×` (`> 1.0`)** | `0.2015` | **`61.46%` (FAIL P6)** |

\* Note: Adding `Testing-QA-4` reduced `Coder-12` retention slightly from `99.8%` (`Union-2`) to `98.11%` (`Union-3`), consistent with `Testing-QA-4`'s non-zero cross-activation on coding tokens ($\rho(\text{QA}\to\text{Coder}) = 0.1611\times$, `33.46%` fire rate).

**Explicit Scope & Falsified Alternative:** *Condition $\max_{j \neq i} \rho(j \to i) < 1.0$ is evaluated across all `6/6` ordered pairs of `Union-3` (`3` domains) and remains a candidate necessary condition, not yet a validated universal law. The hidden-state Grassmann subspace alternative (`k = 16`) was empirically falsified (`0.2015 < 0.2292` ranked the failure backwards).*

---

## 4. Law 2 Self-Falsification, Binary Indicator Ablation & Safe-Cartridge Bias (`F-045`, Blocker C Closed)

**Exact Denominator Clarification:** Every cartridge is evaluated across `4` protected domains (`stem`, `gsm8k_math`, `ifeval`, `bfcl_tool`). The `4`-cartridge unseen holdout split contains `16` pairs (`3` pos / `13` safe; `law2_L05_plus_L639` `holdout R² = +0.6399`, `TP=3/3, FN=0/3, TN=12/13`). Below is the **full 11-cartridge Leave-One-Cartridge-Out (LOCO-11) cross-validation** across all **`44` pairs** (`5` positive drift pairs `> +0.003`, `39` safe pairs; exact NNLS; `10,000` cartridge-level permutations; [l2_loco11_binary_ablation.json](../results/l2_loco11_binary_ablation.json)):

| Predictor Model (Exact NNLS fit on 10 cartridges → predict held-out 11th) | LOCO-11 $R^2$ | LOCO-11 MAE | 95% PI | Confusion (`44` pairs: `5` pos / `39` safe) | 10k-Shuffle `perm_p` | Safe-Pair Bias (vs `0` / vs raw) | Verdict |
|---|---:|---:|---:|---|---:|---:|---|
| `weight_mean_dW2` (naive `\|\|ΔW\|\|²`) | **−0.1347** | `0.00301` | `±0.01017` | `TP=0, FN=5, FP=4, TN=35` | `0.56054` | `+0.00164` / `+0.00405` | **FALSIFIED** |
| **`binary_1[L00–L05 open]` (Audit #2 Blocker C ablation)** | **−0.1231** | `0.00292` | `±0.01012` | **`TP=0, FN=5, FP=0, TN=39`** | **`0.39776`** | `+0.00158` / `+0.00400` | **FALSIFIED** |
| **`law2_unweighted`** (our original T1 Law 2!) | **−0.1003** | `0.00308` | `±0.01002` | `TP=1, FN=4, FP=10, TN=29` | `0.04660` | `+0.00187` / `+0.00428` | **SELF-FALSIFIED** |
| `weight_max_dW2` | `+0.1223` | `0.00292` | `±0.00894` | `TP=5, FN=0, FP=15, TN=24` | `0.00790` | `+0.00208` / `+0.00450` | High FP (`38%`) |
| **`L05_continuous_only` ($\sum_{l=0}^5 p_{\text{touch},l}\cdot\text{relRMS}_l^2$)** | **`+0.5787`** | **`0.00150`** | **`±0.00620`** | **`TP=5, FN=0, FP=3, TN=36`** | **`0.00550`** | **`+0.00094` / `+0.00336`** | **SUPPORTED\*** |
| `law2_L05_plus_L639` (Exact NNLS sets `w_L639 = 0.0`) | `+0.5787` | `0.00150` | `±0.00620` | `TP=5, FN=0, FP=3, TN=36` | `0.00550` | `+0.00094` / `+0.00336` | Identical to `L05_only` |

- **Blocker C Verdict & Unconstrained OLS Check (`C1`):** The continuous early-layer locality product (`L05_continuous_only`) **beats the binary indicator `1[L00–L05 open]` by `+0.7018 R²`** (`+0.5787` vs `-0.1231`, `TP 5/5` vs `0/5`), because `Coder-12` (`×3`), `Coder-4`, `Estonian-4 L0`, and `Estonian-4 L2` all had `L00–L05` unfrozen yet stayed safe owing to small early-layer locality exposure (`<= 0.0309` vs `Cyber-4` `0.0548–0.2059`). Checking unconstrained 2-variable OLS across all `44` pairs reveals:
  - Clamped `max(0, obs)` (no intercept): `w_L05 = +0.08625` (`95% CI [+0.07063, +0.10186]`), **`w_L639 = -0.00792` (`SE = 0.00427`, `95% CI [-0.01653, +0.00069]`)** (with intercept: `w_L639 = -0.00642`, `95% CI [-0.01871, +0.00587]`).
  - Raw `obs_delta` (no intercept): `w_L05 = +0.09802` (`95% CI [+0.07942, +0.11662]`), **`w_L639 = -0.03891` (`SE = 0.00508`, `95% CI [-0.04917, -0.02865]`)** (reflecting slight positive transfer on `L06–L39`).
  - **Honest Interpretation:** In the raw-outcome fit `w_L639` is **significantly negative** (`t = -7.66`, CI excludes `0`). In the clamped-outcome fit it is negative but **not significantly so** (`t = -1.85`, CI includes `0`). Under either fit, NNLS's `w_L639 = 0.0` is a **non-negativity boundary clamp** rather than an established unconstrained zero (`L06–L39` contributes no positive drift in this dataset).
- **Exact Permutation Unit & p-Floor (`C2`):** *(Note: table column `10k-Shuffle perm_p` reports 10,000-shuffle Monte Carlo across all 6 rows; exact finite-population enumerations below are computed specifically for the winning predictor `L05_continuous_only`.)* Across the `11` cartridges, there are `4` distinct clamped 4-domain outcome vectors (`cyber4_L0_33_ctrl`, `cyber4_L0_39`, `estonian4_L0_39`, and `8` identical all-zero cartridges). Exact finite-population enumerations (no Monte Carlo approximation needed):
  1. **Continuous 4-domain outcome vectors permuted across 11 cartridges ($11 \times 10 \times 9 = 990$ distinct assignments, floor `1/990 = 0.00101`):** **`5 / 990 ⇒ exact p = 0.00505`** (matching 10k Monte Carlo `0.00550`).
  2. **Ordered 2 bad continuous vectors ignoring tiny `+0.00021` on Estonian ($11 \times 10 = 110$, floor `1/110 = 0.00909`):** **`1 / 110 ⇒ exact p = 0.00909`** (`0/110` strictly exceed observed `R²`).
  3. **Unordered binary bad-pair classification ($\binom{11}{2} = 55$, floor `1/55 = 0.01818`):** `L05_continuous_only` ranks the exact 2 bad cartridges `#1` and `#2` ⇒ **`exact p = 1 / 55 = 0.01818`**.
- **Explicit Safe-Cartridge Overprediction Bias (Reported Separately from MAE):** Because NNLS cannot output negative transfer (`obs < 0`), `L05_continuous_only` has a systematic conservative positive bias on the `39` safe pairs (`+0.00094 nats` vs clamped `max(0, obs)`; `+0.00336 nats` vs raw `obs`). Specifically across the `12` `coder12` pairs (`s250`, `s500`, `final`), mean predicted drift is `+0.00159 nats` vs mean raw observed `-0.00403 nats` (never crosses `+0.003` alarm → `FP = 0` on `coder12`).
- \* **Explicit Sample-Size Caveat:** All `5` positive drift events (`obs_delta > +0.003`) come from **`n_bad = 2` cartridges** (`cyber4_L0_39` and `cyber4_L0_33_ctrl`, single failure mode). Out-of-domain generalization to non-Cyber early-layer failures remains open (`L2 CLOSED-with-scope`, not a general drift predictor).

---

## 5. Pooled Stable-Item McNemar & Multiple-Comparison Disclosure (`Blocker B Closed`)

### A. Stable-Item Paired McNemar (`S1 == S2` Only, `6` Benchmarks × `2` Candidates = `12` Comparisons)

| Benchmark (`N`) | Stable (`S1 == S2`) | Disc. Excl. | Base Stable | `Coder-12` (`b` / `c`, net, exact `p`) | `Union-3` (`b` / `c`, net, exact `p`) | Family-Wise Status (`m = 12`) |
|---|---:|---:|---:|---|---|---|
| **`LCB-Medium` (`52`)** *(Post-Hoc Primary)* | `45` | `7` | `4` | `b=7, c=2` (`+5`, `p = 0.17969`) | **`b=6, c=0` (`+6`, `p = 0.03125`)** | **Nominal `p < 0.05`; FAILS FWER/FDR** |
| `LCB-All` (`175`) *(Exploratory)* | `156` | `19` | `37` | `b=11, c=3` (`+8`, `p = 0.05737`) | `b=10, c=2` (`+8`, `p = 0.03857`) | Exploratory; fails FWER/FDR |
| `LCB-Easy` (`43`) *(Exploratory)* | `36` | `7` | `32` | `b=2, c=1` (`+1`, `p = 1.00000`) | `b=2, c=2` (`+0`, `p = 1.00000`) | Ceiling / flat |
| `LCB-Hard` (`80`) *(Exploratory)* | `75` | `5` | `1` | `b=2, c=0` (`+2`, `p = 0.50000`) | `b=2, c=0` (`+2`, `p = 0.50000`) | **Unmoved** (needs RFT/STaR) |
| `GPQA` (`198`) *(Exploratory)* | `137` | `61` | `80` | `b=18, c=16` (`+2`, `p = 0.86417`) | `b=12, c=10` (`+2`, `p = 0.83181`) | Preserved (`+2`) |
| `GSM8K` (`1319`) *(Exploratory)* | `1291` | `28` | `1194` | `b=8, c=9` (`-1`, `p = 1.00000`) | `b=7, c=14` (`-7`, `p = 0.18925`) | `p > 0.18` (ns); nominal `P4` FAIL on full set (`-3` vs `0`-floor) |

**Mandatory Multiple-Testing & Direction Disclosure (Blocker B):**
1. **Pooled stable-item filtering weakened the headline result:** removing seed-selection bias moved `Union-3 LCB-Medium` from `p = 0.00781` (single-seed comparator) to **`p = 0.03125`** (`b=6, c=0`).
2. **Zero benchmarks survive family-wise correction across the 12 comparisons:** `p_min = 0.03125 > Bonferroni α/12 = 0.00417`; Holm step-1 and Benjamini-Hochberg FDR (`q = 0.05`, rank-1 threshold `0.00417`) all fail.
3. **Post-Hoc Primary Endpoint Disclosure:** `LCB-Medium (Union-3, stable-item)` is designated as the single primary endpoint (`p = 0.03125` unadjusted), with all other 11 comparisons explicitly labeled exploratory. **Primary endpoint declared post-hoc; confirmatory replication on a fresh problem split (`40–80` new contest problems, `EXP-0017`) is required before asserting a confirmatory coding improvement.**

### B. Estonian (`et`) `L0 vs L2 vs L6` Layer Ablation (`EXP-0013` — Data Confound vs Architecture)

| Variant (`771` rows × `2` epochs, `base_asst_nll = 0.3136`, `148` unseen holdout seqs / `34,016` tok) | Holdout `ASST_d` Mean | 95% Sequence Bootstrap CI | Protected `P1` | Interpretation |
|---|---:|---:|:---:|---|
| `Estonian-4 L00–L39` (early layers OPEN) | `+0.00862` | `[+0.00163, +0.01605]` | PASS | Overfit (`UTILITY FAIL`) |
| `Estonian-4 L02–L39` | `+0.01007` | `[+0.00299, +0.01754]` | PASS | Overfit (`UTILITY FAIL`) |
| `Estonian-4 L06–L39` (early layers FROZEN) | `+0.01003` | `[+0.00365, +0.01678]` | PASS | Overfit (`UTILITY FAIL`) |

`L0 ≈ L2 ≈ L6` rules out an early-layer architectural cause in this regime, and is **consistent with either a data confound (`base_nll = 0.3136`, `771` rows) or a schedule × 2-epoch mismatch** (violating the 1-epoch unique-data doctrine). A 1-epoch Estonian control (`EXP-0013b`) is required to disambiguate the two. Because high-NLL (`>= 0.70`) natural-language morphology also remains untested, **"Universal" is removed from L5** (now *"Early-Layer Hub Instability (`L00–L05`)"*).

---

## 6. Demotion of `P2` (`Top-k Jaccard`) to Diagnostic & Honest Scoreboard

| Cartridge Class | Max Protected `ASST_d` | `Top-10 Jaccard` Mean Range (`4` prot. domains) | Worst Layer `J_min` | True Safety Status |
|---|---:|---:|---:|:---:|
| Safe `Coder-12` | `-0.00205` | `0.8989 – 0.9564` | `0.8599` | **SAFE (PASS)** |
| No-Op `v2` (`0.08%` touch) | `+0.00048` | `0.9031 – 0.9610` | `0.8499` | **NO-OP (UTILITY FAIL)** |
| Harmful `Cyber-4 L0–39` | `+0.01460` | `0.7801 – 0.8672` | `0.7110` | **HARMFUL (P1 FAIL)** |
| Harmful `Cyber-4 L0–33` | `+0.02204` | `0.7706 – 0.8642` | `0.6961` | **HARMFUL (P1 FAIL)** |

**Corrected Demotion Rationale (Audit #2 Fix):** Comparing like-for-like metrics, both `J_min` (`0.6961–0.7110` harmful vs `0.8499–0.8599` safe/no-op) and `J_mean` (`0.7706–0.8672` harmful vs `0.8989–0.9610` safe) **cleanly separate** the `n = 4` cartridges without overlap. However, the original pre-registered threshold **`J >= 0.99` is invalid** (even the `0.08%`-touch no-op fails at `J_min = 0.8499` due to Law 6 `#10/#11` ties), and any recalibrated threshold (e.g., `J_min >= 0.80`) is **not out-of-sample validated at `n = 4`**. Therefore, **Protocol Amendment #2 demotes `P2` from a PASS/FAIL gate (`checks`) to a non-gating `diagnostics` metric** pending out-of-sample threshold validation (enforced mechanically by `certificate.schema.json` Draft 2020-12 and `validate_certificate.py --self-test`).

### Explicit Population Denominators for Every False-Safe Bound
- **Real-32B Certified/Rejected Cartridges (`n = 7`):** `4/4` bad caught fail-closed (`v2`, `Cyber-4 L0–39`, `Testing-QA-4`, `Estonian-4`), `3/3` good certified (`Coder-12`, `Cyber-4 L6–39`, `Union-2`), **`0 / 7` false-safe** (`95%` Clopper-Pearson upper bound = **`34.8%`**).
- **Real-32B OOS Predictor (`4`-cartridge holdout = `16` pairs, `3` pos / `13` safe; `LOCO-11` = `44` pairs, `5` pos / `39` safe):** **`0 / 3`** false-safe on holdout (`0 / 5` in `LOCO-11`, from `n_bad = 2` cartridges).
- **Synthetic T1 Holdout (`n = 16` independent runs):** **`0 / 16` false-safe** (`95%` Clopper-Pearson upper bound = **`17.1%`**).

### Honest Exit Criteria Scoreboard (`R0–R6` & `P1–P9`)
- **CLOSED / PASS:** `R0/R1` (Metrology + `2×2` Factorial + `sm_120` `vllm#52525` & `VLLM_BATCH_INVARIANT=1` Verification), `R2` (`C-001..C-008`), `R3` (Laws `L1` [rank-dependent evacuation confirmed + mechanically enforced FP32 `1.31 MB` gate mandate], `L6`, `L5` matched `34/40` domain-scoped, `L2` **CLOSED-with-scope** [`n_bad = 2`, single failure mode; exact perm `5/990` & `1/55`; unconstrained OLS `w_L639` reported], `L4` kernel), `R5` (Negative Results), `R6` (JSON Schema Draft 2020-12 `v1.4` fail-closed certificates with 6 mutation tests), `P1`, `P7/P8/P9`.
- **DEMOTED TO DIAGNOSTIC:** `P2` (`Top-k Jaccard`, Amendment #2 — recalibrated threshold unvalidated out-of-sample).
- **HONESTLY OPEN / CANDIDATE / FAIL:** `L7` (**`OPEN / CANDIDATE`**: `max_j ρ < 1.0` evaluated across `6/6` ordered pairs / `3` domains, not yet OOS-validated), `R4` (**`OPEN: 2 / 3` certified domains**), `Union-3 P6` (**`FAIL: 61.46% < 70%`**), `Union-3` nominal `P4` (**`FAIL` vs `0`-floor**, `GSM8K -3`, stable-item `p = 0.189`), `Testing-QA-4` (`UTILITY FAIL: -0.0272 > -0.030`), `Estonian-4` (`UTILITY FAIL: +0.0086`, 1-epoch control `EXP-0013b` open), Confirmatory LCB Holdout (`EXP-0017`, Blocker B), Production `WSD` vs `EXP-0016` `constant` (`OPEN RECIPE CONFLICT`, pending `EXP-0016-T3`).

---

## 7. Companion Study Summary: `EXP-0016-T1 v3` (LR Schedule × 1-Epoch Unique Data, `F-043`)
Confound-free `240`-run LR-envelope study (`3` seeds, `5` peak LRs `2e-4..2e-3`) demonstrated a clean **sign-flip crossover**: **`constant` LR beat `cosine` by `1.32×` on mean learning (`-1.7481` vs `-1.3205`) at matched `ProtKL ≈ 0.107`** in `unique_1ep_sequential` (and beat `cosine`, `wsd_15pct`, and `constant_ema` across all three 1-epoch unique regimes; `unique_1ep_iid`: `-1.9060` vs `-1.7507` at lower `ProtKL 0.104 vs 0.113`), whereas **`cosine` won in `repeated_4ep_iid`** (`-1.0111` vs `-0.9615`). *(Diagnostic note: the `11.0×` final-window `W4` ratio `-1.3998` vs `-0.1267` is a terminal-LR schedule-endpoint artifact where cosine LR approaches zero, and is reported as diagnostic only.)* **Production Recipe Conflict:** Production currently uses `WSD`, which this T1 study ranks below `constant` — real-LLM T3 confirmation on `OLMoE-1B-7B` (`EXP-0016-T3`) is required before changing the production schedule.
