# Licensing & IP Boundary Notice (`NOTICE-weights.md`)

This repository separates **open scientific methodology and public-model
reproduction** from **proprietary production assets**:

## 1. Open Assets Included in This Repository
- **Code, Evaluation Harness, Reproduction Scripts & Schemas** (`harness/cached_paired_nll_gate.py`, `public_repro/reproduce_olmoe.py`, `public_repro/reproduce_qwen35_base_l1_l6.py`, `public_repro/reproduce_marlin_moe_nondeterminism.py`, `public_repro/exp_0016_lr_schedule_unique_epoch_t1.py`, `reproduce.sh`, `certificates/certificate.schema.json`, `certificates/validate_certificate.py`): Licensed under **Apache-2.0** (`LICENSE-CODE`).
- **Technical Report, Executive Summary, Confounds, Glossary & Redacted Example Certificates** (`report/TECHNICAL_REPORT.md`, `report/EXECUTIVE_SUMMARY_FI.md`, `ledger/CONFOUNDS.md`, `ledger/GLOSSARY.md`, `certificates/example_*.yaml`): Licensed under **CC-BY-4.0** (`LICENSE-REPORT`).
- **Public Model Reproduction Data & Benchmark Diagnostics** (`atlas_public/olmoe_1b_7b_atlas.parquet`, `results/olmoe_reproduction_results.json`, `results/qwen35_base_l1_l6.json`, `results/marlin_moe_kernel_microrepro.json`, `results/cross_quant_determinism_summary.json`, `results/audit1_2x2_determinism_factorial.json`, `results/gpqa_temp0_batch1_determinism.json`, `results/l2_loco11_binary_ablation.json`, `results/l7_functional_overlap_and_ucast.json`, `results/exp_0016_lr_schedule_t1.json`): Public model outputs derived strictly from publicly available Apache-2.0 models (`allenai/OLMoE-1B-7B-0125-Instruct`, `Qwen/Qwen3.5-35B-A3B-Base`, `dark-side-of-the-code/Qwen3-Coder-30B-A3B-Instruct-AWQ`) and public Hugging Face datasets (`google-research-datasets/mbpp`, `openai/gsm8k`, `allenai/sciq`, `Salesforce/wikitext`), plus aggregated non-reconstructible diagnostics.

## 2. Closed Assets Explicitly Excluded (No Implied License)
Nothing in this repository grants — expressly or by implication — any license or right to:
1. Proprietary Kataguru BF16 or NVFP4 base model weights.
2. Proprietary Kataguru expert cartridge weights (`.safetensors`).
3. Proprietary Kataguru per-layer expert-ID selection manifests (`*.json`).
4. Proprietary Kataguru training corpora, SQLite Golden Bank rows, or dataset mixture recipes (`DATA_S1` / `DATA_S2`).
5. The full per-expert Kataguru routing atlas (`atlas.parquet`). Only aggregated, non-reconstructible summary statistics are reported for the private 32B validation substrate.

All proprietary assets remain strictly All Rights Reserved by their respective owner.
