#!/usr/bin/env bash
# Single-command open reproduction of MoE Functional Locality Laws (L1, L6, C-006, Public Atlas)
# on 1x 24GB+ GPU (reference runtime: ~16 min on 1x NVIDIA RTX 5090 32GB in BF16) using 100%
# public Apache-2.0 models and public Hugging Face datasets.
#
# Determinism & Runtime Pinning (Audit #1 / #4):
#   Audit #1 proved that EVEN WITH TP=1, batch=1, temp=0, --enforce-eager, and
#   --no-enable-prefix-caching pinned, default Marlin MoE serving in vLLM 0.29.0
#   remains non-deterministic (upstream issue vllm#52525; mitigated by Triton MoE
#   backends / VLLM_BATCH_INVARIANT=1 or fixed token alignment).
#   This script runs PyTorch/HF Transformers single-forward BF16 NLL evaluation
#   (runtime=hf_transformers_single_forward; effective pins: SEED, PYTHONHASHSEED,
#   CUBLAS_WORKSPACE_CONFIG; vLLM serving flags recorded as inert metadata only).
#
# Usage:
#   ./reproduce.sh --verify            # Fast CPU self-check: JSON Schema + fail-closed cert tests + syntax checks
#   ./reproduce.sh                     # Run 1-GPU OLMoE-1B-7B reproduction (~16 min on 1x RTX 5090 BF16)
#   MODEL_QWEN35=Qwen/Qwen3.5-35B-A3B-Base ./reproduce.sh   # Also run 4-GPU Qwen3.5-Base L1+L6 bonus row
#
# License: Apache-2.0 (see LICENSE-CODE)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"

# --- Determinism Pinning (Audit #1) ---
export TP_SIZE="${TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export TEMPERATURE="${TEMPERATURE:-0}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export SEED="${SEED:-0}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

MODEL_OLMOE="${MODEL_OLMOE:-allenai/OLMoE-1B-7B-0125-Instruct}"
MODEL_QWEN35="${MODEL_QWEN35:-}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/repro_output}"

run_verify() {
    echo "=== [--verify] Schema Validation, Fail-Closed Negative Cert Tests, Math Golden Values & Syntax Checks ==="
    "${PY}" -m py_compile \
        "${SCRIPT_DIR}/harness/cached_paired_nll_gate.py" \
        "${SCRIPT_DIR}/public_repro/reproduce_olmoe.py" \
        "${SCRIPT_DIR}/public_repro/reproduce_qwen35_base_l1_l6.py" \
        "${SCRIPT_DIR}/public_repro/reproduce_marlin_moe_nondeterminism.py" \
        "${SCRIPT_DIR}/public_repro/exp_0016_lr_schedule_unique_epoch_t1.py" \
        "${SCRIPT_DIR}/certificates/validate_certificate.py"
    echo "[PASS] py_compile succeeded for all 6 Python modules"
    "${PY}" "${SCRIPT_DIR}/certificates/validate_certificate.py" --self-test
    "${PY}" - << 'PYEOF'
import math
def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
def clopper_pearson_upper_zero(n: int, alpha: float = 0.05) -> float:
    return 1.0 - (alpha ** (1.0 / n))
assert abs(exact_mcnemar(8, 0) - 0.0078125) < 1e-12
assert abs(exact_mcnemar(6, 0) - 0.03125) < 1e-12
assert abs(exact_mcnemar(10, 2) - 0.03857421875) < 1e-12
assert abs(exact_mcnemar(11, 3) - 0.057373046875) < 1e-12
assert abs(clopper_pearson_upper_zero(7) - 0.348163655) < 1e-6
assert abs(clopper_pearson_upper_zero(16) - 0.170749723) < 1e-6
print("[PASS] Golden-value math unit tests verified (McNemar & Clopper-Pearson)")
PYEOF
    echo "[✓] ALL --verify CHECKS PASSED!"
}

if [ "${1:-}" = "--verify" ]; then
    run_verify
    exit 0
fi

mkdir -p "${OUT_DIR}"
run_verify

echo "=== Recording Exact Environment Fingerprint -> ${OUT_DIR}/env.txt ==="
"${PY}" - << EOF > "${OUT_DIR}/env.txt"
import platform, torch
try:
    import transformers; tf_v = transformers.__version__
except Exception: tf_v = "unavailable"
try:
    import vllm; vllm_v = vllm.__version__
except Exception: vllm_v = "unavailable"
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={tf_v}")
print(f"vllm={vllm_v}")
print(f"gpu0={gpu}")
print(f"determinism_pins=TP_SIZE=${TP_SIZE},BATCH_SIZE=${BATCH_SIZE},TEMPERATURE=${TEMPERATURE},ENFORCE_EAGER=${ENFORCE_EAGER},SEED=${SEED}")
EOF
cat "${OUT_DIR}/env.txt"

echo "=== [1/2] Single-GPU Public Reproduction (BF16, ~16 min on 1x RTX 5090): ${MODEL_OLMOE} ==="
"${PY}" -u "${SCRIPT_DIR}/public_repro/reproduce_olmoe.py" \
    --model-dir "${MODEL_OLMOE}" \
    --out-dir "${OUT_DIR}/olmoe_1b_7b"

# Attach explicit determinism metadata block to OLMoE output JSON
"${PY}" - << EOF
import json
from pathlib import Path
p = Path("${OUT_DIR}/olmoe_1b_7b/olmoe_reproduction_results.json")
if p.exists():
    d = json.loads(p.read_text())
    d["determinism"] = {
        "runtime": "hf_transformers_single_forward",
        "pins_effective": ["SEED", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"],
        "pins_inert": ["TP_SIZE", "ENFORCE_EAGER", "BATCH_SIZE", "TEMPERATURE"],
        "pins_inert_note": "vLLM serving flags recorded for environment parity only; OLMoE reproduction uses PyTorch/Transformers single-forward BF16 NLL.",
        "seed": int("${SEED}"),
        "precision": "bf16_single_forward_nll",
        "flip_rate": None,
        "flip_rate_note": "N/A — single-forward BF16 NLL evaluation, no autoregressive generation",
    }
    p.write_text(json.dumps(d, indent=2))
    print(f"[✓] Attached determinism metadata -> {p}")
EOF

if [ -n "${MODEL_QWEN35}" ]; then
    echo "=== [2/2] Multi-GPU Public Bonus Row (BF16): ${MODEL_QWEN35} ==="
    "${PY}" -u "${SCRIPT_DIR}/public_repro/reproduce_qwen35_base_l1_l6.py" \
        --model-dir "${MODEL_QWEN35}" \
        --out-json "${OUT_DIR}/qwen35_base_l1_l6.json"
else
    echo "=== [2/2] Skipping optional Qwen3.5-35B-A3B-Base bonus row (set MODEL_QWEN35=Qwen/Qwen3.5-35B-A3B-Base to enable) ==="
fi

echo "[✓] Public reproduction complete -> ${OUT_DIR}"
