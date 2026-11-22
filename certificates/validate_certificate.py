#!/usr/bin/env python3
"""
Fail-Closed Certificate Validator (JSON Schema Draft 2020-12 + Semantic Invariants).
License: Apache-2.0 (see LICENSE-CODE)

Usage:
  python3 certificates/validate_certificate.py certificates/example_*.yaml
  python3 certificates/validate_certificate.py --self-test
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import List, Tuple

import yaml
from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parent / "certificate.schema.json"


def load_doc(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(raw)
    else:
        obj = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name}: top-level document must be a mapping, got {type(obj).__name__}")
    return obj


def validate_cert_obj(cert: dict, schema: dict) -> List[str]:
    errors: List[str] = []
    validator = Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(cert), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"SCHEMA[{loc}]: {err.message}")

    # Fail-closed semantic invariants (beyond structural JSON Schema)
    verdict = cert.get("verdict")
    failed_gates = cert.get("failed_gates") or []
    checks = cert.get("checks") or []
    check_fails = [c for c in checks if isinstance(c, dict) and c.get("status") == "FAIL"]

    if verdict == "CERTIFIED_PASS":
        if failed_gates:
            errors.append(f"SEMANTIC: verdict=CERTIFIED_PASS but failed_gates is non-empty: {failed_gates}")
        if check_fails:
            errors.append(f"SEMANTIC: verdict=CERTIFIED_PASS but {len(check_fails)} check(s) have status=FAIL")
    elif verdict == "REJECTED_FAIL_CLOSED":
        if not failed_gates and not check_fails:
            errors.append("SEMANTIC: verdict=REJECTED_FAIL_CLOSED requires at least one failed_gates entry or FAIL check")

    for c in checks:
        if isinstance(c, dict) and c.get("gate") == "P2":
            errors.append("SEMANTIC: P2 (Top-k Jaccard) is demoted per Law 6 and MUST appear only in 'diagnostics', not 'checks'")

    # Mechanical FP32 gate-row enforcement for u-bearing cartridges (Audit #4 B3)
    engine = cert.get("engine") or {}
    if cert.get("carries_neutral_u") is True and engine.get("router_dtype") != "float32":
        errors.append(
            "SEMANTIC: carries_neutral_u=true requires engine.router_dtype='float32' "
            "(bfloat16 casting introduces +0.00281 bias_err = 93.7% of +0.003 P1 budget)"
        )

    # Mechanical symmetric flip_rate invariants (Audit #4 B4 + Audit #5 A)
    nondet = cert.get("nondeterminism") or {}
    if nondet.get("measured_on") == "n_a_single_forward" and nondet.get("flip_rate") is not None:
        errors.append("SEMANTIC: measured_on='n_a_single_forward' requires flip_rate=null (not hardcoded 0.0)")
    if nondet.get("measured_on") in ("base_substrate", "candidate") and nondet.get("flip_rate") is None:
        errors.append(f"SEMANTIC: measured_on='{nondet.get('measured_on')}' requires numeric flip_rate (null forbidden)")

    return errors


def run_self_test(schema: dict) -> bool:
    cert_dir = SCHEMA_PATH.parent
    examples = sorted(cert_dir.glob("example_*.yaml"))
    if not examples:
        print("[FAIL] No example_*.yaml certificates found for self-test!")
        return False

    all_ok = True
    sample_valid = None
    for p in examples:
        doc = load_doc(p)
        errs = validate_cert_obj(doc, schema)
        if errs:
            print(f"[FAIL] Valid example failed validation: {p.name}")
            for e in errs:
                print(f"  - {e}")
            all_ok = False
        else:
            print(f"[PASS] Valid example passed: {p.name} (verdict={doc['verdict']})")
            if doc["verdict"] == "CERTIFIED_PASS" and sample_valid is None:
                sample_valid = doc

    if sample_valid is None:
        print("[FAIL] Could not find a CERTIFIED_PASS example for negative mutation tests!")
        return False

    # Positive test: carries_neutral_u=True with float32 MUST PASS
    pos_u = copy.deepcopy(sample_valid)
    pos_u["mode"] = "add"
    pos_u["carries_neutral_u"] = True
    pos_u["engine"]["router_dtype"] = "float32"
    pos_errs = validate_cert_obj(pos_u, schema)
    if pos_errs:
        print(f"[FAIL] Valid neutral-u float32 cert rejected: {pos_errs}")
        all_ok = False
    else:
        print("[PASS] Positive check passed: add + carries_neutral_u=true + router_dtype=float32")

    # 6 Deliberately malformed negative tests — ALL MUST BE REJECTED FAIL-CLOSED!
    mutations: List[Tuple[str, dict]] = []

    m1 = copy.deepcopy(sample_valid)
    del m1["controller"]
    mutations.append(("missing_controller_block", m1))

    m2 = copy.deepcopy(sample_valid)
    m2["checks"].append({"gate": "P2", "metric": "stem_jaccard", "observed": 0.95, "limit": ">= 0.85", "status": "PASS"})
    mutations.append(("p2_forbidden_in_checks", m2))

    m3 = copy.deepcopy(sample_valid)
    m3["failed_gates"] = ["UTILITY"]
    mutations.append(("pass_verdict_with_failed_gate", m3))

    m4 = copy.deepcopy(sample_valid)
    m4["curriculum"]["injection_source"] = "unverified_random_scrape"
    mutations.append(("invalid_injection_source_enum", m4))

    m5 = copy.deepcopy(sample_valid)
    m5["mode"] = "add"
    m5["carries_neutral_u"] = True
    m5["engine"]["router_dtype"] = "bfloat16"
    mutations.append(("neutral_u_with_bf16_router_rejected", m5))

    m6 = copy.deepcopy(sample_valid)
    m6["nondeterminism"]["measured_on"] = "n_a_single_forward"
    m6["nondeterminism"]["flip_rate"] = 0.0
    m6["nondeterminism"]["flip_rate_note"] = "invalid hardcoded zero"
    mutations.append(("hardcoded_zero_flip_rate_on_single_forward_rejected", m6))

    m7 = copy.deepcopy(sample_valid)
    m7["nondeterminism"]["measured_on"] = "candidate"
    m7["nondeterminism"]["flip_rate"] = None
    mutations.append(("null_flip_rate_on_candidate_rejected", m7))

    for name, bad_doc in mutations:
        errs = validate_cert_obj(bad_doc, schema)
        if errs:
            print(f"[PASS] Malformed cert REJECTED fail-closed ({name}): {errs[0]}")
        else:
            print(f"[FAIL] Malformed cert WRONGLY ACCEPTED ({name})!")
            all_ok = False

    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="Certificate YAML/JSON files to validate")
    ap.add_argument("--self-test", action="store_true", help="Validate all example certs + negative malformed-cert tests")
    args = ap.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)

    if args.self_test:
        ok = run_self_test(schema)
        print("[✓] SELF-TEST PASSED" if ok else "[✗] SELF-TEST FAILED")
        return 0 if ok else 1

    if not args.files:
        ap.print_help()
        return 2

    rc = 0
    for f in args.files:
        p = Path(f)
        try:
            doc = load_doc(p)
            errs = validate_cert_obj(doc, schema)
        except Exception as exc:
            errs = [f"PARSE_ERROR: {exc}"]
        if errs:
            print(f"[REJECTED] {p}:")
            for e in errs:
                print(f"  - {e}")
            rc = 1
        else:
            print(f"[VALID] {p} (verdict={doc['verdict']})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
