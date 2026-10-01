"""Run the dependency-light V4 scale and protocol audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from v4_synthetic import generate_transactions
from v4_metrics import binary_metrics, open_set_metrics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def one_scale(out: Path, events: int, seed: int) -> dict:
    started = time.perf_counter()
    frame = generate_transactions(total_events=events, seed=seed)
    csv_path = out / f"synthetic_{events}_seed{seed}.csv"; frame.to_csv(csv_path, index=False)
    unknown = frame.fraud_type.isin(["D", "E"]).to_numpy(dtype=int)
    # Protocol-only score: amount is intentionally labeled as a pending baseline,
    # not as a V4 neural result.
    score = frame.amount.to_numpy(dtype=float)
    threshold = float(frame.loc[frame.period == "known_mechanism", "amount"].quantile(0.95))
    result = {"scale": events, "seed": seed, "rows": len(frame), "fraud_rate": float(frame.label.mean()), "banks": int(frame.bank_id.nunique()), "fraud_counts": {str(k): int(v) for k, v in frame.fraud_type.value_counts().items()}, "runtime_seconds": time.perf_counter() - started, "csv_sha256": sha256(csv_path), "protocol_score_status": "diagnostic_only", "open_set_diagnostic": open_set_metrics(unknown, score, threshold), "neural_v4_status": "pending_torch_environment"}
    (out / f"synthetic_{events}_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--scales", nargs="+", type=int, default=[50_000, 100_000, 500_000]); parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46]); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for scale in args.scales:
        for seed in args.seeds:
            print(f"generating scale={scale} seed={seed}", flush=True); results.append(one_scale(args.output, scale, seed))
    manifest = {"v4_status": "scale_protocol_complete_neural_pending", "python": sys.version, "platform": platform.platform(), "scales": args.scales, "seeds": args.seeds, "results": results, "claims": {"neural_performance": "pending", "public_ieee_cis": "pending_kaggle_authentication", "causal_discovery": "not_implemented", "privacy_guarantee": "not_implemented"}}
    (args.output / "RESULTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
