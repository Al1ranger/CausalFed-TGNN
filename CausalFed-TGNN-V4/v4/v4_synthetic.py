"""Controlled scalable synthetic transaction streams for V4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(
    clients: int = 4,
    total_events: int = 50_000,
    seed: int = 42,
    fraud_rate: float = 0.012,
    unknown_start: float = 0.70,
    device_reuse: float = 0.35,
    merchant_concentration: float = 0.35,
    burstiness: float = 0.55,
    label_delay: int = 0,
    bank_heterogeneity: float = 0.20,
) -> pd.DataFrame:
    """Generate a chronological bank stream with explicit mechanism controls.

    The generator is deterministic for a fixed argument tuple. D/E are withheld
    mechanisms after ``unknown_start``; they are still present as labels so the
    evaluation protocol can measure open-set recognition without using them for
    fitting or threshold calibration.
    """
    if clients < 2 or total_events < clients or not 0 < fraud_rate < 1:
        raise ValueError("invalid clients, total_events, or fraud_rate")
    if not 0 < unknown_start < 1:
        raise ValueError("unknown_start must be in (0, 1)")
    rng = np.random.default_rng(seed)
    per_client = total_events // clients
    counts = np.full(clients, per_client, dtype=int)
    counts[: total_events % clients] += 1
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for bank, n in enumerate(counts):
        n_customers = max(32, int(np.sqrt(n) * 3))
        n_merchants = max(20, int(np.sqrt(n) * 2))
        n_devices = max(40, int(np.sqrt(n) * 3.5))
        n_locations = max(16, int(np.sqrt(n)))
        for i in range(int(n)):
            phase = i / max(n - 1, 1)
            late = phase >= unknown_start
            burst = rng.random() < burstiness * (0.25 + 0.75 * late)
            customer = int(rng.integers(0, n_customers))
            merchant_hot = rng.random() < merchant_concentration
            merchant = int(rng.integers(0, max(2, int(n_merchants * 0.2)))) if merchant_hot else int(rng.integers(0, n_merchants))
            reused = rng.random() < device_reuse
            device = int(rng.integers(0, max(2, int(n_devices * 0.25)))) if reused else int(rng.integers(0, n_devices))
            location = int(rng.integers(0, n_locations))
            amount = float(rng.lognormal(3.5 + (0.25 if burst else 0.0), 0.9))
            bank_shift = bank_heterogeneity * (bank - (clients - 1) / 2)
            risk = -4.0 + bank_shift + 0.55 * (amount > 90) + 0.50 * reused + 0.45 * merchant_hot
            if late:
                risk += 0.55 + 0.55 * (location >= int(n_locations * 0.75))
            probability = 1.0 / (1.0 + np.exp(-risk))
            label = int(rng.random() < max(0.001, min(0.95, probability + fraud_rate - 0.012)))
            if not label:
                fraud_type = "legitimate"
            elif late and location >= int(n_locations * 0.75):
                fraud_type = "E"
            elif late and merchant_hot:
                fraud_type = "D"
            elif amount > 90:
                fraud_type = "A"
            elif reused:
                fraud_type = "B"
            else:
                fraud_type = "C"
            timestamp = start + pd.to_timedelta(i * 1440 + bank, unit="m")
            rows.append({
                "transaction_id": f"B{bank}-T{i}",
                "bank_id": f"Bank_{chr(65 + bank)}",
                "customer_id": f"B{bank}-C{customer}",
                "account_id": f"B{bank}-A{customer}",
                "merchant_id": f"M{merchant}",
                "device_id": f"D{device}",
                "location": f"L{location}",
                "amount": round(amount, 2),
                "timestamp": timestamp.isoformat(),
                "label": label,
                "fraud_type": fraud_type,
                "period": "unknown_mechanism" if late else "known_mechanism",
                "label_available_at": (timestamp + pd.to_timedelta(label_delay, unit="m")).isoformat(),
            })
    return pd.DataFrame(rows)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=50_000)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--unknown-start", type=float, default=0.70)
    parser.add_argument("--label-delay", type=int, default=0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_transactions(args.clients, args.events, args.seed, args.fraud_rate, args.unknown_start, label_delay=args.label_delay)
    frame.to_csv(args.output, index=False)
    meta = {"events": len(frame), "clients": args.clients, "seed": args.seed, "fraud_rate": float(frame.label.mean()), "fraud_counts": frame.fraud_type.value_counts().to_dict(), "csv_sha256": fingerprint(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
