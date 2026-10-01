"""Optional V4 model components.

This module imports without PyTorch so protocol and data work can run in a
dependency-light environment. The neural components activate when the research
environment provides torch.
"""

from __future__ import annotations

import numpy as np


def novelty_scores(logits, prototypes=None):
    """Return energy, max-logit margin, and optional prototype distance."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    energy = -(logits.max(axis=1) + np.log(np.exp(shifted).sum(axis=1)))
    ordered = np.sort(logits, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if logits.shape[1] > 1 else ordered[:, -1]
    result = {"energy": energy, "max_logit_margin": margin}
    if prototypes is not None:
        p = np.asarray(prototypes, dtype=float)
        result["prototype_distance"] = np.min(((logits[:, None, :] - p[None, :, :]) ** 2).sum(axis=2) ** 0.5, axis=1)
    return result


def environment_batch_indices(environments, batch_size, minimum_environments=2, seed=42):
    """Construct batches that make the invariant penalty nonzero when possible."""
    rng = np.random.default_rng(seed)
    env = np.asarray(environments)
    groups = {value: np.flatnonzero(env == value).tolist() for value in np.unique(env)}
    if len(groups) < minimum_environments:
        return [np.arange(len(env)).tolist()]
    pools = list(groups.values()); output = []
    while any(pools):
        active = [pool for pool in pools if pool]
        if not active: break
        chosen = [pool.pop(rng.integers(len(pool))) for pool in active[:minimum_environments]]
        while len(chosen) < min(batch_size, len(env)) and any(pools):
            active = [pool for pool in pools if pool]
            if not active: break
            pool = active[int(rng.integers(len(active)))]
            chosen.append(pool.pop(rng.integers(len(pool))))
        output.append(chosen)
    return output


def time_features(elapsed_days, dimensions=8):
    """Deterministic elapsed-time features for the optional V4 time path."""
    t = np.asarray(elapsed_days, dtype=float).reshape(-1, 1)
    frequencies = np.logspace(-2, 0, num=max(1, dimensions // 2))
    phase = np.log1p(np.maximum(t, 0.0)) * frequencies.reshape(1, -1)
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)[:, :dimensions]
