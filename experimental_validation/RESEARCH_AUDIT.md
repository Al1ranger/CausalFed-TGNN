# Research audit

## Repository structure
The workspace contains V3 manuscript artifacts, V4 generator/protocol utilities,
and `scientific_analysis` with executed pooled logistic regressions. The recursive
inventory is saved in `repository_inventory.json`. No Git checkout was found at
the workspace root; source hashes will identify this implementation.

## Existing implemented modules
V4 provides synthetic generation, diagnostic metrics, elapsed-time features,
novelty utilities, and environment batching. It does not implement a trainable
neural encoder, graph builder, federated optimizer, or continual learner.
`scientific_analysis/run_analysis.py` implements corrected tie-aware metrics,
chronological partitions, and NumPy logistic regression.

## Existing completed evidence
Historical manifests describe 15 streams at 50k/100k/500k, seeds 42–46, and 30
logistic fits. These are historical records until independently hash checked.
Legacy V4 novelty metrics are diagnostic only and must not be imported as neural
results. The supplied manuscript correctly describes its graph figures as
conceptual and its neural/federated evaluations as pending.

## Missing implementations
Trainable MLP, graph baselines, HGT, representation decomposition, experiment
runner, open-set calibration, federated algorithms, leakage tests, checkpoint
validation, attacks, continual adaptation, and public-data adapters.

## Missing dependencies
Bundled Python 3.12.14 initially lacked torch, torch_geometric, sklearn, scipy,
pytest, and psutil. Project-local installation was requested. Python 3.12 is used
instead of preferred 3.11 to reuse the provided runtime. Exact installed versions
will be recorded after import verification.

## Data availability
All 15 original CSV paths are present in V4/runs/v4_scale. A PaySim `.part` file
is not a validated dataset. IEEE-CIS credentials and accepted terms are not supplied.

## Reproducibility risks
Original CSVs are bank-major, not globally chronological. Reproduce the existing
per-bank rank/(n-1) partitions. Treat merchant/device/location IDs as client-local
because the generator reuses their strings across banks. Customer and account
are redundant aliases in the generator. Never expose fraud_type, period, label,
future entity degree, or label_available_at as predictive input. Entity histories
must be event-specific snapshots, never full-period aggregates. All original
bank event gaps are documented as fixed daily increments; entity revisit gaps
can still vary. Same-bank temporal tests do not establish unseen-bank transfer.

## Experimental blockers
Neural dependencies initially unavailable; installation underway. No existing
trainable architecture can be resumed. Public-data and privacy claims remain
unsupported. Compute feasibility requires profiling rather than assumed limits.

## Proposed execution plan
1. Verify hashes, supports, ordering, and partitions for all immutable streams.
2. Implement causal entity-history snapshots and tested message-passing models.
3. Train compact centralized models and matched component ablations, retaining
   validation-selected checkpoints and independent novelty scores.
4. Compare simulated four-client FedAvg/FedProx/SCAFFOLD and median aggregation.
5. Execute matched attacks if feasible; keep other incomplete modules explicit.
6. Generate machine-readable records, tables, plots, and manuscript updates from
   measured outputs only. Document the implemented finite-history architecture
   as an operationalization, not validation of every conceptual module.
