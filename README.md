# CausalFed TGNN

Reproducible finite-history graph-learning screening study for synthetic financial
fraud detection. **Study status: partial.** The broader conceptual framework is
not experimentally complete.

## Recorded evidence

- 15 immutable synthetic streams: 50k, 100k, 500k events; seeds 42–46; four banks.
- 180 centralized neural runs, including MLP, history-matched MLP, GraphSAGE,
  temporal GraphSAGE, HGT, a decomposed full variant, and matched ablations.
- 20 clean four-client simulations: FedAvg, FedProx, SCAFFOLD, coordinate median.
- 30 matched label-flipping, update-poisoning, and Sybil simulations.
- 30 historical logistic fits independently hash checked and reconstructed.
- All 230 new runs verified from saved source/config/checkpoint/prediction/log
  hashes and reconstructed metrics. Fourteen software tests passed.

The full variant's known-class AUROC at 500k is **0.5061 ± 0.0137** (five-seed mean
and sample SD). These weak scores are retained. This repository makes no claim of
predictive superiority, causal identification, privacy, or real-world validation.
Eight epochs/rounds constitute a fixed screening budget without convergence claims.

## Artifacts

- [Updated manuscript](experimental_validation/CausalFed_TGNN_Experimental_Validation_Complete.docx)
- [Canonical manifest](experimental_validation/RESULTS_MANIFEST.json)
- [Experiment status](experimental_validation/EXPERIMENT_STATUS.md)
- [Measured results](experimental_validation/RESULTS_SUMMARY.md)
- [Exact methods](experimental_validation/EXPERIMENTAL_METHODS.md)
- [Tests](experimental_validation/TEST_REPORT.md)
- [Document QA](experimental_validation/DOCUMENT_QA.md)
- [Limitations](experimental_validation/LIMITATIONS_AND_OPEN_ITEMS.md)
- [Tables](experimental_validation/results/tables)
- [Figures](experimental_validation/results/figures)

The filename ending in Complete is the requested artifact name, not a completion
claim. Public-data empirical validation, continual adaptation, realistic variable
arrival stress tests, and unseen-bank transfer remain open.
Manuscript layout passed a full rendered-page review; hashes and page count are
recorded in experimental_validation/DOCUMENT_QA.json.

## Verify or reproduce

Use Windows, PowerShell, and Python 3.12. Recorded paths use Windows separators.
Install the pinned dependencies into a virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r experimental_validation/requirements-lock.txt
gh release download evidence-v1 --repo Al1ranger/CausalFed-TGNN --pattern "causalfed-evidence-v1.zip.chunk*" --dir evidence
.venv/Scripts/python experimental_validation/restore_evidence.py
.venv/Scripts/python experimental_validation/verify_results.py
.venv/Scripts/python experimental_validation/audit_historical.py
.venv/Scripts/python -m pytest experimental_validation/tests -q
.venv/Scripts/python experimental_validation/report.py
```

The evidence archive contains the immutable inputs and all neural/historical
predictions. Large transfers failed, so 197 verified 2 MB release chunks are used.
The public [evidence-v1 release](https://github.com/Al1ranger/CausalFed-TGNN/releases/tag/evidence-v1)
contains all chunks; every GitHub SHA-256 digest matches its local hash.
restore_evidence.py checks chunk/archive/member hashes and restores all paths.
Archive/member hashes are in RELEASE_ASSET_MANIFEST.json; chunk hashes are in
evidence/PARTS_MANIFEST.json. Publication verification is recorded in
experimental_validation/GITHUB_PUBLICATION.json.
Checkpoints, source snapshots, run records, plots, and tables are tracked in Git.
Dependency binaries and temporary document-render pages are excluded.

For **new independent fits**, use a fresh sibling copy without results/runs or
worker run records; the runner otherwise resumes by skipping existing IDs. Exact
commands and recorded optimization/threshold choices are in
[REPRODUCTION.md](experimental_validation/REPRODUCTION.md).

The original raw files and historical manifests are preserved byte for byte.
Historical absolute host paths in provenance records describe where the study ran;
current artifact paths are repository relative. No credentials are included.
