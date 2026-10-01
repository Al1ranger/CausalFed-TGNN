# Reproduction

Run from PowerShell in the repository root. Required immutable input CSVs and
their historical manifest remain in `CausalFed-TGNN-V4\runs\v4_scale`.
For the public repository, first restore the evidence release and use Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r experimental_validation/requirements-lock.txt
gh release download evidence-v1 --repo Al1ranger/CausalFed-TGNN --pattern "causalfed-evidence-v1.zip" --dir .
.venv/Scripts/python -m zipfile -e causalfed-evidence-v1.zip .
$researchPython = (Resolve-Path .venv/Scripts/python.exe).Path
```

The study host used the bundled runtime and project-local packages below. On a
fresh clone, keep `$researchPython` set to the virtual environment and omit the
host-specific package install and PYTHONPATH lines.

```powershell
$researchPython = 'C:\Users\ehsan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $researchPython -m pip install --target experimental_validation/packages -r experimental_validation/requirements.txt
& $researchPython -X utf8 experimental_validation/data.py
$env:PYTHONPATH = 'D:\Study\Papers\p7\experimental_validation\packages'
& $researchPython -m pytest experimental_validation/tests -q
& $researchPython -X utf8 experimental_validation/run.py --scales 50000 100000 500000 --epochs 8
& $researchPython -X utf8 experimental_validation/worker.py --scale 50000 --seeds 42 43 44 45 46 --models full --methods fedavg fedprox scaffold median --name fed_clean
& $researchPython -X utf8 experimental_validation/worker.py --scale 50000 --seeds 42 43 44 45 46 --models full --methods fedavg median --attacks label_flip poison sybil --name fed_attacks
& $researchPython -X utf8 experimental_validation/audit_historical.py
& $researchPython -X utf8 experimental_validation/verify_results.py
& $researchPython -X utf8 experimental_validation/report.py
Remove-Item Env:PYTHONPATH
& $researchPython -X utf8 experimental_validation/update_manuscript.py
```

Use `requirements-lock.txt` when available for transitive package pins.
The last command uses python-docx. It reads the repository's manuscript source,
falling back to the supplied Downloads DOCX on the original study host, and
creates a new DOCX under experimental_validation; it never overwrites the input.
Document rendering is described in `DOCUMENT_QA.md` after it is executed.

Existing run IDs are resumed by skipping recorded outputs; this does not refit
them. To independently refit without deleting evidence, create a new sibling
directory, copy only the top-level Python sources and tests, install dependencies
into its `packages` subdirectory, and substitute its name in the commands above.
Its parent must still contain the same immutable V4 and scientific_analysis input
directories. Do not copy old `results`, `worker_*`, or pilot artifacts into that
fresh directory. Match package versions and CPU thread count before comparing
checkpoints. Wall times and process-memory peaks will differ across runs.

The executed run was accelerated with disjoint 500k workers:

```powershell
& $researchPython -X utf8 experimental_validation/worker.py --scale 500000 --seeds 42 43 --name large_a
& $researchPython -X utf8 experimental_validation/worker.py --scale 500000 --seeds 44 45 46 --name large_b
& $researchPython -X utf8 experimental_validation/worker.py --scale 500000 --seeds 46 --name large_c
```

These worker files retain their checkpoints/predictions and publish completed JSON
records to the canonical results/runs folder. Their relative artifact paths are
authoritative; do not relocate one JSON file without its referenced artifacts.
`pilot_pre_freeze` contains initial exploratory runs, excluded from final summaries.

Raw saved logits/probabilities use floating-point representations. Read historical
float64 CSV predictions with `float_precision='round_trip'` when reconstructing
threshold decisions; the default fast CSV parser can shift a boundary by one
floating-point unit. No historical metric is silently changed for that parser
artifact. All verified comparisons and their tolerances are in the verifier code.
To refit the historical logistic baselines, copy scientific_analysis/run_analysis.py
into a fresh sibling directory (without its results), retain the same V4 inputs
in the parent, then run `& $researchPython -X utf8 sibling_name/run_analysis.py`.
Use a fresh sibling: the historical runner writes into its own results directory.

Neural score columns must be restored to their original float32 dtype before
threshold comparisons, as implemented by verify_results.py and report.py.
