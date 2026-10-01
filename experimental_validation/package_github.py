"""Publishable mirror and verified release archive; original evidence is untouched."""
import bootstrap
import json
import shutil
import zipfile
from pathlib import Path
from data import ROOT,sha

DEST=ROOT.parent/'CausalFed-TGNN-GitHub'
ASSETS=ROOT.parent/'github_release_assets'

def copy(path,relative):
    target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)

def main():
    DEST.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.txt','.docx','.ps1','.log'] and not p.name.startswith(('download_renderer','renderer_')):
            copy(p,Path('experimental_validation')/p.name)
    manifest=json.loads((ROOT/'RESULTS_MANIFEST.json').read_text())
    for folder in ['tests','results/runs','results/tables','results/figures']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                copy(p,Path('experimental_validation')/p.relative_to(ROOT))
    archive_inputs={}
    for r in manifest['runs']:
        assert r['status']=='COMPLETE'
        for key in ['checkpoint','log']:
            path=ROOT/r[key];copy(path,Path('experimental_validation')/r[key])
        path=ROOT/r['predictions']
        archive_inputs['experimental_validation/'+r['predictions'].replace('\\','/')]=path
        for path in (ROOT/r['source_snapshot']).glob('*.py'):
            copy(path,Path('experimental_validation')/path.relative_to(ROOT))
    v4=ROOT.parent/'CausalFed-TGNN-V4'
    for p in (v4/'v4').glob('*.py'):copy(p,Path('CausalFed-TGNN-V4/v4')/p.name)
    for p in (v4/'runs/v4_scale').glob('*.json'):copy(p,Path('CausalFed-TGNN-V4/runs/v4_scale')/p.name)
    source=json.loads((v4/'runs/v4_scale/RESULTS_MANIFEST.json').read_text())
    for r in source['results']:
        path=v4/f'runs/v4_scale/synthetic_{r["scale"]}_seed{r["seed"]}.csv'
        assert sha(path)==r['csv_sha256']
        archive_inputs[path.relative_to(ROOT.parent).as_posix()]=path
    old=ROOT.parent/'scientific_analysis'
    copy(old/'run_analysis.py',Path('scientific_analysis/run_analysis.py'))
    copy(old/'results/ANALYSIS_MANIFEST.json',Path('scientific_analysis/results/ANALYSIS_MANIFEST.json'))
    historical=json.loads((old/'results/ANALYSIS_MANIFEST.json').read_text())
    for name in historical['result_hashes']:
        path=old/'results'/name
        assert sha(path)==historical['result_hashes'][name]
        if name.startswith('predictions_'):
            archive_inputs[path.relative_to(ROOT.parent).as_posix()]=path
        else:copy(path,path.relative_to(ROOT.parent))
    for r in historical['records']:
        for item in r['models'].values():
            p=old/'results'/item['file'];copy(p,p.relative_to(ROOT.parent))
    original=Path('C:/Users/ehsan/Downloads/CausalFed_TGNN_Scientific_Analysis_Edited_Final.docx')
    copy(original,Path('manuscript')/original.name)
    configurations={r['configuration_hash']:r['configuration'] for r in manifest['runs']}
    for key,config in configurations.items():
        target=DEST/'configs'/f'{key}.json';target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(config,indent=2))
    (DEST/'.gitignore').write_text('**/__pycache__/\n**/.pytest_cache/\n**/packages/\n.venv/\n*.zip\n'+
        'experimental_validation/qa_*/\nexperimental_validation/pilot_pre_freeze/\n'+
        'experimental_validation/**/predictions/\nCausalFed-TGNN-V4/runs/v4_scale/*.csv\nscientific_analysis/results/predictions_*.csv\n')
    (DEST/'README.md').write_text('''# CausalFed TGNN

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
- [Document QA limitation](experimental_validation/DOCUMENT_QA.md)
- [Limitations](experimental_validation/LIMITATIONS_AND_OPEN_ITEMS.md)
- [Tables](experimental_validation/results/tables)
- [Figures](experimental_validation/results/figures)

The filename ending in Complete is the requested artifact name, not a completion
claim. Public-data empirical validation, continual adaptation, realistic variable
arrival stress tests, and unseen-bank transfer remain open.
Manuscript layout is unverified: automated Word export stalled and the LibreOffice
fallback could not be obtained. Inspect pagination before publication use.

## Verify or reproduce

Use Python 3.12. Install the pinned dependencies into a virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r experimental_validation/requirements-lock.txt
gh release download evidence-v1 --repo Al1ranger/CausalFed-TGNN --pattern "causalfed-evidence-v1.zip" --dir .
.venv/Scripts/python -m zipfile -e causalfed-evidence-v1.zip .
.venv/Scripts/python experimental_validation/verify_results.py
.venv/Scripts/python experimental_validation/audit_historical.py
.venv/Scripts/python -m pytest experimental_validation/tests -q
.venv/Scripts/python experimental_validation/report.py
```

The release archive contains the immutable inputs and all neural/historical
predictions. Extract it into the repository root to restore every referenced path.
Its SHA-256 and member hashes are recorded in RELEASE_ASSET_MANIFEST.json.
Checkpoints, source snapshots, run records, plots, and tables are tracked in Git.
Dependency binaries and temporary document-render pages are excluded.

For **new independent fits**, use a fresh sibling copy without results/runs or
worker run records; the runner otherwise resumes by skipping existing IDs. Exact
commands and recorded optimization/threshold choices are in
[REPRODUCTION.md](experimental_validation/REPRODUCTION.md).

The original raw files and historical manifests are preserved byte for byte.
Historical absolute host paths in provenance records describe where the study ran;
current artifact paths are repository relative. No credentials are included.
''',encoding='utf-8')
    archive=ASSETS/'causalfed-evidence-v1.zip'
    members=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,path in sorted(archive_inputs.items()):
            z.write(path,name);members.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    asset=dict(filename=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),members=members,
       scope='15 immutable synthetic streams, 230 neural prediction CSVs, and 30 historical prediction CSVs')
    (DEST/'RELEASE_ASSET_MANIFEST.json').write_text(json.dumps(asset,indent=2))
    (ASSETS/'SHA256SUMS.txt').write_text(f'{asset["sha256"]}  {archive.name}\n')
    (DEST/'PUBLISH_CHECK.json').write_text(json.dumps(dict(files_copied=True,release_members=len(members),
       zip_crc_check='PASS',git_artifact_scope='source, documents, manifests, checkpoints, logs, tables, figures',
       archive_bytes=asset['bytes'],archive_sha256=asset['sha256']),indent=2))
    print(json.dumps(dict(repository=str(DEST),archive=str(archive),archive_bytes=asset['bytes'],members=len(members))))

if __name__=='__main__':main()
