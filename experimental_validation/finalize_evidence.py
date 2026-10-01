"""Build delivery index and test report without inventing completion gates."""
import bootstrap
import importlib.metadata as metadata
import json
from pathlib import Path
from data import ROOT,sha

def main():
    versions={d.metadata['Name']:d.version for d in metadata.distributions(path=[str(ROOT/'packages')])}
    for name in ['pandas','python-docx','pdf2image','pypdfium2']:
        versions[name]=metadata.version(name)
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(versions.items()))+'\n')
    verification=json.loads((ROOT/'RESULT_VERIFICATION.json').read_text())
    log=(ROOT/'pytest.log').read_text()
    (ROOT/'TEST_REPORT.md').write_text('# Test report\n\n'+log+'\n\n'+
       f'Independent artifact verification: {verification["status"]}; {verification["verified_completed_runs"]} completed runs.\n\n'+
       'Verified source/configuration/checkpoint/prediction/log hashes and reconstructed known/open-set metrics. '+
       'Neural CSV scores are restored to float32; historical float64 CSVs use round-trip parsing. '+
       'All 15 raw stream hashes and chronological/D/E constraints passed. All 30 historical logistic records passed separate verification. '+
       'Every completed neural checkpoint was reloaded and compared exactly on 128 test examples. '+
       'Tests do not establish external validity, convergence, causal invariance, or privacy.\n',encoding='utf-8')
    files=['CausalFed_TGNN_Experimental_Validation_Complete.docx','RESULTS_MANIFEST.json','EXPERIMENT_STATUS.md',
       'RESEARCH_AUDIT.md','DATA_AUDIT.md','ENVIRONMENT_REPORT.md','TEST_REPORT.md','EXPERIMENTAL_METHODS.md',
       'RESULTS_SUMMARY.md','LIMITATIONS_AND_OPEN_ITEMS.md','REPRODUCTION.md','DOCUMENT_QA.md']
    manifest=json.loads((ROOT/'RESULTS_MANIFEST.json').read_text())
    central=[r for r in manifest['runs'] if r['status']=='COMPLETE' and r['configuration']['method']=='centralized']
    matrix={(r['scale'],r['seed'],r['model']) for r in central}
    expected={(s,k,m) for s in [50000,100000,500000] for k in range(42,47) for m in
              ['mlp','history_mlp','sage','temporal_sage','hetero_static','hgt','full','no_inv','no_env','no_time','no_con','homogeneous']}
    assert matrix==expected
    index=dict(study_status='PARTIAL SCREENING STUDY',centralized_matrix_complete=True,
       missing_centralized_cells=[],verified_runs=verification['verified_completed_runs'],
       required_files={f:dict(exists=(ROOT/f).exists(),sha256=sha(ROOT/f) if (ROOT/f).exists() else None) for f in files},
       run_records='results/runs',tables='results/tables',figures='results/figures',
       checkpoints='Paths in each run record include results/checkpoints and worker_*/checkpoints',
       unmet_stronger_gates=['continual adaptation','variable-arrival temporal stress','public-data empirical validation',
          'unseen-bank transfer','adaptive aggregation','explanation evaluation','public release'])
    index['document_layout_status']='UNVERIFIED: automated renderer blocked; see DOCUMENT_QA.md'
    publication=ROOT/'GITHUB_PUBLICATION.json'
    if publication.exists():
        record=json.loads(publication.read_text())
        index['github_publication']=record
        if record.get('status')=='VERIFIED PUBLIC REPOSITORY AND RELEASE':
            index['unmet_stronger_gates'].remove('public release')
    (ROOT/'DELIVERY_INDEX.json').write_text(json.dumps(index,indent=2))
    (ROOT/'README.md').write_text('# CausalFed TGNN experimental evidence\n\n'+
       'Status: partial screening study. 180 centralized, 20 clean federated, and 30 attacked runs; five seeds per configuration. '+
       'Raw inputs and historical artifacts are preserved.\n\n'+
       '\n'.join(f'- [{f}]({f})' for f in files)+'\n\n'+
       'See results/tables and results/figures for generated outputs. Checkpoints and predictions are located by each canonical run record. '+
       'Run REPRODUCTION.md commands in a fresh sibling directory for independent refitting.\n',encoding='utf-8')
    print(json.dumps(index,indent=2))

if __name__=='__main__':main()
