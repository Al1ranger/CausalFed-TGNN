"""Verify copied artifact hashes, release member hashes, and DOCX structure."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'CausalFed-TGNN-GitHub'

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def main():
    for name in ['finalize_evidence.py', 'REPRODUCTION.md', 'DOCUMENT_QA.md',
                 'TEST_REPORT.md', 'DELIVERY_INDEX.json', 'README.md',
                 'render_final.ps1', 'package_github.py', 'requirements-lock.txt',
                 'validate_publication.py']:
        shutil.copy2(ROOT / name, REPO / 'experimental_validation' / name)
    doc = ROOT / 'CausalFed_TGNN_Experimental_Validation_Complete.docx'
    with zipfile.ZipFile(doc) as z:
        assert z.testzip() is None
        xml = ET.fromstring(z.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    assert len(xml.findall('.//w:tbl', ns)) == 18
    runs = json.loads((ROOT / 'RESULTS_MANIFEST.json').read_text())['runs']
    for r in runs:
        for name, key in [('checkpoint', 'checkpoint_hash'), ('log', 'log_hash')]:
            assert sha(REPO / 'experimental_validation' / r[name]) == r[key]
        source = REPO / 'experimental_validation' / r['source_snapshot']
        original = ROOT / r['source_snapshot']
        for path in original.glob('*.py'):
            assert sha(source / path.name) == sha(path)
    asset = json.loads((REPO / 'RELEASE_ASSET_MANIFEST.json').read_text())
    for member in asset['members']:
        member['path'] = member['path'].replace('\\', '/')
    (REPO / 'RELEASE_ASSET_MANIFEST.json').write_text(json.dumps(asset, indent=2))
    archive = ROOT.parent / 'github_release_assets' / asset['filename']
    assert sha(archive) == asset['sha256']
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist()) == len(asset['members']) == 275
        for member in asset['members']:
            with z.open(member['path']) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == member['sha256']
        names = set(z.namelist())
        for r in runs:
            assert 'experimental_validation/' + r['predictions'].replace('\\', '/') in names
    result = dict(status='PASS', copied_runs=len(runs), release_members=275,
                  docx_zip_integrity='PASS', docx_xml_tables=18,
                  document_layout='UNVERIFIED: renderer blocked',
                  all_release_member_sha256='PASS', archive_sha256=asset['sha256'])
    (REPO / 'PUBLISH_CHECK.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
