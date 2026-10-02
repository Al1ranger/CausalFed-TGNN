"""Restore verified raw inputs and predictions from repository archive parts."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def main():
    manifest = json.loads((ROOT / 'RELEASE_ASSET_MANIFEST.json').read_text())
    archive = ROOT / manifest['filename']
    if not archive.exists() or sha(archive) != manifest['sha256']:
        parts = json.loads((ROOT / 'evidence/PARTS_MANIFEST.json').read_text())
        assert parts['archive_sha256'] == manifest['sha256']
        with archive.open('wb') as target:
            for part in parts['parts']:
                source = ROOT / part['path']
                assert source.stat().st_size == part['bytes']
                assert sha(source) == part['sha256'], source
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, target, 1048576)
    assert sha(archive) == manifest['sha256']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {m['path'] for m in manifest['members']}
        for member in manifest['members']:
            target = (ROOT / member['path']).resolve()
            assert target.is_relative_to(ROOT.resolve())
            if target.exists() and sha(target) == member['sha256']:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member['path']) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 1048576)
            assert sha(target) == member['sha256'], target
    print(f'PASS: restored {len(manifest["members"])} verified evidence files')

if __name__ == '__main__':
    main()
