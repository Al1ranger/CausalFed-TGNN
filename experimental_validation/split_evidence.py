"""Prepare small Git-compatible archive parts without changing recorded evidence."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'CausalFed-TGNN-GitHub'

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def main():
    manifest = json.loads((REPO / 'RELEASE_ASSET_MANIFEST.json').read_text())
    archive = ROOT.parent / 'github_release_assets' / manifest['filename']
    assert sha(archive) == manifest['sha256']
    folder = ROOT.parent / 'github_release_assets/evidence_chunks'
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    with archive.open('rb') as source:
        while data := source.read(2 * 1024 * 1024):
            target = folder / f'{manifest["filename"]}.chunk{len(parts)+1:03d}'
            target.write_bytes(data)
            parts.append(dict(path='evidence/' + target.name,
                              bytes=len(data), sha256=sha(target)))
    assert sum(p['bytes'] for p in parts) == manifest['bytes']
    result = dict(archive_sha256=manifest['sha256'], archive_bytes=manifest['bytes'],
                  parts=parts, reason='Large transfers failed; 2 MB release chunks are used')
    (folder / 'PARTS_MANIFEST.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(folder / 'PARTS_MANIFEST.json', REPO / 'evidence/PARTS_MANIFEST.json')
    for name in ['restore_evidence.py', 'split_evidence.py', 'REPRODUCTION.md']:
        shutil.copy2(ROOT / name, REPO / 'experimental_validation' / name)
    print(json.dumps(dict(parts=len(parts), bytes=manifest['bytes'])))

if __name__ == '__main__':
    main()
