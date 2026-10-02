"""Windows console LibreOffice export and faithful PDFium page rasterization."""
from pathlib import Path
import json
import subprocess
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
DOC = ROOT / 'CausalFed_TGNN_Experimental_Validation_Complete.docx'
OUT = ROOT / 'qa_final'
OUT.mkdir(exist_ok=True)
PROFILE = ROOT / 'renderer_profile_console'
command = [str(ROOT / 'renderer_local/program/soffice.com'),
    '-env:UserInstallation=' + PROFILE.as_uri(), '--headless', '--norestore',
    '--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(OUT), str(DOC)]
print('Exporting manuscript with LibreOffice console frontend', flush=True)
result = subprocess.run(command, capture_output=True, text=True, timeout=300)
print(result.stdout, flush=True)
print(result.stderr, flush=True)
assert result.returncode == 0
pdf_path = OUT / (DOC.stem + '.pdf')
assert pdf_path.exists() and pdf_path.stat().st_size > 0
pdf = pdfium.PdfDocument(pdf_path)
texts = []
for i in range(len(pdf)):
    page = pdf[i]
    page.render(scale=2).to_pil().save(OUT / f'page-{i+1}.png')
    texts.append(dict(page=i+1, text=page.get_textpage().get_text_range()))
    page.close()
(OUT / 'page_text.json').write_text(json.dumps(texts, indent=2), encoding='utf-8')
print(f'Rendered {len(pdf)} manuscript pages', flush=True)
