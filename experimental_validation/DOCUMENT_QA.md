# Document and figure QA

Status: DOCUMENT LAYOUT UNVERIFIED — RENDERER BLOCKED.

The edited DOCX is preserved. It was generated with python-docx from a new copy of
the supplied source, with paragraph edits logged in MANUSCRIPT_CHANGELOG.json and
six tables plus one measured figure generated from verified results. ZIP integrity
and document XML structure are checked separately from visual pagination.

The packaged LibreOffice renderer could not run because soffice was unavailable.
Native Word automation opened the manuscript read-only, but repagination, direct
PDF export, and SaveAs2 PDF conversion stalled. The task-owned automation processes
were stopped. Downloading a workspace-local LibreOffice fallback failed through
multiple official mirrors and transports. No PDF or page images are presented as
a verified manuscript render. Manual layout review remains necessary before use
as a publication-ready manuscript.

Fourteen measured figures were generated as SVG, PDF, and PNG. Their overview was
visually inspected for readable labels, legends, and overlap. All plotted numbers
come from the machine-readable tables. This figure inspection does not verify the
DOCX's pagination, captions, table flow, or original conceptual figures.

To finish document QA on a machine with Word, use render_final.ps1. Alternatively,
convert the DOCX with LibreOffice and inspect every rendered page. Do not interpret
the requested filename ending in Complete as proof of study or layout completion.
