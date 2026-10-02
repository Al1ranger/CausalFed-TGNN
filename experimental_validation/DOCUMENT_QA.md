# Document and figure QA

Status: PASS — final rendered pages visually reviewed. Exact hashes and page count
are recorded in DOCUMENT_QA.json.

The edited DOCX is preserved. It was generated with python-docx from a new copy of
the supplied source, with paragraph edits logged in MANUSCRIPT_CHANGELOG.json and
six tables plus one measured figure generated from verified results. ZIP integrity
and document XML structure are checked separately from visual pagination.

Initial Word export and renderer downloads failed. A subsequent range download of
the official LibreOffice 26.2.6 installer passed its published SHA-256 check. The
workspace-local console frontend exported the edited DOCX to PDF; PDFium rendered
every page at twice its native scale for visual inspection. No global installation
was required. The source document remains unchanged.

Review corrected the orphaned section 5.4 heading by moving its page break from
the Table 3 caption to the heading. New measured tables use consistent borders,
header shading, alternating rows, alignment, and repeated headers. Final pages
were inspected for clipped text, overlaps, caption placement, and table flow.

Fourteen measured figures were generated as SVG, PDF, and PNG. Their overview was
visually inspected for readable labels, legends, and overlap. All plotted numbers
come from the machine-readable tables. The separate final manuscript review also
checked pagination, captions, table flow, and the retained conceptual figures.

The executed fallback is render_pdfium.py. Temporary PDF/page images and renderer
binaries are excluded from delivery. The requested filename ending in Complete
does not establish scientific completion: the broader screening study is partial.
