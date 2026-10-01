$ErrorActionPreference = 'Stop'
$researchRoot = $PSScriptRoot
$researchSource = Join-Path $researchRoot 'CausalFed_TGNN_Experimental_Validation_Complete.docx'
$researchPdf = Join-Path $researchRoot 'qa_final\manuscript.pdf'
New-Item -ItemType Directory -Path (Split-Path -Parent $researchPdf) -Force | Out-Null
Write-Output 'Starting Word automation'
$researchWord = New-Object -ComObject Word.Application
Write-Output 'Word automation initialized'
$researchWord.Visible = $false
$researchWord.DisplayAlerts = 0
$researchWord.AutomationSecurity = 3
$researchWord.Options.UpdateLinksAtOpen = $false
$researchWord.Options.UpdateFieldsAtPrint = $false
$researchWord.Options.UpdateLinksAtPrint = $false
$researchWord.Options.Pagination = $false
$researchWord.ActivePrinter = 'Microsoft Print to PDF'
try {
    Write-Output 'Opening manuscript read-only'
    $researchPaper = $researchWord.Documents.Open($researchSource, $false, $true, $false)
    Write-Output 'Manuscript opened'
    $researchPaper.ShowRevisions = $false
    $researchPaper.PrintRevisions = $false
    Write-Output 'Exporting PDF directly'
    $researchPaper.ExportAsFixedFormat($researchPdf, 17, $false, 1)
    Write-Output 'PDF exported'
    $researchPaper.Close(0)
} finally { $researchWord.Quit() }
