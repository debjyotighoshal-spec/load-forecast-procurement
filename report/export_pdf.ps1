# Export a DOCX or PPTX to PDF with Microsoft Office.
# Office may open files under some folders read-only, so the export works on a
# copy in %TEMP% and the PDF is copied back next to the source file.
param([Parameter(Mandatory = $true)][string]$Src)
$Src = (Resolve-Path $Src).Path
$Pdf = [IO.Path]::ChangeExtension($Src, ".pdf")
$work = Join-Path $env:TEMP ("lfp_export_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $work | Out-Null
$tmpSrc = Join-Path $work ([IO.Path]::GetFileName($Src))
$tmpPdf = [IO.Path]::ChangeExtension($tmpSrc, ".pdf")
Copy-Item $Src $tmpSrc -Force
if (Test-Path $tmpPdf) { Remove-Item $tmpPdf -Force }
if ($Src -like "*.docx") {
    $app = New-Object -ComObject Word.Application
    $app.Visible = $false
    $app.DisplayAlerts = 0
    try {
        $doc = $app.Documents.Open($tmpSrc, $false, $true, $false)
        $doc.ExportAsFixedFormat($tmpPdf, 17)     # wdExportFormatPDF
        $doc.Close(0)
    } finally { $app.Quit() }
} else {
    $app = New-Object -ComObject PowerPoint.Application
    try {
        # FileName, ReadOnly, Untitled, WithWindow
        $pres = $app.Presentations.Open($tmpSrc, -1, 0, 0)
        $pres.SaveAs($tmpPdf, 32)                 # ppSaveAsPDF
        $pres.Close()
    } finally { $app.Quit() }
}
Copy-Item $tmpPdf $Pdf -Force -ErrorAction Stop
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
"saved $Pdf"
