<#
  Build the Chinese edition PDFs of the RLHF Book.

  Usage (run from the repository root, rlhf-book/):
    .\zh\build.ps1          # build every translated chapter
    .\zh\build.ps1 01       # build only chapter 01
    .\zh\build.ps1 -Book    # assemble all chapters into one book PDF
    .\zh\build.ps1 -Docx    # assemble all chapters into one Word (.docx)

  Differences from the English build:
    - xelatex instead of pdflatex (required for CJK)
    - zh/metadata.yml sets the Chinese title and SimSun/SimHei/KaiTi fonts
    - book/templates/pdf.tex already ships a xeCJK branch, so no template edits

  Messages are ASCII on purpose: Windows PowerShell 5.1 reads .ps1 files as
  ANSI unless they carry a BOM, which garbles non-ASCII string literals.
#>

param(
    [string]$Chapter = "",
    [switch]$Book,
    [switch]$Docx
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$crossref = "build\_tools\pandoc-crossref.exe"
$strip    = "build\_tools\strip-labels.lua"

if (-not (Test-Path $crossref)) { throw "missing $crossref" }

$srcDir = "zh\chapters"
$outDir = "zh\build"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# ---------------------------------------------------------------- book mode
if ($Book) {
    $parts = Get-ChildItem "$srcDir\*.md" | Sort-Object Name
    if (-not $parts) { throw "no chapters found to assemble" }

    $joined = Join-Path $outDir "_book-joined.md"
    $sb = New-Object System.Text.StringBuilder
    foreach ($p in $parts) {
        [void]$sb.AppendLine((Get-Content -LiteralPath $p.FullName -Raw -Encoding UTF8))
        [void]$sb.AppendLine("")
    }
    [System.IO.File]::WriteAllText((Resolve-Path $outDir).Path + "\_book-joined.md",
                                   $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))

    $out = Join-Path $outDir "rlhf-book-zh.pdf"
    Write-Host ("assembling {0} chapters ..." -f $parts.Count) -NoNewline
    $sw = [Diagnostics.Stopwatch]::StartNew()

    $log = & pandoc $joined `
        --toc --toc-depth 2 `
        --metadata-file zh/metadata.yml `
        --filter $crossref `
        --lua-filter $strip `
        --bibliography=book/chapters/bib.bib --citeproc --csl=book/templates/ieee.csl `
        --metadata "date=2026-09" `
        --template book/templates/pdf.tex `
        --pdf-engine xelatex `
        --resource-path=book `
        -o $out 2>&1

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) {
        Write-Host " FAILED" -ForegroundColor Red
        $log | Where-Object { $_ -match "rror|undefined|^! " } | Select-Object -Last 15 |
            ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
        exit 1
    }
    $kb = [math]::Round((Get-Item $out).Length / 1KB)
    Write-Host (" ok  {0} KB  {1:N1}s" -f $kb, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- docx mode
if ($Docx) {
    $parts = Get-ChildItem "$srcDir\*.md" | Sort-Object Name
    if (-not $parts) { throw "no chapters found to assemble" }

    $joined = Join-Path $outDir "_book-joined-docx.md"
    $sb = New-Object System.Text.StringBuilder
    foreach ($p in $parts) {
        [void]$sb.AppendLine((Get-Content -LiteralPath $p.FullName -Raw -Encoding UTF8))
        [void]$sb.AppendLine("")
    }
    [System.IO.File]::WriteAllText((Resolve-Path $outDir).Path + "\_book-joined-docx.md",
                                   $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))

    $ref = "zh\templates\docx-zh.docx"
    if (-not (Test-Path $ref)) {
        Write-Host "缺少 $ref，正在生成 ..." -ForegroundColor DarkYellow
        & python zh\tools\make-docx-reference.py
    }

    $out = Join-Path $outDir "rlhf-book-zh.docx"
    Write-Host ("assembling {0} chapters -> docx ..." -f $parts.Count) -NoNewline
    $sw = [Diagnostics.Stopwatch]::StartNew()

    $log = & pandoc $joined `
        --toc --toc-depth 2 `
        --metadata-file zh/metadata.yml `
        --metadata-file zh/metadata-docx.yml `
        --filter $crossref `
        --lua-filter $strip `
        --bibliography=book/chapters/bib.bib --citeproc --csl=book/templates/ieee.csl `
        --metadata "date=2026-09" `
        --reference-doc $ref `
        --resource-path=book `
        -o $out 2>&1

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) {
        Write-Host " FAILED" -ForegroundColor Red
        $log | Where-Object { $_ -match "rror|undefined|^! " } | Select-Object -Last 15 |
            ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
        exit 1
    }
    $kb = [math]::Round((Get-Item $out).Length / 1KB)
    Write-Host (" ok  {0} KB  {1:N1}s" -f $kb, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    # pandoc 不会保留参考文档里的 sectPr，页脚需要在产物上补回来
    & python zh\tools\add-docx-footer.py $out
    exit 0
}

$files = if ($Chapter) {
    Get-ChildItem "$srcDir\$Chapter-*.md" | Sort-Object Name
} else {
    Get-ChildItem "$srcDir\*.md" | Sort-Object Name
}

if (-not $files) { throw "no chapters found to build" }

$failed = 0
foreach ($f in $files) {
    $out = Join-Path $outDir ($f.BaseName + ".pdf")
    Write-Host ("building {0} ..." -f $f.Name) -NoNewline
    $sw = [Diagnostics.Stopwatch]::StartNew()

    # Native stderr (pandoc warnings) must not be treated as a terminating error,
    # so keep the preference at Continue and judge success by $LASTEXITCODE.
    $log = & pandoc $f.FullName `
        --metadata-file zh/metadata.yml `
        --filter $crossref `
        --lua-filter $strip `
        --bibliography=book/chapters/bib.bib --citeproc --csl=book/templates/ieee.csl `
        --metadata "date=2026-09" `
        --template book/templates/pdf.tex `
        --pdf-engine xelatex `
        --resource-path=book `
        -o $out 2>&1

    $problems = $log | Where-Object { $_ -match "rror|undefined|^! " }

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) {
        Write-Host " FAILED" -ForegroundColor Red
        $problems | Select-Object -Last 12 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
        $failed++
    } else {
        $kb = [math]::Round((Get-Item $out).Length / 1KB)
        Write-Host (" ok  {0} KB  {1:N1}s" -f $kb, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
        if ($problems) { $problems | Select-Object -First 3 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow } }
    }
}

if ($failed -gt 0) { exit 1 }
