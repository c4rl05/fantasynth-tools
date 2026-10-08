<#
.SYNOPSIS
  Create or refresh the music-events Python environments in the workspace.

.DESCRIPTION
  The code lives in the repo (music-events\). The venvs, models, audio, outputs and the
  per-track config.json live in the WORKSPACE folder outside it, resolved by workspace.py:
  MUSIC_EVENTS_WORKSPACE if set, else <main checkout>\..\..\music-events if that folder
  already exists, else <main checkout>\..\music-events (a sibling of the clone).

  Creates, or refreshes when it already exists:
    <workspace>\venv-main   py -3.13, requirements-main.txt, CUDA 12.8 torch
    <workspace>\venv-bp     py -3.10, requirements-bp.txt, Basic Pitch on ONNX
  then checks that torch sees CUDA and that basic_pitch + onnxruntime import.

  Idempotent: an existing venv of the right Python version is reused and its packages are
  brought to the pinned versions. Nothing is ever deleted: a venv with the wrong Python
  version stops the script, and audio\, out\ and models\ are only created when missing.

.PARAMETER DryRun
  Print every command instead of running it. Nothing is created or installed.
  (-WhatIf does the same.)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File music-events\setup.ps1 -DryRun

.EXAMPLE
  $env:MUSIC_EVENTS_WORKSPACE = 'D:\scratch\ae'; powershell -ExecutionPolicy Bypass -File music-events\setup.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
if ($WhatIfPreference) { $DryRun = $true }

$ToolDir = $PSScriptRoot
$Cu128 = 'https://download.pytorch.org/whl/cu128'
$ReqMain = Join-Path $ToolDir 'requirements-main.txt'
$ReqBp = Join-Path $ToolDir 'requirements-bp.txt'
$Constraints = Join-Path $ToolDir 'constraints-main.txt'

function Format-Command {
    param([string]$Exe, [string[]]$Arguments)
    $parts = @($Exe) + $Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    return ($parts -join ' ')
}

# Run a native command (or only print it under -DryRun). Output goes to the host, never into
# the caller's return value, and failure is judged by the exit code alone: pip writes warnings
# to stderr, which Windows PowerShell 5.1 would otherwise turn into terminating errors.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    $shown = Format-Command $Exe $Arguments
    if ($DryRun) {
        Write-Host "  [dry-run] $shown"
        return
    }
    Write-Host "  > $shown"
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE from: $shown" }
}

function New-DirIfMissing {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) { return }
    if ($DryRun) {
        Write-Host "  [dry-run] mkdir $Path"
        return
    }
    New-Item -ItemType Directory -Path $Path -WhatIf:$false | Out-Null
    Write-Host "  created $Path"
}

# Returns the venv's python.exe, creating the venv when it does not exist yet.
function Initialize-Venv {
    param([string]$Name, [string]$Version)
    $dir = Join-Path $Workspace $Name
    $py = Join-Path $dir 'Scripts\python.exe'
    if (Test-Path -LiteralPath $py) {
        $ErrorActionPreference = 'Continue'
        $have = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($LASTEXITCODE -ne 0) { throw "$py does not run. Move $dir aside by hand (this script never deletes) and re-run." }
        if ("$have".Trim() -ne $Version) {
            throw "$Name is Python $have, expected $Version. Move $dir aside by hand (this script never deletes) and re-run."
        }
        Write-Host "  $Name exists (Python $have): reusing it, refreshing packages"
    } else {
        Invoke-Native 'py' @("-$Version", '-m', 'venv', $dir)
    }
    return $py
}

Write-Host '== music-events setup'
if ($DryRun) { Write-Host '   DRY RUN: commands are printed, nothing is created or installed' }

# ---------------------------------------------------------------- prerequisites
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'The py launcher is missing. Install Python 3.13 and 3.10 from python.org (the launcher comes with them).'
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is missing: it is needed to find the workspace and to install adtof-pytorch from GitHub.'
}
foreach ($tool in 'ffmpeg', 'ffprobe') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Warning "$tool is not on PATH. run_all.py's decode stage, render_check.py and export_app.py --flac need it."
    }
}

# ---------------------------------------------------------------- workspace
# One rule, in one place: ask workspace.py (it honours MUSIC_EVENTS_WORKSPACE itself).
$ErrorActionPreference = 'Continue'
$Workspace = & py -3.13 -c 'import sys; sys.path.insert(0, sys.argv[1]); import workspace; print(workspace.WORKSPACE)' $ToolDir
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($code -ne 0 -or -not $Workspace) {
    throw "Could not resolve the workspace with py -3.13 and workspace.py (exit $code). Is Python 3.13 installed?"
}
$Workspace = "$Workspace".Trim()
Write-Host "   tool dir:  $ToolDir"
Write-Host "   workspace: $Workspace"

New-DirIfMissing $Workspace
foreach ($sub in 'audio', 'out', 'models') { New-DirIfMissing (Join-Path $Workspace $sub) }

# ---------------------------------------------------------------- venv-main
Write-Host '== venv-main (Python 3.13, CUDA torch)'
$PyMain = Initialize-Venv 'venv-main' '3.13'
Invoke-Native $PyMain @('-m', 'pip', 'install', '--upgrade', 'pip')
# ALWAYS with the constraints and the cu128 index: without them pip resolves torch from
# PyPI, which is CPU-only on Windows.
Invoke-Native $PyMain @('-m', 'pip', 'install', '-r', $ReqMain, '-c', $Constraints, '--extra-index-url', $Cu128)

Write-Host '   checking CUDA'
Invoke-Native $PyMain @('-c', "import sys, torch; ok = torch.cuda.is_available(); print('   torch', torch.__version__, '| CUDA', torch.version.cuda, '| available', ok); ok or sys.exit('CUDA NOT AVAILABLE in venv-main (torch ' + torch.__version__ + '). A version without +cu128 means a CPU torch was swapped in: reinstall with -c constraints-main.txt and the cu128 index.'); print('   device', torch.cuda.get_device_name(0), '| capability sm_%d%d' % torch.cuda.get_device_capability(0))")

# ---------------------------------------------------------------- venv-bp
Write-Host '== venv-bp (Python 3.10, Basic Pitch on ONNX)'
$PyBp = Initialize-Venv 'venv-bp' '3.10'
Invoke-Native $PyBp @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Native $PyBp @('-m', 'pip', 'install', '-r', $ReqBp)

Write-Host '   checking basic_pitch + onnxruntime'
Invoke-Native $PyBp @('-c', "import sys, onnxruntime, basic_pitch; from basic_pitch import ICASSP_2022_MODEL_PATH as p; print('   onnxruntime', onnxruntime.__version__, '| model', p); str(p).endswith('.onnx') or sys.exit('basic_pitch did not pick its ONNX model')")

# ---------------------------------------------------------------- done
Write-Host ''
if ($DryRun) {
    Write-Host '== dry run finished: nothing was changed'
} else {
    Write-Host '== ready. Model weights download on first use (see README.md).'
}
$Config = Join-Path $Workspace 'config.json'
if (-not (Test-Path -LiteralPath $Config)) {
    Write-Host "   config: $Config does not exist yet. Copy $(Join-Path $ToolDir 'config.example.json') there"
    Write-Host '           and fill it in (app checkout, tracks) before --all or export_app.py.'
}
Write-Host "   run:    $(Join-Path $Workspace 'venv-main\Scripts\python.exe') music-events\run_all.py `"<track.mp3>`""
Write-Host "   viewer: $(Join-Path $Workspace 'venv-main\Scripts\python.exe') music-events\serve.py"
