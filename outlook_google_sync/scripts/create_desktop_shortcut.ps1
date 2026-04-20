<#
.SYNOPSIS
    Outlook → Google カレンダー同期ツールのデスクトップショートカットを作成する。

.DESCRIPTION
    本スクリプトは ``scripts\run_gui.bat`` を起動するショートカットを
    現在のユーザーのデスクトップ上に生成する。ソース直実行のため、
    コードを変更すれば次回起動から即反映され、再ビルドは不要。

.NOTES
    実行例（PowerShell）:
        PowerShell -ExecutionPolicy Bypass -File .\scripts\create_desktop_shortcut.ps1

    ``-Name`` や ``-IconPath`` を指定すると表示名・アイコンを変更できる。
#>

[CmdletBinding()]
param(
    [string] $Name     = 'Outlook→Google 同期',
    [string] $IconPath = ''
)

$ErrorActionPreference = 'Stop'

$root    = Split-Path -Parent $PSScriptRoot
$target  = Join-Path $root 'scripts\run_gui.bat'
$workdir = $root

if (-not (Test-Path -LiteralPath $target)) {
    throw "ターゲットが見つかりません: $target"
}

$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop ("{0}.lnk" -f $Name)

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnkPath)
$sc.TargetPath       = $target
$sc.WorkingDirectory = $workdir
$sc.WindowStyle      = 7   # 7 = 最小化で起動（黒い cmd 窓が邪魔にならない）
$sc.Description      = 'Outlook → Google カレンダー同期 GUI'

if ($IconPath -and (Test-Path -LiteralPath $IconPath)) {
    $sc.IconLocation = (Resolve-Path -LiteralPath $IconPath).Path
}

$sc.Save()

Write-Host ("ショートカットを作成しました: {0}" -f $lnkPath)
Write-Host ("  TargetPath       = {0}" -f $target)
Write-Host ("  WorkingDirectory = {0}" -f $workdir)
