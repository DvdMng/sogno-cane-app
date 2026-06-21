@echo off
setlocal
REM ===========================================================================
REM  SOGNO_CANE - Installer con un click.
REM  Scarica l'app completa su questo PC, attiva l'auto-aggiornamento e crea i
REM  collegamenti. Basta fare doppio click su questo file.
REM
REM  CONFIGURA QUI SOTTO l'indirizzo del tuo manifest (version.json):
REM ===========================================================================
if not defined MANIFEST_URL set "MANIFEST_URL=https://github.com/DvdMng/sogno-cane-app/releases/latest/download/version.json"
REM ===========================================================================

title SOGNO_CANE Setup
echo.
echo   SOGNO_CANE - installazione in corso...
echo.

REM Esegue la parte PowerShell che si trova in fondo a questo stesso file,
REM dopo la riga marcatore #__PSCODE__ . PowerShell legge il file, prende le
REM righe dopo il marcatore e le esegue (cosi' restano leggibili e modificabili).
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $f=Get-Content -LiteralPath '%~f0'; $i=[Array]::IndexOf($f,'#__PSCODE__'); if($i -lt 0){Write-Host 'marker non trovato'; exit 1}; $code=($f[($i+1)..($f.Count-1)] -join [Environment]::NewLine); Invoke-Expression $code"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo   Installazione NON riuscita ^(codice %RC%^). Controlla la connessione e
  echo   che MANIFEST_URL sia corretto, poi riprova.
  echo.
  pause
)
endlocal
exit /b %RC%

#__PSCODE__
# ---------------------------------------------------------------------------
# Installer PowerShell (eseguito dalla parte batch qui sopra).
# ---------------------------------------------------------------------------
$ProgressPreference = 'SilentlyContinue'
$manifest = $env:MANIFEST_URL
if (-not $manifest -or $manifest -like '*UTENTE/REPO*') {
    Write-Host 'ERRORE: imposta MANIFEST_URL nel file .bat con il tuo version.json.'
    exit 1
}
$installDir = if ($env:SOGNO_CANE_INSTALL_DIR) { $env:SOGNO_CANE_INSTALL_DIR } `
              else { Join-Path $env:LOCALAPPDATA 'SOGNO_CANE' }

Write-Host "Controllo versione: $manifest"
$info = Invoke-RestMethod -Uri $manifest -UseBasicParsing
$bundleUrl = if ($info.bundle_url) { $info.bundle_url } else { $env:BUNDLE_URL }
if (-not $bundleUrl) { Write-Host 'ERRORE: il manifest non contiene bundle_url.'; exit 1 }
Write-Host ("Versione disponibile: " + $info.version)

$zip = Join-Path $env:TEMP 'sogno_cane_bundle.zip'
Write-Host "Scarico l'app... ($bundleUrl)"
Invoke-WebRequest -Uri $bundleUrl -OutFile $zip -UseBasicParsing
if ($info.bundle_sha256) {
    $h = (Get-FileHash -Path $zip -Algorithm SHA256).Hash
    if ($h -ne ([string]$info.bundle_sha256).ToUpper()) {
        Write-Host 'ERRORE: checksum del download non valido (file corrotto?).'
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
        exit 1
    }
}

Write-Host "Installo in: $installDir"
if (Test-Path $installDir) { Remove-Item $installDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Expand-Archive -Path $zip -DestinationPath $installDir -Force
Remove-Item $zip -Force -ErrorAction SilentlyContinue

$start = Get-ChildItem -Path $installDir -Recurse -Filter 'START.bat' |
         Select-Object -First 1
if (-not $start) { Write-Host 'ERRORE: START.bat non trovato nel pacchetto.'; exit 1 }
$appDir = $start.Directory.FullName

# Attiva l'auto-aggiornamento scrivendo le impostazioni utente.
$homeDir = if ($env:SOGNO_CANE_HOME) { $env:SOGNO_CANE_HOME } `
           else { Join-Path $env:USERPROFILE '.sogno_cane' }
New-Item -ItemType Directory -Force -Path $homeDir | Out-Null
$settingsPath = Join-Path $homeDir 'settings.json'
$settings = @{}
if (Test-Path $settingsPath) {
    try {
        $obj = Get-Content $settingsPath -Raw | ConvertFrom-Json
        foreach ($p in $obj.PSObject.Properties) { $settings[$p.Name] = $p.Value }
    } catch {}
}
$settings['update_url'] = $manifest
$settings['auto_update'] = $true
# Scrive UTF-8 SENZA BOM (Set-Content -Encoding UTF8 di PowerShell 5.1
# aggiungerebbe un BOM che rompe il parsing JSON dell'app).
[System.IO.File]::WriteAllText($settingsPath, ($settings | ConvertTo-Json -Depth 8))
Write-Host 'Auto-aggiornamento attivato.'

# Collegamenti su Desktop e nel menu Start.
if (-not $env:SOGNO_CANE_SETUP_NOSHORTCUT) {
    $shell = New-Object -ComObject WScript.Shell
    $targets = @(
        [Environment]::GetFolderPath('Desktop'),
        (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
    )
    foreach ($loc in $targets) {
        try {
            $lnk = $shell.CreateShortcut((Join-Path $loc 'SOGNO_CANE.lnk'))
            $lnk.TargetPath = $start.FullName
            $lnk.WorkingDirectory = $appDir
            $lnk.Description = 'SOGNO_CANE - EEG to MIDI'
            $lnk.Save()
        } catch {}
    }
    Write-Host 'Collegamenti creati (Desktop e menu Start).'
}

Write-Host ''
Write-Host 'Installazione completata.'
if (-not $env:SOGNO_CANE_SETUP_NOLAUNCH) {
    Write-Host 'Avvio SOGNO_CANE...'
    Start-Process -FilePath $start.FullName -WorkingDirectory $appDir
}
exit 0
