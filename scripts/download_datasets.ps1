[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12 -bor [System.Net.SecurityProtocolType]::Tls13

$ROOT     = "D:\SIH26143_OilSpill"
$RAW_DIR  = "$ROOT\data\raw"
$META_DIR = "$ROOT\data\metadata"
$LOG_FILE = "$META_DIR\download_log.csv"

if (!(Test-Path $LOG_FILE)) {
    "timestamp,dataset_id,dataset_name,source_url,target_folder,status" | Set-Content $LOG_FILE -Encoding UTF8
}

function Log-Action($id, $name, $url, $folder, $status) {
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$timestamp,$id,`"$name`",`"$url`",`"$folder`",$status" | Add-Content $LOG_FILE -Encoding UTF8
}

function Download-And-Extract-File($id, $name, $url, $targetDir, $archiveName) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "[*] Processing: $name" -ForegroundColor Yellow
    Write-Host "========================================================" -ForegroundColor Cyan

    if (!(Test-Path $targetDir)) { 
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null 
    }
    
    $archivePath = Join-Path $targetDir $archiveName

    try {
        # 1. Download with real-time progress via curl
        if (Test-Path $archivePath) {
            $sizeMB = [math]::Round(((Get-Item $archivePath).Length / 1MB), 2)
            Write-Host "[i] Archive already exists ($sizeMB MB): $archivePath (Skipping download)" -ForegroundColor DarkGray
        } else {
            Write-Host "[↓] Downloading: $name" -ForegroundColor Green
            Write-Host "[i] URL: $url" -ForegroundColor DarkGray
            
            # Use native curl.exe for progress display, redirect following (-L), and retry logic
            & curl.exe -L --fail --retry 3 -o "$archivePath" "$url"
            
            if ($LASTEXITCODE -ne 0 -or !(Test-Path $archivePath)) {
                throw "curl failed to download file with exit code $LASTEXITCODE"
            }

            Log-Action $id $name $url $targetDir "DOWNLOADED"
            Write-Host "[✓] Download complete!" -ForegroundColor Green
        }

        # 2. Extract using tar.exe (handles Zip64 and >2GB files seamlessly)
        Write-Host "[⟳] Extracting archive..." -ForegroundColor Yellow
        
        if ($archiveName.EndsWith(".zip")) {
            & tar.exe -xf "$archivePath" -C "$targetDir"
        } elseif ($archiveName.EndsWith(".tar.gz") -or $archiveName.EndsWith(".tgz") -or $archiveName.EndsWith(".tar")) {
            & tar.exe -xzf "$archivePath" -C "$targetDir"
        }
        
        Log-Action $id $name $url $targetDir "EXTRACTED_SUCCESS"
        Write-Host "[✓] Successfully extracted to: $targetDir" -ForegroundColor Green

    } catch {
        Write-Host "[✗] Error processing $id : $($_.Exception.Message)" -ForegroundColor Red
        Log-Action $id $name $url $targetDir "FAILED: $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------
# 1. Gulf of Mexico Sentinel-1 Dataset (Zenodo: 4672426)
#    Queries Zenodo API to get the exact file download URL dynamically
# ---------------------------------------------------------------------
try {
    Write-Host "[i] Querying Zenodo API for record 4672426 (Gulf of Mexico)..." -ForegroundColor DarkCyan
    $zenodoMeta = Invoke-RestMethod -Uri "https://zenodo.org/api/records/4672426" -TimeoutSec 30
    $files = $zenodoMeta.files
    foreach ($f in $files) {
        $fName = $f.key
        $fUrl  = $f.links.self
        Download-And-Extract-File `
            -id "gulf_mexico" `
            -name "Gulf of Mexico Sentinel-1 ($fName)" `
            -url $fUrl `
            -targetDir "$RAW_DIR\gulf_mexico" `
            -archiveName $fName
    }
} catch {
    Write-Host "[!] API query failed, falling back to direct Zenodo URL..." -ForegroundColor Yellow
    Download-And-Extract-File `
        -id "gulf_mexico" `
        -name "Gulf of Mexico Sentinel-1 Oil Spill Dataset" `
        -url "https://zenodo.org/records/4672426/files/oil_spill_dataset.zip?download=1" `
        -targetDir "$RAW_DIR\gulf_mexico" `
        -archiveName "gulf_mexico_s1.zip"
}

# ---------------------------------------------------------------------
# 2. Eastern Mediterranean Sentinel-1 Dataset (PANGAEA: 980773)
# ---------------------------------------------------------------------
Download-And-Extract-File `
    -id "eastern_med" `
    -name "Eastern Mediterranean S1 Oil-Slick and Look-Alike Dataset" `
    -url "https://download.pangaea.de/dataset/980773/files/dataset.zip" `
    -targetDir "$RAW_DIR\eastern_med" `
    -archiveName "eastern_med_pangaea.zip"

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host " [✓] TARGET DATASETS DOWNLOADED AND EXTRACTED!" -ForegroundColor Green
Write-Host " Log file: $LOG_FILE" -ForegroundColor Green
Write-Host "========================================================`n" -ForegroundColor Green
