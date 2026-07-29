# PowerShell script for Windows single-click execution
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🏥 Healthcare AI Assistant — Windows Single-Click Pipeline Launcher" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if (-not (Test-Path ".env")) {
    Write-Host "⚠️ .env file not found! Copying .env.example to .env..." -ForegroundColor Yellow
    Copy-Item .env.example .env
}

# Check if Docker daemon is running
$dockerDaemonRunning = $false
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerDaemonRunning = $true }
} catch {}

if ($dockerDaemonRunning) {
    Write-Host "🐳 Docker daemon detected. Building & launching container..." -ForegroundColor Green
    docker-compose up --build -d
    Write-Host "🎉 Container launched! Access at http://localhost:8501" -ForegroundColor Green
} else {
    Write-Host "ℹ️ Docker daemon not running. Executing pipeline in Python environment..." -ForegroundColor Yellow
    $env:PYTHONPATH = "."
    $env:OMP_NUM_THREADS = "1"
    $env:TOKENIZERS_PARALLELISM = "false"

    if (-not (Test-Path ".venv")) {
        Write-Host "📦 Creating virtual environment (.venv)..." -ForegroundColor Yellow
        python -m venv .venv
    }

    Write-Host "⚡ Verifying requirements from requirements.txt..." -ForegroundColor Cyan
    & .venv\Scripts\python.exe -m pip install --quiet -r requirements.txt

    $pythonBin = ".venv\Scripts\python.exe"

    if (Test-Path "mplus_topics_2026-07-28.xml") {
        Write-Host "📄 Step 1/5: Extracting MedlinePlus XML..." -ForegroundColor Cyan
        & $pythonBin scripts/extract_medlineplus.py --xml mplus_topics_2026-07-28.xml --output data/raw/medical_kb_raw.json
    }

    if (Test-Path "data/raw/medical_kb_raw.json") {
        Write-Host "📄 Step 2/5: Building embedding documents..." -ForegroundColor Cyan
        & $pythonBin scripts/build_embedding_documents.py
        
        Write-Host "📄 Step 3/5: Chunking documents..." -ForegroundColor Cyan
        & $pythonBin scripts/chunk_documents.py --input data/raw/medical_kb_raw.json --output data/processed/chunked_documents.json
    }

    if (Test-Path "data/processed/chunked_documents.json") {
        Write-Host "🧠 Step 4/5: Generating embeddings & indexing into ChromaDB..." -ForegroundColor Cyan
        & $pythonBin app/vectorstore/embed_documents.py
    }

    Write-Host "🚀 Step 5/5: Starting Healthcare AI Assistant Streamlit Web App..." -ForegroundColor Green
    & $pythonBin -m streamlit run app.py --server.port 8501
}
