@echo off
echo ======================================================================
echo 🏥 Healthcare AI Assistant — Windows Single-Click Launcher
echo ======================================================================

if not exist .env (
    echo ⚠️ Copying .env.example to .env...
    copy .env.example .env
)

docker info >nul 2>&1
if %errorlevel% == 0 (
    echo 🐳 Launching with Docker Compose...
    docker-compose up --build -d
    echo 🎉 Container running at http://localhost:8501
) else (
    echo ℹ️ Docker not running. Executing pipeline in Python...
    set PYTHONPATH=.
    set OMP_NUM_THREADS=1
    set TOKENIZERS_PARALLELISM=false

    if not exist .venv (
        echo 📦 Creating virtual environment .venv...
        python -m venv .venv
    )
    
    echo ⚡ Verifying requirements from requirements.txt...
    .venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
    
    if exist mplus_topics_2026-07-28.xml .venv\Scripts\python.exe scripts\extract_medlineplus.py --xml mplus_topics_2026-07-28.xml --output data/raw/medical_kb_raw.json
    if exist data\raw\medical_kb_raw.json .venv\Scripts\python.exe scripts\build_embedding_documents.py
    if exist data\raw\medical_kb_raw.json .venv\Scripts\python.exe scripts\chunk_documents.py --input data/raw/medical_kb_raw.json --output data/processed/chunked_documents.json
    if exist data\processed\chunked_documents.json .venv\Scripts\python.exe app\vectorstore\embed_documents.py
    
    echo 🚀 Starting Streamlit UI...
    .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
)
