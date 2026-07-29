#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "🏥 Launching Healthcare AI Assistant Single-Click Pipeline..."
echo "======================================================================"

if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found! Copying .env.example to .env..."
    cp .env.example .env
    echo "🔑 Please verify your NVIDIA_API_KEY is present in .env"
fi

chmod +x entrypoint.sh

# Check if Docker daemon is running
if docker info >/dev/null 2>&1; then
    echo "🐳 Docker daemon is running. Launching in Docker container..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up --build -d
    elif docker compose version &> /dev/null 2>&1; then
        docker compose up --build -d
    else
        echo "📦 Building container image with docker build..."
        docker build -t medical_chatbot .
        docker rm -f medical_chatbot_app 2>/dev/null || true
        echo "🚀 Launching container..."
        docker run -d \
            --name medical_chatbot_app \
            -p 8501:8501 \
            --env-file .env \
            -v "$(pwd)/data/chroma:/app/data/chroma" \
            medical_chatbot
    fi
    echo "======================================================================"
    echo "🎉 Healthcare AI Assistant Docker container launched successfully!"
    echo "🌐 Access the application at: http://localhost:8501"
    echo "======================================================================"
else
    echo "ℹ️ Docker daemon is not running. Executing single-click pipeline locally..."
    echo "======================================================================"
    
    # Environment variables
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONPATH=.

    # Auto-create virtual environment if missing
    if [ ! -d ".venv" ]; then
        echo "📦 Creating virtual environment (.venv)..."
        python3 -m venv .venv
    fi

    # Auto-install/verify requirements
    echo "⚡ Verifying requirements from requirements.txt..."
    .venv/bin/python -m pip install --quiet -r requirements.txt

    PYTHON_BIN=".venv/bin/python"

    # Step 1: XML Extraction
    if [ -f "mplus_topics_2026-07-28.xml" ]; then
        echo "📄 Step 1/5: Extracting MedlinePlus XML knowledge base..."
        $PYTHON_BIN scripts/extract_medlineplus.py --xml mplus_topics_2026-07-28.xml --output data/raw/medical_kb_raw.json
    fi

    # Step 2: Build Embedding Documents
    if [ -f "data/raw/medical_kb_raw.json" ]; then
        echo "📄 Step 2/5: Building embedding documents..."
        $PYTHON_BIN scripts/build_embedding_documents.py
    fi

    # Step 3: Chunk Documents
    if [ -f "data/raw/medical_kb_raw.json" ]; then
        echo "📄 Step 3/5: Chunking documents into tokens..."
        $PYTHON_BIN scripts/chunk_documents.py --input data/raw/medical_kb_raw.json --output data/processed/chunked_documents.json
    fi

    # Step 4: Index & Embed Chunks into ChromaDB
    if [ -f "data/processed/chunked_documents.json" ]; then
        echo "🧠 Step 4/5: Generating embeddings & indexing into ChromaDB..."
        $PYTHON_BIN app/vectorstore/embed_documents.py
    fi

    # Step 5: Launch Streamlit Web Application
    echo "🚀 Step 5/5: Starting Healthcare AI Assistant Streamlit Web App..."
    pkill -f "streamlit run" 2>/dev/null || true
    exec $PYTHON_BIN -m streamlit run app.py --server.port 8501
fi
