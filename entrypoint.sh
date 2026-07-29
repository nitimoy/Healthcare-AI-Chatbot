#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "🏥 Healthcare AI Assistant — End-to-End Pipeline & Server Entrypoint"
echo "======================================================================"

# Environment variables for PyTorch / HuggingFace thread safety
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

# Step 1: XML Extraction
if [ -f "mplus_topics_2026-07-28.xml" ]; then
    echo "📄 Step 1/5: Extracting MedlinePlus XML knowledge base..."
    python scripts/extract_medlineplus.py --xml mplus_topics_2026-07-28.xml --output data/raw/medical_kb_raw.json
else
    echo "⚠️ Warning: mplus_topics_2026-07-28.xml not found, skipping extraction."
fi

# Step 2: Build Embedding Documents
if [ -f "data/raw/medical_kb_raw.json" ]; then
    echo "📄 Step 2/5: Building embedding documents..."
    python scripts/build_embedding_documents.py
fi

# Step 3: Chunk Documents
if [ -f "data/raw/medical_kb_raw.json" ]; then
    echo "📄 Step 3/5: Chunking documents into tokens..."
    python scripts/chunk_documents.py --input data/raw/medical_kb_raw.json --output data/processed/chunked_documents.json
fi

# Step 4: Index & Embed Chunks into ChromaDB
if [ -f "data/processed/chunked_documents.json" ]; then
    echo "🧠 Step 4/5: Generating embeddings & indexing into ChromaDB..."
    python app/vectorstore/embed_documents.py
fi

# Step 5: Launch Streamlit Web Application
echo "🚀 Step 5/5: Starting Healthcare AI Assistant Streamlit Web App..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
