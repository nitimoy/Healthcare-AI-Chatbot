# Healthcare AI Chatbot — Production-Grade RAG & Voice Interface

An enterprise-ready **Healthcare AI Chatbot** built with Retrieval-Augmented Generation (RAG), browser-native **"Ask by Voice"** input, dual-tier **Safety Guardrails**, and evidence-grounded medical knowledge from **MedlinePlus** (U.S. National Library of Medicine).

Powered by **BAAI/bge-m3** embeddings · **ChromaDB** vector store · **Meta Llama 3.1 8B via NVIDIA NIM** · Dark glassmorphic **Streamlit UI**

---

## 🎬 Demo Video

**▶ Watch the full demo on YouTube:** [https://www.youtube.com/watch?v=XZyWeYzL7io](https://www.youtube.com/watch?v=XZyWeYzL7io)

The demo covers:
- Live voice query using the "Ask by Voice" floating button
- Real-time token streaming responses
- Emergency guardrail interception in action
- Source citation cards and conversation memory
- Cross-platform single-click launch

---

## 📋 Assignment Requirements & Compliance Matrix

| Requirement / Enhancement | Status | Implementation Details |
|---|---|---|
| **Common Symptoms & Diseases** | ✅ Covered | 1,017 MedlinePlus topics covering symptoms, conditions, treatments |
| **Healthy Lifestyle & Nutrition** | ✅ Covered | Dedicated retrieval of diet, exercise, and preventive care topics |
| **First-Aid Guidance** | ✅ Covered | Burn treatment, poison response, and first-aid chunks indexed |
| **Medical Disclaimer** | ✅ Enforced | Mandatory disclaimer appended to every LLM response by system prompt |
| **Frontend — Streamlit** | ✅ Implemented | Dark glassmorphism UI, streaming, suggestion pills, sidebar, auto-scroll |
| **Backend — Python** | ✅ Implemented | Fully modular Python 3.11 architecture (`app/chatbot/`, `preprocessing/`) |
| **LLM Selection** | ✅ Implemented | **Meta Llama 3.1 8B via NVIDIA NIM** — switched from GLM-5.2 (too slow); Llama 3.1 8B is significantly faster with OpenAI-compatible streaming |
| **RAG Pipeline** | ✅ Implemented | BAAI/bge-m3 dense retrieval → ChromaDB HNSW cosine search → 2,010 semantic chunks |
| **Vector Database** | ✅ Implemented | **ChromaDB** with batched idempotent upserts and HNSW indexing |
| **Medical Knowledge Base** | ✅ Implemented | Streaming-parsed MedlinePlus XML — 1,017 topics, MeSH-categorised |
| **Prompt Engineering** | ✅ Implemented | Grounding constraints, medical tone rules, context injection templates |
| **Context-Aware Conversations** | ✅ Implemented | Rolling 5-exchange **ConversationMemory** for multi-turn coherence |
| **Chat History** | ✅ Implemented | Full session-scoped chat history in Streamlit `session_state` |
| **Response Guardrails** | ✅ Implemented | Dual-tier regex interceptor — Emergency Bypass + Unsafe Request Redirection |
| **Citation of Sources** | ✅ Implemented | Expandable cards with MedlinePlus URLs, topic titles, MeSH headings |
| **Voice Input ("Ask by Voice")** | ✅ Implemented | Browser-native Web Speech API · FAB button · Conservative filler sanitizer · 5s auto-submit |
| **Automated Test Suite** | ✅ Implemented | **25 passing tests** covering guardrails, retrieval, and voice sanitization |
| **Containerization & One-Click Run** | ✅ Implemented | `run_docker.sh` (macOS/Linux) · `run_win.ps1` · `run_win.bat` · Docker Compose |

---

## 🚀 Quick Start (Single-Click Execution)

Runs the full pipeline (XML parse → chunk → embed → launch) automatically:

### Linux / macOS:
```bash
./run_docker.sh
```

### Windows (PowerShell):
```powershell
.\run_win.ps1
```

### Windows (Command Prompt):
```cmd
run_win.bat
```

> Scripts auto-detect Docker availability and fall back to local Python execution if Docker is not running.

---

## 💻 Manual Setup

### 1. Environment Setup

```bash
git clone <repo-url>
cd Medical-Chatbot
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** First install downloads PyTorch (~500 MB) and BAAI/bge-m3 (~1.2 GB). Subsequent runs use local cache.

### 2. API Key Configuration

```bash
cp .env.example .env
```

Edit `.env` and set your [NVIDIA NIM API key](https://build.nvidia.com):
```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
```

### 3. Data Ingestion Pipeline (One-Time)

```bash
# Parse MedlinePlus XML → structured JSON
PYTHONPATH=. python scripts/extract_medlineplus.py

# Build embedding documents with MeSH metadata
PYTHONPATH=. python scripts/build_embedding_documents.py

# Token-aware sliding-window chunking (384 tokens max)
PYTHONPATH=. python scripts/chunk_documents.py

# Generate BGE-M3 embeddings & index into ChromaDB (~10-15 min on CPU)
PYTHONPATH=. python app/vectorstore/embed_documents.py
```

### 4. Launch the App

```bash
PYTHONPATH=. streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in **Chrome or Edge** (required for voice input).

---

## 🧪 Automated Test Suite

```bash
PYTHONPATH=. python tests/run_tests.py
```

**25 tests covering:**
| Category | Count | Status |
|---|---|---|
| Emergency Guardrail Interception | 8 | ✅ All PASSED |
| Unsafe Clinical Request Redirection | 6 | ✅ All PASSED |
| RAG Evidence & Retrieval Cohesion | 6 | ✅ All PASSED |
| Voice Transcript Sanitization | 5 | ✅ All PASSED |
| **Total** | **25** | **🎉 100% PASS** |

---

## 📂 Project Folder Structure

```
Medical-Chatbot/
├── app/
│   ├── chatbot/
│   │   ├── generator.py       # RAG orchestrator: guardrails → retrieval → memory → LLM stream
│   │   ├── retriever.py       # ChromaDB query engine (BGE-M3 dense embeddings)
│   │   ├── guardrails.py      # Dual-tier safety interceptor
│   │   ├── memory.py          # Rolling 5-exchange ConversationMemory
│   │   ├── voice.py           # VoiceSession model + clean_voice_transcript() sanitizer
│   │   └── prompts.py         # System prompt templates + context injection
│   ├── vectorstore/
│   │   └── embed_documents.py # Batched idempotent ChromaDB indexing engine
│   └── config/
│       └── settings.py        # All tuneable constants (env-driven)
├── preprocessing/             # Offline ingestion pipeline
│   ├── parser.py              # C-accelerated lxml XML stream parser
│   ├── cleaner.py             # ftfy text normalizer + lxml.html stripper
│   ├── splitter.py            # Header / Summary / Footer section splitter
│   ├── chunker.py             # Rust-backed semantic-text-splitter (384 tokens)
│   └── metadata.py            # MeSH provenance schema builder
├── scripts/                   # CLI pipeline entry points
│   ├── extract_medlineplus.py
│   ├── build_embedding_documents.py
│   └── chunk_documents.py
├── tests/
│   ├── run_tests.py           # Unified test runner
│   ├── test_evaluator.py      # Guardrail + evidence evaluation tests
│   └── test_voice.py          # Voice sanitizer unit tests
├── data/
│   ├── raw/                   # medical_kb_raw.json (parsed topics)
│   ├── processed/             # chunked_documents.json (2,010 chunks)
│   └── chroma/                # ChromaDB persistent vector store
├── app.py                     # Streamlit UI entry point
├── PIPELINE_ARCHITECTURE.md   # End-to-end architecture & slide outline
├── LOGIC_DOCUMENTATION.md     # Query processing, prompt logic, assumptions
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── README.md
```

---

## ⚙️ Configuration Reference

All settings in [`app/config/settings.py`](app/config/settings.py), overridable via `.env`:

| Parameter | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace sentence-transformer model |
| `LLM_MODEL` | `meta/llama-3.1-8b-instruct` | Generation model via NVIDIA NIM Engine |
| `TOP_K` | `5` | Chunks retrieved per query |
| `TEMPERATURE` | `0.3` | LLM temperature (low = factually conservative) |
| `MAX_OUTPUT_TOKENS` | `1024` | Maximum response token budget |
| `CHUNK_SIZE` | `384` | Sub-word token limit per chunk |
| `MEMORY_WINDOW_SIZE` | `5` | Conversational exchange history depth |

---

## 🤖 LLM Selection & Rationale

| Option Considered | Decision | Reasoning |
|---|---|---|
| **OpenAI GPT-4o** | ❌ Not used | Requires paid API; higher cost per token |
| **Google Gemini Pro** | ❌ Not used | Good alternative but NVIDIA NIM offers simpler OpenAI-compatible interface |
| **Mistral (local/API)** | ❌ Not used | Lower benchmark scores on medical Q&A tasks |
| **GLM-5.2 via NVIDIA NIM** | ❌ Tried, rejected | Trialled first — noticeably slow response times; replaced for better user experience |
| **Meta Llama 3.1 8B via NVIDIA NIM** ✅ | **Selected** | Significantly faster than GLM-5.2; free-tier NVIDIA NIM Engine; OpenAI SDK–compatible streaming; swappable via `.env` with no code changes |

The model is fully **swappable without code changes** — set `LLM_MODEL` in `.env` to any NVIDIA NIM hosted model.

---

## 🛡️ Safety & Guardrails Architecture

```
           User Query ("Ask by Voice" or Typed)
                           │
                           ▼
             ┌─────────────────────────┐
             │   Tier 1: Guardrails    │
             │  (O(n) Regex Intercept) │
             └───────────┬─────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
[Emergency Detected]          [Unsafe Clinical Request]
 Chest pain, anaphylaxis,      Medication doses,
 severe bleeding, poison        disease prediction
          │                             │
          ▼                             ▼
 Instant 911/ER Notice       Redirect to Doctor
 (0ms, $0 LLM cost)          (0ms, $0 LLM cost)
          │                             │
          └──────────────┬──────────────┘
                         │ Safe Educational Query
                         ▼
             ┌─────────────────────────┐
             │   Tier 2: RAG Pipeline  │
             │  BGE-M3 + ChromaDB LLM  │
             └─────────────────────────┘
```

---

## 🎤 "Ask by Voice" Feature

- **Browser-Native:** Uses `streamlit_mic_recorder` (Web Speech API) for zero-cost, zero-backend speech-to-text — triggered via the floating **"Ask by Voice"** pill button.
- **Conservative Sanitizer:** `clean_voice_transcript()` removes *only* leading filler words ("uh", "um", "well") — preserves mid-sentence medical terms (e.g. "What foods are *like* asthma triggers?" is kept intact).
- **FAB Positioning:** Fixed-position floating button above the chat input bar using JS DOM manipulation (`position: fixed; bottom: 85px; right: 22px`).
- **Auto-Submit:** Optional 5-second silence detection toggle auto-submits the question.
- **Auto-Scroll:** Smooth JS scroll to bottom of chat after each token stream completes.

---

## 💡 Innovation Highlights (Beyond Requirements)

| Innovation | Description |
|---|---|
| **Dual-Tier Guardrails** | Deterministic O(n) regex interceptor — emergency bypass at 0ms latency, $0 LLM cost |
| **Conservative Voice Sanitizer** | Preserves medical meaning while stripping only leading hesitation fillers |
| **Floating Voice FAB via JS DOM** | Mic button repositioned using `MutationObserver` JS without breaking Streamlit's iframe model |
| **Idempotent Vector Indexing** | `upsert`-based ChromaDB indexing — safe to re-run without creating duplicate vectors |
| **10.7x Data Compression** | `semantic-text-splitter` reduced chunked data from 40 MB → 3.7 MB with zero information loss |
| **Type-Safe RAG Bridge** | `RetrievedChunk` dataclass prevents runtime key errors between ChromaDB output and UI |
| **Auto-Scroll UX** | JS-injected smooth scroll to answer after streaming, via `st.components.v1.html(height=0)` |
| **Cross-Platform One-Click Run** | Unified launch scripts for Docker, macOS, Linux, Windows CMD & PowerShell |

---

## 📄 Medical Disclaimer

This application is for **educational and informational purposes only**. It does not diagnose, prescribe, or replace professional medical advice. Always consult a qualified healthcare provider for medical concerns.

**Data Source:** [MedlinePlus](https://medlineplus.gov) — U.S. National Library of Medicine, NIH.
