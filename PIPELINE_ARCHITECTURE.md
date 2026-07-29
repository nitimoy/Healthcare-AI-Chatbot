# Healthcare AI Chatbot: System Architecture & Technical Specifications

This document covers the **complete end-to-end architecture**, technology stack rationale, pipeline workflows, LLM selection, safety guardrails, voice interface design, challenges faced, and the 4–5 slide presentation guide.

---

## 1. System Overview & Architecture Diagram

The system operates in two decoupled phases: **Offline Data Ingestion** (one-time pipeline) and **Real-Time RAG + Voice Execution** (per query).

```
══════════════════════════════════════════════════════════════════════════════
  PHASE 1 — OFFLINE DATA INGESTION (One-Time Pipeline)
══════════════════════════════════════════════════════════════════════════════

  [MedlinePlus XML (30 MB)]
        │
        ▼ lxml iterparse streaming
  [XML Parser]  →  [ftfy + lxml.html Cleaner]  →  [Header/Summary/Footer Splitter]
                                                            │
                                                            ▼ semantic-text-splitter (Rust)
                                              [Token-Aware Sliding Chunker — 384 tokens]
                                                            │
                                                            ▼ BAAI/bge-m3 (HuggingFace)
                                              [Dense Embedding Generator — 1024-dim vectors]
                                                            │
                                                            ▼ batched upsert
                                              [ChromaDB Persistent Vector Store (HNSW)]

══════════════════════════════════════════════════════════════════════════════
  PHASE 2 — REAL-TIME RAG, VOICE & SAFETY (Per User Query)
══════════════════════════════════════════════════════════════════════════════

  User Input
  (Typed in chat OR spoken via "Ask by Voice" button)
        │
        ▼
  [Voice Sanitizer — clean_voice_transcript()]
   → Strips leading fillers ("uh", "um")
   → Preserves medical terms mid-sentence
        │
        ▼
  [Dual-Tier Guardrails — O(n) Regex Interceptor]
   → Tier 1A Emergency?  → Returns 911/ER notice immediately (0ms, $0 LLM)
   → Tier 1B Unsafe?     → Returns physician redirection (0ms, $0 LLM)
   → Safe? ──────────────────────────────────────────────────────────────────┐
                                                                             │
                                                                             ▼
                                                         [BGE-M3 Query Embedding (1024-dim)]
                                                                             │
                                                                             ▼
                                                   [ChromaDB HNSW Cosine Search — Top-K=5 Chunks]
                                                                             │
                                                                             ▼
                                                   [Context Builder]
                                                    → Inject Top-K MedlinePlus chunks
                                                    → Append rolling memory (last 5 exchanges)
                                                    → Apply system prompt constraints
                                                                             │
                                                                             ▼
                                                   [Meta Llama 3.1 8B via NVIDIA NIM Engine]
                                                    → Streaming token generation
                                                    → Mandatory disclaimer in prompt
                                                                             │
                                                                             ▼
                                                   [Streamlit UI]
                                                    → Real-time token stream render
                                                    → Expandable source citation cards
                                                    → Auto-scroll to bottom of chat
```

---

## 2. Tech Stack & Rationale

### 2.1 Embedding & Vector Store

| Component | Technology | Rationale |
|---|---|---|
| **Embedding Model** | `BAAI/bge-m3` (HuggingFace) | State-of-the-art MTEB benchmark scores for medical semantic search; multilingual; runs CPU-only |
| **Vector Database** | `ChromaDB` | Embedded, zero-infrastructure vector DB; HNSW cosine similarity; idempotent batched `upsert` prevents duplicate vectors |

### 2.2 LLM Selection & Rationale

| Candidate Model | Decision | Reasoning |
|---|---|---|
| **OpenAI GPT-4o** | ❌ Not selected | Paid API with high per-token costs; overkill for grounded RAG tasks |
| **Google Gemini Pro** | ❌ Not selected | Alternative endpoint; NVIDIA NIM offers lower latency for streaming |
| **Meta Llama 3 (local)** | ❌ Not selected | Requires local GPU (8–16 GB VRAM); impractical for CPU-only deployment |
| **Mistral (local/API)** | ❌ Not selected | Lower benchmark scores on medical question-answering |
| **GLM-5.2 via NVIDIA NIM** | ❌ Tried, rejected | Trialled first; noticeably slow response latency; replaced for better UX |
| **Meta Llama 3.1 8B via NVIDIA NIM** | ✅ **Selected** | Significantly faster than GLM-5.2; free-tier NVIDIA NIM Engine; OpenAI SDK–compatible streaming; swappable via `.env` — no code changes needed |

### 2.3 Preprocessing Pipeline Stack

| Stage | Technology | Rationale |
|---|---|---|
| **XML Parsing** | `lxml.etree iterparse` | C-accelerated streaming — 5-10x faster than stdlib; handles 30 MB XML with minimal RAM |
| **Text Cleaning** | `ftfy` + `lxml.html` | Fixes UTF-8 mojibake; safely strips HTML markup without losing valid text |
| **Chunking** | `semantic-text-splitter` (Rust) | True token-aware sliding window with sentence-boundary alignment; 384 sub-word token hard cap |
| **Serialization** | `orjson` | 3-10x faster than stdlib `json`; native UTF-8; handles large chunk files efficiently |

### 2.4 Frontend & Voice

| Component | Technology | Rationale |
|---|---|---|
| **UI Framework** | `Streamlit` | Rapid professional chat interface; native streaming; sidebar; session state |
| **Voice Input** | `streamlit_mic_recorder` (Web Speech API) | Browser-native, zero-cost, zero-backend speech recognition; works in Chrome & Edge |

---

## 3. Application Workflow (Step-by-Step)

1. **User submits query** — typed into the chat box *or* spoken via the floating **"Ask by Voice"** button.
2. **Voice sanitization** — `clean_voice_transcript()` strips leading filler words ("uh", "um", "well") while preserving all medical terms and question structure.
3. **Guardrails intercept** — regex matcher evaluates query against emergency patterns and unsafe clinical request patterns. If matched, returns a pre-built response instantly without touching the vector DB or LLM.
4. **Dense retrieval** — query is converted to a 1024-dimensional BGE-M3 embedding; ChromaDB returns Top-K=5 chunks ranked by HNSW cosine similarity.
5. **Prompt assembly** — retrieved chunks are injected as formatted MedlinePlus context blocks; the last 5 conversational exchanges from `ConversationMemory` are appended; the system prompt enforces grounding, tone, and mandatory disclaimer.
6. **LLM streaming** — the assembled prompt is sent to **Meta Llama 3.1 8B** via NVIDIA NIM Engine; tokens stream in real-time via the OpenAI-compatible client.
7. **UI rendering** — Streamlit renders each token chunk as it arrives; after generation completes, expandable source citation cards are displayed and the page auto-scrolls to the response.

---

## 4. Prompt Engineering Strategy

### 4.1 System Prompt Design (Enforced Constraints)
```
Role: You are Healthcare AI, an evidence-grounded health education assistant.

Rules:
1. GROUNDING — Answer ONLY from the provided MedlinePlus context. If insufficient, say so clearly.
2. NO DIAGNOSIS — Never provide personal diagnoses, prescribe drugs, or modify dosages.
3. TONE — Empathetic, clear, accessible (8th grade reading level).
4. FORMAT — Markdown headings, bullet points, and key takeaways.
5. DISCLAIMER — Always append a concise medical disclaimer to every answer.
```

### 4.2 Context Injection Template
```
--- MEDLINEPLUS CONTEXT ---
[Source 1: <Topic Title> | MeSH: <Category>]
URL: https://medlineplus.gov/<topic>.html
<chunk content>
---
[Source 2: ...]
--- END CONTEXT ---

Conversation History:
User: <previous question>
Assistant: <previous answer>
...

Current Question: <user query>
```

---

## 5. Safety Measures & Guardrails

### 5.1 Tier 1A — Emergency Bypass
- **Pattern triggers:** `chest pain`, `stroke`, `can't breathe`, `anaphylaxis`, `severe bleeding`, `ingested poison`, `unresponsive`, `suicidal`, `overdose`, etc.
- **Response:** Pre-built emergency notice with Poison Control (1-800-222-1222) and 911 instruction.
- **Cost:** $0 LLM tokens · 0ms latency.

### 5.2 Tier 1B — Unsafe Request Redirection
- **Pattern triggers:** `diagnose me`, `prescribe`, `increase my dose`, `stop taking my medication`, `cancer risk`, `alternative to my inhaler`, etc.
- **Response:** Empathetic redirection to consult a licensed healthcare provider.
- **Cost:** $0 LLM tokens · 0ms latency.

### 5.3 Tier 2 — System Prompt Constraints
- Forces LLM to cite only retrieved context chunks.
- Prohibits diagnostic language, dosage recommendations, and predictions.
- Injects mandatory medical disclaimer into every response.

---

## 6. Challenges Faced & Solutions

| Challenge | Root Cause | Solution Implemented |
|---|---|---|
| **Voice button not activating** | Streamlit sandboxed iframe blocks microphone access when using `st.components.v1.html` with `srcdoc` | Switched to `streamlit_mic_recorder` (pre-built component with proper browser permissions) |
| **Large white box under voice button** | `streamlit_mic_recorder` iframe defaulted to ~100px height with white background | JS `MutationObserver` constrains iframe to `38px × 165px` with `background: transparent; overflow: hidden` |
| **Voice transcript reprocessing on reruns** | Streamlit re-runs all code on every widget interaction | `st.session_state.last_processed_voice` deduplication guard — skips already-processed transcripts |
| **Over-sanitizing voice queries** | Initial sanitizer removed words like "like" globally, changing medical meaning | Rewrote `clean_voice_transcript()` to strip *only leading* fillers — never mid-sentence words |
| **Data bloating in chunking** | Legacy custom chunker produced 40 MB output with chunks exceeding token limits | Replaced with `semantic-text-splitter` (Rust) — output compressed to 3.7 MB; 0 chunks over 384 tokens |
| **Auto-scroll not working during streaming** | Streamlit incremental DOM updates reset scroll position mid-stream | Injected JS with multiple `setTimeout` delays (150ms, 600ms, 1200ms) to catch streaming render frames |
| **Dropdown/input background invisible** | Dark theme CSS not targeting BaseWeb internal shadow DOM selectors | Targeted specific `data-baseweb` and `data-testid` selectors with `!important` overrides |
| **ChromaDB duplicate vectors on re-run** | Script used `add()` instead of `upsert()` | Switched to batched `upsert()` with existing ID check — safe to re-run without producing duplicates |

---

## 7. Evaluation & Testing Metrics

| Test Suite Category | Count | Result | Key Scenarios |
|---|---|---|---|
| **Guardrails — Emergency** | 8 | ✅ PASSED | Anaphylaxis, gasping infant, poison, unresponsive child, arterial bleeding |
| **Guardrails — Unsafe Clinical** | 6 | ✅ PASSED | Insulin dose, chemo stopping, disease prediction, herbal alternative therapy |
| **RAG Evidence & Retrieval** | 6 | ✅ PASSED | Metadata alignment, content availability, token limits, negative regressions |
| **Voice Sanitization** | 5 | ✅ PASSED | Medical term retention, leading filler removal, empty input, edge cases |
| **TOTAL** | **25** | **🎉 100% PASS** | Zero regressions |

---

## 8. Architecture Presentation Guide (4–5 Slides)

### Slide 1 — System Overview & Medical RAG Flow
- Full end-to-end flow: User (Voice/Text) → Guardrails → BGE-M3 Retrieval → ChromaDB → Llama 3.1 8B → Streamlit UI
- Core value: Grounded medical education · Zero-hallucination guardrails · Voice-native interaction

### Slide 2 — Tech Stack & LLM Rationale
- BAAI/bge-m3 + ChromaDB → dense HNSW semantic matching over 2,010 chunks
- **Meta Llama 3.1 8B via NVIDIA NIM Engine** → fastest option tested (replaced GLM-5.2); free-tier; OpenAI-compatible streaming; swappable via `.env`
- Streamlit + Web Speech API → floating FAB button, real-time stream, auto-scroll
- lxml + ftfy + semantic-text-splitter → C/Rust-accelerated ingestion pipeline

### Slide 3 — Prompt Engineering & Dual-Tier Safety
- Dual-tier guardrails: O(n) regex interceptor at 0ms / $0 cost for emergencies
- System prompt: Context grounding + no-diagnosis constraint + mandatory disclaimer
- Memory: Sliding 5-exchange `ConversationMemory` for multi-turn coherence
- Show real example: "chest pain" → instant emergency response (no LLM call)

### Slide 4 — Data Ingestion & Knowledge Base
- Raw input: MedlinePlus XML (30 MB, 1,017 healthcare topics, MeSH-categorized)
- Pipeline: lxml → ftfy → splitter → semantic-text-splitter → bge-m3 → ChromaDB
- Result: 2,010 self-contained 384-token chunks · 0 token budget violations
- Optimization: 40 MB → 3.7 MB (10.7× compression via semantic chunker)

### Slide 5 — Key Achievements, Challenges & Demo
- Demo video: [https://www.youtube.com/watch?v=XZyWeYzL7io](https://www.youtube.com/watch?v=XZyWeYzL7io)
- 25 automated tests passing (guardrails, RAG, voice)
- Challenges: Voice iframe sandboxing → `streamlit_mic_recorder`; data bloat → Rust chunker; scroll during stream → multi-delay JS injection
- Cross-platform: Docker · macOS/Linux shell · Windows PS/CMD
