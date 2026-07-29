# Healthcare AI Chatbot: Logic & Implementation Documentation

This document explains **how the chatbot processes user queries**, how prompts are customised, how responses are generated, all safety measures, and assumptions made during development.

---

## 1. Query Processing — End-to-End Workflow

When a user submits a query (typed in the chat box **or** spoken via the **"Ask by Voice"** floating button), the application executes the following deterministic pipeline:

```
[User Query ("Ask by Voice" / Typed)]
         │
         ▼
[Step 1: Voice Transcript Sanitization]   ← strips leading fillers only
         │
         ▼
[Step 2: Dual-Tier Guardrail Check]       ← O(n) regex, 0ms latency
         │ Safe query
         ▼
[Step 3: Dense Vector Retrieval]          ← BGE-M3 + ChromaDB HNSW search
         │ Top-K MedlinePlus chunks
         ▼
[Step 4: Context & Memory Assembly]       ← system prompt + history + chunks
         │ Full prompt payload
         ▼
[Step 5: LLM Streaming Generation]        ← Meta Llama 3.1 8B via NVIDIA NIM Engine
         │ Token stream + source metadata
         ▼
[Step 6: UI Rendering & Auto-Scroll]      ← Streamlit streaming + citation cards
```

---

### Step 1: "Ask by Voice" Input Sanitization (`app/chatbot/voice.py`)

- Activated when the user clicks the floating **"Ask by Voice"** microphone button.
- The browser-native **Web Speech API** (via `streamlit_mic_recorder`) converts spoken audio to text in real-time inside the browser — no backend audio processing, zero cost.
- The raw transcript is passed through **`clean_voice_transcript()`**:
  - ✅ **Strips** leading conversational hesitation fillers: *"Uh..."*, *"Um..."*, *"Well..."*, *"You know..."*, *"So..."*
  - ✅ **Preserves** all mid-sentence content including critical medical terms
  - ✅ **Example preserved:** `"What foods are like asthma triggers?"` — the word *"like"* is **not** removed because it is mid-sentence
  - ✅ **Example stripped:** `"Um... so what causes diabetes?"` → `"what causes diabetes?"`
- The cleaned transcript populates the chat input field for review and submission.
- A **`last_processed_voice`** deduplication guard in `session_state` prevents the same transcript from being re-submitted on Streamlit reruns.

---

### Step 2: Deterministic Safety Guardrail Check (`app/chatbot/guardrails.py`)

The guardrail system runs **before** any database lookup or LLM call. It uses O(n) regex pattern matching — no ML inference, zero latency, zero API cost.

#### Tier 1A — Emergency Interception
- **Trigger examples:** *"chest pain"*, *"can't breathe"*, *"unresponsive"*, *"heavy bleeding"*, *"ingested poison"*, *"anaphylaxis"*, *"suicidal"*, *"overdose"*
- **Response:** Pre-formatted emergency notice including Poison Control (1-800-222-1222) and 911 instruction.
- **LLM call:** ❌ None. Returns instantly.

#### Tier 1B — Unsafe Clinical Request Redirection
- **Trigger examples:** *"diagnose me"*, *"prescribe"*, *"increase my dose"*, *"stop taking my medication"*, *"will I get cancer"*, *"give my child herbal medicine"*
- **Response:** Empathetic message redirecting the user to consult a licensed healthcare provider.
- **LLM call:** ❌ None. Returns instantly.

#### Safe Query Path
- If neither tier is triggered, the query proceeds to the RAG retrieval pipeline.

---

### Step 3: Dense Vector Retrieval (`app/chatbot/retriever.py`)

- The query text is embedded using **BAAI/bge-m3** (HuggingFace `sentence-transformers`) into a **1024-dimensional dense vector**.
- ChromaDB performs **HNSW cosine similarity search** against the 2,010 indexed MedlinePlus chunks.
- Returns the **Top-K = 5** most semantically similar chunks as type-safe `RetrievedChunk` dataclass objects containing:
  - `content` — cleaned chunk text
  - `title` — MedlinePlus topic title
  - `url` — canonical MedlinePlus URL
  - `mesh_category` — MeSH classification heading
  - `score` — cosine similarity score

---

### Step 4: Context & Memory Assembly (`app/chatbot/prompts.py` + `app/chatbot/memory.py`)

- **Conversation Memory:** `ConversationMemory` stores the last **5 user–assistant exchange pairs** in a rolling deque. These are prepended to each new prompt to enable multi-turn coherent follow-ups (e.g., *"Tell me more about that last point."*).
- **Context Injection:** The Top-K chunks are formatted as structured MedlinePlus context blocks:
  ```
  --- MEDLINEPLUS CONTEXT ---
  [Source 1: Asthma | MeSH: Respiratory Tract Diseases]
  URL: https://medlineplus.gov/asthma.html
  <chunk text>
  ---
  ```
- The complete prompt payload combines:
  1. System prompt (grounding rules, tone, disclaimer requirement)
  2. Formatted context blocks
  3. Rolling conversation history
  4. Current user question

---

### Step 5: Streaming Token Generation (`app/chatbot/generator.py`)

- The prompt payload is dispatched to **`meta/llama-3.1-8b-instruct`** via the **NVIDIA NIM Engine** (`https://integrate.api.nvidia.com/v1`).
- The OpenAI SDK client (`openai.OpenAI`) is used with `stream=True` for token-by-token delivery.
- **Temperature:** `0.3` — low value ensures factually conservative, low-variance medical answers.
- **Max Tokens:** `1024` — sufficient for detailed answers without runaway length.
- Tokens stream directly into Streamlit's `st.write_stream()` for real-time rendering.
- After streaming completes, **expandable source citation cards** are rendered showing each `RetrievedChunk`'s URL, topic title, and MeSH category.

---

### Step 6: UI Rendering & Auto-Scroll (`app.py`)

- **Auto-Scroll:** After the token stream completes, a `st.components.v1.html(height=0)` JS block fires at 150ms, 600ms, and 1200ms delays to catch the incremental Streamlit DOM renders and scroll the main chat container smoothly to the bottom.
- **Chat History:** All `user` and `assistant` turns are appended to `st.session_state.messages` and re-rendered on each rerun with role-appropriate avatars (🧑 / 🏥).

---

## 2. Prompt Engineering Strategy

### 2.1 System Prompt — Enforced Medical Constraints

```markdown
You are Healthcare AI, an evidence-grounded health education assistant
powered by the MedlinePlus knowledge base from the U.S. National Library of Medicine.

STRICT RULES:
1. GROUNDING: Answer ONLY using the provided MedlinePlus context chunks.
   If the context is insufficient, say clearly: "I don't have enough information..."
2. NO DIAGNOSIS: Never diagnose conditions, prescribe drugs, or recommend dosages.
3. NO PREDICTIONS: Never state cancer risk, life expectancy, or personal prognosis.
4. TONE: Empathetic, professional, accessible (8th grade reading level).
5. FORMAT: Use clear markdown headings, bullet points, and a "Key Points" summary.
6. DISCLAIMER: End every response with: "⚕️ This is health education only. Always
   consult a qualified healthcare provider for personal medical advice."
```

### 2.2 Customisation Points
- **Model swappable:** Change `LLM_MODEL` in `.env` — no code changes required.
- **Temperature configurable:** Lower (`0.1`) for stricter accuracy; higher (`0.7`) for more conversational tone.
- **Top-K configurable:** Increase `TOP_K` for broader context at the cost of prompt length.
- **Memory depth configurable:** `MEMORY_WINDOW_SIZE` controls how many past turns are included.

---

## 3. Response Generation Logic

| Parameter | Value | Effect |
|---|---|---|
| Model | `meta/llama-3.1-8b-instruct` via NVIDIA NIM Engine | Significantly faster than GLM-5.2; free-tier |
| Temperature | `0.3` | Low variance → medically conservative |
| Max Tokens | `1024` | Detailed answer without truncation |
| Streaming | `True` | Real-time token-by-token display |
| Context Window | Top-5 chunks (~1,600 tokens) | Relevant MedlinePlus passages only |
| Memory | Last 5 exchanges | Coherent multi-turn conversations |

Every response is backed by the specific `RetrievedChunk` objects used — shown as clickable source cards beneath the answer, enabling the user to verify the source directly on MedlinePlus.gov.

---

## 4. Safety Measures & Validation

| Tier | Component | Method | Latency | LLM Cost |
|---|---|---|---|---|
| **1A — Emergency** | `guardrails.py` | O(n) regex · 20+ emergency keyword patterns | **0ms** | **$0** |
| **1B — Unsafe Request** | `guardrails.py` | O(n) regex · 15+ unsafe clinical request patterns | **0ms** | **$0** |
| **2 — Prompt Grounding** | `prompts.py` | System prompt rules — context-only answers, no diagnosis | Streaming | Standard token cost |
| **3 — UI Disclaimer** | `app.py` | Mandatory disclaimer banner in sidebar + every response footer | Rendered in UI | N/A |
| **4 — Test Validation** | `tests/run_tests.py` | 25 automated unit and regression tests | CI/CD | N/A |

### 4.1 Test Coverage Summary (25 Tests — All Passing)

```
✅ Poisoning Emergency                  ✅ Pediatric Respiratory Distress
✅ Unresponsive/Unconscious Emergency   ✅ Anaphylaxis Emergency
✅ Heavy Bleeding Emergency             ✅ Medication Modification Refusal
✅ Multi-Condition Diet Refusal         ✅ Disease Risk Prediction Refusal
✅ Treatment Substitution Refusal       ✅ Personal Drug Interaction Refusal
✅ RAG Content Availability Rule        ✅ RAG Evidence Cohesion (Pass/Fail)
✅ RAG Metadata Alignment Rule          ✅ RAG Diversity & Quality Rule
✅ Negative Regression (Safe Questions) ✅ Voice Sanitizer Medical Preservation
```

---

## 5. Development Assumptions

1. **Educational Scope:** The chatbot is built strictly as a **health education assistant** — not a diagnostic medical device. Every design decision enforces this constraint.
2. **Browser Requirement for Voice:** The **"Ask by Voice"** feature requires **Google Chrome** or **Microsoft Edge** (the only browsers that fully support the Web Speech API). Safari and Firefox are not supported.
3. **Data Provenance:** The knowledge base is derived exclusively from the **U.S. National Library of Medicine's MedlinePlus** topic directory. Medical facts are sourced as-is without modification.
4. **API Access Required:** The LLM backend requires network access to NVIDIA NIM (`integrate.api.nvidia.com`) and a valid `NVIDIA_API_KEY`.
5. **CPU-Only Deployment:** The BAAI/bge-m3 embedding model runs on CPU without GPU. Indexing (~10-15 min on CPU) is a one-time operation; subsequent query embeddings are fast (<1s).
6. **Chunk Freshness:** The knowledge base reflects the MedlinePlus XML snapshot from 2026-07-28. To update, re-run the ingestion pipeline with a new XML export.
7. **No PII Storage:** The application stores no user data, credentials, or conversation history beyond the current browser session (`st.session_state` is ephemeral).
