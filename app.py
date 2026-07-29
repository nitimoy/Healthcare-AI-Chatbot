"""
app.py — Healthcare AI Chatbot (Streamlit UI)

Run:
    PYTHONPATH=. streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Environment fixes for macOS / PyTorch / OpenMP ───────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Healthcare AI Assistant | MedlinePlus RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ultra-Premium CSS Theme ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* Streamlit Main Container background */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e293b 0%, #0f172a 60%, #090d16 100%);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #090d16 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399 !important;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 0.3px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Hero Header Gradient */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #34d399 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        text-align: center;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Metric Cards Grid */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #38bdf8 !important;
    }
    .metric-lbl {
        font-size: 0.74rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Disclaimer Card */
    .disclaimer-card {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 0.8rem;
        color: #fca5a5 !important;
        line-height: 1.5;
    }

    /* Source Cards */
    .source-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.84rem;
        line-height: 1.6;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .source-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
    }
    .source-card a {
        color: #38bdf8 !important;
        font-weight: 600;
        text-decoration: none;
    }
    .source-card a:hover {
        text-decoration: underline;
    }

    .chip-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #7dd3fc !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 0.72rem;
        margin-right: 4px;
        margin-top: 4px;
    }

    /* Prompt Suggestion Buttons */
    /* Markdown text contrast fix */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .stMarkdown h2 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 4px;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    div[data-testid="stColumn"] .stButton > button {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.84rem !important;
        padding: 12px 16px !important;
        text-align: left !important;
        width: 100% !important;
        height: 100% !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stColumn"] .stButton > button:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.15) !important;
    }

    /* Custom Chat Input & Bottom Container Fix */
    div[data-testid="stBottom"], div[data-testid="stBottom"] > div {
        background-color: #090d16 !important;
        background: #090d16 !important;
    }

    .stChatInputContainer, div[data-testid="stChatInput"], div[data-baseweb="base-input"] {
        border-radius: 16px !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        background-color: #1e293b !important;
        background: #1e293b !important;
        backdrop-filter: blur(12px) !important;
    }
    .stChatInputContainer textarea, div[data-baseweb="base-input"] textarea, div[data-baseweb="input"] input {
        color: #f1f5f9 !important;
        background: transparent !important;
    }
    .stChatInputContainer textarea::placeholder {
        color: #94a3b8 !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3) !important;
    }

    /* Selectbox (Voice Language), Text Inputs & Dropdown Popover Reset */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSelectbox"] > div > div,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        background: #1e293b !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }
    div[data-baseweb="select"] * {
        color: #f1f5f9 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #38bdf8 !important;
    }

    /* Dropdown Popover & Option Menu Fix */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
    }
    li[role="option"],
    div[role="option"],
    li[role="option"] *,
    div[role="option"] * {
        color: #f1f5f9 !important;
        background-color: #0f172a !important;
    }
    li[role="option"]:hover,
    div[role="option"]:hover,
    li[aria-selected="true"],
    div[aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }

    /* Text Inputs (Editable Voice Question Box) */
    div[data-testid="stTextInput"] input,
    div[data-baseweb="input"] input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    /* Sidebar Clear Button Styling */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        color: #fca5a5 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(239, 68, 68, 0.3) !important;
        border-color: rgba(239, 68, 68, 0.7) !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.3) !important;
    }

    /* Expander Source Header */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Defaults
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []       # [{role, content, sources?}]
if "pending_q" not in st.session_state:
    st.session_state.pending_q = ""      # populated by suggestion card clicks
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
from app.chatbot.voice import VoiceSession, clean_voice_transcript

if "generator" not in st.session_state:
    st.session_state.generator = None    # MedicalChatGenerator, lazily loaded
if "voice_session" not in st.session_state:
    st.session_state.voice_session = VoiceSession()
if "voice_language" not in st.session_state:
    st.session_state.voice_language = "en-US"
if "auto_submit_voice" not in st.session_state:
    st.session_state.auto_submit_voice = True

SUGGESTION_CARDS = [
    ("🩺 Diabetes Symptoms", "What are the early signs and symptoms of diabetes?"),
    ("🫀 Hypertension Care", "What is high blood pressure and how is it managed?"),
    ("🫁 Asthma Guidance", "What are the common symptoms and triggers of asthma?"),
    ("🔥 Burn First Aid", "How do I treat a minor second-degree burn at home?"),
    ("☀️ Heat Illness", "How is dehydration different from heat stroke?"),
    ("🥗 Heart Healthy Diet", "What foods are recommended for a healthy heart?"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Generator — Server-level cache (BGE-M3 + ChromaDB, loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _get_cached_retriever():
    """Load BGE-M3 + open ChromaDB once per server process (shared across sessions).

    show_spinner=False so this never blocks page rendering.  Loading status
    is communicated via ``st.session_state.retriever_loading``.
    """
    from app.chatbot.retriever import Retriever
    r = Retriever()
    r._get_collection()   # warm-up: SentenceTransformerEmbeddingFunction + ChromaDB
    return r


def get_generator():
    """Return this session’s MedicalChatGenerator.

    The first call builds the generator using the already-cached Retriever
    (or raises a friendly error if the retriever hasn’t loaded yet).
    """
    if st.session_state.generator is None:
        try:
            from app.chatbot.generator import MedicalChatGenerator
            from app.chatbot.memory import ConversationMemory
            retriever = _get_cached_retriever()    # fast once cached
            st.session_state.generator = MedicalChatGenerator(
                retriever=retriever,
                memory=ConversationMemory(),
            )
        except EnvironmentError as exc:
            st.error(
                f"**Configuration error:**\n\n{exc}\n\n"
                "Add your NVIDIA API key to `.env`:\n"
                "```\nNVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx\n```\n"
                "Then restart Streamlit."
            )
            st.stop()
        except Exception as exc:
            import traceback
            st.error(f"**Error loading knowledge base:**\n\n```\n{traceback.format_exc()}\n```")
            st.stop()
    return st.session_state.generator



# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏥 Healthcare AI")
    st.markdown(
        '<div class="status-badge"><span class="status-dot"></span>MedlinePlus Grounded KB</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        '<div class="disclaimer-card">⚠️ <strong>Medical Disclaimer</strong><br>'
        "General health education only. Does <strong>not</strong> diagnose, "
        "prescribe, or replace professional medical advice.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### ⚙️ Retrieval Settings")
    st.session_state.top_k = st.slider(
        "Sources to retrieve (Top-K)",
        min_value=3,
        max_value=10,
        value=st.session_state.top_k,
        help="Higher values fetch more candidate source cards.",
    )

    st.markdown("#### 🎤 Voice Settings")
    lang_choice = st.selectbox(
        "Voice Language",
        options=["English (US)", "English (India)", "English (UK)"],
        index=0 if st.session_state.voice_language == "en-US" else (1 if st.session_state.voice_language == "en-IN" else 2),
    )
    lang_map = {"English (US)": "en-US", "English (India)": "en-IN", "English (UK)": "en-GB"}
    st.session_state.voice_language = lang_map[lang_choice]

    st.session_state.auto_submit_voice = st.toggle(
        "⚡ Auto-Submit Spoken Question",
        value=st.session_state.auto_submit_voice,
        help="When enabled, questions are submitted automatically after speech stops.",
    )

    st.markdown("#### 📊 Knowledge Engine")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">1,017</div>'
            '<div class="metric-lbl">Topics</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">2,010</div>'
            '<div class="metric-lbl">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="metric-card"><div class="metric-val" style="font-size:0.95rem;color:#34d399!important">Meta Llama 3.1 8B</div>'
        '<div class="metric-lbl">NVIDIA NIM Engine</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.generator is not None:
            st.session_state.generator.memory.clear()
        st.rerun()

    st.markdown(
        "<div style='font-size:.74rem;color:#64748b;margin-top:14px;text-align:center'>"
        "Sourced from <a href='https://medlineplus.gov' target='_blank' style='color:#38bdf8'>MedlinePlus</a><br>"
        "U.S. National Library of Medicine</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div style="text-align:center;margin-top:10px;margin-bottom:20px">'
    '<div class="status-badge"><span class="status-dot"></span>Production Healthcare RAG Assistant</div>'
    '<h1 class="hero-title">Healthcare AI Assistant</h1>'
    '<p class="hero-subtitle">Evidence-grounded health education powered by MedlinePlus knowledge base</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Source Card Renderer
# ─────────────────────────────────────────────────────────────────────────────


def render_sources(chunks: list) -> None:
    """Render glassmorphism cards for retrieved source citations."""
    if not chunks:
        return
    with st.expander(f"📚 Verified Sources ({len(chunks)} retrieved)", expanded=False):
        for i, c in enumerate(chunks, 1):
            mesh_chips = "".join(
                [f'<span class="chip-badge">🏷️ {m}</span>' for m in (c.mesh or [])]
            )
            group_chips = "".join(
                [f'<span class="chip-badge">📂 {g}</span>' for g in (c.groups or [])]
            )
            st.markdown(
                f'<div class="source-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<strong>{i}. <a href="{c.url}" target="_blank">{c.title}</a></strong>'
                f'<span style="color:#34d399;font-weight:600;font-size:0.78rem;background:rgba(52,211,153,0.1);padding:2px 8px;border-radius:10px;border:1px solid rgba(52,211,153,0.3)">'
                f'{c.similarity_score:.0%} match</span>'
                f'</div>'
                f'<div style="margin-top:6px;font-size:0.78rem;color:#94a3b8">'
                f'🔗 <a href="{c.url}" target="_blank" style="font-size:0.78rem">{c.url}</a>'
                f'</div>'
                f'<div style="margin-top:6px">{mesh_chips}{group_chips}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Quick Action Hero Cards (Shown when history is empty)
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("<h4 style='color:#e2e8f0;margin-bottom:12px;text-align:center'>💡 Explore Common Healthcare Topics</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for idx, (label, prompt_text) in enumerate(SUGGESTION_CARDS):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(f"{label}\n\n_{prompt_text}_", key=f"hero_card_{idx}"):
                st.session_state.pending_q = prompt_text
                st.rerun()
    st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Warm-up: pre-load BGE-M3 + ChromaDB on first visit (no question needed yet)
# ─────────────────────────────────────────────────────────────────────────────
# On first page load, generator is None. We warm up the cache here so that
# the spinner shows inside the main content area (not blocking the sidebar or
# header). After the load completes we rerun so the page renders in full.
if st.session_state.generator is None and not st.session_state.get("pending_q"):
    with st.spinner("🔄 Loading medical knowledge base & embedding model (first load only)…"):
        get_generator()   # triggers _get_cached_retriever() and caches generator
    st.rerun()            # re-render page cleanly now that model is ready



# ─────────────────────────────────────────────────────────────────────────────
# Render Chat History
# ─────────────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🏥"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ─────────────────────────────────────────────────────────────────────────────
# Input Handling & Voice Component
# ─────────────────────────────────────────────────────────────────────────────

from streamlit_mic_recorder import speech_to_text

# Deduplication guard: prevents re-processing on reruns from other widgets.
if "last_processed_voice" not in st.session_state:
    st.session_state.last_processed_voice = ""

# ── Render voice recorder (hidden in normal flow; JS positions it as FAB) ────
# We keep the widget in the DOM so Streamlit manages the iframe permissions
# and return value. JS below detaches the column from document flow and pins
# it as a floating action button above the chat input.
_vc1, _vc2, _vc3 = st.columns([1, 2, 1])
with _vc2:
    voice_text = speech_to_text(
        language=st.session_state.voice_language,
        start_prompt="🎤 Ask by Voice",
        stop_prompt="🔴 Stop",
        just_once=True,
        use_container_width=True,
        key="st_voice_recorder",
    )

# ── JS: reposition mic button as FAB above chat input (bottom-right corner) ──
st.components.v1.html("""
<script>
(function() {
  var d = window.parent.document;

  function positionMicFAB() {
    // Find the streamlit_mic_recorder iframe by src
    var iframes = d.querySelectorAll('iframe');
    var micFrame = null;
    for (var i = 0; i < iframes.length; i++) {
      var s = iframes[i].src || '';
      if (s.indexOf('streamlit_mic_recorder') !== -1) {
        micFrame = iframes[i];
        break;
      }
    }
    if (!micFrame) return;

    // Walk up to the stColumn and its parent stHorizontalBlock
    var col = micFrame.closest('[data-testid="stColumn"]');
    var row = col && col.closest('[data-testid="stHorizontalBlock"]');

    if (col && col.getAttribute('data-mic-fab') !== '1') {
      col.setAttribute('data-mic-fab', '1');
      col.style.position   = 'fixed';
      col.style.bottom     = '120px';
      col.style.right      = '22px';
      col.style.width      = '190px';
      col.style.zIndex     = '1001';
      col.style.background = 'transparent';
      col.style.padding    = '0';
    }
    if (row && row.getAttribute('data-mic-row') !== '1') {
      row.setAttribute('data-mic-row', '1');
      row.style.height    = '0';
      row.style.overflow  = 'visible';
      row.style.minHeight = '0';
      row.style.margin    = '0';
      row.style.padding   = '0';
    }
  }

  // Run immediately and after Streamlit rerenders
  setTimeout(positionMicFAB, 600);
  setTimeout(positionMicFAB, 1400);

  // MutationObserver re-applies positioning after each Streamlit rerun
  // (Streamlit does incremental DOM updates, not full reloads)
  var obs = new MutationObserver(function() {
    setTimeout(positionMicFAB, 250);
  });
  obs.observe(d.body, { childList: true, subtree: true });
})();
</script>
""", height=0)

# ── Process returned transcript ───────────────────────────────────────────────
_new_voice = (voice_text or "").strip()
if _new_voice and _new_voice != st.session_state.last_processed_voice:
    st.session_state.last_processed_voice = _new_voice
    clean_text = clean_voice_transcript(_new_voice)
    if clean_text:
        if st.session_state.auto_submit_voice:
            # Set pending_q — pickup block below runs in the same script pass.
            st.session_state.voice_session.clean_transcript = ""
            st.session_state.voice_session.state = "Idle"
            st.session_state.pending_q = clean_text
        else:
            st.session_state.voice_session.raw_transcript = _new_voice
            st.session_state.voice_session.clean_transcript = clean_text
            st.session_state.voice_session.state = "Recognised"

# ── Manual review box (auto-submit OFF only) ──────────────────────────────────
if st.session_state.voice_session.clean_transcript:
    lang_label = (
        "English (India)" if st.session_state.voice_language == "en-IN"
        else ("English (UK)" if st.session_state.voice_language == "en-GB" else "English (US)")
    )
    st.markdown(
        f'<div style="font-size:0.83rem;color:#34d399;font-weight:600;margin-top:8px;margin-bottom:4px">'
        f'✓ Recognised ({lang_label}) &nbsp;|&nbsp; Confidence: —'
        f'</div>',
        unsafe_allow_html=True,
    )
    voice_q_edited = st.text_input(
        "🎙️ Spoken Question — Review or Edit before Submitting:",
        value=st.session_state.voice_session.clean_transcript,
        key="live_voice_text_input",
    )
    col_submit, col_clear = st.columns([2, 1])
    with col_submit:
        if st.button("🚀 Submit Spoken Question", key="btn_submit_voice_spoken"):
            st.session_state.pending_q = voice_q_edited
            st.session_state.voice_session.clean_transcript = ""
            st.session_state.voice_session.state = "Idle"
            st.rerun()
    with col_clear:
        if st.button("🔄 Clear", key="btn_clear_voice_spoken"):
            st.session_state.voice_session.clean_transcript = ""
            st.session_state.voice_session.state = "Idle"
            st.rerun()

# ── Pick up the question (voice auto-submit or typed) ─────────────────────────
question = ""
if st.session_state.pending_q:
    question = st.session_state.pending_q
    st.session_state.pending_q = ""

typed = st.chat_input("Ask a health question (e.g. 'What are the symptoms of asthma?')")
if typed and typed.strip():
    question = clean_voice_transcript(typed.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Response Generator Execution
# ─────────────────────────────────────────────────────────────────────────────

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🏥"):
        try:
            gen = get_generator()
            with st.spinner("🧠 Searching MedlinePlus knowledge base & generating answer..."):
                chunks, token_stream = gen.stream_response(
                    question=question,
                    k=st.session_state.top_k,
                )
                full_response = st.write_stream(token_stream)
            render_sources(chunks)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": chunks}
            )
        except Exception as exc:
            import traceback
            err_msg = f"❌ **An error occurred during generation:**\n\n```\n{exc}\n```"
            st.error(err_msg)
            print("STREAMLIT GENERATION ERROR:", exc, file=sys.stderr)
            traceback.print_exc()
            st.session_state.messages.append(
                {"role": "assistant", "content": err_msg, "sources": []}
            )

    # ── Auto-scroll to bottom so answer is immediately visible ────────────────
    st.components.v1.html("""
    <script>
    (function() {
      function scrollBottom() {
        var d = window.parent.document;
        // Scroll the main Streamlit content container
        var main = d.querySelector('section.main');
        if (main) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
        // Also scroll any chat message containers
        d.querySelectorAll('[data-testid="stChatMessageContainer"]').forEach(function(el) {
          el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
        });
      }
      // Fire multiple times to catch async rendering of markdown/stream
      setTimeout(scrollBottom, 150);
      setTimeout(scrollBottom, 600);
      setTimeout(scrollBottom, 1200);
    })();
    </script>
    """, height=0)
