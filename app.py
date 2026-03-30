# app.py
# Run with: streamlit run app.py

import os, sys, json, requests as _req
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

import streamlit as st

# ── Cached heavy loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_embed_model():
    """Load SentenceTransformer once for the whole app session."""
    from embeddings.sentence_embeddings import _get_model
    return _get_model()


@st.cache_resource(show_spinner=False)
def _load_spacy():
    """Load spaCy model once."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


@st.cache_resource(show_spinner=False)
def _load_nltk():
    """Download NLTK data once per session."""
    import nltk
    for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


# Call these immediately so they warm up in the background
# while the user is reading the landing page:
_load_nltk()
# Uncomment the next two lines if startup time is acceptable —
# they pre-warm the models before the user clicks "Process":
# _load_embed_model()
# _load_spacy()


# ── Image helper ─────────────────────────────────────────────────
def _fetch_image(query: str):
    try:
        resp = _req.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ','_')}",
            timeout=5, headers={"User-Agent": "StudyMateAI/1.0"}
        )
        if resp.status_code == 200:
            data  = resp.json()
            thumb = data.get("thumbnail", {}).get("source")
            width = data.get("thumbnail", {}).get("width", 0)
            if thumb and width >= 100:
                return thumb
    except Exception:
        pass
    return None


# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global markdown ── */
.stMarkdown h1{font-size:1.4rem!important;font-weight:800!important;color:#e2e8f0!important;margin:1.2rem 0 0.6rem!important}
.stMarkdown h2{font-size:1.1rem!important;font-weight:700!important;color:#e2e8f0!important;border-bottom:1px solid #1e1e2e;padding-bottom:5px;margin:1.2rem 0 0.5rem!important}
.stMarkdown h3{font-size:0.97rem!important;font-weight:700!important;color:#a5b4fc!important;margin:0.9rem 0 0.35rem!important}
.stMarkdown h4{font-size:0.9rem!important;font-weight:700!important;color:#94a3b8!important;margin:0.7rem 0 0.3rem!important}
.stMarkdown p{color:#cbd5e1!important;font-size:0.93rem!important;line-height:1.85!important;margin:0.35rem 0!important}
.stMarkdown ul,.stMarkdown ol{color:#cbd5e1!important;font-size:0.93rem!important;line-height:1.8!important;padding-left:1.5rem!important;margin:0.3rem 0 0.6rem!important}
.stMarkdown li{margin:5px 0!important;color:#cbd5e1!important}
.stMarkdown li strong{color:#e2e8f0!important}
.stMarkdown strong{color:#e2e8f0!important;font-weight:700!important}
.stMarkdown em{color:#a5b4fc!important}
.stMarkdown code{background:#1e1e2e!important;color:#a5b4fc!important;padding:2px 7px!important;border-radius:5px!important;font-size:0.85rem!important;font-family:monospace!important}
.stMarkdown pre{background:#0d0d14!important;border:1px solid #1e1e2e!important;border-radius:10px!important;padding:1rem!important;overflow-x:auto!important}
.stMarkdown blockquote{border-left:3px solid #6366f1!important;background:rgba(99,102,241,0.06)!important;padding:0.6rem 1.1rem!important;margin:0.6rem 0!important;border-radius:0 10px 10px 0!important}
.stMarkdown blockquote p{color:#a5b4fc!important;font-style:normal!important;font-weight:600!important;margin:0!important}
.stMarkdown table{width:100%!important;border-collapse:collapse!important;margin:0.8rem 0!important;font-size:0.88rem!important}
.stMarkdown th{background:#1a1a2e!important;color:#a5b4fc!important;padding:9px 14px!important;text-align:left!important;font-weight:700!important;border:1px solid #2d2d4e!important}
.stMarkdown td{color:#cbd5e1!important;padding:7px 14px!important;border:1px solid #1e1e2e!important}
.stMarkdown tr:nth-child(even) td{background:rgba(99,102,241,0.03)!important}
.stMarkdown hr{border-color:#1e1e2e!important;margin:1rem 0!important}

/* ── Reset & Base ── */
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif!important;background:#0d0d14!important;color:#e2e8f0!important}
footer{visibility:hidden}
[data-testid="stHeader"]{background:#0a0a10!important;border-bottom:1px solid #1e1e2e!important}
[data-testid="stDecoration"]{display:none!important}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:#0a0a10!important;border-right:1px solid #1e1e2e!important}
[data-testid="stSidebar"] *{color:#cbd5e1!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#f1f5f9!important}
[data-testid="stSidebar"] .stMarkdown p{color:#94a3b8!important;font-size:0.85rem}
[data-testid="stSidebar"] label{color:#94a3b8!important;font-size:0.85rem}
[data-testid="stFileUploader"]{background:#13131f!important;border:1.5px dashed #2d2d4e!important;border-radius:12px!important}
[data-testid="stFileUploaderDropzone"]{background:transparent!important}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:#13131f!important;border-radius:14px!important;padding:5px!important;border:1px solid #1e1e2e!important;gap:4px!important}
.stTabs [data-baseweb="tab"]{border-radius:10px!important;padding:9px 22px!important;font-weight:600!important;font-size:0.88rem!important;color:#64748b!important;background:transparent!important;transition:all 0.2s!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;color:#ffffff!important;box-shadow:0 4px 12px rgba(99,102,241,0.4)!important}

/* ── Buttons ── */
.stButton>button{border-radius:10px!important;font-weight:600!important;font-size:0.88rem!important;border:none!important;transition:all 0.2s ease!important;color:#e2e8f0!important;background:#1e1e2e!important}
.stButton>button:hover{background:#2a2a3e!important;transform:translateY(-1px)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%)!important;color:white!important;box-shadow:0 4px 14px rgba(99,102,241,0.35)!important}
.stButton>button[kind="primary"]:hover{box-shadow:0 6px 20px rgba(99,102,241,0.5)!important;transform:translateY(-2px)!important}

/* ── Inputs ── */
.stTextInput>div>div>input{background:#13131f!important;border:1.5px solid #2d2d4e!important;border-radius:10px!important;color:#e2e8f0!important;font-size:0.9rem!important}
.stTextInput>div>div>input:focus{border-color:#6366f1!important;box-shadow:0 0 0 3px rgba(99,102,241,0.15)!important}
.stTextInput>div>div>input::placeholder{color:#475569!important}
.stSelectbox>div>div{background:#13131f!important;border:1.5px solid #2d2d4e!important;border-radius:10px!important;color:#e2e8f0!important}
.stTextArea textarea{background:#13131f!important;border:1.5px solid #2d2d4e!important;border-radius:10px!important;color:#94a3b8!important}
.stSlider [role="slider"]{background:#6366f1!important}

/* ── Radio ── */
.stRadio label{color:#cbd5e1!important;font-size:0.9rem!important}
.stRadio [data-testid="stMarkdownContainer"] p{color:#cbd5e1!important}

/* ── Expander ── */
[data-testid="stExpander"]{background:#13131f!important;border:1px solid #1e1e2e!important;border-radius:12px!important}
[data-testid="stExpander"] summary{color:#94a3b8!important}
[data-testid="stExpander"] summary:hover{color:#e2e8f0!important}

/* ── Chat ── */
[data-testid="stChatInput"]>div{background:#13131f!important;border:1.5px solid #2d2d4e!important;border-radius:14px!important}
[data-testid="stChatInput"] textarea{background:transparent!important;color:#e2e8f0!important}
[data-testid="stChatMessage"]{background:transparent!important}

/* ── Progress / Spinner ── */
.stProgress>div>div>div{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;border-radius:99px!important}
.stProgress>div>div{background:#1e1e2e!important;border-radius:99px!important}
.stSpinner>div{border-top-color:#6366f1!important}
.stSuccess{background:#052e16!important;border-color:#166534!important;color:#86efac!important}
.stWarning{background:#1c1006!important;border-color:#92400e!important;color:#fcd34d!important}
.stInfo{background:#0c1a35!important;border-color:#1e3a5f!important;color:#93c5fd!important}
.stCaption,[data-testid="stCaptionContainer"] p{color:#475569!important}
hr{border-color:#1e1e2e!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#0d0d14}
::-webkit-scrollbar-thumb{background:#2d2d4e;border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:#6366f1}

/* ── Custom components ── */
.hero{background:linear-gradient(135deg,#0f0f1a 0%,#13131f 40%,#0f1729 100%);border:1px solid #1e1e2e;border-radius:24px;padding:3.5rem 3rem;margin-bottom:2rem;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-80px;right:-80px;width:350px;height:350px;background:radial-gradient(circle,rgba(99,102,241,0.12) 0%,transparent 65%);border-radius:50%;pointer-events:none}
.hero h1{font-size:3rem;font-weight:900;background:linear-gradient(135deg,#e2e8f0 30%,#a5b4fc 70%,#c4b5fd 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 0.8rem 0;line-height:1.1;letter-spacing:-1px}
.hero p{font-size:1.05rem;color:#64748b;max-width:580px;line-height:1.7;margin:0}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);color:#a5b4fc;padding:5px 16px;border-radius:99px;font-size:0.75rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:1.2rem}
.section-title{font-size:1rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;margin:2rem 0 1rem 0;display:flex;align-items:center;gap:8px}
.section-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e1e2e,transparent);margin-left:4px}
.card{background:#13131f;border:1px solid #1e1e2e;border-radius:16px;padding:1.5rem;transition:border-color 0.2s}
.feat-card{background:#13131f;border:1px solid #1e1e2e;border-radius:20px;padding:1.8rem 1.5rem;height:100%;transition:all 0.25s;position:relative;overflow:hidden}
.feat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#6366f1,transparent);opacity:0;transition:opacity 0.25s}
.feat-card:hover{border-color:#3730a3;transform:translateY(-4px);box-shadow:0 12px 30px rgba(99,102,241,0.12)}
.feat-card:hover::before{opacity:1}
.feat-icon{font-size:2.2rem;margin-bottom:1rem}
.feat-title{font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem}
.feat-desc{font-size:0.83rem;color:#64748b;line-height:1.6}
.t-pill{display:inline-block;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);color:#a5b4fc;padding:3px 12px;border-radius:99px;font-size:0.76rem;font-weight:600;margin:3px}
.m-card{background:#13131f;border:1px solid #1e1e2e;border-radius:14px;padding:1.2rem;text-align:center}
.m-val{font-size:2rem;font-weight:800}
.m-lbl{font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px}
.mcq-wrap{background:#13131f;border:1px solid #1e1e2e;border-left:3px solid #6366f1;border-radius:0 14px 14px 0;padding:1.2rem 1.5rem;margin-bottom:10px}
.mcq-num{font-size:0.7rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px}
.mcq-q{font-size:0.97rem;font-weight:600;color:#e2e8f0}
.score-card{background:#13131f;border:1px solid #1e1e2e;border-radius:20px;padding:2rem;text-align:center}
.score-num{font-size:3.5rem;font-weight:900;background:linear-gradient(135deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.score-sub{font-size:0.85rem;color:#475569}
.file-bar{background:#13131f;border:1px solid #1e1e2e;border-radius:16px;padding:1.2rem 1.8rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;margin-bottom:1.5rem}
.fname{color:#e2e8f0;font-size:1rem;font-weight:700}
.flabel{color:#475569;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px}
.fval{color:#a5b4fc;font-size:1.1rem;font-weight:800}
.stat-g{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.stat-cell{background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.12);border-radius:10px;padding:10px 8px;text-align:center}
.stat-v{font-size:1.3rem;font-weight:800;color:#a5b4fc}
.stat-l{font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-top:1px}
.step-track{display:flex;gap:6px;flex-wrap:wrap;margin:0.8rem 0 1.2rem}
.step-item{display:flex;align-items:center;gap:5px;padding:4px 12px;border-radius:99px;font-size:0.78rem;font-weight:600;border:1px solid #1e1e2e;color:#475569;background:#13131f}
.step-item.done{background:rgba(34,197,94,0.08);border-color:rgba(34,197,94,0.2);color:#86efac}
.step-item.active{background:rgba(99,102,241,0.1);border-color:rgba(99,102,241,0.25);color:#a5b4fc}
.step-dot{width:6px;height:6px;border-radius:50%;background:currentColor}
.ent-card{background:#13131f;border:1px solid #1e1e2e;border-radius:14px;padding:1.1rem;min-height:130px}
.exp-box{background:#13131f;border:1px solid #1e1e2e;border-radius:14px;padding:1.4rem 1.6rem;color:#cbd5e1;font-size:0.95rem;line-height:1.8}
.exp-label{font-size:0.7rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px}
.cl-row{border-radius:10px;padding:10px 14px;margin:5px 0;display:flex;gap:10px;align-items:flex-start}
.rag-badge{background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:10px;padding:10px 16px;display:flex;gap:10px;align-items:center;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "raw_text": None, "pipeline_result": None,
        "faiss_index": None, "chunks": [],
        "file_name": None, "chat_history": [],
        "processed": False, "summary_result": None,
        "summary_style": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 StudyMate AI")
    st.markdown("---")

    s1 = "done"   if st.session_state.raw_text  else "active"
    s2 = "done"   if st.session_state.processed else ("active" if st.session_state.raw_text else "")
    s3 = "active" if st.session_state.processed else ""
    st.markdown(
        f'<div class="step-track">'
        f'<div class="step-item {s1}"><span class="step-dot"></span>Upload</div>'
        f'<div class="step-item {s2}"><span class="step-dot"></span>Process</div>'
        f'<div class="step-item {s3}"><span class="step-dot"></span>Explore</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📁 Upload Notes")
    st.caption("You can upload multiple PDFs and TXT files at once")
    uploaded_files = st.file_uploader(
        "PDF or TXT", type=["pdf","txt"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_key = "_".join(sorted(f.name for f in uploaded_files))
        if st.session_state.file_name != new_key:
            from utils.pdf_reader import load_uploaded_file
            combined_text, loaded, failed = "", [], []
            with st.spinner(f"Reading {len(uploaded_files)} file(s)..."):
                for uf in uploaded_files:
                    txt = load_uploaded_file(uf)
                    if txt and len(txt.strip()) > 50:
                        combined_text += f"\n\n=== {uf.name} ===\n\n{txt}"
                        loaded.append(uf.name)
                    else:
                        failed.append(uf.name)

            if combined_text.strip():
                st.session_state.raw_text  = combined_text.strip()
                st.session_state.file_name = new_key
                st.session_state.processed = False
                st.session_state.summary_result = None
                for name in loaded:
                    st.success(f"✅ {name}")
                for name in failed:
                    st.error(f"❌ {name}")
                total_words = len(combined_text.split())
                st.markdown(
                    f'<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);'
                    f'border-radius:10px;padding:8px 12px;margin-top:6px;font-size:0.82rem;color:#a5b4fc;">'
                    f'📚 <b>{len(loaded)} file(s)</b> · {total_words:,} words</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.error("❌ None of the files could be read.")

    if st.session_state.raw_text and not st.session_state.processed:
        st.markdown("---")
        st.markdown("### ⚡ Process Notes")
        st.markdown(
            '<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);'
            'border-radius:10px;padding:10px 14px;margin-bottom:12px;">'
            '<div style="font-size:0.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;'
            'letter-spacing:1px;margin-bottom:8px;">What happens:</div>'
            '<div style="font-size:0.78rem;color:#64748b;line-height:1.8;">'
            '🧹 Clean &amp; tokenize text<br>🔤 Lemmatize &amp; extract keywords<br>'
            '🧠 Generate sentence embeddings<br>🗂️ Build FAISS vector index<br>'
            '✅ All 4 tabs unlocked</div></div>',
            unsafe_allow_html=True,
        )

        if st.button("▶  Start Processing", type="primary", use_container_width=True):
            status_box = st.empty()

            def show_step(step_num, detail, pct):
                icons  = ["🧹","🔤","🧠","🗂️","✅"]
                labels = ["Clean & Tokenize","Lemmatize & Keywords",
                          "Sentence Embeddings","Build FAISS Index","Complete"]
                rows = ""
                for j in range(5):
                    n = j + 1
                    if n < step_num:
                        color, dot, tc = "#22c55e","✓","#86efac"
                    elif n == step_num:
                        color, dot, tc = "#6366f1","●","#a5b4fc"
                    else:
                        color, dot, tc = "#1e1e2e","○","#334155"
                    rows += (
                        f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #0d0d14;">'
                        f'<div style="width:22px;height:22px;border-radius:50%;background:{color};display:flex;'
                        f'align-items:center;justify-content:center;font-size:0.7rem;font-weight:800;color:white;flex-shrink:0;">{dot}</div>'
                        f'<div style="font-size:0.82rem;font-weight:600;color:{tc};">{icons[j]} {labels[j]}</div></div>'
                    )
                status_box.markdown(
                    f'<div style="background:#13131f;border:1px solid #1e1e2e;border-radius:14px;padding:1.2rem 1.4rem;margin:8px 0;">'
                    f'<div style="font-size:0.7rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">'
                    f'Processing · Step {step_num} of 4</div>{rows}'
                    f'<div style="margin-top:12px;background:#0d0d14;border-radius:99px;height:6px;overflow:hidden;">'
                    f'<div style="background:linear-gradient(90deg,#6366f1,#8b5cf6);width:{pct}%;height:100%;border-radius:99px;"></div></div>'
                    f'<div style="font-size:0.78rem;color:#475569;margin-top:8px;">⏳ {detail}</div></div>',
                    unsafe_allow_html=True,
                )

            show_step(1, "Cleaning and tokenizing text...", 10)
            from preprocessing.pipeline import run_preprocessing_pipeline
            result = run_preprocessing_pipeline(st.session_state.raw_text)
            st.session_state.pipeline_result = result

            show_step(2, f"Found {result['sentence_count']} sentences, {len(result['keywords'])} keywords...", 35)
            from generative.explainer import set_notes_context
            set_notes_context(result["sentences"])

            show_step(3, "Loading SentenceTransformer, encoding chunks...", 60)

            show_step(4, "Building FAISS vector index...", 85)
            from rag.indexer import index_document
            idx, chunks = index_document(st.session_state.raw_text, chunk_size=3, overlap=1, save=True)
            st.session_state.faiss_index = idx
            st.session_state.chunks      = chunks

            w, s = result["word_count"], result["sentence_count"]
            kw, ch = len(result["keywords"]), len(chunks)
            status_box.markdown(
                f'<div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);'
                f'border-radius:14px;padding:1.2rem 1.4rem;margin:8px 0;">'
                f'<div style="font-size:0.72rem;font-weight:700;color:#22c55e;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:10px;">✅ Processing Complete!</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
                f'<div style="background:rgba(34,197,94,0.06);border-radius:8px;padding:8px;text-align:center;">'
                f'<div style="font-size:1.2rem;font-weight:800;color:#86efac;">{w:,}</div>'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">Words</div></div>'
                f'<div style="background:rgba(34,197,94,0.06);border-radius:8px;padding:8px;text-align:center;">'
                f'<div style="font-size:1.2rem;font-weight:800;color:#86efac;">{s}</div>'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">Sentences</div></div>'
                f'<div style="background:rgba(34,197,94,0.06);border-radius:8px;padding:8px;text-align:center;">'
                f'<div style="font-size:1.2rem;font-weight:800;color:#86efac;">{kw}</div>'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">Keywords</div></div>'
                f'<div style="background:rgba(34,197,94,0.06);border-radius:8px;padding:8px;text-align:center;">'
                f'<div style="font-size:1.2rem;font-weight:800;color:#86efac;">{ch}</div>'
                f'<div style="font-size:0.65rem;color:#475569;text-transform:uppercase;">Chunks</div></div>'
                f'</div><div style="font-size:0.78rem;color:#475569;margin-top:10px;text-align:center;">'
                f'All 4 tabs unlocked 🚀</div></div>',
                unsafe_allow_html=True,
            )
            time.sleep(1.2)
            st.session_state.processed = True
            st.rerun()

    if st.session_state.processed and st.session_state.pipeline_result:
        r = st.session_state.pipeline_result
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        st.markdown(
            f'<div class="stat-g">'
            f'<div class="stat-cell"><div class="stat-v">{r["word_count"]:,}</div><div class="stat-l">Words</div></div>'
            f'<div class="stat-cell"><div class="stat-v">{r["sentence_count"]}</div><div class="stat-l">Sentences</div></div>'
            f'<div class="stat-cell"><div class="stat-v">{len(r["keywords"])}</div><div class="stat-l">Keywords</div></div>'
            f'<div class="stat-cell"><div class="stat-v">{len(st.session_state.chunks)}</div><div class="stat-l">Chunks</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.processed:
        st.markdown("---")
        if st.button("🔄 Upload New File", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown("---")
    st.caption("Made with 🤍 · Python · NLP · HuggingFace · FAISS · Claude AI")


# ════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ════════════════════════════════════════════════════════════════
if not st.session_state.processed:
    st.markdown(
        '<div class="hero">'
        '<div class="hero-badge">✦ AI-Powered · NLP · Transformers · Agentic RAG</div>'
        '<h1>StudyMate AI</h1>'
        '<p>Upload your study notes and instantly get structured summaries, exam-level quizzes, '
        'named entity maps, and a multi-tool AI agent that answers questions from your notes — '
        'grounded, precise, zero hallucination.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">✦ Features</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    feats = [
        ("📋","Smart Summary","Claude generates structured summaries with ## sections, bold concepts, bullet points, and key takeaways — not just extracted sentences."),
        ("📝","Quiz Generator","Exam-level MCQs mixing Application, Scenario, Cause-Effect, Comparison and Error-Spotting types with plausible distractors and full explanations."),
        ("🔑","Key Concepts","BERT NER extracts entities. Claude explains any concept with definition, mechanism, examples and takeaway. ANN classifies sentence types."),
        ("💬","Agent Chat","Multi-tool AI agent: decides which tools to use, chains search_notes → explain_concept → compare_concepts, shows its reasoning."),
    ]
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],feats):
        col.markdown(
            f'<div class="feat-card"><div class="feat-icon">{icon}</div>'
            f'<div class="feat-title">{title}</div>'
            f'<div class="feat-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">⚙️ Pipeline</div>', unsafe_allow_html=True)
        pipeline_html = '<div style="background:#13131f;border:1px solid #1e1e2e;border-radius:20px;padding:1.8rem;">'
        steps = [
            ("1","#6366f1","Upload","PDF or TXT notes uploaded via the sidebar"),
            ("2","#6366f1","NLP Pipeline","Clean → Tokenize → Stopwords → Lemmatize → Keywords"),
            ("3","#6366f1","Embed + Index","SentenceTransformers → 384-dim vectors → FAISS"),
            ("4","#22c55e","Explore","All 4 tabs unlocked — summarize, quiz, chat, classify"),
        ]
        for num,color,title,desc in steps:
            line = "" if num=="4" else f'<div style="width:2px;height:24px;background:linear-gradient(180deg,{color}60,#1e1e2e);margin:3px auto 3px 15px;"></div>'
            pipeline_html += (
                f'<div style="display:flex;gap:14px;align-items:flex-start;">'
                f'<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">'
                f'<div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,{color},{color}99);'
                f'display:flex;align-items:center;justify-content:center;font-size:0.78rem;font-weight:800;color:white;">{num}</div>'
                f'{line}</div>'
                f'<div style="padding-top:5px;margin-bottom:4px;">'
                f'<div style="font-weight:700;color:#e2e8f0;font-size:0.9rem;">{title}</div>'
                f'<div style="color:#475569;font-size:0.81rem;margin-top:2px;line-height:1.5;">{desc}</div>'
                f'</div></div>'
            )
        pipeline_html += '</div>'
        st.markdown(pipeline_html, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        p = lambda bg,border,color,text: (
            f'<span style="background:{bg};border:1px solid {border};color:{color};'
            f'padding:4px 13px;border-radius:99px;font-size:0.78rem;font-weight:600;'
            f'margin:3px;display:inline-block;">{text}</span>'
        )
        stack_html = '<div style="background:#13131f;border:1px solid #1e1e2e;border-radius:20px;padding:1.8rem;">'
        groups = [
            ("NLP",[p("rgba(99,102,241,0.1)","rgba(99,102,241,0.25)","#a5b4fc","NLTK"),p("rgba(99,102,241,0.1)","rgba(99,102,241,0.25)","#a5b4fc","spaCy"),p("rgba(99,102,241,0.1)","rgba(99,102,241,0.25)","#a5b4fc","Lemmatization"),p("rgba(99,102,241,0.1)","rgba(99,102,241,0.25)","#a5b4fc","POS Tagging")]),
            ("AI Models",[p("rgba(139,92,246,0.1)","rgba(139,92,246,0.25)","#c4b5fd","BERT NER"),p("rgba(139,92,246,0.1)","rgba(139,92,246,0.25)","#c4b5fd","SentenceTransformers"),p("rgba(139,92,246,0.1)","rgba(139,92,246,0.25)","#c4b5fd","ANN · CNN · LSTM")]),
            ("Vector Search",[p("rgba(34,197,94,0.08)","rgba(34,197,94,0.2)","#86efac","FAISS"),p("rgba(34,197,94,0.08)","rgba(34,197,94,0.2)","#86efac","Cosine Similarity"),p("rgba(34,197,94,0.08)","rgba(34,197,94,0.2)","#86efac","Semantic Chunking")]),
            ("Framework",[p("rgba(251,146,60,0.08)","rgba(251,146,60,0.2)","#fdba74","Streamlit"),p("rgba(251,146,60,0.08)","rgba(251,146,60,0.2)","#fdba74","PyTorch"),p("rgba(251,146,60,0.08)","rgba(251,146,60,0.2)","#fdba74","Ollama + LLaMA 3.2 3B")]),
        ]
        for grp_name,pills in groups:
            stack_html += (
                f'<div style="margin-bottom:14px;">'
                f'<div style="font-size:0.67rem;font-weight:700;color:#334155;'
                f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:7px;">{grp_name}</div>'
                f'<div>{"".join(pills)}</div></div>'
            )
        stack_html += '</div>'
        st.markdown(stack_html, unsafe_allow_html=True)

    # About
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📖 About the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card">'
        '<p style="color:#94a3b8;font-size:0.93rem;line-height:1.9;margin:0 0 1rem 0;">'
        '<span style="color:#e2e8f0;font-weight:700;">StudyMate AI</span> is a '
        '<span style="color:#a5b4fc;font-weight:600;">GenAI-powered smart study assistant</span> '
        'designed to transform raw educational content into structured and interactive learning material. '
        'The system takes study notes as input and automatically generates summaries, key concepts, '
        'and quiz questions to help students revise more effectively.</p>'
        '<p style="color:#94a3b8;font-size:0.93rem;line-height:1.9;margin:0 0 1rem 0;">'
        'The project uses <span style="color:#a5b4fc;font-weight:600;">Natural Language Processing</span> '
        'techniques for text preprocessing, deep learning models for classification, and '
        'transformer-based generative AI for producing summaries and questions. It also incorporates '
        'embeddings and retrieval-based generation to allow users to interact with their notes.</p>'
        '<p style="color:#94a3b8;font-size:0.93rem;line-height:1.9;margin:0;">'
        'This tool reduces the time spent on manual note-making and helps students understand '
        'concepts faster by converting passive study material into an '
        '<span style="color:#a5b4fc;font-weight:600;">active learning experience</span>.</p></div>',
        unsafe_allow_html=True,
    )

    # Team
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">👥 Team Contributions</div>', unsafe_allow_html=True)
    team = [
        ("AS","Ashi Srivastava","#6366f1","Data collection and preprocessing pipeline including tokenization, stopword removal, and lemmatization."),
        ("PN","Parth Nawal","#8b5cf6","Implementation of text embeddings and Named Entity Recognition (NER)."),
        ("SK","Simran Karan Bora","#a78bfa","Development and training of deep learning models including ANN, CNN, and LSTM."),
        ("VB","Vidushi Bhadauria","#c4b5fd","Utility functions, main application (app.py), integration of all modules, and compiling the complete project."),
        ("AR","Aryama Sharma","#7c3aed","Generative AI module and Retrieval-Augmented Generation (RAG) implementation."),
    ]
    col_a,col_b = st.columns(2)
    for i,(initials,name,color,role) in enumerate(team):
        col = col_a if i%2==0 else col_b
        col.markdown(
            f'<div class="card" style="margin-bottom:10px;display:flex;gap:14px;align-items:flex-start;">'
            f'<div style="width:42px;height:42px;border-radius:12px;flex-shrink:0;'
            f'background:linear-gradient(135deg,{color}33,{color}15);border:1px solid {color}40;'
            f'display:flex;align-items:center;justify-content:center;font-size:0.78rem;font-weight:800;color:{color};">'
            f'{initials}</div>'
            f'<div><div style="font-weight:700;color:#e2e8f0;font-size:0.92rem;margin-bottom:3px;">{name}</div>'
            f'<div style="color:#64748b;font-size:0.82rem;line-height:1.5;">{role}</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈  Upload your notes in the sidebar to unlock all features", icon="🚀")
    st.stop()


# ════════════════════════════════════════════════════════════════
#  FILE INFO BAR
# ════════════════════════════════════════════════════════════════
r = st.session_state.pipeline_result
_raw_key   = st.session_state.file_name or ""
_file_list = [f for f in _raw_key.split("_") if f]
_num_files = len(_file_list)
_icon      = "📚" if _num_files > 1 else "📄"
_label     = "Active Documents" if _num_files > 1 else "Active Document"
_fname     = " · ".join(_file_list) if _num_files > 1 else _raw_key
_badge     = (
    f'<span style="background:rgba(99,102,241,0.15);color:#a5b4fc;padding:2px 8px;'
    f'border-radius:99px;font-size:0.68rem;margin-left:6px;">{_num_files} files</span>'
) if _num_files > 1 else ""

st.markdown(
    '<div class="file-bar">'
    '<div style="flex:1;min-width:0;">'
    f'<div style="color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">{_label}{_badge}</div>'
    f'<div class="fname" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_icon} {_fname}</div>'
    '</div>'
    '<div style="display:flex;gap:24px;flex-shrink:0;">'
    f'<div><div class="flabel">Words</div><div class="fval">{r["word_count"]:,}</div></div>'
    f'<div><div class="flabel">Sentences</div><div class="fval">{r["sentence_count"]}</div></div>'
    f'<div><div class="flabel">Keywords</div><div class="fval">{len(r["keywords"])}</div></div>'
    f'<div><div class="flabel">Chunks</div><div class="fval">{len(st.session_state.chunks)}</div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

tab1,tab2,tab3,tab4 = st.tabs(["📋  Summary","📝  Quiz Generator","🔑  Key Concepts","💬  Chat with Notes"])


# ════════════════════════════════════════════════════════════════
#  TAB 1 — SUMMARY
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.08);'
        'border:1px solid rgba(99,102,241,0.2);border-radius:99px;padding:5px 14px;margin-bottom:1rem;">'
        '<span style="width:7px;height:7px;background:#6366f1;border-radius:50%;display:inline-block;'
        'box-shadow:0 0 6px #6366f1;"></span>'
        '<span style="font-size:0.78rem;font-weight:600;color:#a5b4fc;">Claude AI · Summarization Agent</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">📋 Smart Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.88rem;margin-bottom:1.5rem;">'
        'Claude reads your notes and generates a <b style="color:#94a3b8;">structured, '
        'section-by-section summary</b> — with headers, bold concepts, bullet points, '
        'and a key takeaway. Not just extracted sentences.</p>',
        unsafe_allow_html=True,
    )

    col1,col2 = st.columns([3,1])
    with col1:
        style = st.selectbox("",["concise","detailed","bullets"],
            label_visibility="collapsed",
            format_func=lambda x:{
                "concise":"⚡ Concise — Overview + Key Points + Takeaway",
                "detailed":"📖 Detailed — Full breakdown with concepts & examples",
                "bullets":"🔘 Bullet Points — Scannable grouped facts",
            }[x])
    with col2:
        gen_btn = st.button("✨ Generate",type="primary",use_container_width=True)

    if gen_btn:
        from generative.summarizer import summarize_text
        with st.spinner("🤖 Claude is reading and summarising your notes..."):
            summary = summarize_text(st.session_state.raw_text, style=style)
        st.session_state["summary_result"] = summary
        st.session_state["summary_style"]  = style

    if st.session_state.get("summary_result"):
        summary = st.session_state["summary_result"]
        orig    = r["word_count"]
        sumw    = len(summary.split())
        redu    = max(0, round((1-sumw/orig)*100))

        m1,m2,m3 = st.columns(3)
        m1.markdown(f'<div class="m-card"><div class="m-val" style="color:#6366f1;">{orig:,}</div><div class="m-lbl">Original Words</div></div>',unsafe_allow_html=True)
        m2.markdown(f'<div class="m-card"><div class="m-val" style="color:#22c55e;">{sumw:,}</div><div class="m-lbl">Summary Words</div></div>',unsafe_allow_html=True)
        m3.markdown(f'<div class="m-card"><div class="m-val" style="color:#f59e0b;">{redu}%</div><div class="m-lbl">Compression</div></div>',unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(summary)

        st.markdown("<br>", unsafe_allow_html=True)
        col_d,col_r2 = st.columns([1,3])
        with col_d:
            st.download_button("⬇️ Download Summary",summary,"summary.txt","text/plain",use_container_width=True)
        with col_r2:
            if st.button("🔄 Re-generate",use_container_width=True):
                st.session_state["summary_result"] = None
                st.rerun()

    with st.expander("🔍 View Raw Extracted Text"):
        st.text_area("",st.session_state.raw_text[:3000]+"...",height=220,disabled=True,label_visibility="collapsed")


# ════════════════════════════════════════════════════════════════
#  TAB 2 — QUIZ
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.08);'
        'border:1px solid rgba(99,102,241,0.2);border-radius:99px;padding:5px 14px;margin-bottom:1rem;">'
        '<span style="width:7px;height:7px;background:#6366f1;border-radius:50%;display:inline-block;'
        'box-shadow:0 0 6px #6366f1;"></span>'
        '<span style="font-size:0.78rem;font-weight:600;color:#a5b4fc;">Claude AI · Advanced Quiz Agent</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">📝 Quiz Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.88rem;margin-bottom:1.5rem;">'
        'Generates exam-level questions mixing Application, Scenario, Cause-Effect, Comparison '
        'and Error-Spotting types — with plausible distractors, full explanations, '
        'and visual aids where helpful.</p>',
        unsafe_allow_html=True,
    )

    col1,col2 = st.columns([3,1])
    with col1:
        num_q = st.slider("Number of questions",2,10,5)
    with col2:
        st.markdown("<br>",unsafe_allow_html=True)
        quiz_btn = st.button("🎯 Generate Quiz",type="primary",use_container_width=True)

    if quiz_btn:
        st.session_state["revealed"] = {}
        from generative.quiz_generator import generate_quiz
        with st.spinner("🤖 Claude is crafting advanced exam questions..."):
            quiz = generate_quiz(r["sentences"], num_questions=num_q)
        if not quiz:
            st.warning("⚠️ Could not generate questions. Try uploading a longer document.")
        else:
            has_img = sum(1 for q in quiz if q.get("image_url"))
            st.success(f"✅ {len(quiz)} questions ready · {has_img} with visual aids")
            st.session_state["current_quiz"] = quiz

    if st.session_state.get("current_quiz"):
        quiz = st.session_state["current_quiz"]
        if "revealed" not in st.session_state:
            st.session_state["revealed"] = {}

        qtype_meta = {
            "application"  :("#6366f1","Application"),
            "scenario"     :("#8b5cf6","Scenario"),
            "cause_effect" :("#f59e0b","Cause & Effect"),
            "conceptual"   :("#3b82f6","Conceptual"),
            "comparison"   :("#22c55e","Comparison"),
            "error_spotting":("#ef4444","Error Spotting"),
        }

        st.markdown("---")
        st.markdown(f'<div class="section-title">Your Quiz — {len(quiz)} Questions</div>',unsafe_allow_html=True)
        st.markdown('<p style="color:#475569;font-size:0.82rem;margin-bottom:1rem;">Pick your answer then click <b>Reveal Answer</b> to see the full explanation.</p>',unsafe_allow_html=True)

        user_answers = {}

        for i,mcq in enumerate(quiz,1):
            is_revealed  = st.session_state["revealed"].get(i,False)
            has_image    = bool(mcq.get("image_url"))
            qtype        = mcq.get("question_type","conceptual")
            qtype_color, qtype_label = qtype_meta.get(qtype,("#475569","MCQ"))

            if has_image:
                q_col,img_col = st.columns([3,1])
            else:
                q_col = st.container()
                img_col = None

            with q_col:
                badge = (
                    f'<span style="background:{qtype_color}18;border:1px solid {qtype_color}40;'
                    f'color:{qtype_color};padding:2px 10px;border-radius:99px;font-size:0.68rem;'
                    f'font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-right:8px;">{qtype_label}</span>'
                )
                st.markdown(
                    f'<div style="background:#13131f;border:1px solid #1e1e2e;border-left:3px solid {qtype_color};'
                    f'border-radius:0 14px 14px 0;padding:1.1rem 1.4rem;margin-bottom:10px;">'
                    f'<div style="margin-bottom:6px;">{badge}'
                    f'<span style="font-size:0.68rem;color:#334155;">Q{i} of {len(quiz)}</span></div>'
                    f'<div style="font-size:0.97rem;font-weight:600;color:#e2e8f0;line-height:1.5;">{mcq["question"]}</div></div>',
                    unsafe_allow_html=True,
                )
                opts = [f"{l})  {t}" for l,t in mcq["options"].items() if t]
                if opts:
                    sel = st.radio("",opts,key=f"q_{i}",label_visibility="collapsed")
                    user_answers[i] = sel[0]

            if has_image and img_col is not None:
                with img_col:
                    topic_label = mcq.get("image_topic") or mcq.get("topic","")
                    st.image(mcq["image_url"],caption=topic_label.title(),use_container_width=True)
                    st.markdown('<div style="font-size:0.68rem;color:#334155;text-align:center;margin-top:2px;">📷 Visual Aid</div>',unsafe_allow_html=True)

            btn_c,_ = st.columns([1,4])
            with btn_c:
                lbl = "🙈 Hide" if is_revealed else "👁️ Reveal Answer"
                if st.button(lbl,key=f"rev_{i}",use_container_width=True):
                    st.session_state["revealed"][i] = not is_revealed
                    st.rerun()

            if is_revealed:
                correct_letter = mcq["answer"]
                correct_text   = mcq["options"].get(correct_letter,"")
                user_letter    = user_answers.get(i,"")
                is_correct     = user_letter == correct_letter
                explanation    = mcq.get("explanation","").strip()

                if user_letter:
                    if is_correct:
                        bg,border,left = "rgba(34,197,94,0.07)","rgba(34,197,94,0.2)","#22c55e"
                        verdict,vc = "✅ Correct!","#86efac"
                    else:
                        bg,border,left = "rgba(239,68,68,0.07)","rgba(239,68,68,0.2)","#ef4444"
                        verdict,vc = f"❌ Incorrect — you chose {user_letter})","#fca5a5"
                else:
                    bg,border,left = "rgba(99,102,241,0.07)","rgba(99,102,241,0.2)","#6366f1"
                    verdict,vc = "ℹ️ No answer selected","#a5b4fc"

                reveal = (
                    f'<div style="background:{bg};border:1px solid {border};border-left:3px solid {left};'
                    f'border-radius:0 12px 12px 0;padding:1rem 1.3rem;margin:6px 0;">'
                    f'<div style="font-size:0.7rem;font-weight:700;color:{vc};text-transform:uppercase;'
                    f'letter-spacing:1px;margin-bottom:8px;">{verdict}</div>'
                    f'<div style="color:#e2e8f0;font-weight:700;font-size:0.95rem;margin-bottom:10px;">'
                    f'✔ Correct: {correct_letter}) {correct_text}</div>'
                )
                if explanation:
                    reveal += (
                        f'<div style="background:rgba(0,0,0,0.25);border-radius:10px;padding:0.8rem 1rem;">'
                        f'<div style="font-size:0.68rem;font-weight:700;color:#475569;text-transform:uppercase;'
                        f'letter-spacing:1px;margin-bottom:6px;">💡 Why this answer is correct</div>'
                        f'<div style="color:#94a3b8;font-size:0.87rem;line-height:1.65;">{explanation}</div></div>'
                    )
                reveal += '</div>'
                st.markdown(reveal,unsafe_allow_html=True)

            st.markdown('<hr style="border-color:#1e1e2e;margin:1.2rem 0;">',unsafe_allow_html=True)

        num_rev = sum(1 for idx in range(1,len(quiz)+1) if st.session_state["revealed"].get(idx,False))

        if num_rev == len(quiz) and user_answers:
            score = sum(1 for idx,m in enumerate(quiz,1) if user_answers.get(idx)==m["answer"])
            pct   = round(score/len(quiz)*100)
            st.markdown("---")
            cs,cb = st.columns([1,2])
            with cs:
                emoji = "🌟" if pct>=80 else ("👍" if pct>=60 else "📚")
                msg   = "Excellent!" if pct>=80 else ("Good effort!" if pct>=60 else "Keep studying!")
                st.markdown(f'<div class="score-card"><div class="score-num">{pct}%</div><div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:4px 0;">{score}/{len(quiz)} {emoji}</div><div class="score-sub">{msg}</div></div>',unsafe_allow_html=True)
            with cb:
                bar_c = "#22c55e" if pct>=80 else ("#f59e0b" if pct>=60 else "#ef4444")
                st.markdown(
                    f'<div class="card" style="margin-top:0;"><div style="font-weight:700;color:#e2e8f0;margin-bottom:14px;">Performance</div>'
                    f'<div style="background:#0d0d14;border-radius:99px;height:12px;overflow:hidden;margin-bottom:10px;">'
                    f'<div style="background:{bar_c};width:{pct}%;height:100%;border-radius:99px;"></div></div>'
                    f'<div style="display:flex;justify-content:space-between;">'
                    f'<span style="color:#86efac;font-size:0.85rem;font-weight:600;">✅ {score} correct</span>'
                    f'<span style="color:#fca5a5;font-size:0.85rem;font-weight:600;">❌ {len(quiz)-score} wrong</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        elif num_rev < len(quiz):
            st.markdown(
                f'<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);'
                f'border-radius:10px;padding:10px 16px;color:#6366f1;font-size:0.85rem;font-weight:600;">'
                f'👁️ Reveal all {len(quiz)} answers to see your score ({num_rev}/{len(quiz)} revealed)</div>',
                unsafe_allow_html=True,
            )

        from generative.quiz_generator import format_quiz_for_display
        st.download_button("⬇️ Download Quiz",format_quiz_for_display(quiz),"quiz.txt","text/plain")


# ════════════════════════════════════════════════════════════════
#  TAB 3 — KEY CONCEPTS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.08);'
        'border:1px solid rgba(99,102,241,0.2);border-radius:99px;padding:5px 14px;margin-bottom:1rem;">'
        '<span style="width:7px;height:7px;background:#6366f1;border-radius:50%;display:inline-block;'
        'box-shadow:0 0 6px #6366f1;"></span>'
        '<span style="font-size:0.78rem;font-weight:600;color:#a5b4fc;">Claude AI · Knowledge Extraction Agent</span></div>',
        unsafe_allow_html=True,
    )

    # Keywords
    st.markdown('<div class="section-title">🏷️ Top Keywords</div>',unsafe_allow_html=True)
    kws = r["keywords"]
    kw_html = (
        '<div style="background:#13131f;border:1px solid #1e1e2e;border-radius:14px;'
        'padding:1.2rem 1.5rem;margin-bottom:1rem;">'
        '<div style="font-size:0.72rem;font-weight:600;color:#475569;margin-bottom:10px;">'
        'Extracted via TF-IDF frequency after stopword removal and lemmatization</div>'
    )
    if kws:
        kw_html += "".join([
            f'<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);'
            f'color:#a5b4fc;padding:5px 14px;border-radius:99px;font-size:0.82rem;'
            f'font-weight:600;margin:3px;display:inline-block;">{k}</span>'
            for k in kws
        ])
    kw_html += '</div>'
    st.markdown(kw_html,unsafe_allow_html=True)

    # NER
    st.markdown('<div class="section-title">🔍 Named Entity Recognition</div>',unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;font-size:0.82rem;margin-bottom:1rem;">BERT (CoNLL-2003 fine-tuned) extracts People, Organizations, Locations, and Misc entities.</p>',unsafe_allow_html=True)

    if st.button("🤖 Run BERT NER",type="primary"):
        from ner.ner_extractor import extract_entities
        with st.spinner("BERT scanning entities..."):
            ents = extract_entities(st.session_state.raw_text)
            st.session_state["ner_entities"] = ents

    if st.session_state.get("ner_entities"):
        ents = st.session_state["ner_entities"]
        meta = {
            "PERSON"       :("#1d4ed8","#bfdbfe","👤"),
            "ORGANIZATION" :("#7c3aed","#ddd6fe","🏢"),
            "LOCATION"     :("#047857","#a7f3d0","📍"),
            "MISCELLANEOUS":("#b45309","#fde68a","🔖"),
        }
        cols = st.columns(4)
        for i,(etype,(accent,tag_color,icon)) in enumerate(meta.items()):
            words = ents.get(etype,[])
            with cols[i]:
                card = (
                    f'<div style="background:#13131f;border:1px solid #1e1e2e;border-top:2px solid {accent}50;'
                    f'border-radius:14px;padding:1rem;min-height:120px;">'
                    f'<div style="font-size:0.68rem;font-weight:700;color:{tag_color};text-transform:uppercase;'
                    f'letter-spacing:1px;margin-bottom:8px;">{icon} {etype} · {len(words)}</div>'
                )
                if words:
                    card += "".join([
                        f'<span style="background:{tag_color}18;border:1px solid {tag_color}40;'
                        f'color:{tag_color};padding:3px 9px;border-radius:99px;font-size:0.78rem;'
                        f'font-weight:600;margin:2px;display:inline-block;">{w}</span>'
                        for w in words
                    ])
                else:
                    card += '<span style="color:#334155;font-size:0.82rem;">None found</span>'
                card += '</div>'
                st.markdown(card,unsafe_allow_html=True)

    # Concept Explainer
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Concept Explainer</div>',unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.82rem;margin-bottom:1rem;">'
        'Claude finds relevant passages from your notes and generates a <b style="color:#94a3b8;">'
        'deep, structured explanation</b> — definition, mechanism, examples, takeaway.</p>',
        unsafe_allow_html=True,
    )

    col1,col2,col3 = st.columns([3,1,1])
    with col1:
        concept = st.text_input("",placeholder="e.g., backpropagation, neural network...",label_visibility="collapsed")
    with col2:
        exstyle = st.selectbox("",["simple","technical","example"],label_visibility="collapsed",
            format_func=lambda x:{"simple":"🧒 Simple","technical":"🔬 Technical","example":"🌍 Example"}[x])
    with col3:
        expl_btn = st.button("💬 Explain",type="primary",use_container_width=True)

    if expl_btn:
        if concept.strip():
            from generative.explainer import explain_concept
            with st.spinner(f"🤖 Claude is explaining '{concept}'..."):
                exp = explain_concept(concept,style=exstyle)
            style_labels = {"simple":"🧒 Simple","technical":"🔬 Technical Deep-Dive","example":"🌍 Example-Based"}
            st.markdown(
                f'<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.08);'
                f'border:1px solid rgba(99,102,241,0.2);border-radius:99px;padding:4px 12px;margin-bottom:8px;">'
                f'<span style="font-size:0.75rem;font-weight:600;color:#a5b4fc;">'
                f'🔍 {concept.title()} · {style_labels.get(exstyle,"")}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(exp)
        else:
            st.warning("⚠️ Please enter a concept.")

    # Quick explain buttons
    if kws:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<p style="color:#475569;font-size:0.78rem;font-weight:600;margin-bottom:8px;">⚡ Quick explain a keyword:</p>',unsafe_allow_html=True)
        kw_cols = st.columns(min(len(kws[:6]),6))
        for i,(col,kw) in enumerate(zip(kw_cols,kws[:6])):
            if col.button(kw,key=f"qe_{i}",use_container_width=True):
                st.session_state["quick_explain"] = kw
                st.rerun()

    if st.session_state.get("quick_explain"):
        kw = st.session_state.pop("quick_explain")
        from generative.explainer import explain_concept
        with st.spinner(f"🤖 Explaining '{kw}'..."):
            exp = explain_concept(kw,style="simple")
        st.markdown(
            f'<div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.15);'
            f'border-radius:12px;padding:1rem 1.3rem;margin-top:8px;">'
            f'<div style="font-size:0.7rem;font-weight:700;color:#6366f1;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:8px;">⚡ Quick Explain: {kw.title()}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(exp)
        st.markdown("</div>",unsafe_allow_html=True)

    # Sentence Classification
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏷️ Sentence Classification</div>',unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;font-size:0.82rem;margin-bottom:1rem;">ANN classifier tags each sentence as Definition / Concept / Example / Important Point.</p>',unsafe_allow_html=True)

    if st.button("🤖 Classify Sentences"):
        if not os.path.exists("models/saved/ann_model.pt"):
            st.warning("⚠️ Run `python -m models.train_all` in terminal first.")
        else:
            import torch
            from models.ann_model import load_model
            from models.data_prep  import ID_TO_LABEL
            from embeddings.sentence_embeddings import embed_sentences
            sents = r["sentences"][:15]
            with st.spinner("Classifying..."):
                mdl  = load_model("models/saved/ann_model.pt")
                vecs = embed_sentences(sents)
                with torch.no_grad():
                    preds = torch.argmax(mdl(torch.FloatTensor(vecs)),dim=1).numpy()
            lmeta = {
                "definition"     :("#3b82f6","rgba(59,130,246,0.08)","📘","Definition"),
                "concept"        :("#8b5cf6","rgba(139,92,246,0.08)","💡","Concept"),
                "example"        :("#22c55e","rgba(34,197,94,0.08)","📝","Example"),
                "important_point":("#f59e0b","rgba(245,158,11,0.08)","⭐","Important"),
            }
            for s,p in zip(sents,preds):
                label = ID_TO_LABEL[p]
                ac,bg,icon,display = lmeta.get(label,("#475569","rgba(71,85,105,0.1)","🔖","Other"))
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {ac}25;border-left:2px solid {ac};'
                    f'border-radius:0 10px 10px 0;padding:9px 14px;margin:5px 0;display:flex;gap:10px;align-items:flex-start;">'
                    f'<span style="flex-shrink:0;font-size:0.95rem;">{icon}</span>'
                    f'<div><span style="font-size:0.68rem;font-weight:700;color:{ac};text-transform:uppercase;'
                    f'letter-spacing:1px;">{display}</span>'
                    f'<div style="color:#cbd5e1;font-size:0.88rem;margin-top:2px;">{s}</div></div></div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════
#  TAB 4 — CHAT (Multi-tool Agent)
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(34,197,94,0.06);'
        'border:1px solid rgba(34,197,94,0.15);border-radius:99px;padding:5px 14px;margin-bottom:1rem;">'
        '<span style="width:7px;height:7px;background:#22c55e;border-radius:50%;display:inline-block;'
        'box-shadow:0 0 6px #22c55e;"></span>'
        '<span style="font-size:0.78rem;font-weight:600;color:#86efac;">● Online · RAG + Claude · Agentic Q&amp;A</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">💬 Chat with Your Notes</div>',unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);'
        'border-radius:12px;padding:12px 16px;margin-bottom:1rem;display:flex;gap:12px;align-items:flex-start;">'
        '<span style="font-size:1.1rem;flex-shrink:0;">🧠</span>'
        '<div style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">'
        '<b style="color:#a5b4fc;">Agentic RAG:</b> The agent decides which tools to use — '
        'search_notes, explain_concept, compare_concepts, find_examples, list_key_facts — '
        'chains them together, and synthesises a <b style="color:#a5b4fc;">structured answer from your notes only</b>.</div></div>',
        unsafe_allow_html=True,
    )

    # Suggested questions
    sugg_kws  = r["keywords"][:5]
    templates = ["What is {}?","How does {} work?","Compare {} with related concepts.",
                 "Why is {} important?","Give an example of {}."]
    sugg_list = [templates[i%len(templates)].format(kw) for i,kw in enumerate(sugg_kws)]

    st.markdown('<p style="color:#475569;font-size:0.82rem;font-weight:600;margin-bottom:8px;">💡 Suggested questions:</p>',unsafe_allow_html=True)
    row1 = sugg_list[:3]
    row2 = sugg_list[3:]
    for row in [row1,row2]:
        if row:
            cols = st.columns(len(row))
            for col,s in zip(cols,row):
                if col.button(s,key=f"s_{sugg_list.index(s)}",use_container_width=True):
                    st.session_state["pending_query"] = s
                    st.rerun()

    st.markdown("<br>",unsafe_allow_html=True)

    TOOL_LABELS = {
        "search_notes"    :("🔍","Searching notes"),
        "explain_concept" :("💡","Explaining concept"),
        "compare_concepts":("⚖️","Comparing concepts"),
        "find_examples"   :("📌","Finding examples"),
        "list_key_facts"  :("📋","Listing key facts"),
    }

    # Chat history
    if not st.session_state.chat_history:
        st.markdown(
            '<div style="text-align:center;padding:3rem 2rem;">'
            '<div style="font-size:3rem;margin-bottom:1rem;">💬</div>'
            '<div style="font-size:1rem;font-weight:700;color:#334155;">No conversation yet</div>'
            '<div style="font-size:0.85rem;color:#1e293b;margin-top:6px;">'
            'Click a suggestion or type below to start</div></div>',
            unsafe_allow_html=True,
        )
    else:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(f"**{turn['question']}**")
            with st.chat_message("assistant",avatar="🤖"):
                st.markdown(turn["answer"])
                tool_calls = turn.get("tool_calls",[])
                if tool_calls:
                    with st.expander(f"🧠 Agent reasoning — {len(tool_calls)} tool call(s)"):
                        for tc in tool_calls:
                            icon,label = TOOL_LABELS.get(tc["tool"],("🔧","Tool"))
                            inp_str = json.dumps(tc["input"])[:100]+"..."
                            st.markdown(
                                f'<div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:8px;'
                                f'padding:7px 12px;margin:4px 0;font-size:0.78rem;color:#475569;font-family:monospace;">'
                                f'<span style="color:#6366f1;font-weight:700;">{icon} {label}</span> · {inp_str}</div>',
                                unsafe_allow_html=True,
                            )

    # Chat input
    user_query = st.chat_input("Ask anything about your notes...")
    if "pending_query" in st.session_state:
        user_query = st.session_state.pop("pending_query")

    if user_query:

        with st.chat_message("user"):
            st.markdown(f"**{user_query}**")

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching notes · ✍️ Generating structured answer..."):
                from rag.agent import answer_question
                res = answer_question(
                    question = user_query,
                    index    = st.session_state.faiss_index,
                    chunks   = st.session_state.chunks,
                    top_k    = 5,
                )

            # Show tool calls if any
            tool_calls = res.get("tool_calls", [])
            if tool_calls:
                with st.expander(f"🧠 Agent reasoning — {len(tool_calls)} step(s)", expanded=False):
                    for tc in tool_calls:
                        icon, label = TOOL_LABELS.get(tc["tool"], ("🔧","Tool call"))
                        inp_str = json.dumps(tc["input"], ensure_ascii=False)[:120]
                        st.markdown(
                            f'<div style="background:#0d0d14;border:1px solid #1e1e2e;border-radius:10px;'
                            f'padding:0.8rem 1rem;margin:5px 0;">'
                            f'<div style="font-size:0.7rem;font-weight:700;color:#6366f1;text-transform:uppercase;'
                            f'letter-spacing:1px;margin-bottom:4px;">{icon} {label}</div>'
                            f'<div style="font-size:0.78rem;color:#475569;font-family:monospace;'
                            f'word-break:break-all;">{inp_str}</div></div>',
                            unsafe_allow_html=True,
                        )

            st.markdown(res.get("answer", "No answer generated."))

        st.session_state.chat_history.append({
            "question"  : user_query,
            "answer"    : res.get("answer",""),
            "answerable": res.get("answerable",False),
            "tool_calls": res.get("tool_calls",[]),
        })

    if st.session_state.chat_history:
        st.markdown("<br>",unsafe_allow_html=True)
        col_clr,_ = st.columns([1,4])
        with col_clr:
            if st.button("🗑️ Clear Chat History",use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()