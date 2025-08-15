# app.py — RAG: PDF → classify → OCR (Tesseract, optional EasyOCR overlay) → normalize → embed → Supabase → search
# Modes: Keyword (exact/normalized), Keyword (fuzzy w/ pg_trgm or local fallback), Semantic (RPC or local)

import io
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output as TessOutput
from PIL import Image, ImageOps, ImageFilter

from sentence_transformers import SentenceTransformer
from supabase import create_client, ClientOptions
import supabase as _sb

# Optional, used for local fuzzy fallback:
try:
    from rapidfuzz.fuzz import partial_ratio
except Exception:
    partial_ratio = None  # We'll handle gracefully if not installed


# ===================== Build / Environment =====================
BUILD_ID = "full-stack-ocr-adaptive-2025-08-15"

# Keep CPU stable on Streamlit Cloud
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# EasyOCR (optional) + persistent cache to avoid re-downloads
EASYOCR_AVAILABLE = True
try:
    import easyocr  # heavy; we load the model lazily via cache_resource
except Exception:
    EASYOCR_AVAILABLE = False

EASYOCR_MODEL_DIR = Path(".easyocr_models")
EASYOCR_MODEL_DIR.mkdir(exist_ok=True)


# ===================== Streamlit Page =====================
st.set_page_config(page_title="RAG (OCR + Supabase)", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
      .stTabs [data-baseweb="tab"] {
        height: 42px; padding: 0 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15);
        background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(0,0,0,0.08));
        backdrop-filter: blur(6px);
      }
      .stTabs [data-baseweb="tab"]:hover {border-color: rgba(255,255,255,0.35);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG App — OCR → Embeddings → Supabase Search")


# ===================== Sidebar: setup & toggles =====================
with st.sidebar:
    st.header("Setup & Status")
    st.caption(f"Build: `{BUILD_ID}`")
    st.caption(f"supabase-py version: `{_sb.__version__}`")

    # Supabase secrets check
    if "SUPABASE_URL" in st.secrets:
        st.caption("SUPABASE_URL: " + st.secrets["SUPABASE_URL"])

    # Tesseract languages present
    try:
        langs = pytesseract.get_languages(config="")
        st.caption("Tesseract languages: " + ", ".join(sorted(langs)))
    except Exception:
        st.caption("Tesseract language list unavailable")

    # EasyOCR cache + enable toggle + warm-up
    if EASYOCR_AVAILABLE:
        try:
            files = list(EASYOCR_MODEL_DIR.glob("**/*"))
            st.caption(f"EasyOCR cache: {EASYOCR_MODEL_DIR} ({len(files)} files)")
        except Exception:
            st.caption("EasyOCR cache not accessible")
    else:
        st.caption("EasyOCR not installed (handwriting overlay disabled)")

    st.markdown("---")
    st.subheader("OCR Options")

    preset = st.selectbox(
        "Language preset",
        ["English (printed)", "English (handwritten)", "Arabic (ara)", "Chinese (Simplified)", "Chinese (Traditional)"],
        index=0,
        help="This biases page handling. You can still enable handwriting overlay below.",
    )

    # Defaults by preset
    if preset == "English (printed)":
        default_zoom = 3.0
        default_intensive = False
    elif preset == "English (handwritten)":
        default_zoom = 3.5
        default_intensive = True
    else:
        default_zoom = 3.5
        default_intensive = True

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        base_zoom = st.slider("Render zoom (DPI proxy)", 2.0, 4.5, default_zoom, 0.5)
        force_ocr = st.checkbox("Force OCR (ignore digital text)", False)
        max_pages_debug = st.number_input("Limit pages (0=all)", min_value=0, value=0, step=1)
    with col_sb2:
        intensive = st.checkbox("Intensive (retry if low conf)", default_intensive)
        timeout_sec = st.slider("OCR timeout/page (sec)", 5, 60, 20)

    enable_easyocr = st.toggle(
        "Enable handwriting overlay (EasyOCR)", value=False,
        help="When ON, scanned/annotated pages also run EasyOCR. First run will download models into .easyocr_models."
    )

    def _warmup():
        try:
            _ = __import__("easyocr")
            load_easyocr_reader(["en"])
            st.success("EasyOCR models warmed up.")
        except Exception as e:
            st.error(f"Warm-up failed: {e}")

    st.button(
        "Warm-up handwriting models",
        help="Proactively download EasyOCR models so ingest doesn’t stall.",
        on_click=_warmup,
        disabled=not EASYOCR_AVAILABLE,
    )

    # Optional vocabulary file (.txt) passed to Tesseract as user-words
    user_words_path: Optional[Path] = None
    use_vocab = st.checkbox("Use custom vocabulary (.txt)", value=False)
    if use_vocab:
        st.caption("Upload a .txt with one term per line (e.g., Gen. Transporting)")
        user_words_file = st.file_uploader("Vocabulary (TXT)", type=["txt"], key="user_words")
        if user_words_file is not None:
            user_words_path = Path("user_words.txt").absolute()
            with open(user_words_path, "wb") as f:
                f.write(user_words_file.read())
            st.caption(f"Custom vocabulary saved to {user_words_path}")


# ===================== Supabase =====================
def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        st.sidebar.error("❌ Supabase not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY in Settings → Secrets.")
        return None
    try:
        return create_client(url, key, options=ClientOptions())
    except Exception as e:
        st.sidebar.error(f"Supabase init failed: {e}")
        return None


_supa = get_supabase_client()
if _supa:
    try:
        _res = _supa.table("document_chunks").select("id", count="exact").limit(1).execute()
        st.sidebar.success(f"✅ Supabase connected. `document_chunks` rows: {_res.count}")
    except Exception as e:
        st.sidebar.error(f"❌ Supabase connection failed: {e}")


# ===================== Normalization helpers =====================
ZW_REMOVE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None)  # soft hyphen + zero-widths
PUNCT_MAP = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "\u00A0": " ", "·": " "
})
_PUNCT_STRIP_RE = re.compile(r"[.,:;|/\\()\[\]{}<>•·…]+")
_SPACE_COLLAPSE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """
    Aggressive normalization for keyword matching (no 'corrections', just cleanup).
    """
    if not s:
        return ""
    s = s.replace("-\n", "")          # de-hyphenate end-of-line
    s = s.replace("\n", " ")          # flatten newlines
    s = s.translate(ZW_REMOVE).translate(PUNCT_MAP)
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_STRIP_RE.sub(" ", s)   # turn .,;:/()[]{}<>… into spaces
    s = _SPACE_COLLAPSE_RE.sub(" ", s).strip()
    return s.lower()


def escape_for_or_filter(s: str) -> str:
    # PostgREST .or_ uses comma as separator; escape commas in query
    return s.replace(",", r"\,")


# ===================== Models =====================
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    # 384-dim, light & solid for retrieval
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=True)
def load_easyocr_reader(langs: List[str]):
    """Load EasyOCR with a persistent local cache to prevent repeated downloads."""
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr is not installed. Add `easyocr` to requirements.txt")
    return easyocr.Reader(
        langs, gpu=False, verbose=False,
        model_storage_directory=str(EASYOCR_MODEL_DIR),
        download_enabled=True,
    )


# ===================== Rendering & Preprocessing =====================
def render_page_to_image(page, zoom: float = 3.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def preprocess_image(
    img: Image.Image,
    do_denoise: bool = True,
    do_autocontrast: bool = True,
    binarize: bool = True,
    thresh: int = 180,
    to_gray: bool = True,
) -> Image.Image:
    if to_gray:
        img = img.convert("L")
    if do_denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    if do_autocontrast:
        img = ImageOps.autocontrast(img)
    if binarize:
        img = img.point(lambda p: 255 if p > thresh else 0)
    return img


# ===================== OCR engines =====================
def tesseract_ocr(
    page,
    zoom: float,
    lang: str,
    psm: str,
    do_denoise: bool,
    do_autocontrast: bool,
    binarize: bool,
    thresh: int,
    user_words_path: Optional[Path] = None,
    timeout_sec: int = 20,
) -> Tuple[str, float]:
    """
    Tesseract OCR with confidence and a hard timeout.
    Returns (text, avg_conf).
    """
    img = render_page_to_image(page, zoom=zoom)
    img = preprocess_image(img, do_denoise, do_autocontrast, binarize, thresh, to_gray=True)

    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    if user_words_path:
        config += f" --user-words {user_words_path}"

    try:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=TessOutput.DICT, timeout=timeout_sec
        )
        words = [w for w in data.get("text", []) if isinstance(w, str) and w.strip()]
        confs = [int(c) for c in data.get("conf", []) if c not in (None, "", "-1")]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        text = " ".join(words) if words else ""
    except Exception:
        try:
            text = pytesseract.image_to_string(img, lang=lang, config=config, timeout=max(5, timeout_sec // 2))
        except Exception:
            text = ""
        return (text or "").strip(), 0.0

    if not text.strip():
        try:
            text2 = pytesseract.image_to_string(img, lang=lang, config=config, timeout=timeout_sec)
            text = text2 or text
        except Exception:
            pass

    return (text or "").strip(), avg_conf


def easyocr_ocr(page, zoom: float, langs: List[str], allowlist: Optional[str] = None) -> Tuple[str, float]:
    """Handwriting / mixed text OCR via EasyOCR. Returns (text, 'avg confidence')."""
    try:
        reader = load_easyocr_reader(langs)
    except Exception as e:
        st.info(f"Handwriting OCR not ready ({e}). Falling back to printed OCR only for this page.")
        return "", 0.0

    img = render_page_to_image(page, zoom=zoom).convert("RGB")
    results = reader.readtext(
        np.array(img),
        detail=1, paragraph=True, decoder="greedy",
        allowlist=allowlist if allowlist else None,
    )
    texts, confs = [], []
    for item in results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        txt = (item[1] or "").strip()
        conf = float(item[2]) if item[2] is not None else 0.0
        if txt:
            texts.append(txt)
            confs.append(conf)
    joined = "\n".join(texts).strip()
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return joined, avg_conf


# ===================== Page classification & adaptive OCR =====================
def classify_page(page) -> Dict[str, Any]:
    """Lightweight classifier: digital text vs scanned/annotated."""
    info = {"has_digital_text": False, "digital_chars": 0, "image_count": 0}
    try:
        raw = (page.get_text() or "").strip()
        info["digital_chars"] = len(raw)
        info["has_digital_text"] = info["digital_chars"] >= 50  # enough to trust it's digital
    except Exception:
        pass
    try:
        info["image_count"] = len(page.get_images(full=True) or [])
    except Exception:
        pass
    return info


def ocr_page_auto(
    page,
    preset: str,
    user_words_path: Optional[Path],
    base_zoom: float,
    timeout_sec: int,
    strict: bool,
    allowlist: Optional[str],
    enable_easyocr_flag: bool,
) -> Dict[str, Any]:
    """
    Strategy:
      1) If not force_ocr and page has digital text → trust that (avoid OCR errors).
      2) Else run Tesseract tuned per preset.
      3) If page looks scanned/annotated OR conf low → optionally overlay EasyOCR (if enabled).
      4) Choose the output that maximizes normalized coverage without 'correcting' words.
    """
    info = classify_page(page)
    conf_used = 0.0

    # 1) Use digital text if available and not forcing OCR
    if not force_ocr and info["has_digital_text"]:
        try:
            digital = (page.get_text() or "").strip()
            if digital:
                return {"text": digital, "engine": "digital", "conf": 100.0, "meta": info}
        except Exception:
            pass

    # 2) Printed OCR (Tesseract)
    if preset == "Arabic (ara)":
        t_text, t_conf = tesseract_ocr(page, base_zoom, "ara", "4", True, True, True, 180, user_words_path, timeout_sec)
    elif preset == "Chinese (Simplified)":
        t_text, t_conf = tesseract_ocr(page, base_zoom, "chi_sim", "4", True, True, True, 180, user_words_path, timeout_sec)
    elif preset == "Chinese (Traditional)":
        t_text, t_conf = tesseract_ocr(page, base_zoom, "chi_tra", "4", True, True, True, 180, user_words_path, timeout_sec)
    elif preset == "English (handwritten)":
        # Handwriting preset still tries printed OCR first (many forms have printed + scribbles)
        t_text, t_conf = tesseract_ocr(page, base_zoom, "eng", "4", True, True, True, 175, user_words_path, timeout_sec)
    else:  # English (printed)
        t_text, t_conf = tesseract_ocr(page, base_zoom, "eng", "6", True, True, True, 180, user_words_path, timeout_sec)

    best_text = t_text
    best_conf = t_conf
    engine = "tesseract"

    # 3) Optional handwriting overlay (EasyOCR) if scanned/annotated or low conf
    need_easy = (info["image_count"] > 0) or (t_conf < 70)
    if enable_easyocr_flag and need_easy and EASYOCR_AVAILABLE:
        # Language selection for EasyOCR
        if preset.startswith("Chinese"):
            langs = ["ch_sim"] if "Simplified" in preset else ["ch_tra"]
        elif preset.startswith("Arabic"):
            langs = ["ar"]
        else:
            langs = ["en"]  # most work is English

        e_text, e_conf = easyocr_ocr(page, max(3.5, base_zoom), langs=langs, allowlist=allowlist)
        # Choose between outputs by normalized coverage (length) first, then conf
        t_norm = normalize_text(t_text)
        e_norm = normalize_text(e_text)
        if len(e_norm) > len(t_norm) and len(e_norm) >= 10:
            best_text = e_text
            best_conf = e_conf
            engine = "easyocr"
        # If intensive, and printed conf low, try a second Tesseract pass with different psm
        if intensive and best_conf < 65:
            t2_text, t2_conf = tesseract_ocr(page, base_zoom + 0.5, "eng", "4", True, True, True, 175, user_words_path, timeout_sec)
            if normalize_text(t2_text).__len__() > normalize_text(best_text).__len__():
                best_text, best_conf, engine = t2_text, t2_conf, "tesseract-psm4"

    return {"text": best_text, "engine": engine, "conf": best_conf, "meta": info}


# ===================== Extractors =====================
def extract_text_from_pdf(file_bytes, preset: str, user_words_path: Optional[Path],
                          base_zoom: float, timeout_sec: int,
                          strict: bool, allowlist: Optional[str],
                          enable_easyocr_flag: bool, max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    total = len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    prog = st.progress(0, text=f"Extracting text 0/{limit} pages...")
    for i, page in enumerate(doc):
        if i >= limit:
            break
        try:
            cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, strict, allowlist, enable_easyocr_flag)
            parts.append(cand["text"])
        except Exception as e:
            parts.append(f"[Error reading page {i+1}: {e}]")
        finally:
            prog.progress((i + 1) / limit, text=f"Extracting text {i+1}/{limit} pages...")
    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(parts).strip()


def extract_pages_with_metadata(file_bytes, document_name, preset: str, user_words_path: Optional[Path],
                                base_zoom: float, timeout_sec: int, strict: bool,
                                allowlist: Optional[str], enable_easyocr_flag: bool, max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    pages = []
    prog = st.progress(0, text=f"Splitting into page chunks 0/{limit}...")
    for i, page in enumerate(doc):
        if i >= limit:
            break
        try:
            cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, strict, allowlist, enable_easyocr_flag)
            pages.append({
                "document_name": document_name,
                "page_number": i + 1,
                "text": cand["text"],
                "ocr_engine": cand["engine"],
                "ocr_conf": float(cand["conf"]),
                "has_digital_text": bool(cand["meta"].get("has_digital_text", False)),
                "image_count": int(cand["meta"].get("image_count", 0)),
            })
        except Exception as e:
            pages.append({"document_name": document_name, "page_number": i + 1, "text": f"[Error: {e}]", "ocr_engine": "error"})
        finally:
            prog.progress((i + 1) / limit, text=f"Splitting into page chunks {i+1}/{limit}...")
    doc.close()
    return pages


# ===================== Embeddings =====================
def embed_texts(model, texts):
    emb = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb


# ===================== UI Tabs =====================
tabs = st.tabs(["Ingest", "Search", "Page Inspector", "Notes"])

# ---------------- Ingest ----------------
with tabs[0]:
    st.subheader("Step 1 — Upload & Extract")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

    # Optional allowlist for EasyOCR (digits & ASCII letters common in forms)
    allowlist = st.text_input(
        "Optional character allowlist (EasyOCR only)",
        value="",  # e.g. "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-:/,"
        help="Leave blank for none. This can reduce misreads like 'Gen.' → 'German'."
    )

    if uploaded_file:
        st.info(f"**Selected:** {uploaded_file.name} • {uploaded_file.size/1024:.1f} KB")
        if st.button("1) Process PDF", type="primary"):
            with st.spinner("Processing..."):
                try:
                    file_bytes = uploaded_file.read()
                    st.session_state["last_pdf_bytes"] = file_bytes
                    st.session_state["last_pdf_name"] = uploaded_file.name

                    full_text = extract_text_from_pdf(
                        file_bytes, preset, user_words_path, base_zoom, timeout_sec,
                        strict=True, allowlist=allowlist or None,
                        enable_easyocr_flag=enable_easyocr, max_pages=max_pages_debug
                    )
                    page_chunks = extract_pages_with_metadata(
                        file_bytes, uploaded_file.name, preset, user_words_path, base_zoom, timeout_sec,
                        strict=True, allowlist=allowlist or None,
                        enable_easyocr_flag=enable_easyocr, max_pages=max_pages_debug
                    )
                    st.session_state["pages"] = page_chunks
                    st.session_state["document_name"] = uploaded_file.name

                    st.success("Extraction complete. Page-level chunks ready.")
                    with st.expander("Preview first pages", expanded=False):
                        for rec in page_chunks[:5]:
                            st.markdown(f"**{rec['document_name']} — Page {rec['page_number']}**  "
                                        f"(engine: {rec.get('ocr_engine','?')}, conf: {rec.get('ocr_conf',0):.1f}, "
                                        f"images: {rec.get('image_count',0)}, digital: {rec.get('has_digital_text')})")
                            preview = (rec["text"] or "").replace("\n", " ")
                            st.write((preview[:500] + ("..." if len(preview) > 500 else "")) or "_(empty page)_")
                            st.divider()

                    with st.expander("Full Extracted Text", expanded=False):
                        st.text_area("PDF Text Content", full_text or "", height=300)

                except Exception as e:
                    st.error("PDF processing failed.")
                    st.exception(e)

    st.subheader("Step 2 — Embed & Upload to Supabase")
    if not _supa:
        st.warning("Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY in Settings → Secrets.")
    elif "pages" not in st.session_state or not st.session_state["pages"]:
        st.info("No pages detected yet. Complete Step 1 first, then return here.")
    else:
        st.write(
            f"Pages ready to index: **{len(st.session_state['pages'])}** "
            f"(from **{st.session_state.get('document_name','unknown')}**)"
        )
        colA, colB = st.columns(2)
        with colA:
            delete_first = st.checkbox("Delete existing rows for this document", False)
        with colB:
            st.caption("Use this to avoid duplicates if re-indexing the same PDF.")

        if st.button("2) Generate embeddings & upload", type="primary"):
            try:
                # ensure text_norm column exists (silent if not)
                try:
                    _ = _supa.table("document_chunks").select("text_norm").limit(1).execute()
                except Exception:
                    st.warning("Column `text_norm` not found. Create it: ALTER TABLE document_chunks ADD COLUMN text_norm text;")

                if delete_first:
                    try:
                        _ = _supa.rpc("delete_document", {"p_document_name": st.session_state.get("document_name", "")}).execute()
                        st.info("Old rows for this document deleted (if any).")
                    except Exception:
                        _supa.table("document_chunks").delete().eq("document_name", st.session_state.get("document_name", "")).execute()
                        st.info("Old rows deleted via fallback.")

                model = load_embedding_model()
                pages = st.session_state["pages"]
                texts = [p["text"] if p["text"] else "" for p in pages]

                st.info("Generating embeddings…")
                emb = embed_texts(model, texts)  # (n, 384)

                rows = []
                for p, vec in zip(pages, emb):
                    raw = p["text"] or ""
                    rows.append(
                        {
                            "document_name": p["document_name"],
                            "page_number": p["page_number"],
                            "text": raw,
                            "text_norm": normalize_text(raw),
                            "embedding": vec.tolist(),
                            "ocr_engine": p.get("ocr_engine"),
                            "ocr_conf": p.get("ocr_conf"),
                            "has_digital_text": p.get("has_digital_text"),
                            "image_count": p.get("image_count"),
                        }
                    )

                st.info("Uploading to Supabase…")
                BATCH = 100
                inserted = 0
                for i in range(0, len(rows), BATCH):
                    batch = rows[i: i + BATCH]
                    res = _supa.table("document_chunks").insert(batch).execute()
                    if getattr(res, "data", None) is None:
                        st.error(
                            f"No data returned for batch {i+1}-{i+len(batch)}. "
                            f"Check RLS (disable or add anon INSERT/SELECT policies) and that the table exists."
                        )
                        st.write(res)
                        st.stop()
                    inserted += len(res.data)
                    st.write(f"Inserted rows {i+1}–{i+len(batch)} (total {inserted})")
                    time.sleep(0.05)

                st.success(f"All chunks uploaded to Supabase. Total inserted: {inserted}")
                st.info("If you don't see data in Table Editor, ensure RLS is disabled or anon policies allow INSERT/SELECT.")

            except Exception as e:
                st.error("Embedding or upload failed.")
                st.exception(e)


# ---------------- Search ----------------
with tabs[1]:
    st.subheader("Step 3 — Search (Keyword / Fuzzy / Semantic)")

    if not _supa:
        st.info("Configure Supabase first to enable search.")
    else:
        # Optional: filter by document
        try:
            docs = _supa.table("document_chunks").select("document_name").execute()
            doc_names = sorted({r["document_name"] for r in (docs.data or []) if r.get("document_name")})
        except Exception:
            doc_names = []
        selected_doc = st.selectbox("Limit to document (optional)", ["All"] + doc_names)

        query = st.text_input("Enter your query (e.g., Gen. Transporting)")
        mode = st.radio(
            "Search type",
            ["Keyword (exact/normalized)", "Keyword (fuzzy)", "Semantic (meaning match)"],
            index=0,
            horizontal=True,
        )
        top_k = st.slider("Results to show", 5, 200, 50)
        fetch_all = st.checkbox("Return all matches (exact/normalized only)")
        max_scan = st.number_input("Local fallback: max rows to scan", 100, 20000, 5000, step=100)

        # Quick checks
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Row count"):
                try:
                    res = _supa.table("document_chunks").select("id", count="exact").limit(1).execute()
                    st.write(f"Row count: {res.count}")
                except Exception as e:
                    st.error(f"Count failed: {e}")
        with c2:
            if st.button("Show 3 newest rows"):
                try:
                    res = (
                        _supa.table("document_chunks")
                        .select("document_name,page_number,text")
                        .order("id", desc=True)
                        .limit(3)
                        .execute()
                    )
                    st.write(res.data)
                except Exception as e:
                    st.error(f"Preview failed: {e}")

        if st.button("3) Search", type="primary"):
            if not query:
                st.warning("Please enter a query.")
            else:
                try:
                    results: List[Dict[str, Any]] = []
                    total = None

                    if mode.startswith("Keyword (exact/normalized)"):
                        norm_query = normalize_text(query)
                        oq = escape_for_or_filter(query)
                        onq = escape_for_or_filter(norm_query)

                        base = _supa.table("document_chunks").select("document_name,page_number,text", count="exact")
                        base = base.or_(f"text.ilike.%{oq}%,text_norm.ilike.%{onq}%")
                        if selected_doc != "All":
                            base = base.eq("document_name", selected_doc)

                        if fetch_all:
                            page_size = 100
                            first = base.order("document_name").order("page_number").range(0, page_size - 1).execute()
                            total = getattr(first, "count", None)
                            results = list(first.data or [])
                            offset = page_size
                            while total is not None and offset < total:
                                chunk = (
                                    base.order("document_name")
                                        .order("page_number")
                                        .range(offset, min(offset + page_size - 1, total - 1))
                                        .execute()
                                )
                                results.extend(chunk.data or [])
                                offset += page_size
                        else:
                            res = base.order("document_name").order("page_number").limit(top_k).execute()
                            results = getattr(res, "data", []) or []
                            total = getattr(res, "count", None)

                        results.sort(key=lambda r: (r.get("document_name", ""), r.get("page_number") or 0))
                        st.write(f"Matches found: {total if total is not None else len(results)}")

                    elif mode.startswith("Keyword (fuzzy)"):
                        norm_query = normalize_text(query)

                        # Try RPC first (if you created pg_trgm functions)
                        rpc_ok = True
                        try:
                            if selected_doc == "All":
                                res = _supa.rpc("fuzzy_find_chunks_all",
                                                {"qnorm": norm_query, "limit_n": top_k}).execute()
                            else:
                                res = _supa.rpc("fuzzy_find_chunks_in_doc",
                                                {"docname": selected_doc, "qnorm": norm_query, "limit_n": top_k}).execute()
                            results = getattr(res, "data", []) or []
                            results.sort(key=lambda r: (-r.get("sim", 0.0), r.get("document_name", ""), r.get("page_number") or 0))
                            st.write(f"Top {len(results)} fuzzy matches (server-side).")
                        except Exception:
                            rpc_ok = False

                        if not rpc_ok:
                            # Local fallback
                            if selected_doc == "All":
                                fetched: List[Dict[str, Any]] = []
                                offset = 0
                                page_size = 1000
                                while offset < max_scan:
                                    chunk = (
                                        _supa.table("document_chunks")
                                        .select("document_name,page_number,text")
                                        .order("document_name")
                                        .order("page_number")
                                        .range(offset, offset + page_size - 1)
                                        .execute()
                                    )
                                    data = chunk.data or []
                                    if not data:
                                        break
                                    fetched.extend(data)
                                    offset += len(data)
                                    if len(fetched) >= max_scan:
                                        break
                                results = local_fuzzy_search(fetched, query, top_k=top_k)
                            else:
                                fetched = []
                                offset = 0
                                page_size = 2000
                                while offset < max_scan:
                                    chunk = (
                                        _supa.table("document_chunks")
                                        .select("document_name,page_number,text")
                                        .eq("document_name", selected_doc)
                                        .order("page_number")
                                        .range(offset, offset + page_size - 1)
                                        .execute()
                                    )
                                    data = chunk.data or []
                                    if not data:
                                        break
                                    fetched.extend(data)
                                    offset += len(data)
                                    if len(fetched) >= max_scan:
                                        break
                                results = local_fuzzy_search(fetched, query, top_k=top_k)

                            st.write(f"Top {len(results)} fuzzy matches (local fallback).")

                    else:
                        # Semantic (RPC first, local fallback else)
                        model = load_embedding_model()
                        qv = model.encode([query])[0].astype(np.float32)
                        qv /= (np.linalg.norm(qv) + 1e-12)

                        rpc_ok = True
                        try:
                            if selected_doc == "All":
                                res = _supa.rpc(
                                    "find_similar_chunks",
                                    {"query_embedding": qv.tolist(), "match_count": top_k},
                                ).execute()
                            else:
                                res = _supa.rpc(
                                    "find_similar_chunks_in_doc",
                                    {"doc_name": selected_doc, "query_embedding": qv.tolist(), "match_count": top_k},
                                ).execute()
                            results = getattr(res, "data", []) or []
                            results.sort(key=lambda r: (r.get("document_name", ""), r.get("page_number") or 0))
                            st.write(f"Top {len(results)} semantic matches (server-side).")
                        except Exception:
                            rpc_ok = False

                        if not rpc_ok:
                            # Local semantic: fetch embeddings and score
                            fetched = []
                            offset = 0
                            page_size = 500
                            while offset < max_scan:
                                q = _supa.table("document_chunks").select("document_name,page_number,text,embedding")
                                if selected_doc != "All":
                                    q = q.eq("document_name", selected_doc)
                                q = q.order("document_name").order("page_number").range(offset, offset + page_size - 1)
                                chunk = q.execute()
                                data = chunk.data or []
                                if not data:
                                    break
                                fetched.extend([d for d in data if isinstance(d.get("embedding"), list)])
                                offset += len(data)
                                if len(fetched) >= max_scan:
                                    break
                            results = local_semantic_search(fetched, qv, top_k=top_k)
                            st.write(f"Top {len(results)} semantic matches (local fallback).")

                    # Render results
                    if not results:
                        st.info("No results found.")
                    else:
                        for i, r in enumerate(results, 1):
                            doc = r.get("document_name", "Unknown")
                            page = r.get("page_number", "?")
                            text = (r.get("text") or "").replace("\n", " ")
                            snippet = text[:400] + ("..." if len(text) > 400 else "")
                            st.markdown(f"**{i}. {doc} — Page {page}**")
                            st.write(snippet or "_(empty page)_")
                            if "sim" in r:
                                st.caption(f"Similarity: {r['sim']:.3f}")
                            st.divider()

                except Exception as e:
                    st.error("Search failed.")
                    st.exception(e)


# ---------------- Page Inspector ----------------
with tabs[2]:
    st.subheader("🔍 Page Inspector & Re-OCR")
    if not _supa:
        st.info("Configure Supabase first.")
    else:
        # Pick a document & page to inspect
        try:
            _docs = _supa.table("document_chunks").select("document_name").execute()
            _doc_names = sorted({r["document_name"] for r in (_docs.data or []) if r.get("document_name")})
        except Exception:
            _doc_names = []
        doc_to_fix = st.selectbox("Select document", _doc_names)
        page_to_fix = st.number_input("Page number", min_value=1, value=3, step=1)
        query_debug = st.text_input("Optional test keyword", value="Gen. Transporting")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Fetch row"):
                try:
                    r = (_supa.table("document_chunks")
                         .select("id,document_name,page_number,text,embedding")
                         .eq("document_name", doc_to_fix)
                         .eq("page_number", page_to_fix)
                         .limit(1)
                         .execute())
                    row = (r.data or [None])[0]
                    if not row:
                        st.error("No row found for that document/page.")
                    else:
                        st.session_state["_inspect_row"] = row
                        raw = row.get("text") or ""
                        norm = normalize_text(raw)
                        st.success("Row loaded.")
                        st.markdown("**Raw OCR (first 800 chars):**")
                        st.write((raw[:800] + ("..." if len(raw) > 800 else "")) or "_(empty)_")
                        st.markdown("**Normalized (first 800 chars):**")
                        st.write((norm[:800] + ("..." if len(norm) > 800 else "")) or "_(empty)_")
                        if query_debug:
                            qn = normalize_text(query_debug)
                            contains_raw = query_debug.lower() in (raw.lower())
                            contains_norm = qn in norm
                            st.write(f"Contains (raw) = {contains_raw}  |  Contains (normalized) = {contains_norm}")
                            try:
                                from rapidfuzz.fuzz import partial_ratio as _pr
                                score = _pr(qn, norm) / 100.0
                                st.write(f"Fuzzy score (partial_ratio vs normalized): {score:.3f}")
                            except Exception:
                                st.caption("Install rapidfuzz for fuzzy score in-app.")
                except Exception as e:
                    st.error("Fetch failed.")
                    st.exception(e)

        with colB:
            # Re-OCR just this page, then update DB row
            if st.button("Try tougher OCR variants"):
                bytes_ok = (
                    st.session_state.get("last_pdf_bytes") is not None
                    and st.session_state.get("last_pdf_name") == doc_to_fix
                )
                if not bytes_ok:
                    st.warning("Re-upload this exact PDF in Ingest so the app holds its bytes in memory, then come back.")
                else:
                    try:
                        pdf_bytes = st.session_state["last_pdf_bytes"]
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        if page_to_fix < 1 or page_to_fix > len(doc):
                            st.error(f"PDF has {len(doc)} pages — page {page_to_fix} is out of range.")
                        else:
                            page = doc[page_to_fix - 1]
                            variants = [
                                ("eng", "6", 3.0),
                                ("eng", "7", 3.5),
                                ("eng", "11", 3.5),
                                ("eng", "4", 4.0),   # block layout
                            ]
                            tried = []
                            for lang, psm, zoom in variants:
                                txt, conf = tesseract_ocr(
                                    page, zoom=float(zoom), lang=lang, psm=psm,
                                    do_denoise=True, do_autocontrast=True, binarize=True, thresh=175,
                                    user_words_path=user_words_path, timeout_sec=25
                                )
                                nrm = normalize_text(txt)
                                score = 0.0
                                if query_debug:
                                    try:
                                        from rapidfuzz.fuzz import partial_ratio as _pr
                                        score = _pr(normalize_text(query_debug), nrm) / 100.0
                                    except Exception:
                                        score = 1.0 if normalize_text(query_debug) in nrm else 0.0
                                tried.append({
                                    "lang": lang, "psm": psm, "zoom": zoom,
                                    "conf": conf, "len": len(nrm), "score": score,
                                    "text": txt, "text_norm": nrm,
                                })
                            doc.close()

                            tried.sort(key=lambda d: (-d["score"], -d["len"], -d["conf"]))
                            st.success(f"Tried {len(tried)} variants. Showing top 3:")
                            for i, cand in enumerate(tried[:3], 1):
                                st.markdown(f"**#{i} lang={cand['lang']} psm={cand['psm']} zoom={cand['zoom']} "
                                            f"(score={cand['score']:.3f}, conf={cand['conf']:.1f}, len={cand['len']})**")
                                prev = cand["text_norm"][:500] + ("..." if len(cand["text_norm"]) > 500 else "")
                                st.write(prev or "_(empty)_")

                            st.session_state["_re_ocr_candidates"] = tried
                    except Exception as e:
                        st.error("Re-OCR failed.")
                        st.exception(e)

        row = st.session_state.get("_inspect_row")
        cands = st.session_state.get("_re_ocr_candidates", [])
        if row and cands:
            pick = st.selectbox(
                "Pick a candidate to save back to Supabase",
                [f"#{i+1}: lang={d['lang']} psm={d['psm']} zoom={d['zoom']} (score={d['score']:.3f}, conf={d['conf']:.1f})"
                 for i, d in enumerate(cands[:5])]
            )
            idx = [f"#{i+1}: lang={d['lang']} psm={d['psm']} zoom={d['zoom']} (score={d['score']:.3f}, conf={d['conf']:.1f})"
                   for i, d in enumerate(cands[:5])].index(pick)
            chosen = cands[idx]
            if st.button("✅ Update this page in Supabase (text, text_norm, embedding)"):
                try:
                    model = load_embedding_model()
                    vec = model.encode([chosen["text"]])[0].astype(np.float32)
                    vec /= (np.linalg.norm(vec) + 1e-12)
                    _supa.table("document_chunks").update({
                        "text": chosen["text"],
                        "text_norm": chosen["text_norm"],
                        "embedding": vec.tolist(),
                    }).eq("id", row["id"]).execute()
                    st.success("Page updated. Re-run your search.")
                except Exception as e:
                    st.error("Update failed.")
                    st.exception(e)


# ---------------- Notes ----------------
with tabs[3]:
    st.markdown(
        """
        **Notes & Tips**
        - Keyword search hits both **raw** and **normalized** text (hyphenation joined, zero-width removed, quotes/dashes normalized).
        - **Fuzzy** search uses Postgres `pg_trgm` RPC if present; otherwise it falls back to a local fuzzy scorer (add `rapidfuzz` for speed).
        - For long PDFs, set **Limit pages** for quick tests; enable **Intensive** if a page looks under-read.
        - Ensure RLS is disabled or anon policies allow INSERT/SELECT on `document_chunks`.
        - For best fidelity, keep **handwriting overlay OFF** unless you truly need it; it’s slower and can misread some printed tokens.
        """
    )


# ===================== Local helpers: fuzzy/semantic fallbacks =====================
def local_fuzzy_search(rows: List[Dict[str, Any]], query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    qn = normalize_text(query)
    if not qn:
        return []
    results = []
    use_rf = partial_ratio is not None
    tokens = [t for t in qn.split() if len(t) >= 3]
    for r in rows:
        tn = normalize_text(r.get("text", ""))
        if tokens and not any(t in tn for t in tokens):
            continue
        if use_rf:
            score = partial_ratio(qn, tn)
        else:
            score = 100 if qn in tn else (80 if tokens and tn.find(tokens[0]) >= 0 else 0)
        out = {k: r.get(k) for k in ("document_name", "page_number", "text")}
        out["sim"] = float(score) / 100.0
        results.append(out)
    results.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return results[:top_k]


def local_semantic_search(rows: List[Dict[str, Any]], query_vec: np.ndarray, top_k: int = 50) -> List[Dict[str, Any]]:
    embs = [np.array(r["embedding"], dtype=np.float32) for r in rows if r.get("embedding") is not None]
    if not embs:
        return []
    M = np.vstack(embs)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    q = query_vec.astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    sims = M @ q

    kept = []
    j = 0
    for r in rows:
        if r.get("embedding") is None:
            continue
        sim = float(sims[j])
        kept.append({
            "document_name": r.get("document_name"),
            "page_number": r.get("page_number"),
            "text": r.get("text"),
            "sim": sim
        })
        j += 1

    kept.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return kept
