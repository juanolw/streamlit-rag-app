# app.py — RAG: PDF → OCR → normalize → embed → Supabase → search
# Modes: Keyword (exact/normalized), Keyword (fuzzy w/ pg_trgm or local fallback), Semantic (RPC or local)

import io
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

# ===================== DEBUG: Build tag + asc scanner =====================
BUILD_ID = "full-python-stack-ocr-fuzzy-2025-08-14"
try:
    with open(__file__, "r", encoding="utf-8") as _f:
        _src = _f.read()
    ASC_OCCURRENCES = len(re.findall(r"\border\s*\([^)]*asc\s*=", _src))
except Exception:
    ASC_OCCURRENCES = -1
# ==========================================================================

st.set_page_config(page_title="RAG (OCR + Supabase)", layout="wide")
st.title("RAG App — PDF → OCR → Embeddings → Supabase Search")

# ---------------- Sidebar: environment checks ----------------
with st.sidebar:
    st.header("Setup checks")
    st.write("Build:", BUILD_ID)
    st.write("`asc=` occurrences in this running file:", ASC_OCCURRENCES)
    st.write("supabase-py version:", _sb.__version__)
    if "SUPABASE_URL" in st.secrets:
        st.caption("SUPABASE_URL: " + st.secrets["SUPABASE_URL"])
    try:
        langs = pytesseract.get_languages(config="")
        st.caption("Tesseract languages: " + ", ".join(sorted(langs)))
    except Exception:
        st.caption("Tesseract language list unavailable")

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
        st.sidebar.success(f"✅ Supabase connected. document_chunks rows: {_res.count}")
    except Exception as e:
        st.sidebar.error(f"❌ Supabase connection failed: {e}")

# ---------------- Normalization helpers ----------------
ZW_REMOVE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None)  # soft hyphen + zero-widths
PUNCT_MAP = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "\u00A0": " ", "·": " "
})
_PUNCT_STRIP_RE = re.compile(r"[.,:;|/\\()\[\]{}<>•·…]+")
_SPACE_COLLAPSE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """
    Aggressive normalization for keyword matching:
    - join hyphenated line breaks, flatten newlines
    - remove zero-width & soft hyphen
    - normalize quotes/dashes; strip most punctuation to spaces
    - NFKC unicode normalize; collapse spaces; lowercase
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

# ---------------- Caches ----------------
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    # 384-dim, light & solid for retrieval
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_trocr_handwritten():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    return processor, model, device

# ---------------- Page render & preprocessing ----------------
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

# ---------------- OCR engines ----------------
def tesseract_ocr(
    page,
    zoom: float,
    lang: str,
    psm: str,
    do_denoise: bool,
    do_autocontrast: bool,
    binarize: bool,
    thresh: int,
    user_words_path: Optional[str] = None,
    timeout_sec: int = 20,
) -> Tuple[str, float]:
    """
    Tesseract OCR with confidence and a hard timeout.
    Returns (text, avg_conf). On timeout/failure, returns best-effort text with low conf.
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
        # Fallback to plain string with shorter timeout
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

def trocr_ocr(page, zoom: float) -> str:
    processor, model, device = load_trocr_handwritten()
    import torch
    img = render_page_to_image(page, zoom=zoom).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_length=512)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip()

# ---------------- OCR strategy ----------------
def ocr_strategy(
    page,
    preset: str,
    user_words_path: Optional[str],
    force_ocr: bool,
    base_zoom: float,
    use_trocr: bool,
    timeout_sec: int,
    psm_eng: str = "6",
    psm_rtl_cjk: str = "4",
    intensive: bool = False,
) -> str:
    if preset == "English (printed)":
        text, conf = tesseract_ocr(page, base_zoom, "eng", psm_eng, True, True, True, 180, user_words_path, timeout_sec)
        if conf < 70 and intensive:
            text2, conf2 = tesseract_ocr(page, base_zoom + 0.5, "eng", "4", True, True, True, 180, user_words_path, timeout_sec)
            return text2 if conf2 > conf else text
        return text

    if preset == "English (handwritten)":
        if use_trocr:
            try:
                text = trocr_ocr(page, max(3.0, base_zoom))
                if len(text) >= 4:
                    return text
            except Exception:
                pass
        text, conf = tesseract_ocr(page, base_zoom + 0.5, "eng", "4", True, True, True, 175, user_words_path, timeout_sec)
        if conf < 65 and intensive:
            text2, conf2 = tesseract_ocr(page, base_zoom + 1.0, "eng", "3", True, True, True, 170, user_words_path, timeout_sec)
            return text2 if conf2 > conf else text
        return text

    if preset == "Arabic (ara)":
        text, conf = tesseract_ocr(page, base_zoom, "ara", psm_rtl_cjk, True, True, True, 180, user_words_path, timeout_sec)
        if conf < 65 and intensive:
            text2, conf2 = tesseract_ocr(page, base_zoom + 0.5, "ara", "3", True, True, True, 175, user_words_path, timeout_sec)
            return text2 if conf2 > conf else text
        return text

    if preset == "Chinese (Simplified)":
        text, conf = tesseract_ocr(page, base_zoom, "chi_sim", psm_rtl_cjk, True, True, True, 180, user_words_path, timeout_sec)
        if conf < 65 and intensive:
            text2, conf2 = tesseract_ocr(page, base_zoom + 0.5, "chi_sim", "3", True, True, True, 175, user_words_path, timeout_sec)
            return text2 if conf2 > conf else text
        return text

    if preset == "Chinese (Traditional)":
        text, conf = tesseract_ocr(page, base_zoom, "chi_tra", psm_rtl_cjk, True, True, True, 180, user_words_path, timeout_sec)
        if conf < 65 and intensive:
            text2, conf2 = tesseract_ocr(page, base_zoom + 0.5, "chi_tra", "3", True, True, True, 175, user_words_path, timeout_sec)
            return text2 if conf2 > conf else text
        return text

    text, _ = tesseract_ocr(page, base_zoom, "eng", "6", True, True, True, 180, user_words_path, timeout_sec)
    return text

# ---------------- Extractors ----------------
def extract_text_from_pdf(file_bytes, preset: str, user_words_path: Optional[str],
                          force_ocr: bool, intensive: bool, base_zoom: float,
                          use_trocr: bool, timeout_sec: int, max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    total = len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    prog = st.progress(0, text=f"Extracting text 0/{limit} pages...")
    for i, page in enumerate(doc):
        if i >= limit:
            break
        try:
            text = ""
            if not force_ocr:
                text = (page.get_text() or "").strip()
            if not text:
                text = ocr_strategy(page, preset, user_words_path, force_ocr, base_zoom,
                                    use_trocr=use_trocr, timeout_sec=timeout_sec, intensive=intensive)
            parts.append(text)
        except Exception as e:
            parts.append(f"[Error reading page {i+1}: {e}]")
        finally:
            prog.progress((i + 1) / limit, text=f"Extracting text {i+1}/{limit} pages...")
    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(parts).strip()

def extract_pages_with_metadata(file_bytes, document_name, preset: str, user_words_path: Optional[str],
                                force_ocr: bool, intensive: bool, base_zoom: float,
                                use_trocr: bool, timeout_sec: int, max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    pages = []
    prog = st.progress(0, text=f"Splitting into page chunks 0/{limit}...")
    for i, page in enumerate(doc):
        if i >= limit:
            break
        try:
            text = ""
            if not force_ocr:
                text = (page.get_text() or "").strip()
            if not text:
                text = ocr_strategy(page, preset, user_words_path, force_ocr, base_zoom,
                                    use_trocr=use_trocr, timeout_sec=timeout_sec, intensive=intensive)
            pages.append({"document_name": document_name, "page_number": i + 1, "text": text})
        except Exception as e:
            pages.append({"document_name": document_name, "page_number": i + 1, "text": f"[Error: {e}]"})
        finally:
            prog.progress((i + 1) / limit, text=f"Splitting into page chunks {i+1}/{limit}...")
    doc.close()
    return pages

# ---------------- Embeddings ----------------
def embed_texts(model, texts):
    emb = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb

# ---------------- OCR options UI ----------------
with st.expander("⚙️ OCR Language & Options", expanded=True):
    preset = st.selectbox(
        "Language preset",
        ["English (printed)", "English (handwritten)", "Arabic (ara)", "Chinese (Simplified)", "Chinese (Traditional)"],
        index=0
    )
    if preset == "English (printed)":
        default_zoom = 3.0; default_intensive = False
    elif preset == "English (handwritten)":
        default_zoom = 3.5; default_intensive = True
    else:
        default_zoom = 3.5; default_intensive = True

    col1, col2 = st.columns(2)
    with col1:
        base_zoom = st.slider("Render zoom (DPI proxy)", 2.0, 4.5, default_zoom, 0.5)
        force_ocr = st.checkbox("Force OCR on all pages (ignore embedded text)", False)
        max_pages_debug = st.number_input("Limit pages to process (0 = all)", min_value=0, value=0, step=1)
    with col2:
        intensive = st.checkbox("Intensive mode (retry with higher DPI if low confidence)", default_intensive)
        timeout_sec = st.slider("OCR timeout per page (sec)", 5, 60, 20)
        use_trocr = st.checkbox("Use handwriting engine (TrOCR) when preset = English (handwritten)", value=False)

    # Optional vocabulary file (.txt)
    user_words_path = None
    use_vocab = st.checkbox("Use custom vocabulary (.txt only)", value=False)
    if use_vocab:
        st.caption("Upload a .txt with one term per line (e.g., 'Gen. Transporting')")
        user_words_file = st.file_uploader("Upload vocabulary file (TXT only)", type=["txt"], key="user_words")
        if user_words_file is not None:
            user_words_path = Path("user_words.txt").absolute()
            with open(user_words_path, "wb") as f:
                f.write(user_words_file.read())
            st.caption(f"Custom vocabulary saved to {user_words_path}")

# ---------------- STEP 1: Upload & Extract ----------------
st.header("Step 1: Upload & Extract")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file:
    st.info(f"**Selected:** {uploaded_file.name} • {uploaded_file.size/1024:.1f} KB")
    if st.button("1) Process PDF", type="primary"):
        with st.spinner("Processing..."):
            try:
                file_bytes = uploaded_file.read()
                full_text = extract_text_from_pdf(
                    file_bytes, preset, user_words_path, force_ocr, intensive, base_zoom,
                    use_trocr=use_trocr, timeout_sec=timeout_sec, max_pages=max_pages_debug
                )
                page_chunks = extract_pages_with_metadata(
                    file_bytes, uploaded_file.name, preset, user_words_path, force_ocr, intensive, base_zoom,
                    use_trocr=use_trocr, timeout_sec=timeout_sec, max_pages=max_pages_debug
                )
                st.session_state["pages"] = page_chunks
                st.session_state["document_name"] = uploaded_file.name

                st.success("Extraction complete. Page-level chunks ready.")
                st.subheader("Per-page chunks (preview)")
                for rec in page_chunks[:5]:
                    st.markdown(f"**{rec['document_name']} — Page {rec['page_number']}**")
                    preview = (rec["text"] or "").replace("\n", " ")
                    st.write((preview[:500] + ("..." if len(preview) > 500 else "")) or "_(empty page)_")
                    st.divider()

                st.subheader("Full Extracted Text")
                st.text_area("PDF Text Content", full_text or "", height=300)
            except Exception as e:
                st.error("PDF processing failed.")
                st.exception(e)

# ---------------- STEP 2: Embed & Upload to Supabase ----------------
st.header("Step 2: Embed & Upload to Supabase")

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
        delete_first = st.checkbox("Delete existing rows for this document before upload", False)
    with colB:
        st.caption("Check to avoid duplicates if re-indexing the same PDF.")

    if st.button("2) Generate embeddings & upload", type="primary"):
        try:
            # ensure text_norm column exists (silent if not)
            try:
                _ = _supa.table("document_chunks").select("text_norm").limit(1).execute()
            except Exception:
                st.warning("Column text_norm not found. Create it in Supabase: ALTER TABLE document_chunks ADD COLUMN text_norm text;")

            if delete_first:
                try:
                    _ = _supa.rpc(
                        "delete_document", {"p_document_name": st.session_state.get("document_name", "")}
                    ).execute()
                    st.info("Old rows for this document deleted (if any).")
                except Exception:
                    _supa.table("document_chunks").delete().eq(
                        "document_name", st.session_state.get("document_name", "")
                    ).execute()
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
                    }
                )

            st.info("Uploading to Supabase…")
            BATCH = 100
            inserted = 0
            for i in range(0, len(rows), BATCH):
                batch = rows[i : i + BATCH]
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
            st.info(
                "If you don't see data in Table Editor, ensure RLS is disabled or anon policies allow INSERT/SELECT."
            )
        except Exception as e:
            st.error("Embedding or upload failed.")
            st.exception(e)

    # ---- Maintenance: Backfill text_norm for existing rows in this doc ----
    with st.expander("Maintenance: Backfill text_norm for this document"):
        st.caption("Use this if you indexed pages before this update so older rows get `text_norm`.")
        if st.button("Backfill now"):
            try:
                docname = st.session_state.get("document_name", "")
                if not docname:
                    st.warning("No document selected.")
                else:
                    total_updated = 0
                    page = 0
                    page_size = 500
                    while True:
                        q = (_supa.table("document_chunks")
                             .select("id,text")
                             .eq("document_name", docname)
                             .range(page * page_size, page * page_size + page_size - 1))
                        res = q.execute()
                        rows = res.data or []
                        if not rows:
                            break
                        for r in rows:
                            tid = r["id"]; raw = r.get("text") or ""
                            tn = normalize_text(raw)
                            _supa.table("document_chunks").update({"text_norm": tn}).eq("id", tid).execute()
                            total_updated += 1
                        page += 1
                    st.success(f"Backfilled text_norm for {total_updated} rows in '{docname}'.")
            except Exception as e:
                st.error("Backfill failed.")
                st.exception(e)

# ---------------- Helpers: local fuzzy + semantic fallback ----------------
def local_fuzzy_search(rows: List[Dict[str, Any]], query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Fuzzy search across rows (each with text_norm). Uses rapidfuzz.partial_ratio if available,
    else Python difflib fallback (slower).
    """
    qn = normalize_text(query)
    if not qn:
        return []
    results = []
    use_rf = partial_ratio is not None
    # Prefilter by token presence to reduce work
    tokens = [t for t in qn.split() if len(t) >= 3]
    for r in rows:
        tn = normalize_text(r.get("text", ""))
        if tokens and not any(t in tn for t in tokens):
            continue
        if use_rf:
            score = partial_ratio(qn, tn)
        else:
            # simple heuristic if rapidfuzz not installed
            score = 100 if qn in tn else (80 if tn.find(tokens[0]) >= 0 else 0) if tokens else (100 if qn in tn else 0)
        results.append({**r, "sim": float(score) / 100.0})
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
    idxs = np.argsort(-sims)[:top_k]
    # Need to map back to rows; ensure alignment
    kept = []
    j = 0
    for r in rows:
        if r.get("embedding") is None:
            continue
        if j in idxs:
            kept.append({k: r[k] for k in ("document_name", "page_number", "text") if k in r} | {"sim": float(sims[j])})
        j += 1
    kept.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return kept

# ---------------- STEP 3: Search (Keyword / Fuzzy / Semantic) ----------------
st.header("Step 3: Search (Keyword / Semantic)")

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
        index=0
    )
    top_k = st.slider("Results to show", 5, 200, 50)
    fetch_all = st.checkbox("Return all matches (only applies to exact/normalized)")
    max_scan = st.number_input("Max rows to scan in local fuzzy/semantic fallback", 100, 20000, 5000, step=100)

    # Quick checks
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Check row count"):
            try:
                res = _supa.table("document_chunks").select("id", count="exact").limit(1).execute()
                st.write(f"Row count: {res.count}")
            except Exception as e:
                st.error(f"Count failed: {e}")
    with c2:
        if st.button("Show 3 most recent rows"):
            try:
                res = (
                    _supa.table("document_chunks")
                    .select("document_name,page_number,text")
                    .order("id", desc=True)  # newest first (valid kwarg)
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
                    # Exact-ish: search raw TEXT and TEXT_NORM with ILIKE (fast & simple)
                    norm_query = normalize_text(query)
                    oq = escape_for_or_filter(query)
                    onq = escape_for_or_filter(norm_query)

                    base = _supa.table("document_chunks").select("document_name,page_number,text", count="exact")
                    base = base.or_(f"text.ilike.%{oq}%,text_norm.ilike.%{onq}%")
                    if selected_doc != "All":
                        base = base.eq("document_name", selected_doc)

                    if fetch_all:
                        page_size = 100
                        first = (
                            base.order("document_name")
                                .order("page_number")
                                .range(0, page_size - 1).execute()
                        )
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
                        # Fields from RPC: document_name, page_number, text, sim
                        results.sort(key=lambda r: (-r.get("sim", 0.0), r.get("document_name", ""), r.get("page_number") or 0))
                        st.write(f"Top {len(results)} fuzzy matches (server-side).")
                    except Exception:
                        rpc_ok = False

                    # Local fallback if RPC not available
                    if not rpc_ok:
                        # Fetch candidates
                        if selected_doc == "All":
                            # Page through to max_scan
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
                            # Restrict to one document
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
                            # keep only rows with embeddings
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

# ---------------- Footer ----------------
with st.expander("Notes & Tips"):
    st.markdown(
        "- Keyword search hits both **raw** and **normalized** text (hyphenation joined, zero-width removed, quotes/dashes normalized).\n"
        "- **Fuzzy** search uses Postgres `pg_trgm` RPC if present; otherwise it falls back to a local fuzzy scorer (add `rapidfuzz` for speed).\n"
        "- For long PDFs, set **Limit pages** for quick tests; enable **Intensive** if a page looks under-read.\n"
        "- Ensure RLS is disabled or anon policies allow INSERT/SELECT on `document_chunks`."
    )
