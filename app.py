# app.py — Single-page RAG with enhanced OCR + options
import io
import os
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path

# Embeddings + DB
from sentence_transformers import SentenceTransformer
from supabase import create_client
from supabase.client import ClientOptions
import supabase as _sb

st.set_page_config(page_title="RAG: PDF → Embeddings → Search", layout="wide")
st.title("RAG App: PDF Upload → Embeddings → Search (Enhanced OCR)")

# ================== Sidebar Setup Checks ==================
with st.sidebar:
    st.header("Setup checks")
    st.write("supabase-py version:", _sb.__version__)
    has_url = "SUPABASE_URL" in st.secrets
    has_key = "SUPABASE_ANON_KEY" in st.secrets
    st.write("Has SUPABASE_URL:", has_url)
    if has_url:
        st.write("SUPABASE_URL:", st.secrets["SUPABASE_URL"])
    st.write("Has SUPABASE_ANON_KEY:", has_key)

def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
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
else:
    st.sidebar.error("❌ Supabase not configured. Add secrets in Settings → Secrets.")

# ================== Caches ==================
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    # 384-dim sentence embeddings (free, lightweight)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================== OCR Helper ==================
def ocr_from_page(page, zoom=3.0, lang="eng", psm="6",
                  do_denoise=True, do_autocontrast=True,
                  binarize=True, thresh=180, user_words_path=None):
    """
    Higher-quality OCR for a single PDF page:
    - Renders at higher DPI (zoom 3.0 ≈ ~216 DPI; 4.0 ≈ ~288 DPI).
    - Optional denoise + autocontrast + binarize to sharpen text.
    - Tesseract tuned with configurable PSM (page segmentation mode).
    """
    # 1) Render page at higher resolution for better character shapes
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # no alpha channel
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")  # grayscale

    # 2) Preprocessing
    if do_denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    if do_autocontrast:
        img = ImageOps.autocontrast(img)
    if binarize:
        # simple global threshold; tweak (140–220) per doc quality
        img = img.point(lambda p: 255 if p > thresh else 0)

    # 3) Tesseract config
    # --oem 1 = LSTM engine, --psm X controls layout assumptions
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    if user_words_path:
        config += f" --user-words {user_words_path}"

    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return (text or "").strip()

# ================== Extractors ==================
def extract_text_from_pdf(file_bytes, ocr_opts, force_ocr=False):
    """Whole-document text; per-page OCR fallback or force OCR."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    total = len(doc)
    prog = st.progress(0, text=f"Extracting text 0/{total} pages...")
    for i, page in enumerate(doc):
        try:
            text = ""
            if not force_ocr:
                text = (page.get_text() or "").strip()
            if not text:
                text = ocr_from_page(
                    page,
                    zoom=ocr_opts["zoom"],
                    lang=ocr_opts["lang"],
                    psm=ocr_opts["psm"],
                    do_denoise=ocr_opts["do_denoise"],
                    do_autocontrast=ocr_opts["do_autocontrast"],
                    binarize=ocr_opts["binarize"],
                    thresh=ocr_opts["thresh"],
                    user_words_path=ocr_opts["user_words_path"]
                )
            parts.append(text)
        except Exception as e:
            parts.append(f"[Error reading page {i+1}: {e}]")
        finally:
            prog.progress((i + 1) / total, text=f"Extracting text {i+1}/{total} pages...")
    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(parts).strip()

def extract_pages_with_metadata(file_bytes, document_name, ocr_opts, force_ocr=False):
    """One chunk per page with {document_name, page_number, text}."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    pages = []
    prog = st.progress(0, text=f"Splitting into page chunks 0/{total}...")
    for i, page in enumerate(doc):
        text = ""
        if not force_ocr:
            text = (page.get_text() or "").strip()
        if not text:
            text = ocr_from_page(
                page,
                zoom=ocr_opts["zoom"],
                lang=ocr_opts["lang"],
                psm=ocr_opts["psm"],
                do_denoise=ocr_opts["do_denoise"],
                do_autocontrast=ocr_opts["do_autocontrast"],
                binarize=ocr_opts["binarize"],
                thresh=ocr_opts["thresh"],
                user_words_path=ocr_opts["user_words_path"]
            )
        pages.append({
            "document_name": document_name,
            "page_number": i + 1,
            "text": text
        })
        prog.progress((i + 1) / total, text=f"Splitting into page chunks {i+1}/{total}...")
    doc.close()
    return pages

# ================== Embeddings ==================
def embed_texts(model, texts):
    """Return L2-normalized embeddings (n x 384)."""
    emb = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb

# ================== OCR Options UI ==================
with st.expander("⚙️ OCR Options (tune if extraction is wrong)", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        zoom = st.slider("Render zoom (DPI proxy)", 2.0, 4.0, 3.0, 0.5,
                         help="Higher → sharper OCR but slower (2≈144DPI, 3≈216DPI, 4≈288DPI).")
        psm_choice = st.selectbox(
            "Tesseract PSM (layout mode)",
            options=[("6 - single uniform block (default)", "6"),
                     ("4 - single column of text", "4"),
                     ("3 - fully automatic", "3"),
                     ("11 - sparse text", "11"),
                     ("12 - sparse text with OSD", "12")],
            index=0, format_func=lambda x: x[0]
        )[1]
    with col2:
        lang = st.text_input("Language code", "eng",
                             help="Install extra packs in packages.txt (e.g., tesseract-ocr-fra).")
        do_denoise = st.checkbox("Denoise (median)", True)
        do_autocontrast = st.checkbox("Auto-contrast", True)
    with col3:
        binarize = st.checkbox("Binarize (black/white)", True)
        thresh = st.slider("Binarize threshold", 140, 220, 180, 1)
        force_ocr = st.checkbox("Force OCR on all pages", False,
                                help="Ignore embedded text and OCR everything.")

    # Optional custom vocabulary (.txt, one term per line)
    user_words_file = st.file_uploader("Optional: custom vocabulary file (user_words.txt)", type=["txt"])
    user_words_path = None
    if user_words_file is not None:
        # Save to a temp path in the working directory
        user_words_path = Path("user_words.txt").absolute()
        with open(user_words_path, "wb") as f:
            f.write(user_words_file.read())
        st.caption(f"Custom vocabulary saved to {user_words_path}")

ocr_opts = {
    "zoom": float(zoom),
    "lang": lang.strip() or "eng",
    "psm": psm_choice,
    "do_denoise": do_denoise,
    "do_autocontrast": do_autocontrast,
    "binarize": binarize,
    "thresh": int(thresh),
    "user_words_path": str(user_words_path) if user_words_path else None
}

# ================== STEP 1: Upload & Extract ==================
st.header("Step 1: Upload & Extract")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file:
    st.info(f"**Selected:** {uploaded_file.name} • {uploaded_file.size/1024:.1f} KB")
    if st.button("1) Process PDF", type="primary"):
        with st.spinner("Processing..."):
            try:
                file_bytes = uploaded_file.read()
                full_text = extract_text_from_pdf(file_bytes, ocr_opts, force_ocr=force_ocr)
                page_chunks = extract_pages_with_metadata(file_bytes, uploaded_file.name, ocr_opts, force_ocr=force_ocr)
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

# ================== STEP 2: Embed & Upload to Supabase ==================
st.header("Step 2: Embed & Upload to Supabase")

if not _supa:
    st.warning("Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY in Settings → Secrets.")
elif "pages" not in st.session_state or not st.session_state["pages"]:
    st.info("No pages detected yet. Complete Step 1 first, then return here.")
else:
    st.write(f"Pages ready to index: **{len(st.session_state['pages'])}** "
             f"(from **{st.session_state.get('document_name','unknown')}**)")
    colA, colB = st.columns(2)
    with colA:
        delete_first = st.checkbox("Delete existing rows for this document before upload", False)
    with colB:
        st.caption("Check to avoid duplicates if re-indexing the same PDF.")

    if st.button("2) Generate embeddings & upload", type="primary"):
        try:
            if delete_first:
                try:
                    _ = _supa.rpc("delete_document", {"p_document_name": st.session_state.get("document_name", "")}).execute()
                    st.info("Old rows for this document deleted (if any).")
                except Exception:
                    # helper may not exist; fall back to raw delete
                    _supa.table("document_chunks").delete().eq("document_name", st.session_state.get("document_name", "")).execute()
                    st.info("Old rows deleted via fallback.")

            model = load_embedding_model()
            pages = st.session_state["pages"]
            texts = [p["text"] if p["text"] else "" for p in pages]

            st.info("Generating embeddings…")
            emb = embed_texts(model, texts)  # (n, 384)

            rows = []
            for p, vec in zip(pages, emb):
                rows.append({
                    "document_name": p["document_name"],
                    "page_number": p["page_number"],
                    "text": p["text"],
                    "embedding": vec.tolist()
                })

            st.info("Uploading to Supabase…")
            BATCH = 100
            inserted = 0
            for i in range(0, len(rows), BATCH):
                batch = rows[i:i+BATCH]
                res = _supa.table("document_chunks").insert(batch).execute()
                if getattr(res, "data", None) is None:
                    st.error(f"No data returned for batch {i+1}-{i+len(batch)}. "
                             f"Check RLS (disable or add insert/select policies) and that the table exists.")
                    st.write(res)
                    st.stop()
                inserted += len(res.data)
                st.write(f"Inserted rows {i+1}–{i+len(batch)} (total {inserted})")
                time.sleep(0.05)

            st.success(f"All chunks uploaded to Supabase. Total inserted: {inserted}")
            st.info("If you don't see data in Table Editor, ensure RLS is disabled on 'document_chunks' or policies allow anon INSERT/SELECT.")
        except Exception as e:
            st.error("Embedding or upload failed.")
            st.exception(e)

# ================== STEP 3: Search ==================
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

    query = st.text_input("Enter your query")
    mode = st.radio("Search type", ["Keyword (exact text match)", "Semantic (meaning match)"])
    top_k = st.slider("Results to show", 1, 20, 5)

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
                q = _supa.table("document_chunks").select("document_name,page_number,text").order("id", desc=True).limit(3)
                res = q.execute()
                st.write(res.data)
            except Exception as e:
                st.error(f"Preview failed: {e}")

    if st.button("3) Search", type="primary"):
        if not query:
            st.warning("Please enter a query.")
        else:
            try:
                results = []
                if mode.startswith("Keyword"):
                    q = _supa.table("document_chunks").select("document_name,page_number,text").ilike("text", f"%{query}%")
                    if selected_doc != "All":
                        q = q.eq("document_name", selected_doc)
                    res = q.limit(top_k).execute()
                    results = getattr(res, "data", []) or []
                else:
                    # Semantic: embed query, call RPC(s)
                    model = load_embedding_model()
                    qv = model.encode([query])[0].astype(np.float32)
                    qv /= (np.linalg.norm(qv) + 1e-12)

                    if selected_doc == "All":
                        res = _supa.rpc("find_similar_chunks", {
                            "query_embedding": qv.tolist(),
                            "match_count": top_k
                        }).execute()
                    else:
                        # filtered RPC (created in the full SQL I gave you)
                        res = _supa.rpc("find_similar_chunks_in_doc", {
                            "doc_name": selected_doc,
                            "query_embedding": qv.tolist(),
                            "match_count": top_k
                        }).execute()
                    results = getattr(res, "data", []) or []

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
                        st.divider()
            except Exception as e:
                st.error("Search failed.")
                st.exception(e)

# ================== Footer Tips ==================
with st.expander("Free-tier tips"):
    st.markdown(
        "- Supabase free DB is ~500 MB; fine for thousands of pages.\n"
        "- If inserts fail, check **RLS** (disable, or add anon SELECT/INSERT policies).\n"
        "- This app sleeps when idle; no cost while dormant.\n"
        "- If OCR is still off, raise **zoom** to 4.0 or try **PSM 4** (single column)."
    )
