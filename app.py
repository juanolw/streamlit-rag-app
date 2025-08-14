# app.py — RAG with English-first OCR (handwriting optional), Arabic/Chinese presets, Supabase search
import io
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output as TessOutput
from PIL import Image, ImageOps, ImageFilter

# Embeddings + DB
from sentence_transformers import SentenceTransformer
from supabase import create_client
from supabase.client import ClientOptions
import supabase as _sb

BUILD_ID = "fix-asc-2025-08-14"
import inspect, os, re
try:
    with open(__file__, "r", encoding="utf-8") as _f:
        _code = _f.read()
    _asc_hits = len(re.findall(r"\basc\s*=", _code))
except Exception:
    _asc_hits = -1

with st.sidebar:
    st.write("Build:", BUILD_ID)
    st.write("asc= occurrences in this running file:", _asc_hits)
    st.write("Script path:", __file__)

# ---------- Page ----------
st.set_page_config(page_title="RAG: PDF → Embeddings → Search (OCR-first)", layout="wide")
st.title("RAG App: PDF Upload → Embeddings → Search — English-first OCR (Handwriting), Arabic/Chinese")

# ---------- Sidebar setup checks ----------
with st.sidebar:
    st.header("Setup checks")
    st.write("supabase-py version:", _sb.__version__)
    try:
        langs = pytesseract.get_languages(config="")
        st.caption("Tesseract languages found: " + ", ".join(sorted(langs)))
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

# ---------- Caches ----------
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim

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

# ---------- Render & preprocess ----------
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

# ---------- OCR engines ----------
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
    Returns (text, avg_conf). If timeout/failure, returns best-effort text with low conf.
    """
    img = render_page_to_image(page, zoom=zoom)
    img = preprocess_image(img, do_denoise, do_autocontrast, binarize, thresh, to_gray=True)

    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    if user_words_path:
        config += f" --user-words {user_words_path}"

    # Try data (conf) first
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=TessOutput.DICT, timeout=timeout_sec
        )
        words = [w for w in data.get("text", []) if isinstance(w, str) and w.strip()]
        confs = [int(c) for c in data.get("conf", []) if c not in (None, "", "-1")]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        text = " ".join(words) if words else ""
    except Exception:
        # Fallback to plain string (shorter timeout)
        try:
            text = pytesseract.image_to_string(img, lang=lang, config=config, timeout=max(5, timeout_sec // 2))
        except Exception:
            text = ""
        return (text or "").strip(), 0.0

    # If empty string, try plain string once (keep avg_conf)
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

# ---------- OCR strategy ----------
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
    """
    - English (printed): Tesseract PSM 6; escalate if low confidence and intensive=True.
    - English (handwritten): TrOCR if use_trocr=True; else Tesseract tuned for messy text.
    - Arabic/Chinese: Tesseract PSM 4; escalate if low confidence and intensive=True.
    """
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

# ---------- Extractors ----------
def extract_text_from_pdf(file_bytes, preset: str, user_words_path: Optional[str],
                          force_ocr: bool, intensive: bool, base_zoom: float,
                          use_trocr: bool, timeout_sec: int, max_pages: int = 0):
    """Whole-document text; tries embedded first unless force_ocr, then strategy."""
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
    """One chunk per page with {document_name, page_number, text} using strategy."""
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

# ---------- Embeddings ----------
def embed_texts(model, texts):
    emb = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb

# ---------- OCR Presets & Options ----------
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

    # Optional vocabulary file (.txt only) — tucked behind a toggle
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

# ---------- STEP 1: Upload & Extract ----------
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

                empty_pages = [p["page_number"] for p in page_chunks if not (p["text"] or "").strip()]
                if empty_pages:
                    st.warning(f"OCR returned empty text on pages: {empty_pages}. Try higher zoom or keep Intensive ON.")

                st.subheader("Full Extracted Text")
                st.text_area("PDF Text Content", full_text or "", height=300)
            except Exception as e:
                st.error("PDF processing failed.")
                st.exception(e)

# ---------- STEP 2: Embed & Upload to Supabase ----------
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
                rows.append(
                    {
                        "document_name": p["document_name"],
                        "page_number": p["page_number"],
                        "text": p["text"],
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
                        f"Check RLS (disable or add insert/select policies) and that the table exists."
                    )
                    st.write(res)
                    st.stop()
                inserted += len(res.data)
                st.write(f"Inserted rows {i+1}–{i+len(batch)} (total {inserted})")
                time.sleep(0.05)

            st.success(f"All chunks uploaded to Supabase. Total inserted: {inserted}")
            st.info(
                "If you don't see data in Table Editor, ensure RLS is disabled on 'document_chunks' "
                "or policies allow anon INSERT/SELECT."
            )
        except Exception as e:
            st.error("Embedding or upload failed.")
            st.exception(e)

# ---------- STEP 3: Search (Keyword / Semantic) ----------
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
    top_k = st.slider("Results to show (when not returning all)", 5, 200, 50)
    fetch_all = st.checkbox("Return all keyword matches (may be slower)")

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
                    .order("id", desc=True)  # newest first
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
                results = []
                total = None

                if mode.startswith("Keyword"):
                    base = (
                        _supa.table("document_chunks")
                        .select("document_name,page_number,text", count="exact")
                        .ilike("text", f"%{query}%")
                    )
                    if selected_doc != "All":
                        base = base.eq("document_name", selected_doc)

                    if fetch_all:
                        # Fetch ALL matches in pages of 100 (ascending by default)
                        page_size = 100
                        first = (
                            base.order("document_name")
                                .order("page_number")
                                .range(0, page_size - 1)
                                .execute()
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
                        res = (
                            base.order("document_name")
                                .order("page_number")
                                .limit(top_k)
                                .execute()
                        )
                        results = getattr(res, "data", []) or []
                        total = getattr(res, "count", None)

                    results.sort(key=lambda r: (r.get("document_name", ""), r.get("page_number") or 0))
                    st.write(f"Matches found: {total if total is not None else len(results)}")

                else:
                    # Semantic: embed query, call RPC(s)
                    model = load_embedding_model()
                    qv = model.encode([query])[0].astype(np.float32)
                    qv /= (np.linalg.norm(qv) + 1e-12)

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
                        st.divider()

            except Exception as e:
                st.error("Search failed.")
                st.exception(e)

# ---------- Footer ----------
with st.expander("Free-tier tips"):
    st.markdown(
        "- For quick tests on long PDFs, set **Limit pages** above to 5–10.\n"
        "- English handwriting uses TrOCR; first load may take time. Only enable when needed.\n"
        "- For Arabic/Chinese, keep **Intensive** ON and try zoom 3.5–4.0 if scans are poor.\n"
        "- Supabase free DB (~500 MB) is fine for thousands of pages. Disable RLS or add anon policies."
    )
