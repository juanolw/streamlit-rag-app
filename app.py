# app.py — Simple, auto-detect OCR + Supabase indexing + search
# - Per page, try native PDF text; if not usable, OCR automatically.
# - OCR auto-detects script (Latin/Arabic/Chinese) via Tesseract OSD.
# - No handwriting toggles; robust defaults.
# - Embeddings via fastembed (384-dim, small + fast).
# - Vector table assumed: document_chunks(document_name text, page_number int, text text, text_norm text, embedding vector(384))

import io
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output as TessOutput
from PIL import Image, ImageOps, ImageFilter

from supabase import create_client
import supabase as _sb

from fastembed import TextEmbedding
from rapidfuzz.fuzz import partial_ratio

# --------------------------- Config ---------------------------
APP_BUILD = "auto-ocr-slim-2025-08-15"

st.set_page_config(page_title="RAG (Auto OCR + Supabase)", layout="wide")

# --------------------------- Sidebar: env checks ---------------------------
with st.sidebar:
    st.subheader("Environment")
    st.caption(f"Build: {APP_BUILD}")
    st.caption(f"supabase-py: {_sb.__version__}")

    if "SUPABASE_URL" in st.secrets:
        st.caption("SUPABASE_URL ✓")
    else:
        st.error("Missing SUPABASE_URL secret")

    if "SUPABASE_ANON_KEY" in st.secrets:
        st.caption("SUPABASE_ANON_KEY ✓")
    else:
        st.error("Missing SUPABASE_ANON_KEY secret")

    try:
        langs = pytesseract.get_languages(config="")
        st.caption("Tesseract languages: " + ", ".join(sorted(langs)))
    except Exception:
        st.caption("Tesseract language list unavailable")

# --------------------------- Supabase client ---------------------------
def get_supabase():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.sidebar.error(f"Supabase init failed: {e}")
        return None

supa = get_supabase()
if supa:
    try:
        res = supa.table("document_chunks").select("id", count="exact").limit(1).execute()
        st.sidebar.success(f"Supabase: document_chunks rows={res.count}")
    except Exception as e:
        st.sidebar.error(f"Supabase check failed: {e}")

# --------------------------- Utilities ---------------------------
ZW_REMOVE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None)
PUNCT_MAP = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00A0": " ", "·": " "
})
_PUNCT_STRIP_RE = re.compile(r"[.,:;|/\\()\[\]{}<>•·…]+")
_SPACE_COLLAPSE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("-\n", "")
    s = s.replace("\n", " ")
    s = s.translate(ZW_REMOVE).translate(PUNCT_MAP)
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_STRIP_RE.sub(" ", s)
    s = _SPACE_COLLAPSE_RE.sub(" ", s).strip()
    return s.lower()

def render_page_image(pdf_page, zoom: float = 3.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # robust defaults: grayscale -> median denoise -> autocontrast -> light binarize
    img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img)
    img = img.point(lambda p: 255 if p > 180 else 0)
    return img

def try_native_text(page: fitz.Page) -> Tuple[str, int]:
    """
    Try to extract digital text. If it's long enough & alphanumeric-rich, accept it.
    Returns (text, quality_score 0..100)
    """
    text = (page.get_text() or "").strip()
    if not text:
        return "", 0
    # Heuristic: require at least 60 characters and at least 30% letters/digits
    letters = sum(ch.isalnum() for ch in text)
    score = int(min(100, (len(text) / 2) + (letters * 0.5)))
    if len(text) >= 60 and letters / max(1, len(text)) >= 0.3:
        return text, min(100, score)
    return "", 0

def detect_script_lang(img: Image.Image) -> str:
    """
    Use Tesseract OSD to guess script, then map to language code.
    """
    try:
        osd = pytesseract.image_to_osd(img)
        # Look for a line like: "Script: Latin"
        m = re.search(r"Script:\s*([A-Za-z0-9_]+)", osd)
        script = (m.group(1).lower() if m else "")
    except Exception:
        script = ""

    if "arab" in script:
        return "ara"
    if "han" in script or "chinese" in script or "hang" in script:
        # OSD sometimes returns "Han", "HanS" or "HanT"
        if "t" in script:
            return "chi_tra"
        return "chi_sim"
    # default to English
    return "eng"

def tesseract_ocr(img: Image.Image, lang: str, psm: str) -> Tuple[str, float]:
    cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=TessOutput.DICT, timeout=25)
        words = [w for w in data.get("text", []) if isinstance(w, str) and w.strip()]
        confs = [int(c) for c in data.get("conf", []) if c not in (None, "", "-1")]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        text = " ".join(words) if words else ""
    except Exception:
        try:
            text = pytesseract.image_to_string(img, lang=lang, config=cfg, timeout=25)
        except Exception:
            text = ""
        return (text or "").strip(), 0.0
    return (text or "").strip(), avg_conf

def auto_ocr_page(page: fitz.Page) -> Tuple[str, Dict[str, Any]]:
    """
    End-to-end per-page:
    1) try native text
    2) if not enough, render → preprocess → OSD → OCR (smart fallback)
    """
    # 1) native
    text, q = try_native_text(page)
    if text:
        return text, {"source": "digital", "lang": "n/a", "conf": 100.0}

    # 2) OCR path
    img = render_page_image(page, zoom=3.0)
    img = preprocess_for_ocr(img)

    lang = detect_script_lang(img)
    # First pass: assume a block of text (psm 6)
    txt, conf = tesseract_ocr(img, lang=lang, psm="6")

    # If confidence is poor, try alternative PSMS that help tables/forms or sparse notes
    if conf < 60:
        txt2, conf2 = tesseract_ocr(img, lang=lang, psm="4")     # block, columns
        if conf2 > conf:
            txt, conf = txt2, conf2
    if conf < 55:
        txt3, conf3 = tesseract_ocr(img, lang=lang, psm="11")    # sparse text
        if conf3 > conf:
            txt, conf = txt3, conf3

    return (txt or "").strip(), {"source": "ocr", "lang": lang, "conf": float(conf)}

# --------------------------- Embeddings (fast + light) ---------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    # 384-dim, good for retrieval; no torch
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    vecs = list(embedder.embed(texts, batch_size=64))
    arr = np.asarray(vecs, dtype=np.float32)
    # normalize for cosine
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr

# --------------------------- UI: Tabs ---------------------------
st.markdown(
    """
    <style>
    .big-tabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .big-tabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 12px;
        background: #f2f2f7;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
tabs = st.tabs(["Ingest", "Search", "QA"])

# ========================= Ingest =========================
with tabs[0]:
    st.subheader("Ingest PDFs (auto-detect OCR)")

    files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    do_index = st.checkbox("Index into Supabase after processing", value=True)
    start = st.button("Process PDFs", type="primary")

    if start and files:
        report_rows: List[Dict[str, Any]] = []
        all_rows_for_db: List[Dict[str, Any]] = []

        for f_idx, f in enumerate(files, 1):
            st.write(f"**{f_idx}. {f.name}**")
            pdf_bytes = f.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pbar = st.progress(0.0, text=f"Processing {f.name} (0/{len(doc)} pages)")

            for i, page in enumerate(doc, 1):
                try:
                    text, info = auto_ocr_page(page)
                except Exception as e:
                    text, info = f"[Error OCR page {i}: {e}]", {"source": "error", "lang": "n/a", "conf": 0.0}

                # Keep a compact report row
                report_rows.append({
                    "document": f.name,
                    "page": i,
                    "source": info.get("source"),
                    "lang": info.get("lang"),
                    "ocr_conf": round(info.get("conf", 0.0), 1),
                    "chars": len(text),
                    "warning": "LOW_CONF" if info.get("source") == "ocr" and info.get("conf", 0.0) < 55 else "",
                })

                # Prepare DB row (no schema changes required)
                all_rows_for_db.append({
                    "document_name": f.name,
                    "page_number": i,
                    "text": text,
                    "text_norm": normalize_text(text),
                })

                pbar.progress(i / len(doc), text=f"Processing {f.name} ({i}/{len(doc)} pages)")

            doc.close()
            st.success(f"Completed: {f.name}")

        # Show a small summary table (first 200 rows for readability)
        st.write("Summary (first 200 rows):")
        st.dataframe(report_rows[:200], use_container_width=True)

        # Downloadable full report
        import pandas as pd
        df = pd.DataFrame(report_rows)
        st.download_button(
            "Download OCR Quality Report (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ocr_report.csv",
            mime="text/csv",
        )

        # Optional: index into Supabase
        if do_index:
            if not supa:
                st.error("Supabase not configured (add secrets). Skipping index.")
            else:
                try:
                    # (Optional) delete prior rows for any of these docs to avoid duplicates
                    docs = sorted({r["document_name"] for r in all_rows_for_db})
                    for dname in docs:
                        supa.table("document_chunks").delete().eq("document_name", dname).execute()

                    # Generate embeddings in batches, upload in batches
                    texts = [r["text"] for r in all_rows_for_db]
                    st.info("Generating embeddings…")
                    vecs = embed_texts(texts)  # (n, 384)

                    BATCH = 100
                    to_insert = []
                    for r, v in zip(all_rows_for_db, vecs):
                        to_insert.append({
                            "document_name": r["document_name"],
                            "page_number": r["page_number"],
                            "text": r["text"],
                            "text_norm": r["text_norm"],
                            "embedding": v.tolist(),
                        })
                    st.info("Uploading to Supabase…")
                    inserted = 0
                    for i in range(0, len(to_insert), BATCH):
                        chunk = to_insert[i:i+BATCH]
                        res = supa.table("document_chunks").insert(chunk).execute()
                        if getattr(res, "data", None) is None:
                            st.error("Insert returned no data (check RLS / table schema).")
                            st.stop()
                        inserted += len(res.data)
                        st.write(f"Inserted {inserted}/{len(to_insert)}")
                        time.sleep(0.02)
                    st.success(f"Index complete. Total rows inserted: {inserted}")
                except Exception as e:
                    st.error("Indexing failed.")
                    st.exception(e)

# ========================= Search =========================
with tabs[1]:
    st.subheader("Search")
    if not supa:
        st.info("Add Supabase secrets to enable search.")
    else:
        # Optional document filter
        try:
            docs = supa.table("document_chunks").select("document_name").execute()
            doc_names = sorted({r["document_name"] for r in (docs.data or []) if r.get("document_name")})
        except Exception:
            doc_names = []
        doc_filter = st.selectbox("Filter by document (optional)", ["All"] + doc_names)

        q = st.text_input("Query (e.g., Gen. Transporting)")
        mode = st.radio("Mode", ["Keyword", "Fuzzy", "Semantic"], horizontal=True)
        top_k = st.slider("Results", 5, 100, 30)

        if st.button("Run Search", type="primary"):
            if not q:
                st.warning("Enter a query.")
            else:
                try:
                    results: List[Dict[str, Any]] = []

                    if mode == "Keyword":
                        norm = normalize_text(q)
                        oq = q.replace(",", r"\,")
                        on = norm.replace(",", r"\,")
                        base = supa.table("document_chunks").select("document_name,page_number,text", count="exact")
                        base = base.or_(f"text.ilike.%{oq}%,text_norm.ilike.%{on}%")
                        if doc_filter != "All":
                            base = base.eq("document_name", doc_filter)
                        res = base.order("document_name").order("page_number").limit(top_k).execute()
                        results = getattr(res, "data", []) or []

                    elif mode == "Fuzzy":
                        # Fetch candidates (limit for speed)
                        page_size = 2000
                        fetched: List[Dict[str, Any]] = []
                        offset = 0
                        while offset < page_size:
                            qy = supa.table("document_chunks").select("document_name,page_number,text")
                            if doc_filter != "All":
                                qy = qy.eq("document_name", doc_filter)
                            qy = qy.order("document_name").order("page_number").range(offset, offset + 999)
                            chunk = qy.execute()
                            data = chunk.data or []
                            if not data:
                                break
                            fetched.extend(data)
                            if len(data) < 1000:
                                break
                            offset += 1000

                        qn = normalize_text(q)
                        scored = []
                        for r in fetched:
                            tn = normalize_text(r.get("text", ""))
                            if not tn:
                                continue
                            score = partial_ratio(qn, tn) / 100.0
                            if score > 0.6:
                                scored.append({**r, "sim": score})
                        scored.sort(key=lambda x: (-x["sim"], x["document_name"], x.get("page_number", 0)))
                        results = scored[:top_k]

                    else:  # Semantic
                        # Build query vector
                        emb = embed_texts([q])[0].astype(np.float32)
                        base = supa.rpc(
                            "find_similar_chunks",
                            {"query_embedding": emb.tolist(), "match_count": top_k}
                        )
                        if doc_filter != "All":
                            # If you also created a per-document RPC, call it here; otherwise filter client-side
                            res = supa.table("document_chunks").select("document_name,page_number,text,embedding").eq(
                                "document_name", doc_filter
                            ).limit(5000).execute()
                            data = res.data or []
                            from math import inf
                            M = []
                            kept = []
                            for r in data:
                                vec = r.get("embedding")
                                if not isinstance(vec, list):
                                    continue
                                v = np.asarray(vec, dtype=np.float32)
                                v /= (np.linalg.norm(v) + 1e-12)
                                M.append(v)
                                kept.append(r)
                            if M:
                                M = np.vstack(M)
                                scores = M @ emb
                                order = np.argsort(-scores)[:top_k]
                                results = []
                                for j in order:
                                    rr = kept[int(j)]
                                    results.append({
                                        "document_name": rr["document_name"],
                                        "page_number": rr["page_number"],
                                        "text": rr["text"],
                                        "sim": float(scores[int(j)])
                                    })
                            else:
                                results = []
                        else:
                            res = base.execute()
                            results = getattr(res, "data", []) or []

                    if not results:
                        st.info("No results.")
                    else:
                        for i, r in enumerate(results, 1):
                            doc = r.get("document_name", "Unknown")
                            page = r.get("page_number", "?")
                            text = (r.get("text") or "").replace("\n", " ")
                            snippet = text[:500] + ("..." if len(text) > 500 else "")
                            st.markdown(f"**{i}. {doc} — Page {page}**")
                            st.write(snippet or "_(empty)_")
                            if "sim" in r:
                                st.caption(f"score: {r['sim']:.3f}")
                            st.divider()
                except Exception as e:
                    st.error("Search failed.")
                    st.exception(e)

# ========================= QA (how to use) =========================
with tabs[2]:
    st.subheader("QA (How to use)")
    st.markdown(
        """
        **What this does now:**  
        - Retrieves the most relevant pages with *Keyword*, *Fuzzy*, or *Semantic* search.  
        - Presents the text snippets and exact page refs so you can quote or copy/paste.

        **If you’d like LLM answers later:**  
        - We can plug in a low-cost API (or local model) that reads the top N retrieved pages and drafts an
          answer with citations. For now this tab explains the workflow:
        
        1. Use **Search** to find pages.  
        2. Open your source PDF and verify the paragraph/page.  
        3. Draft your findings & paste the citations into your report.
        """
    )
