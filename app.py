# app.py — Clean UI: Tabs = Ingest | Search | QA
# Pipeline: PDF → OCR (auto-QA) → normalize → embed → Supabase → search
# Notes:
# - No use of 'asc=' anywhere (compatible with supabase-py >=2.5).
# - QA metrics & thresholds auto-flag weak pages during ingestion.
# - Sidebar holds OCR options to keep main UI uncluttered.

import io
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output as TessOutput
from PIL import Image, ImageOps, ImageFilter

from sentence_transformers import SentenceTransformer
from supabase import create_client, ClientOptions
import supabase as _sb

# Optional: fuzzy fallback
try:
    from rapidfuzz.fuzz import partial_ratio
except Exception:
    partial_ratio = None

BUILD_ID = "tabs-clean-qa-2025-08-14"
st.set_page_config(page_title="RAG (Clean UI: Ingest | Search | QA)", layout="wide")

# ---------------- Sidebar: environment & OCR options ----------------
with st.sidebar:
    st.header("Setup & OCR Options")
    st.caption(f"Build: {BUILD_ID}")
    st.caption(f"supabase-py: {_sb.__version__}")
    try:
        langs = pytesseract.get_languages(config="")
        st.caption("Tesseract languages: " + ", ".join(sorted(langs)))
    except Exception:
        st.caption("Tesseract language list unavailable")

    # OCR options (global; apply in Ingest and QA re-OCR)
    preset = st.selectbox(
        "Language preset",
        ["English (printed)", "English (handwritten)", "Arabic (ara)", "Chinese (Simplified)", "Chinese (Traditional)"],
        index=0
    )
    default_zoom = 3.0 if preset == "English (printed)" else 3.5
    base_zoom = st.slider("Render zoom (DPI proxy)", 2.0, 4.5, default_zoom, 0.5)
    timeout_sec = st.slider("OCR timeout per page (sec)", 5, 60, 20)
    use_trocr = st.checkbox("Use TrOCR for English (handwritten)", value=False)

    # Optional vocabulary file (.txt) — helps with domain terms
    user_words_path = None
    use_vocab = st.checkbox("Use custom vocabulary (.txt)", value=False)
    if use_vocab:
        vocab_file = st.file_uploader("Upload vocabulary (TXT, one term per line)", type=["txt"], key="user_words")
        if vocab_file is not None:
            user_words_path = Path("user_words.txt").absolute()
            with open(user_words_path, "wb") as f:
                f.write(vocab_file.read())
            st.caption(f"Custom vocab saved to {user_words_path}")

# ---------------- Supabase client ----------------
def get_supabase_client():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        st.sidebar.error("❌ Add SUPABASE_URL and SUPABASE_ANON_KEY in Settings → Secrets.")
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
        st.sidebar.success(f"✅ Supabase connected. Rows: {_res.count}")
    except Exception as e:
        st.sidebar.error(f"❌ Supabase connection failed: {e}")

# ---------------- Normalization helpers ----------------
ZW_REMOVE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None)  # soft hyphen + zero-width
PUNCT_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00A0": " ", "·": " "})
_PUNCT_STRIP_RE = re.compile(r"[.,:;|/\\()\[\]{}<>•·…]+")
_SPACE_COLLAPSE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("-\n", "").replace("\n", " ")
    s = s.translate(ZW_REMOVE).translate(PUNCT_MAP)
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_STRIP_RE.sub(" ", s)
    s = _SPACE_COLLAPSE_RE.sub(" ", s).strip()
    return s.lower()

def escape_for_or_filter(s: str) -> str:
    return s.replace(",", r"\,")

# ---------------- Caches ----------------
@st.cache_resource(show_spinner=True)
def load_embedding_model():
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

# ---------------- OCR engines & QA ----------------
def render_page_to_image(page, zoom: float = 3.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def preprocess_image(img: Image.Image, do_denoise=True, do_autocontrast=True, binarize=True, thresh=175, to_gray=True):
    if to_gray:
        img = img.convert("L")
    if do_denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    if do_autocontrast:
        img = ImageOps.autocontrast(img)
    if binarize:
        img = img.point(lambda p: 255 if p > thresh else 0)
    return img

def tesseract_ocr(page, zoom: float, lang: str, psm: str, user_words_path: Optional[str], timeout_sec: int = 20) -> Tuple[str, float]:
    img = render_page_to_image(page, zoom=zoom)
    img = preprocess_image(img, True, True, True, 175, to_gray=True)
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    if user_words_path:
        config += f" --user-words {user_words_path}"
    try:
        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=TessOutput.DICT, timeout=timeout_sec)
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

def trocr_ocr(page, zoom: float) -> str:
    processor, model, device = load_trocr_handwritten()
    import torch
    img = render_page_to_image(page, zoom=zoom).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_length=512)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip()

THRESH = {
    "English (printed)":        {"min_conf": 70, "min_len": 15, "max_noisy": 0.60},
    "English (handwritten)":    {"min_conf": 60, "min_len": 10, "max_noisy": 0.70},
    "Arabic (ara)":             {"min_conf": 60, "min_len": 10, "max_noisy": 0.70},
    "Chinese (Simplified)":     {"min_conf": 60, "min_len": 10, "max_noisy": 0.70},
    "Chinese (Traditional)":    {"min_conf": 60, "min_len": 10, "max_noisy": 0.70},
}

def qc_metrics(text_norm: str) -> Dict[str, Any]:
    n = len(text_norm)
    alnum = sum(ch.isalnum() for ch in text_norm)
    non_alnum_ratio = 1.0 - (alnum / n) if n else 1.0
    return {"text_len": n, "non_alnum_ratio": non_alnum_ratio}

def make_flags(avg_conf: float, text_norm: str, preset: str) -> List[str]:
    t = THRESH.get(preset, THRESH["English (printed)"])
    m = qc_metrics(text_norm)
    flags = []
    if avg_conf < t["min_conf"]:
        flags.append("low_conf")
    if m["text_len"] < t["min_len"]:
        flags.append("very_short")
    if m["non_alnum_ratio"] > t["max_noisy"]:
        flags.append("noisy_text")
    if not text_norm:
        flags.append("empty")
    return flags

def ocr_variants_for_preset(preset: str, use_trocr: bool, base_zoom: float):
    if preset == "English (printed)":
        return [("tess-eng", {"lang": "eng", "psm": "6",  "zoom": base_zoom}),
                ("tess-eng", {"lang": "eng", "psm": "4",  "zoom": base_zoom + 0.5}),
                ("tess-eng", {"lang": "eng", "psm": "11", "zoom": base_zoom + 0.5})]
    if preset == "English (handwritten)":
        v = []
        if use_trocr:
            v.append(("trocr", {"zoom": max(3.0, base_zoom)}))
        v += [("tess-eng", {"lang": "eng", "psm": "4",  "zoom": base_zoom + 0.5}),
              ("tess-eng", {"lang": "eng", "psm": "3",  "zoom": base_zoom + 1.0})]
        return v
    if preset == "Arabic (ara)":
        return [("tess", {"lang": "ara", "psm": "4", "zoom": base_zoom}),
                ("tess", {"lang": "ara", "psm": "3", "zoom": base_zoom + 0.5})]
    if preset == "Chinese (Simplified)":
        return [("tess", {"lang": "chi_sim", "psm": "4", "zoom": base_zoom}),
                ("tess", {"lang": "chi_sim", "psm": "3", "zoom": base_zoom + 0.5})]
    if preset == "Chinese (Traditional)":
        return [("tess", {"lang": "chi_tra", "psm": "4", "zoom": base_zoom}),
                ("tess", {"lang": "chi_tra", "psm": "3", "zoom": base_zoom + 0.5})]
    return [("tess-eng", {"lang": "eng", "psm": "6", "zoom": base_zoom})]

def ocr_page_auto(page, preset: str, user_words_path: Optional[str], base_zoom: float, timeout_sec: int, use_trocr: bool, max_attempts: int = 3):
    variants = ocr_variants_for_preset(preset, use_trocr, base_zoom)[:max_attempts]
    best = None
    for idx, (engine, cfg) in enumerate(variants, 1):
        if engine == "trocr":
            try:
                text = trocr_ocr(page, zoom=cfg["zoom"])
                avg_conf = 65.0  # proxy for non-tesseract engine
                text_norm = normalize_text(text)
                flags = make_flags(avg_conf, text_norm, preset)
                cand = dict(text=text, text_norm=text_norm, avg_conf=avg_conf,
                            ocr_engine="trocr", ocr_psm=None, ocr_zoom=float(cfg["zoom"]),
                            ocr_attempts=idx, flags=flags)
            except Exception:
                continue
        else:
            text, avg_conf = tesseract_ocr(page, zoom=float(cfg["zoom"]),
                                           lang=cfg.get("lang", "eng"), psm=cfg.get("psm", "6"),
                                           user_words_path=user_words_path, timeout_sec=timeout_sec)
            text_norm = normalize_text(text)
            flags = make_flags(avg_conf, text_norm, preset)
            cand = dict(text=text, text_norm=text_norm, avg_conf=avg_conf,
                        ocr_engine="tesseract", ocr_psm=int(cfg.get("psm", "6")), ocr_zoom=float(cfg["zoom"]),
                        ocr_attempts=idx, flags=flags)
        if not cand["flags"]:
            return cand
        score = cand["avg_conf"] + min(len(cand["text_norm"]), 200) * 0.2
        if not best or score > best.get("_score", -1):
            cand["_score"] = score
            best = cand
    return best or dict(text="", text_norm="", avg_conf=0.0, ocr_engine="tesseract", ocr_psm=6, ocr_zoom=base_zoom, ocr_attempts=len(variants), flags=["empty"])

# ---------------- Extractors ----------------
def extract_text_from_pdf(file_bytes, preset: str, user_words_path: Optional[str],
                          base_zoom: float, use_trocr: bool, timeout_sec: int, max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts, total = [], len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    prog = st.progress(0, text=f"Extracting text 0/{limit} pages...")
    for i, page in enumerate(doc):
        if i >= limit:
            break
        try:
            embedded = (page.get_text() or "").strip()
            if embedded:
                parts.append(embedded)
            else:
                cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, use_trocr)
                parts.append(cand["text"])
        except Exception as e:
            parts.append(f"[Error reading page {i+1}: {e}]")
        finally:
            prog.progress((i + 1) / limit, text=f"Extracting text {i+1}/{limit} pages...")
    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(parts).strip()

def extract_pages_with_metadata(file_bytes, document_name, preset: str, user_words_path: Optional[str],
                                base_zoom: float, use_trocr: bool, timeout_sec: int, max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    rows = []
    prog = st.progress(0, text=f"OCR & split 0/{limit}...")
    for i, page in enumerate(doc):
        if i >= limit:
            break
        try:
            embedded = (page.get_text() or "").strip()
            if embedded:
                text = embedded
                text_norm = normalize_text(text)
                avg_conf = 80.0
                flags = make_flags(avg_conf, text_norm, preset)
                meta = dict(ocr_engine="embedded", ocr_psm=None, ocr_zoom=None, ocr_attempts=1)
            else:
                cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, use_trocr)
                text, text_norm, avg_conf, flags = cand["text"], cand["text_norm"], cand["avg_conf"], cand["flags"]
                meta = dict(ocr_engine=cand["ocr_engine"], ocr_psm=cand["ocr_psm"], ocr_zoom=cand["ocr_zoom"], ocr_attempts=cand["ocr_attempts"])
            m = qc_metrics(text_norm)
            rows.append({
                "document_name": document_name,
                "page_number": i + 1,
                "text": text,
                "text_norm": text_norm,
                "avg_conf": float(avg_conf),
                "text_len": int(m["text_len"]),
                "non_alnum_ratio": float(m["non_alnum_ratio"]),
                "ocr_engine": meta["ocr_engine"],
                "ocr_psm": meta["ocr_psm"],
                "ocr_zoom": meta["ocr_zoom"],
                "ocr_attempts": meta["ocr_attempts"],
                "ocr_flags": flags,
                "lang_preset": preset,
            })
        except Exception as e:
            rows.append({
                "document_name": document_name,
                "page_number": i + 1,
                "text": f"[Error: {e}]",
                "text_norm": "",
                "avg_conf": 0.0,
                "text_len": 0,
                "non_alnum_ratio": 1.0,
                "ocr_engine": "error",
                "ocr_psm": None,
                "ocr_zoom": None,
                "ocr_attempts": 1,
                "ocr_flags": ["error"],
                "lang_preset": preset,
            })
        finally:
            prog.progress((i + 1) / limit, text=f"OCR & split {i+1}/{limit}...")
    doc.close()
    return rows

# ---------------- Embeddings ----------------
def embed_texts(model, texts):
    emb = model.encode(texts, batch_size=32, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb

# ---------------- Local search helpers ----------------
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
        score = partial_ratio(qn, tn) if use_rf else (100 if qn in tn else 0)
        results.append({**r, "sim": float(score) / 100.0})
    results.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return results[:top_k]

def local_semantic_search(rows: List[Dict[str, Any]], query_vec: np.ndarray, top_k: int = 50) -> List[Dict[str, Any]]:
    embs = [np.array(r["embedding"], dtype=np.float32) for r in rows if isinstance(r.get("embedding"), list)]
    if not embs:
        return []
    M = np.vstack(embs); M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    q = query_vec.astype(np.float32); q /= (np.linalg.norm(q) + 1e-12)
    sims = M @ q
    idxs = np.argsort(-sims)[:top_k]
    kept, j = [], 0
    for r in rows:
        if not isinstance(r.get("embedding"), list):
            continue
        if j in idxs:
            kept.append({k: r[k] for k in ("document_name", "page_number", "text") if k in r} | {"sim": float(sims[j])})
        j += 1
    kept.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return kept

# ========================= UI: Tabs =========================
st.title("RAG App — Ingest | Search | QA")

tab_ingest, tab_search, tab_qa = st.tabs(["📥 Ingest", "🔎 Search", "✅ QA"])

# ---------------- Tab: Ingest ----------------
with tab_ingest:
    st.subheader("Upload & Extract")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

    colA, colB = st.columns([1,1])
    with colA:
        max_pages_debug = st.number_input("Limit pages (0 = all)", min_value=0, value=0, step=1)
    with colB:
        delete_first = st.checkbox("Delete existing rows for this document before upload", False)

    if uploaded_file and st.button("1) Process PDF (OCR + QA)"):
        with st.spinner("Processing..."):
            try:
                file_bytes = uploaded_file.read()
                st.session_state["last_pdf_bytes"] = file_bytes
                st.session_state["last_pdf_name"] = uploaded_file.name

                full_text = extract_text_from_pdf(file_bytes, preset, user_words_path, base_zoom, use_trocr, timeout_sec, max_pages_debug)
                page_chunks = extract_pages_with_metadata(file_bytes, uploaded_file.name, preset, user_words_path, base_zoom, use_trocr, timeout_sec, max_pages_debug)
                st.session_state["pages"] = page_chunks
                st.session_state["document_name"] = uploaded_file.name

                st.success("Extraction complete. Page-level chunks ready.")
                st.caption("Preview (first 5 pages):")
                for rec in page_chunks[:5]:
                    st.markdown(f"**{rec['document_name']} — Page {rec['page_number']}**")
                    preview = (rec["text"] or "").replace("\n", " ")
                    st.write((preview[:500] + ("..." if len(preview) > 500 else "")) or "_(empty page)_")
                    st.caption(f"flags={rec['ocr_flags']}, conf={rec['avg_conf']:.1f}, len={rec['text_len']}, noisy={rec['non_alnum_ratio']:.2f}")
                    st.divider()

                st.text_area("Full Extracted Text (debug)", full_text or "", height=250)
            except Exception as e:
                st.error("PDF processing failed.")
                st.exception(e)

    st.subheader("Embed & Upload to Supabase")
    if not _supa:
        st.warning("Supabase is not configured.")
    elif "pages" not in st.session_state or not st.session_state["pages"]:
        st.info("No pages detected yet. Process a PDF first.")
    else:
        if st.button("2) Generate embeddings & upload", type="primary"):
            try:
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
                        "text_norm": p["text_norm"],
                        "avg_conf": p["avg_conf"],
                        "text_len": p["text_len"],
                        "non_alnum_ratio": p["non_alnum_ratio"],
                        "ocr_engine": p["ocr_engine"],
                        "ocr_psm": p["ocr_psm"],
                        "ocr_zoom": p["ocr_zoom"],
                        "ocr_attempts": p["ocr_attempts"],
                        "ocr_flags": p["ocr_flags"],
                        "lang_preset": p["lang_preset"],
                        "embedding": vec.tolist(),
                    })

                if delete_first:
                    try:
                        _ = _supa.table("document_chunks").delete().eq("document_name", st.session_state.get("document_name", "")).execute()
                        st.info("Old rows deleted.")
                    except Exception as e:
                        st.warning(f"Delete failed: {e}")

                st.info("Uploading to Supabase…")
                BATCH = 100
                inserted = 0
                for i in range(0, len(rows), BATCH):
                    batch = rows[i : i + BATCH]
                    res = _supa.table("document_chunks").insert(batch).execute()
                    if getattr(res, "data", None) is None:
                        st.error("Insert returned no data. Check RLS and table schema.")
                        st.write(res)
                        st.stop()
                    inserted += len(res.data)
                    st.write(f"Inserted rows {i+1}–{i+len(batch)} (total {inserted})")
                    time.sleep(0.05)

                st.success(f"All chunks uploaded. Total inserted: {inserted}")
            except Exception as e:
                st.error("Embedding or upload failed.")
                st.exception(e)

# ---------------- Tab: Search ----------------
with tab_search:
    st.subheader("Search (Keyword / Fuzzy / Semantic)")
    if not _supa:
        st.info("Configure Supabase first to enable search.")
    else:
        try:
            docs = _supa.table("document_chunks").select("document_name").execute()
            doc_names = sorted({r["document_name"] for r in (docs.data or []) if r.get("document_name")})
        except Exception:
            doc_names = []

        colf1, colf2, colf3 = st.columns([2,1,1])
        with colf1:
            query = st.text_input("Query (e.g., Gen. Transporting)")
        with colf2:
            selected_doc = st.selectbox("Filter by doc (optional)", ["All"] + doc_names)
        with colf3:
            only_low_quality = st.checkbox("Only low-quality pages", value=False)

        mode = st.radio("Type", ["Keyword (exact/normalized)", "Keyword (fuzzy)", "Semantic"], index=0, horizontal=True)
        top_k = st.slider("Results to show", 5, 200, 50)
        fetch_all = st.checkbox("Return all matches (only for exact/normalized)")
        max_scan = st.number_input("Max rows for local fallback", 100, 20000, 5000, step=100)

        if st.button("Search", type="primary"):
            if not query:
                st.warning("Please enter a query.")
            else:
                try:
                    results: List[Dict[str, Any]] = []
                    base = _supa.table("document_chunks").select("document_name,page_number,text,avg_conf,text_len", count="exact")
                    if selected_doc != "All":
                        base = base.eq("document_name", selected_doc)
                    if only_low_quality:
                        base = base.or_("avg_conf.lt.65,text_len.lte.10")

                    if mode.startswith("Keyword (exact/normalized)"):
                        norm_query = normalize_text(query)
                        oq = escape_for_or_filter(query)
                        onq = escape_for_or_filter(norm_query)
                        q = base.or_(f"text.ilike.%{oq}%,text_norm.ilike.%{onq}%")
                        if fetch_all:
                            page_size = 100
                            first = q.order("document_name").order("page_number").range(0, page_size - 1).execute()
                            total = getattr(first, "count", None)
                            results = list(first.data or [])
                            offset = page_size
                            while total is not None and offset < total:
                                chunk = q.order("document_name").order("page_number").range(offset, min(offset + page_size - 1, total - 1)).execute()
                                data = chunk.data or []
                                if not data:
                                    break
                                results.extend(data)
                                offset += len(data)
                        else:
                            res = q.order("document_name").order("page_number").limit(top_k).execute()
                            results = getattr(res, "data", []) or []

                    elif mode.startswith("Keyword (fuzzy)"):
                        norm_query = normalize_text(query)
                        rpc_ok = True
                        try:
                            if selected_doc == "All":
                                res = _supa.rpc("fuzzy_find_chunks_all", {"qnorm": norm_query, "limit_n": top_k}).execute()
                            else:
                                res = _supa.rpc("fuzzy_find_chunks_in_doc", {"docname": selected_doc, "qnorm": norm_query, "limit_n": top_k}).execute()
                            results = getattr(res, "data", []) or []
                            results.sort(key=lambda r: (-r.get("sim", 0.0), r.get("document_name", ""), r.get("page_number") or 0))
                        except Exception:
                            rpc_ok = False

                        if not rpc_ok:
                            # local fallback
                            fetched = []
                            offset = 0
                            page_size = 1000
                            while offset < max_scan:
                                q = _supa.table("document_chunks").select("document_name,page_number,text")
                                if selected_doc != "All":
                                    q = q.eq("document_name", selected_doc)
                                if only_low_quality:
                                    q = q.or_("avg_conf.lt.65,text_len.lte.10")
                                q = q.order("document_name").order("page_number").range(offset, offset + page_size - 1)
                                chunk = q.execute()
                                data = chunk.data or []
                                if not data:
                                    break
                                fetched.extend(data)
                                offset += len(data)
                                if len(fetched) >= max_scan:
                                    break
                            results = local_fuzzy_search(fetched, query, top_k=top_k)

                    else:
                        # Semantic
                        model = load_embedding_model()
                        qv = model.encode([query])[0].astype(np.float32)
                        qv /= (np.linalg.norm(qv) + 1e-12)
                        rpc_ok = True
                        try:
                            if selected_doc == "All":
                                res = _supa.rpc("find_similar_chunks", {"query_embedding": qv.tolist(), "match_count": top_k}).execute()
                            else:
                                res = _supa.rpc("find_similar_chunks_in_doc", {"doc_name": selected_doc, "query_embedding": qv.tolist(), "match_count": top_k}).execute()
                            results = getattr(res, "data", []) or []
                            results.sort(key=lambda r: (r.get("document_name", ""), r.get("page_number") or 0))
                        except Exception:
                            rpc_ok = False

                        if not rpc_ok:
                            fetched = []
                            offset = 0
                            page_size = 500
                            while offset < max_scan:
                                q = _supa.table("document_chunks").select("document_name,page_number,text,embedding,avg_conf,text_len")
                                if selected_doc != "All":
                                    q = q.eq("document_name", selected_doc)
                                if only_low_quality:
                                    q = q.or_("avg_conf.lt.65,text_len.lte.10")
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
                            if "avg_conf" in r and "text_len" in r:
                                st.caption(f"conf={r['avg_conf']:.1f}, len={r['text_len']}")
                            st.divider()
                except Exception as e:
                    st.error("Search failed.")
                    st.exception(e)

# ---------------- Tab: QA ----------------
with tab_qa:
    st.subheader("QA Dashboard")

    if not _supa:
        st.info("Configure Supabase first.")
    else:
        # Top line controls
        try:
            docs = _supa.table("document_chunks").select("document_name").execute()
            qa_doc_names = sorted({r["document_name"] for r in (docs.data or []) if r.get("document_name")})
        except Exception:
            qa_doc_names = []

        colq1, colq2, colq3 = st.columns([2,1,1])
        with colq1:
            sel_doc = st.selectbox("Filter by doc (optional)", ["All"] + qa_doc_names, key="qa_doc")
        with colq2:
            min_conf = st.slider("Min conf", 0, 100, 65, key="qa_conf")
        with colq3:
            list_btn = st.button("List flagged pages", key="qa_list")

        if list_btn:
            try:
                q = _supa.table("document_chunks").select(
                    "document_name,page_number,avg_conf,text_len,non_alnum_ratio,ocr_engine,ocr_psm,ocr_zoom,ocr_attempts"
                )
                if sel_doc != "All":
                    q = q.eq("document_name", sel_doc)
                # Treat as flagged if it fails conf/length thresholds
                q = q.or_(f"avg_conf.lt.{min_conf},text_len.lte.10")
                q = q.order("document_name").order("page_number").limit(5000)
                res = q.execute()
                rows = res.data or []
                if not rows:
                    st.success("No flagged pages 🎉")
                else:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", data=csv, file_name="ocr_flagged_pages.csv", mime="text/csv")
            except Exception as e:
                st.error("QA query failed.")
                st.exception(e)

        st.markdown("---")
        st.subheader("Batch re-OCR the last uploaded PDF’s flagged pages")
        st.caption("Upload the PDF in **Ingest → Process PDF** so the app holds its bytes, then run this.")
        if st.button("Re-OCR flagged pages now", key="qa_reocr"):
            try:
                docname = st.session_state.get("last_pdf_name")
                file_bytes = st.session_state.get("last_pdf_bytes")
                if not docname or not file_bytes:
                    st.warning("No PDF in memory. Go to Ingest, process the PDF, then return.")
                else:
                    # Get flagged pages for that doc
                    q = (_supa.table("document_chunks")
                         .select("id,page_number")
                         .eq("document_name", docname)
                         .or_("avg_conf.lt.65,text_len.lte.10")
                         .order("page_number")
                         .limit(5000))
                    flagged = q.execute().data or []
                    if not flagged:
                        st.info("No flagged pages for this document.")
                    else:
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        model = load_embedding_model()
                        updated = 0
                        prog = st.progress(0, text="Re-OCR 0 pages")
                        for i, row in enumerate(flagged, 1):
                            pno = row["page_number"]
                            if 1 <= pno <= len(doc):
                                page = doc[pno - 1]
                                cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, use_trocr, max_attempts=4)
                                vec = model.encode([cand["text"]])[0].astype(np.float32)
                                vec /= (np.linalg.norm(vec) + 1e-12)
                                m = qc_metrics(cand["text_norm"])
                                _supa.table("document_chunks").update({
                                    "text": cand["text"],
                                    "text_norm": cand["text_norm"],
                                    "avg_conf": float(cand["avg_conf"]),
                                    "text_len": int(m["text_len"]),
                                    "non_alnum_ratio": float(m["non_alnum_ratio"]),
                                    "ocr_engine": cand["ocr_engine"],
                                    "ocr_psm": cand["ocr_psm"],
                                    "ocr_zoom": cand["ocr_zoom"],
                                    "ocr_attempts": cand["ocr_attempts"],
                                    "embedding": vec.tolist(),
                                }).eq("id", row["id"]).execute()
                                updated += 1
                            prog.progress(i/len(flagged), text=f"Re-OCR {i}/{len(flagged)} pages")
                        doc.close()
                        st.success(f"Re-OCR complete. Updated {updated} pages.")
            except Exception as e:
                st.error("Batch re-OCR failed.")
                st.exception(e)

        st.markdown("---")
        st.subheader("Single-page Inspector (manual override)")
        try:
            _docs2 = _supa.table("document_chunks").select("document_name").execute()
            insp_docs = sorted({r["document_name"] for r in (_docs2.data or []) if r.get("document_name")})
        except Exception:
            insp_docs = []
        colI1, colI2, colI3 = st.columns([2,1,1])
        with colI1:
            doc_to_fix = st.selectbox("Document", insp_docs, key="ins_doc")
        with colI2:
            page_to_fix = st.number_input("Page #", min_value=1, value=3, step=1, key="ins_page")
        with colI3:
            keyword_test = st.text_input("Keyword (optional)", value="Gen. Transporting", key="ins_kw")

        cA, cB = st.columns(2)
        with cA:
            if st.button("Fetch row", key="btn_fetch_row"):
                try:
                    r = (_supa.table("document_chunks")
                         .select("id,document_name,page_number,text,avg_conf,text_len")
                         .eq("document_name", doc_to_fix)
                         .eq("page_number", page_to_fix)
                         .limit(1).execute())
                    row = (r.data or [None])[0]
                    if not row:
                        st.error("No row found.")
                    else:
                        st.session_state["_inspect_row"] = row
                        raw = row.get("text") or ""
                        norm = normalize_text(raw)
                        st.success(f"Row loaded. conf={row.get('avg_conf')}, len={row.get('text_len')}")
                        st.write((raw[:900] + ("..." if len(raw) > 900 else "")) or "_(empty)_")
                        if keyword_test:
                            qn = normalize_text(keyword_test)
                            st.caption(f"Contains(raw)={keyword_test.lower() in raw.lower()} | Contains(norm)={qn in norm}")
                except Exception as e:
                    st.error("Fetch failed.")
                    st.exception(e)

        with cB:
            if st.button("Try re-OCR variants", key="btn_reocr"):
                ok = (st.session_state.get("last_pdf_bytes") is not None and st.session_state.get("last_pdf_name") == doc_to_fix)
                if not ok:
                    st.warning("Upload & Process the same PDF in Ingest first (so its bytes are in memory).")
                else:
                    try:
                        pdf_bytes = st.session_state["last_pdf_bytes"]
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        if page_to_fix < 1 or page_to_fix > len(doc):
                            st.error(f"PDF has {len(doc)} pages — page {page_to_fix} out of range.")
                        else:
                            page = doc[page_to_fix - 1]
                            variants = [
                                ("eng", "6", 3.0),
                                ("eng", "7", 3.5),
                                ("eng", "11", 3.5),
                                ("eng", "4", 4.0),
                            ]
                            tried = []
                            for lang, psm, zoom in variants:
                                txt, conf = tesseract_ocr(page, zoom=float(zoom), lang=lang, psm=psm,
                                                          user_words_path=user_words_path, timeout_sec=25)
                                tried.append({"lang": lang, "psm": psm, "zoom": zoom, "conf": conf,
                                              "text": txt, "text_norm": normalize_text(txt), "len": len(normalize_text(txt))})
                            doc.close()
                            tried.sort(key=lambda d: (-d["len"], -d["conf"]))
                            for i, cand in enumerate(tried[:3], 1):
                                st.markdown(f"**#{i} lang={cand['lang']} psm={cand['psm']} zoom={cand['zoom']} (conf={cand['conf']:.1f}, len={cand['len']})**")
                                st.write(cand["text_norm"][:600] + ("..." if len(cand["text_norm"]) > 600 else ""))
                            st.session_state["_re_ocr_candidates"] = tried
                    except Exception as e:
                        st.error("Re-OCR failed.")
                        st.exception(e)

        row = st.session_state.get("_inspect_row")
        cands = st.session_state.get("_re_ocr_candidates", [])
        if row and cands:
            labels = [f"#{i+1}: {d['lang']} psm={d['psm']} zoom={d['zoom']} (conf={d['conf']:.1f}, len={d['len']})" for i, d in enumerate(cands[:5])]
            pick = st.selectbox("Pick candidate to save", labels, key="pick_cand")
            idx = labels.index(pick); chosen = cands[idx]
            if st.button("✅ Update this page", key="btn_update_page"):
                try:
                    model = load_embedding_model()
                    vec = model.encode([chosen["text"]])[0].astype(np.float32)
                    vec /= (np.linalg.norm(vec) + 1e-12)
                    m = qc_metrics(chosen["text_norm"])
                    _supa.table("document_chunks").update({
                        "text": chosen["text"],
                        "text_norm": chosen["text_norm"],
                        "avg_conf": float(chosen["conf"]),
                        "text_len": int(m["text_len"]),
                        "non_alnum_ratio": float(m["non_alnum_ratio"]),
                        "ocr_engine": "tesseract",
                        "ocr_psm": int(chosen["psm"]),
                        "ocr_zoom": float(chosen["zoom"]),
                        "ocr_attempts": 1,
                        "embedding": vec.tolist(),
                    }).eq("id", row["id"]).execute()
                    st.success("Updated. Re-run search if needed.")
                except Exception as e:
                    st.error("Update failed.")
                    st.exception(e)
