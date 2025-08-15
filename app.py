# app.py — RAG OCR/Search/QA with STRICT OCR mode (no guessing)
# - Printed English (strict): Tesseract with dicts OFF + optional allowlist
# - Handwritten English (strict): EasyOCR (non-generative), avoids hallucinations
# - Non-strict keeps your previous behavior (Tesseract + optional TrOCR/booster)
# - Search & QA unchanged; no .order(..., asc=) usage

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
import altair as alt

# Optional fuzzy
try:
    from rapidfuzz.fuzz import partial_ratio
except Exception:
    partial_ratio = None

# Optional EasyOCR for strict handwriting
EASYOCR_AVAILABLE = True
try:
    import easyocr
except Exception:
    EASYOCR_AVAILABLE = False

BUILD_ID = "strict-ocr-no-guessing-2025-08-15"

st.set_page_config(
    page_title="RAG • OCR | Search | QA",
    page_icon="📚",
    layout="wide",
    menu_items={"About": "RAG for Construction Claims — STRICT OCR (no guessing)"}
)

# ---------------- Global CSS (single sleek tab bar) ----------------
st.markdown(
    """
<style>
:root{
  --bg: var(--background-color);
  --bg2: var(--secondary-background-color);
  --border: rgba(49,51,63,0.14);
  --glow: 0 6px 24px rgba(0,0,0,0.06);
}
.main .block-container { padding-top: 0.8rem; padding-bottom: 4.2rem; }
.stTabs{
  position: sticky; top: 6px; z-index: 50; padding:6px 8px; margin:4px 0 14px 0;
  border:1px solid var(--border); border-radius:16px; background:color-mix(in srgb, var(--bg) 85%, transparent);
  backdrop-filter: blur(10px); box-shadow: var(--glow);
}
.stTabs [role="tablist"]{ gap:10px; }
.stTabs [role="tab"]{
  background: var(--bg2); padding: 12px 18px; border-radius: 999px; font-weight:700; font-size:1rem;
  border:1px solid var(--border); transition: transform .15s ease;
}
.stTabs [role="tab"]:hover{ transform: translateY(-1px); border-color: rgba(49,51,63,0.28); }
.stTabs [role="tab"][aria-selected="true"]{
  color: white; background: linear-gradient(135deg, #111827, #1f2937); border-color: transparent;
  box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.stButton > button, .stDownloadButton > button{
  border-radius:12px; font-weight:700; padding:.5rem .9rem; border:1px solid var(--border);
}
.stTextInput input, .stNumberInput input,
div[data-baseweb="select"] > div, .stFileUploader, .stTextArea textarea{
  border-radius:12px !important; border:1px solid var(--border);
}
.card{ border-radius:16px; padding:14px 16px; background:var(--bg2); border:1px solid var(--border); box-shadow: var(--glow); }
.kpi-title{ font-size:.85rem; color:rgba(49,51,63,.75); margin-bottom:6px; }
.kpi-value{ font-size:1.6rem; font-weight:800; letter-spacing:-.02em; }
.badge{ display:inline-block; padding:2px 10px; border-radius:999px; font-weight:700; font-size:.75rem; }
.badge--ok{ background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; }
.badge--warn{ background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
.badge--err{ background:#fef2f2; color:#991b1b; border:1px solid #fecaca; }
.footer{
  position: fixed; left: 18px; right: 18px; bottom: 12px; z-index: 40; display:flex; align-items:center; justify-content: space-between; gap:12px;
  padding: 10px 14px; border-radius: 12px; background: color-mix(in srgb, var(--bg) 70%, transparent);
  border: 1px solid var(--border); backdrop-filter: blur(8px); font-size:.85rem; box-shadow: var(--glow);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Sidebar: setup & OCR options ----------------
with st.sidebar:
    st.header("Setup")
    st.caption(f"Build **{BUILD_ID}** · supabase-py `{_sb.__version__}`")
    try:
        langs = pytesseract.get_languages(config="")
        st.caption("Tesseract languages: " + ", ".join(sorted(langs)))
    except Exception:
        st.caption("Tesseract language list unavailable")

    st.markdown("---")
    st.header("OCR Options")

    preset = st.selectbox(
        "Language preset",
        ["English (printed)", "English (handwritten)", "Arabic (ara)", "Chinese (Simplified)", "Chinese (Traditional)"],
        index=0
    )

    # STRICT OCR switch
    strict_ocr = st.toggle(
        "Strict OCR (no guessing, no generative)", value=True,
        help="Disables Tesseract dictionaries and TrOCR. For handwriting, uses EasyOCR (non-generative)."
    )

    default_zoom = 3.0 if preset == "English (printed)" else 3.5
    base_zoom = st.slider("Render zoom (DPI proxy)", 2.0, 4.5, default_zoom, 0.5)
    timeout_sec = st.slider("OCR timeout per page (sec)", 5, 60, 20)

    # TrOCR & booster only if NOT strict
    if not strict_ocr:
        use_trocr = st.checkbox("Use TrOCR for English (handwritten)", value=False)
        handwriting_booster = st.checkbox(
            "Handwriting booster (line-by-line TrOCR fallback)",
            value=(preset == "English (handwritten)"),
            help="Uses TrOCR on low-confidence lines; may 'clean up' noisy cursive."
        )
        booster_trigger_conf = st.slider("Booster trigger: line conf <", 0, 100, 65)
    else:
        use_trocr = False
        handwriting_booster = False
        booster_trigger_conf = 65

    # Character allowlist (helps keep punctuation like 'Gen.')
    default_allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:;!?'\"-/()&%#@[]{}+*=°$|\\"
    use_allowlist = st.checkbox("Use character allowlist", value=True)
    allowlist = st.text_input("Allowlist (kept as-is; helpful for strict OCR)", value=default_allowlist) if use_allowlist else None

    # Optional vocab for Tesseract (still non-generative)
    user_words_path = None
    use_vocab = st.checkbox("Use custom vocabulary (.txt)", value=False)
    if use_vocab:
        vf = st.file_uploader("Upload vocabulary (TXT, one term per line)", type=["txt"], key="user_words")
        if vf is not None:
            user_words_path = Path("user_words.txt").absolute()
            with open(user_words_path, "wb") as f:
                f.write(vf.read())
            st.caption(f"Custom vocab saved to {user_words_path}")

    # QA thresholds
    st.markdown("---")
    st.header("QA Thresholds")
    qa_min_conf = st.slider("Min OCR confidence to PASS", 0, 100, 60 if preset == "English (printed)" else 55)
    qa_min_len  = st.slider("Min characters to PASS", 0, 300, 20 if preset == "English (printed)" else 15, step=5)
    qa_max_noisy= st.slider("Max non-alphanumeric ratio to PASS", 0.0, 1.0, 0.85, 0.01)
    st.session_state["qa_min_conf"]  = qa_min_conf
    st.session_state["qa_min_len"]   = qa_min_len
    st.session_state["qa_max_noisy"] = qa_max_noisy
    st.session_state["handwriting_booster"] = handwriting_booster
    st.session_state["booster_trigger_conf"] = booster_trigger_conf
    st.session_state["strict_ocr"] = strict_ocr
    st.session_state["allowlist"] = allowlist

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

# ---------------- Normalization & helpers ----------------
ZW_REMOVE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None)
PUNCT_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00A0": " ", "·": " "})
_PUNCT_STRIP_RE = re.compile(r"[,:;|/\\()\[\]{}<>•·…]+")  # NOTE: we KEEP . and - for raw text; normalized strips many
_SPACE_COLLAPSE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.replace("-\n", "").replace("\n", " ")
    s = s.translate(ZW_REMOVE).translate(PUNCT_MAP)
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT_STRIP_RE.sub(" ", s)
    s = _SPACE_COLLAPSE_RE.sub(" ", s).strip()
    return s.lower()

def escape_for_or_filter(s: str) -> str:
    return s.replace(",", r"\,")

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
    device = torch.device("cpu"); model.to(device)
    return processor, model, device

@st.cache_resource(show_spinner=True)
def load_easyocr_reader(langs: List[str]):
    # langs examples: ["en"], ["en","ar"], ["en","ch_sim"]
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr is not installed. Add `easyocr` to requirements.txt")
    return easyocr.Reader(langs, gpu=False, verbose=False)

# ---------------- OCR engines ----------------
def render_page_to_image(page, zoom: float = 3.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom); pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def preprocess_image(img: Image.Image, do_denoise=True, do_autocontrast=True, binarize=True, thresh=175, to_gray=True):
    if to_gray: img = img.convert("L")
    if do_denoise: img = img.filter(ImageFilter.MedianFilter(size=3))
    if do_autocontrast: img = ImageOps.autocontrast(img)
    if binarize: img = img.point(lambda p: 255 if p > thresh else 0)
    return img

def tesseract_ocr(
    page, zoom: float, lang: str, psm: str,
    user_words_path: Optional[str], timeout_sec: int = 20,
    strict: bool = False, allowlist: Optional[str] = None
) -> Tuple[str, float]:
    img = render_page_to_image(page, zoom=zoom)
    img = preprocess_image(img, True, True, True, 175, True)

    # Base config
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    # STRICT: turn off dictionaries to avoid ‘Gen.’ => ‘German’
    if strict:
        config += " -c load_system_dawg=0 -c load_freq_dawg=0"
        # Avoid character permutations on script words (safer)
        config += " -c language_model_penalty_non_dict_word=0 -c language_model_penalty_case_ok=0"
    if allowlist:
        # keep your punctuation like period/dash/colon if in allowlist
        config += f' -c tessedit_char_whitelist="{allowlist}"'
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

def easyocr_ocr(
    page, zoom: float, langs: List[str], allowlist: Optional[str] = None
) -> Tuple[str, float]:
    # Non-generative CTC OCR, returns text and average confidence
    reader = load_easyocr_reader(langs)
    img = render_page_to_image(page, zoom=zoom).convert("RGB")
    # detail=1 to get confidences
    results = reader.readtext(np.array(img), detail=1, paragraph=True, decoder='greedy',
                              allowlist=allowlist if allowlist else None)
    # results: [(bbox, text, conf), ...]
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

def trocr_ocr(page, zoom: float) -> str:
    processor, model, device = load_trocr_handwritten()
    import torch
    img = render_page_to_image(page, zoom=zoom).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        ids = model.generate(**inputs, max_length=512)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return (text or "").strip()

# (Optional) TrOCR line booster kept for non-strict mode only
def ocr_page_handwriting_boosted(page, base_zoom: float, booster_trigger_conf: int, timeout_sec: int, user_words_path: Optional[str]) -> Dict[str, Any]:
    zoom = max(3.5, base_zoom)
    img = render_page_to_image(page, zoom=zoom).convert("RGB")
    gray = ImageOps.autocontrast(img.convert("L"))
    config = f"--oem 1 --psm 6 -c preserve_interword_spaces=1"
    if user_words_path: config += f" --user-words {user_words_path}"
    try:
        data = pytesseract.image_to_data(gray, lang="eng", config=config, output_type=TessOutput.DICT, timeout=timeout_sec)
    except Exception:
        text = trocr_ocr(page, zoom); tn = normalize_text(text); avg = 70.0 if tn else 0.0
        return dict(text=text, text_norm=tn, avg_conf=avg, ocr_engine="trocr", ocr_psm=None, ocr_zoom=float(zoom), ocr_attempts=1, flags=[])
    n = len(data.get("text", []))
    lines: Dict[tuple, Dict[str, Any]] = {}
    for i in range(n):
        txt = data["text"][i]
        if not isinstance(txt, str) or not txt.strip(): continue
        try: conf = int(data.get("conf", ["-1"]*n)[i])
        except Exception: conf = -1
        if conf < 0: continue
        key = (data.get("block_num",[0]*n)[i], data.get("par_num",[0]*n)[i], data.get("line_num",[0]*n)[i])
        l = int(data.get("left",[0]*n)[i]); t = int(data.get("top",[0]*n)[i])
        w = int(data.get("width",[0]*n)[i]); h = int(data.get("height",[0]*n)[i])
        e = lines.setdefault(key, dict(words=[], confs=[], l=l, t=t, r=l+w, b=t+h))
        e["words"].append(txt.strip()); e["confs"].append(conf)
        e["l"] = min(e["l"], l); e["t"] = min(e["t"], t); e["r"] = max(e["r"], l+w); e["b"] = max(e["b"], t+h)

    if not lines:
        text = trocr_ocr(page, zoom); tn = normalize_text(text); avg = 70.0 if tn else 0.0
        return dict(text=text, text_norm=tn, avg_conf=avg, ocr_engine="trocr", ocr_psm=None, ocr_zoom=float(zoom), ocr_attempts=1, flags=[])

    processor, model, device = load_trocr_handwritten()
    import torch
    out_lines, line_confs = [], []
    for _, ln in sorted(lines.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ltxt = " ".join(ln["words"]); lconf = float(sum(ln["confs"])) / max(len(ln["confs"]), 1)
        if lconf < booster_trigger_conf:
            margin = 4
            l = max(0, ln["l"]-margin); t = max(0, ln["t"]-margin)
            r = min(img.width, ln["r"]+margin); b = min(img.height, ln["b"]+margin)
            crop = img.crop((l,t,r,b))
            with torch.no_grad():
                inputs = processor(images=crop, return_tensors="pt").to(device)
                ids = model.generate(**inputs, max_length=256)
                boosted = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if len(normalize_text(boosted)) > max(len(normalize_text(ltxt)), 3):
                ltxt = boosted; lconf = max(lconf, 75.0)
        out_lines.append(ltxt); line_confs.append(lconf)
    combined = "\n".join(out_lines).strip(); tn = normalize_text(combined)
    avg = float(sum(line_confs))/max(len(line_confs),1) if line_confs else 0.0
    return dict(text=combined, text_norm=tn, avg_conf=avg, ocr_engine="tesseract+trocr-lines", ocr_psm=6, ocr_zoom=float(zoom), ocr_attempts=1, flags=[])

# ---------------- QA helpers ----------------
def qc_metrics(text_norm: str) -> Dict[str, Any]:
    n = len(text_norm); alnum = sum(ch.isalnum() for ch in text_norm)
    non_alnum_ratio = 1.0 - (alnum / n) if n else 1.0
    return {"text_len": n, "non_alnum_ratio": non_alnum_ratio}

def get_thresholds_for(preset: str) -> Dict[str, Any]:
    mc = st.session_state.get("qa_min_conf", 60 if preset == "English (printed)" else 55)
    ml = st.session_state.get("qa_min_len",  20 if preset == "English (printed)" else 15)
    mn = st.session_state.get("qa_max_noisy", 0.85)
    return {"min_conf": mc, "min_len": ml, "max_noisy": mn}

def make_flags(avg_conf: float, text_norm: str, preset: str) -> List[str]:
    t = get_thresholds_for(preset); m = qc_metrics(text_norm)
    flags = []
    if avg_conf < t["min_conf"]: flags.append("low_conf")
    if m["text_len"] < t["min_len"]: flags.append("very_short")
    if m["non_alnum_ratio"] > t["max_noisy"]: flags.append("noisy_text")
    if not text_norm: flags.append("empty")
    return flags

# ---------------- OCR strategy ----------------
def ocr_page_auto(page, preset: str, user_words_path: Optional[str], base_zoom: float, timeout_sec: int,
                  use_trocr: bool, strict: bool, allowlist: Optional[str]) -> Dict[str, Any]:
    """
    Returns dict {text, text_norm, avg_conf, ocr_engine, ocr_psm, ocr_zoom, ocr_attempts, flags}
    """
    # STRICT paths — no generative models, no dictionary corrections expanding words
    if strict:
        if preset == "English (handwritten)":
            if not EASYOCR_AVAILABLE:
                st.warning("Strict handwriting selected but easyocr not installed. Falling back to Tesseract.")
                text, conf = tesseract_ocr(page, max(3.5, base_zoom), "eng", "4", user_words_path, timeout_sec, strict=True, allowlist=allowlist)
            else:
                text, conf = easyocr_ocr(page, max(3.5, base_zoom), langs=["en"], allowlist=allowlist)
            tn = normalize_text(text); flags = make_flags(conf, tn, preset)
            return dict(text=text, text_norm=tn, avg_conf=float(conf), ocr_engine="easyocr" if EASYOCR_AVAILABLE else "tesseract-strict",
                        ocr_psm=None if EASYOCR_AVAILABLE else 4, ocr_zoom=max(3.5, base_zoom), ocr_attempts=1, flags=flags)
        elif preset == "English (printed)":
            text, conf = tesseract_ocr(page, base_zoom, "eng", "6", user_words_path, timeout_sec, strict=True, allowlist=allowlist)
            tn = normalize_text(text); flags = make_flags(conf, tn, preset)
            return dict(text=text, text_norm=tn, avg_conf=float(conf), ocr_engine="tesseract-strict", ocr_psm=6, ocr_zoom=base_zoom, ocr_attempts=1, flags=flags)
        elif preset == "Arabic (ara)":
            # Keep Tesseract; dictionary often helpful for Arabic. We do NOT disable dawgs by default here.
            text, conf = tesseract_ocr(page, max(3.5, base_zoom), "ara", "4", user_words_path, timeout_sec, strict=False, allowlist=allowlist)
            tn = normalize_text(text); flags = make_flags(conf, tn, preset)
            return dict(text=text, text_norm=tn, avg_conf=float(conf), ocr_engine="tesseract", ocr_psm=4, ocr_zoom=max(3.5, base_zoom), ocr_attempts=1, flags=flags)
        elif preset == "Chinese (Simplified)":
            text, conf = tesseract_ocr(page, max(3.5, base_zoom), "chi_sim", "4", user_words_path, timeout_sec, strict=False, allowlist=None)
            tn = normalize_text(text); flags = make_flags(conf, tn, preset)
            return dict(text=text, text_norm=tn, avg_conf=float(conf), ocr_engine="tesseract", ocr_psm=4, ocr_zoom=max(3.5, base_zoom), ocr_attempts=1, flags=flags)
        elif preset == "Chinese (Traditional)":
            text, conf = tesseract_ocr(page, max(3.5, base_zoom), "chi_tra", "4", user_words_path, timeout_sec, strict=False, allowlist=None)
            tn = normalize_text(text); flags = make_flags(conf, tn, preset)
            return dict(text=text, text_norm=tn, avg_conf=float(conf), ocr_engine="tesseract", ocr_psm=4, ocr_zoom=max(3.5, base_zoom), ocr_attempts=1, flags=flags)

    # NON-STRICT behavior (previous flexible pipeline)
    if preset == "English (handwritten)" and st.session_state.get("handwriting_booster", False):
        try:
            cand = ocr_page_handwriting_boosted(page, base_zoom, st.session_state.get("booster_trigger_conf", 65), timeout_sec, user_words_path)
            if len(cand.get("text_norm","")) >= st.session_state.get("qa_min_len", 20):
                return cand
        except Exception:
            pass

    if preset == "English (handwritten)" and use_trocr:
        try:
            text = trocr_ocr(page, zoom=max(3.0, base_zoom)); conf = 65.0
            tn = normalize_text(text); flags = make_flags(conf, tn, preset)
            return dict(text=text, text_norm=tn, avg_conf=conf, ocr_engine="trocr", ocr_psm=None, ocr_zoom=max(3.0, base_zoom), ocr_attempts=1, flags=flags)
        except Exception:
            pass

    # Default Tesseract per preset
    if preset == "English (printed)":
        text, conf = tesseract_ocr(page, base_zoom, "eng", "6", user_words_path, timeout_sec, strict=False, allowlist=allowlist)
    elif preset == "English (handwritten)":
        text, conf = tesseract_ocr(page, base_zoom+0.5, "eng", "4", user_words_path, timeout_sec, strict=False, allowlist=allowlist)
    elif preset == "Arabic (ara)":
        text, conf = tesseract_ocr(page, base_zoom, "ara", "4", user_words_path, timeout_sec, strict=False, allowlist=allowlist)
    elif preset == "Chinese (Simplified)":
        text, conf = tesseract_ocr(page, base_zoom, "chi_sim", "4", user_words_path, timeout_sec, strict=False, allowlist=None)
    elif preset == "Chinese (Traditional)":
        text, conf = tesseract_ocr(page, base_zoom, "chi_tra", "4", user_words_path, timeout_sec, strict=False, allowlist=None)
    else:
        text, conf = tesseract_ocr(page, base_zoom, "eng", "6", user_words_path, timeout_sec, strict=False, allowlist=allowlist)

    tn = normalize_text(text); flags = make_flags(conf, tn, preset)
    return dict(text=text, text_norm=tn, avg_conf=float(conf), ocr_engine="tesseract", ocr_psm=None, ocr_zoom=base_zoom, ocr_attempts=1, flags=flags)

# ---------------- Extraction ----------------
def extract_text_from_pdf(file_bytes, preset: str, user_words_path: Optional[str],
                          base_zoom: float, use_trocr: bool, timeout_sec: int,
                          strict: bool, allowlist: Optional[str], max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts, total = [], len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    prog = st.progress(0, text=f"Extracting text 0/{limit} pages...")
    for i, page in enumerate(doc):
        if i >= limit: break
        try:
            embedded = (page.get_text() or "").strip()
            if embedded and not strict:
                parts.append(embedded)
            else:
                cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, use_trocr, strict, allowlist)
                parts.append(cand["text"])
        except Exception as e:
            parts.append(f"[Error reading page {i+1}: {e}]")
        finally:
            prog.progress((i + 1) / limit, text=f"Extracting text {i+1}/{limit} pages...")
    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(parts).strip()

def extract_pages_with_metadata(file_bytes, document_name, preset: str, user_words_path: Optional[str],
                                base_zoom: float, use_trocr: bool, timeout_sec: int,
                                strict: bool, allowlist: Optional[str], max_pages: int = 0):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    limit = min(total, max_pages) if max_pages and max_pages > 0 else total
    rows = []; prog = st.progress(0, text=f"OCR & split 0/{limit}...")
    for i, page in enumerate(doc):
        if i >= limit: break
        try:
            embedded = (page.get_text() or "").strip()
            if embedded and not strict:
                text = embedded; tn = normalize_text(text); avg = 80.0
                engine, psm, zoom, attempts = "embedded", None, None, 1
            else:
                cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec, use_trocr, strict, allowlist)
                text, tn, avg = cand["text"], cand["text_norm"], cand["avg_conf"]
                engine, psm, zoom, attempts = cand["ocr_engine"], cand["ocr_psm"], cand["ocr_zoom"], cand["ocr_attempts"]
            m = qc_metrics(tn)
            rows.append({
                "document_name": document_name, "page_number": i + 1,
                "text": text, "text_norm": tn,
                "avg_conf": float(avg), "text_len": int(m["text_len"]),
                "non_alnum_ratio": float(m["non_alnum_ratio"]),
                "ocr_engine": engine, "ocr_psm": psm, "ocr_zoom": zoom, "ocr_attempts": attempts,
                "ocr_flags": make_flags(avg, tn, preset), "lang_preset": preset,
            })
        except Exception as e:
            rows.append({
                "document_name": document_name, "page_number": i + 1,
                "text": f"[Error: {e}]", "text_norm": "",
                "avg_conf": 0.0, "text_len": 0, "non_alnum_ratio": 1.0,
                "ocr_engine": "error", "ocr_psm": None, "ocr_zoom": None, "ocr_attempts": 1,
                "ocr_flags": ["error"], "lang_preset": preset,
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
    qn = normalize_text(query); 
    if not qn: return []
    results = []; use_rf = partial_ratio is not None
    tokens = [t for t in qn.split() if len(t) >= 3]
    for r in rows:
        tn = normalize_text(r.get("text", ""))
        if tokens and not any(t in tn for t in tokens): continue
        score = partial_ratio(qn, tn) if use_rf else (100 if qn in tn else 0)
        results.append({**r, "sim": float(score) / 100.0})
    results.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return results[:top_k]

def local_semantic_search(rows: List[Dict[str, Any]], query_vec: np.ndarray, top_k: int = 50) -> List[Dict[str, Any]]:
    embs = [np.array(r["embedding"], dtype=np.float32) for r in rows if isinstance(r.get("embedding"), list)]
    if not embs: return []
    M = np.vstack(embs); M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    q = query_vec.astype(np.float32); q /= (np.linalg.norm(q) + 1e-12)
    sims = M @ q; idxs = np.argsort(-sims)[:top_k]
    kept, j = [], 0
    for r in rows:
        if not isinstance(r.get("embedding"), list): continue
        if j in idxs:
            kept.append({k: r[k] for k in ("document_name", "page_number", "text") if k in r} | {"sim": float(sims[j])})
        j += 1
    kept.sort(key=lambda x: (-x.get("sim", 0.0), x.get("document_name", ""), x.get("page_number") or 0))
    return kept

# ========================= Tabs =========================
tab_ingest, tab_search, tab_qa = st.tabs(["📥 Ingest", "🔎 Search", "✅ QA"])

# -------- Ingest tab --------
with tab_ingest:
    st.subheader("Batch Ingest")
    uploaded_files = st.file_uploader("Drop one or many PDF files", type="pdf", accept_multiple_files=True, key="pdf_batch")

    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        quiet_mode = st.toggle("Quiet mode (summaries only)", value=True)
    with colB:
        limit_preview = st.slider("Preview pages per doc", 0, 3, 1)
    with colC:
        max_pages_each = st.number_input("Limit pages per doc (0 = all)", min_value=0, value=0, step=1)

    colD, colE, colF = st.columns([1.2, 1, 1])
    with colD:
        auto_upload = st.checkbox("Auto-embed & upload to Supabase", value=True)
    with colE:
        delete_existing = st.checkbox("Delete existing rows for each document", value=False)
    with colF:
        batch_size = st.selectbox("Upload batch size", [64, 96, 128, 192], index=2)

    if uploaded_files and st.button("Run ingest on selected file(s)", type="primary"):
        if auto_upload and not _supa:
            st.error("Supabase is not configured."); st.stop()

        model = load_embedding_model() if auto_upload else None
        report_rows: List[Dict[str, Any]] = []; errors_log: List[str] = []
        total_files = len(uploaded_files)
        master_prog = st.progress(0.0, text=f"Processing 0/{total_files} files...")

        for idx, file in enumerate(uploaded_files, start=1):
            docname = file.name
            file_zone = st.container()
            file_prog = file_zone.progress(0.0, text=f"{docname}: starting…")

            try:
                file_bytes = file.read()
                st.session_state["last_pdf_bytes"] = file_bytes
                st.session_state["last_pdf_name"] = docname

                pages = extract_pages_with_metadata(
                    file_bytes=file_bytes,
                    document_name=docname,
                    preset=preset,
                    user_words_path=user_words_path,
                    base_zoom=base_zoom,
                    use_trocr=use_trocr,
                    timeout_sec=timeout_sec,
                    strict=st.session_state["strict_ocr"],
                    allowlist=st.session_state["allowlist"],
                    max_pages=max_pages_each,
                )

                flagged = [p for p in pages if ("low_conf" in p["ocr_flags"] or "very_short" in p["ocr_flags"] or "noisy_text" in p["ocr_flags"] or "empty" in p["ocr_flags"])]
                avg_conf_mean = float(np.mean([p["avg_conf"] for p in pages])) if pages else 0.0
                summary = {
                    "document_name": docname, "pages": len(pages), "flagged": len(flagged),
                    "pass_rate_%": round(100.0 * (len(pages)-len(flagged))/max(len(pages),1), 1),
                    "avg_conf_mean": round(avg_conf_mean, 1),
                    "low_conf": sum(1 for p in pages if "low_conf" in p["ocr_flags"]),
                    "very_short": sum(1 for p in pages if "very_short" in p["ocr_flags"]),
                    "noisy_text": sum(1 for p in pages if "noisy_text" in p["ocr_flags"]),
                    "empty": sum(1 for p in pages if "empty" in p["ocr_flags"]),
                }
                report_rows.append(summary)

                if not quiet_mode and limit_preview > 0:
                    with file_zone.expander(f"Preview • {docname}  ({len(flagged)} flagged / {len(pages)} pages)"):
                        for rec in pages[:limit_preview]:
                            st.markdown(f"**Page {rec['page_number']}** — flags: {rec['ocr_flags']}  •  conf={rec['avg_conf']:.1f}")
                            preview = (rec["text"][:500] + ("..." if len(rec["text"]) > 500 else "")) or "_(empty)_"
                            st.write(preview)
                            st.caption(f"engine={rec['ocr_engine']} psm={rec['ocr_psm']} zoom={rec['ocr_zoom']} attempts={rec['ocr_attempts']}")
                            st.divider()

                file_prog.progress(0.25, text=f"{docname}: OCR complete ({len(pages)} pages).")

                inserted = 0
                if auto_upload and _supa:
                    if delete_existing:
                        try:
                            _supa.table("document_chunks").delete().eq("document_name", docname).execute()
                        except Exception as e:
                            errors_log.append(f"{docname}: delete failed — {e}")

                    texts_batch: List[str] = []; rows_batch: List[Dict[str, Any]] = []

                    def flush_batch_return_count(texts_list: List[str], rows_list: List[Dict[str, Any]]) -> int:
                        if not texts_list: return 0
                        vecs = embed_texts(model, texts_list)
                        payload = []
                        for r, v in zip(rows_list, vecs):
                            payload.append({
                                "document_name": r["document_name"], "page_number": r["page_number"],
                                "text": r["text"], "text_norm": r["text_norm"],
                                "avg_conf": r["avg_conf"], "text_len": r["text_len"], "non_alnum_ratio": r["non_alnum_ratio"],
                                "ocr_engine": r["ocr_engine"], "ocr_psm": r["ocr_psm"], "ocr_zoom": r["ocr_zoom"], "ocr_attempts": r["ocr_attempts"],
                                "ocr_flags": r["ocr_flags"], "lang_preset": r["lang_preset"], "embedding": v.tolist(),
                            })
                        res = _supa.table("document_chunks").insert(payload).execute()
                        return len(res.data or [])

                    for rec in pages:
                        texts_batch.append(rec["text"] or ""); rows_batch.append(rec)
                        if len(texts_batch) >= batch_size:
                            inserted += flush_batch_return_count(texts_batch, rows_batch)
                            texts_batch, rows_batch = [], []
                            frac = 0.25 + 0.60 * (inserted / max(len(pages), 1))
                            file_prog.progress(min(0.90, frac), text=f"{docname}: uploading… {inserted}/{len(pages)}")

                    inserted += flush_batch_return_count(texts_batch, rows_batch)
                    file_prog.progress(0.95, text=f"{docname}: uploaded {inserted} rows.")

                file_prog.progress(1.0, text=f"{docname}: done.")
                file_zone.success(
                    f"✅ {docname} — pages: {summary['pages']} • flagged: {summary['flagged']} • pass-rate: {summary['pass_rate_%']}%"
                )

            except Exception as e:
                errors_log.append(f"{docname}: {e}")
                file_zone.error(f"❌ {docname}: {e}")

            master_prog.progress(idx / total_files, text=f"Processing {idx}/{total_files} files…")

        st.markdown("---")
        st.markdown("### Ingest Report")
        if report_rows:
            df = pd.DataFrame(report_rows).sort_values(["flagged", "pages"], ascending=[False, False])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download report (CSV)", data=csv, file_name="ingest_report.csv", mime="text/csv")

        if errors_log:
            st.error("Some files had issues.")
            err_txt = "\n".join(errors_log)
            st.text(err_txt)
            st.download_button("Download error log", data=err_txt, file_name="ingest_errors.txt", mime="text/plain")
        else:
            st.success("Batch ingest completed with no errors.")

# -------- Search tab --------
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
            try:
                results: List[Dict[str, Any]] = []
                base = _supa.table("document_chunks").select("document_name,page_number,text,avg_conf,text_len", count="exact")
                if selected_doc != "All": base = base.eq("document_name", selected_doc)
                if only_low_quality: base = base.or_("avg_conf.lt.65,text_len.lte.10")

                if mode.startswith("Keyword (exact/normalized)"):
                    norm_query = normalize_text(query)
                    oq = escape_for_or_filter(query); onq = escape_for_or_filter(norm_query)
                    q = base.or_(f"text.ilike.%{oq}%,text_norm.ilike.%{onq}%")
                    if fetch_all:
                        page_size = 100
                        first = q.order("document_name").order("page_number").range(0, page_size - 1).execute()
                        total = getattr(first, "count", None)
                        results = list(first.data or []); offset = page_size
                        while total is not None and offset < total:
                            chunk = q.order("document_name").order("page_number").range(offset, min(offset + page_size - 1, total - 1)).execute()
                            data = chunk.data or []
                            if not data: break
                            results.extend(data); offset += len(data)
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
                        fetched = []; offset = 0; page_size = 1000
                        while offset < max_scan:
                            q = _supa.table("document_chunks").select("document_name,page_number,text")
                            if selected_doc != "All": q = q.eq("document_name", selected_doc)
                            if only_low_quality: q = q.or_("avg_conf.lt.65,text_len.lte.10")
                            q = q.order("document_name").order("page_number").range(offset, offset + page_size - 1)
                            chunk = q.execute(); data = chunk.data or []
                            if not data: break
                            fetched.extend(data); offset += len(data)
                            if len(fetched) >= max_scan: break
                        results = local_fuzzy_search(fetched, query, top_k=top_k)

                else:
                    model = load_embedding_model()
                    qv = model.encode([query])[0].astype(np.float32); qv /= (np.linalg.norm(qv) + 1e-12)
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
                        fetched = []; offset = 0; page_size = 500
                        while offset < max_scan:
                            q = _supa.table("document_chunks").select("document_name,page_number,text,embedding,avg_conf,text_len")
                            if selected_doc != "All": q = q.eq("document_name", selected_doc)
                            if only_low_quality: q = q.or_("avg_conf.lt.65,text_len.lte.10")
                            q = q.order("document_name").order("page_number").range(offset, offset + page_size - 1)
                            chunk = q.execute(); data = chunk.data or []
                            if not data: break
                            fetched.extend([d for d in data if isinstance(d.get("embedding"), list)])
                            offset += len(data)
                            if len(fetched) >= max_scan: break
                        results = local_semantic_search(fetched, qv, top_k=top_k)

                if not results:
                    st.info("No results found.")
                else:
                    for i, r in enumerate(results, 1):
                        doc = r.get("document_name", "Unknown"); page = r.get("page_number", "?")
                        text = (r.get("text") or "").replace("\n", " ")
                        snippet = text[:400] + ("..." if len(text) > 400 else "")
                        st.markdown(f"**{i}. {doc} — Page {page}**")
                        st.write(snippet or "_(empty page)_")
                        if "sim" in r: st.caption(f"Similarity: {r['sim']:.3f}")
                        if "avg_conf" in r and "text_len" in r: st.caption(f"conf={r['avg_conf']:.1f}, len={r['text_len']}")
                        st.divider()
            except Exception as e:
                st.error("Search failed.")
                st.exception(e)

# -------- QA tab --------
with tab_qa:
    st.subheader("Quality Assurance")
    if not _supa:
        st.info("Configure Supabase first.")
    else:
        try:
            docs_res = _supa.table("document_chunks").select("document_name").execute()
            all_docs = sorted({r["document_name"] for r in (docs_res.data or []) if r.get("document_name")})
        except Exception:
            all_docs = []
        try:
            total_pages = _supa.table("document_chunks").select("id", count="exact").limit(1).execute().count or 0
            flagged_pages = _supa.table("document_chunks").select("id", count="exact").or_("avg_conf.lt.65,text_len.lte.10").limit(1).execute().count or 0
        except Exception:
            total_pages, flagged_pages = 0, 0

        pass_rate = (100.0 * (max(total_pages - flagged_pages, 0) / total_pages)) if total_pages else 0.0
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="card"><div class="kpi-title">Documents</div><div class="kpi-value">{len(all_docs)}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card"><div class="kpi-title">Pages Indexed</div><div class="kpi-value">{total_pages}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><div class="kpi-title">Flagged Pages</div><div class="kpi-value">{flagged_pages}</div></div>', unsafe_allow_html=True)
        with c4:
            badge = "badge--ok" if pass_rate >= 90 else ("badge--warn" if pass_rate >= 70 else "badge--err")
            st.markdown(f'<div class="card"><div class="kpi-title">Pass Rate</div><div class="kpi-value">{pass_rate:.1f}%</div><span class="badge {badge}">quality</span></div>', unsafe_allow_html=True)

        st.markdown("---")

        try:
            _docs = _supa.table("document_chunks").select("document_name").execute()
            qa_docs = sorted({r["document_name"] for r in (_docs.data or []) if r.get("document_name")})
        except Exception:
            qa_docs = []
        colq1, colq2, colq3 = st.columns([2,1,1])
        with colq1: sel_doc = st.selectbox("Filter by doc (optional)", ["All"] + qa_docs, key="qa_doc")
        with colq2: min_conf = st.slider("Min conf", 0, 100, 65, key="qa_conf")
        with colq3: list_btn = st.button("List flagged pages", key="qa_list")

        if list_btn:
            try:
                q = _supa.table("document_chunks").select(
                    "document_name,page_number,avg_conf,text_len,non_alnum_ratio,ocr_flags,ocr_engine,ocr_psm,ocr_zoom,ocr_attempts"
                )
                if sel_doc != "All": q = q.eq("document_name", sel_doc)
                q = q.or_(f"avg_conf.lt.{min_conf},text_len.lte.10").order("document_name").order("page_number").limit(5000)
                res = q.execute(); rows = res.data or []
                if not rows:
                    st.success("No flagged pages 🎉")
                else:
                    df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", data=csv, file_name="ocr_flagged_pages.csv", mime="text/csv")
                    try:
                        chart_df = (df.groupby("document_name", as_index=False).size()
                                      .rename(columns={"size": "flagged"})
                                      .sort_values("flagged", ascending=False).head(15))
                        chart = (alt.Chart(chart_df)
                                   .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                                   .encode(x=alt.X("flagged:Q", title="Flagged pages"),
                                           y=alt.Y("document_name:N", sort='-x', title="Document"),
                                           tooltip=["document_name:N","flagged:Q"])
                                   .properties(height=380))
                        st.markdown("**Top flagged documents**")
                        st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        pass
            except Exception as e:
                st.error("QA query failed."); st.exception(e)

        st.markdown("---")
        st.subheader("Batch re-OCR flagged pages (last uploaded PDF)")
        st.caption("Upload & Process the PDF in **Ingest** first so the app holds its bytes.")
        if st.button("Re-OCR flagged pages now", key="qa_reocr"):
            try:
                docname = st.session_state.get("last_pdf_name")
                file_bytes = st.session_state.get("last_pdf_bytes")
                if not docname or not file_bytes:
                    st.warning("No PDF in memory. Go to Ingest → Process PDF, then return.")
                else:
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
                                cand = ocr_page_auto(page, preset, user_words_path, base_zoom, timeout_sec,
                                                     use_trocr=False, strict=st.session_state["strict_ocr"],
                                                     allowlist=st.session_state["allowlist"])
                                vec = model.encode([cand["text"]])[0].astype(np.float32); vec /= (np.linalg.norm(vec) + 1e-12)
                                m = qc_metrics(cand["text_norm"])
                                _supa.table("document_chunks").update({
                                    "text": cand["text"], "text_norm": cand["text_norm"],
                                    "avg_conf": float(cand["avg_conf"]), "text_len": int(m["text_len"]),
                                    "non_alnum_ratio": float(m["non_alnum_ratio"]),
                                    "ocr_engine": cand["ocr_engine"], "ocr_psm": cand["ocr_psm"],
                                    "ocr_zoom": cand["ocr_zoom"], "ocr_attempts": cand["ocr_attempts"],
                                    "embedding": vec.tolist(),
                                }).eq("id", row["id"]).execute()
                                updated += 1
                            prog.progress(i/len(flagged), text=f"Re-OCR {i}/{len(flagged)} pages")
                        doc.close()
                        st.success(f"Re-OCR complete. Updated {updated} pages.")
            except Exception as e:
                st.error("Batch re-OCR failed."); st.exception(e)

# ---------------- Sticky footer ----------------
st.markdown(
    f"""
<div class="footer">
  <div><strong>RAG for Construction Claims — Strict OCR</strong> (no guessing)</div>
  <div>Build <code>{BUILD_ID}</code></div>
</div>
""",
    unsafe_allow_html=True,
)
