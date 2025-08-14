# app.py
import io
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- DEBUG: Supabase connection test ---
from supabase import create_client, Client
from supabase.client import ClientOptions

def get_supabase_client() -> Client | None:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    try:
        # NOTE: pass an explicit empty options object
        return create_client(url, key, options=ClientOptions())
    except Exception as e:
        st.sidebar.error(f"Supabase init failed: {e}")
        return None


# ---------- Helpers ----------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Whole-document text extraction.
    Uses embedded text where available; falls back to OCR per page if needed.
    Returns one big string (for quick viewing).
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    full_text_parts = []
    total_pages = len(doc)
    progress = st.progress(0, text=f"Extracting text 0/{total_pages} pages...")

    for i, page in enumerate(doc):
        try:
            text = (page.get_text() or "").strip()
            if text:
                full_text_parts.append(text)
            else:
                # OCR fallback for this page
                pix = page.get_pixmap()
                png_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(png_bytes))
                ocr_text = (pytesseract.image_to_string(img) or "").strip()
                full_text_parts.append(ocr_text)
        except Exception as page_err:
            full_text_parts.append(f"[Error reading page {i+1}: {page_err}]")
        finally:
            progress.progress((i + 1) / total_pages, text=f"Extracting text {i+1}/{total_pages} pages...")

    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(full_text_parts).strip()


def extract_pages_with_metadata(file_bytes: bytes, document_name: str):
    """
    Page-level chunking for RAG (Step 2.1).
    Returns a list of dicts like:
      {"document_name": <str>, "page_number": <int>, "text": <str>}
    Uses embedded text where available; OCR fallback per page otherwise.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    pages = []
    total_pages = len(doc)
    progress = st.progress(0, text=f"Splitting into page chunks 0/{total_pages}...")

    for i, page in enumerate(doc):
        text = (page.get_text() or "").strip()
        if not text:
            # OCR fallback only for pages without embedded text
            pix = page.get_pixmap()
            png_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(png_bytes))
            text = (pytesseract.image_to_string(img) or "").strip()

        pages.append({
            "document_name": document_name,
            "page_number": i + 1,  # 1-based page numbering
            "text": text
        })
        progress.progress((i + 1) / total_pages, text=f"Splitting into page chunks {i+1}/{total_pages}...")

    doc.close()
    return pages

# ---------- UI ----------

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.info(f"**Selected file:** {uploaded_file.name}  •  Size: {uploaded_file.size/1024:.1f} KB")

    if st.button("Process PDF", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                # Read once and reuse for both functions
                file_bytes = uploaded_file.read()

                # 1) Whole-document extraction for quick viewing (Part 1)
                extracted_text = extract_text_from_pdf(file_bytes)

                # 2) NEW: Page-level chunks with metadata (Part 2 - Step 1)
                page_chunks = extract_pages_with_metadata(file_bytes, uploaded_file.name)

                # Save to session for later steps (embeddings/search)
                st.session_state["pages"] = page_chunks
                st.session_state["document_name"] = uploaded_file.name

                # ---- Output: Preview ----
                st.success("Done! See previews below. Page-level chunks are ready for embeddings/search in the next step.")

                # A) Quick per-page preview (first 5 pages)
                st.subheader("Per‑page chunks (preview)")
                if not page_chunks:
                    st.warning("No page text was extracted (OCR may have struggled).")
                else:
                    for rec in page_chunks[:5]:
                        st.markdown(f"**{rec['document_name']} — Page {rec['page_number']}**")
                        preview = (rec["text"] or "").replace("\n", " ")
                        st.write((preview[:500] + ("..." if len(preview) > 500 else "")) or "_(empty page)_")
                        st.divider()

                # B) Whole-document text preview (scrollable)
                st.subheader("Full Extracted Text")
                st.text_area("PDF Text Content", extracted_text or "", height=300)

            except Exception as e:
                st.error("Something went wrong while processing the PDF.")
                st.exception(e)

else:
    st.caption("Tip: start with a small PDF (1–10 pages). Very large scanned PDFs can be slow on free tiers.")
