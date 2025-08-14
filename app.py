import io
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

st.set_page_config(page_title="PDF Text Extractor (RAG App Part 1)", layout="wide")
st.title("PDF Text Extractor (RAG App Part 1)")
st.write("Upload a PDF file and click **Process PDF** to extract its text. If a page has no embedded text, the app falls back to OCR.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes. Uses embedded text when available, otherwise OCR on rendered page images."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    full_text = []
    total_pages = len(doc)

    # Progress bar
    progress = st.progress(0, text=f"Processing 0/{total_pages} pages...")

    for i, page in enumerate(doc):
        try:
            text = (page.get_text() or "").strip()
            if text:
                full_text.append(text)
            else:
                # Render page to image for OCR
                pix = page.get_pixmap()  # rasterize page
                # Convert pixmap -> PNG bytes -> PIL.Image
                png_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(png_bytes))
                ocr_text = pytesseract.image_to_string(img)
                full_text.append(ocr_text.strip())
        except Exception as page_err:
            # Don't break the whole run on one bad page – record the error and continue
            full_text.append(f"[Error reading page {i+1}: {page_err}]")
        finally:
            progress.progress((i + 1) / total_pages, text=f"Processing {i+1}/{total_pages} pages...")

    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(full_text).strip()

if uploaded_file:
    st.info(f"**Selected file:** {uploaded_file.name}  •  Size: {uploaded_file.size/1024:.1f} KB")
    if st.button("Process PDF", type="primary"):
        with st.spinner("Extracting text..."):
            try:
                file_bytes = uploaded_file.read()
                extracted = extract_text_from_pdf(file_bytes)
                if extracted:
                    st.subheader("Extracted Text")
                    st.text_area("PDF Text Content", extracted, height=400)
                    st.success("Done! You can upload another file or proceed to the next steps later.")
                else:
                    st.warning("No text found. If this is a scanned PDF, OCR may have struggled to read it.")
            except Exception as e:
                st.error("Something went wrong while processing the PDF.")
                st.exception(e)
else:
    st.caption("Tip: start with a small PDF (1–10 pages). Very large scanned PDFs can be slow on free tiers.")

# NEW: page-level extractor that returns one record per page
def extract_pages_with_metadata(file_bytes: bytes, document_name: str):
    """
    Returns a list of dicts: [{document_name, page_number, text}, ...]
    - Uses embedded text when available
    - Falls back to OCR per page if needed
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    pages = []
    total_pages = len(doc)
    progress = st.progress(0, text=f"Splitting into pages 0/{total_pages}...")

    for i, page in enumerate(doc):
        # Try embedded text first
        text = (page.get_text() or "").strip()
        if not text:
            # Fallback to OCR for this page only
            pix = page.get_pixmap()
            png_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(png_bytes))
            text = (pytesseract.image_to_string(img) or "").strip()

        pages.append({
            "document_name": document_name,
            "page_number": i + 1,  # 1-based
            "text": text
        })
        progress.progress((i + 1) / total_pages, text=f"Splitting into pages {i+1}/{total_pages}...")

    doc.close()
    return pages
