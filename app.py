import streamlit as st
import fitz # PyMuPDF
import pytesseract
from PIL import Image
st.title("PDF Text Extractor (RAG App Part 1)")
st.write("Upload a PDF file to extract and display its text content.")
# File uploader accepts only PDF files
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
