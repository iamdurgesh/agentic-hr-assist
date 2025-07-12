# src/resume_parser.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
