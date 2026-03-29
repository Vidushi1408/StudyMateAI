# utils/pdf_reader.py

"""
PDF Reader Utility
------------------
This module handles reading text from PDF files.

We use PyMuPDF (imported as 'fitz') because:
- It's fast and reliable
- Handles multi-page PDFs easily
- Extracts clean text with good formatting

Beginner tip: A PDF is not just text — it's a complex format.
PyMuPDF opens each 'page' like a book page and pulls the text out.
"""

import fitz  # PyMuPDF — installed via 'pip install PyMuPDF'
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF file and returns all its text as one big string.

    Args:
        pdf_path (str): The file path to the PDF (e.g., "data/raw/notes.pdf")

    Returns:
        str: All extracted text combined from every page.
             Returns empty string if file not found or unreadable.
    """

    # Step 1: Check if the file actually exists before trying to open it
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return ""

    extracted_text = ""  # We'll build up the full text here

    try:
        # Step 2: Open the PDF using fitz (PyMuPDF)
        # Think of 'doc' like opening a physical book
        doc = fitz.open(pdf_path)

        print(f"[INFO] PDF loaded: {pdf_path} | Pages: {len(doc)}")

        # Step 3: Loop through each page and extract text
        for page_number in range(len(doc)):
            page = doc[page_number]          # Get the current page
            text = page.get_text("text")     # Extract plain text from page

            # Add a newline between pages so text doesn't run together
            extracted_text += text + "\n"

        doc.close()  # Always close the file after reading!

    except Exception as e:
        # If anything goes wrong, print the error instead of crashing
        print(f"[ERROR] Could not read PDF: {e}")
        return ""

    print(f"[INFO] Extraction complete. Total characters: {len(extracted_text)}")
    return extracted_text.strip()  # .strip() removes leading/trailing whitespace


def extract_text_from_txt(txt_path: str) -> str:
    """
    Reads a plain .txt file and returns its content.

    Args:
        txt_path (str): Path to the text file

    Returns:
        str: Full text content of the file
    """

    if not os.path.exists(txt_path):
        print(f"[ERROR] File not found: {txt_path}")
        return ""

    try:
        # Open with UTF-8 encoding (handles special characters, accents, etc.)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[INFO] Text file loaded. Characters: {len(text)}")
        return text.strip()

    except Exception as e:
        print(f"[ERROR] Could not read text file: {e}")
        return ""


def load_uploaded_file(uploaded_file) -> str:
    """
    Handles files uploaded via Streamlit's file uploader widget.

    Streamlit gives us a file-like object (not a path), so we handle
    both .pdf and .txt formats here.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        str: Extracted text content
    """

    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name  # e.g., "chapter1.pdf"
    file_bytes = uploaded_file.read()  # Read raw bytes

    # Handle PDF files
    if file_name.endswith(".pdf"):
        # fitz can open from bytes directly (no need to save to disk first)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()

    # Handle plain text files
    elif file_name.endswith(".txt"):
        # Decode bytes to string using UTF-8
        return file_bytes.decode("utf-8").strip()

    else:
        print(f"[WARNING] Unsupported file type: {file_name}")
        return ""