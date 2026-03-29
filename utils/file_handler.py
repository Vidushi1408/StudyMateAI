# utils/file_handler.py

"""
File Handler Utility
--------------------
Handles saving and loading text data to/from disk.

Why do we need this?
- NLP preprocessing takes time (tokenizing, lemmatizing, etc.)
- We save the result once and reload it later — much faster!
- This is a basic form of 'caching'.
"""

import os
import json


# Define where processed data goes
PROCESSED_DIR = "data/processed/"
RAW_DIR = "data/raw/"


def ensure_directories():
    """
    Creates the data folders if they don't exist yet.
    Always call this before saving files.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)  # exist_ok=True = no error if folder exists
    os.makedirs(RAW_DIR, exist_ok=True)
    print("[INFO] Data directories are ready.")


def save_raw_text(text: str, filename: str = "uploaded_text.txt") -> str:
    """
    Saves the raw extracted text (before any processing) to disk.

    Args:
        text (str): The raw text to save
        filename (str): What to name the saved file

    Returns:
        str: The full path where the file was saved
    """
    ensure_directories()
    save_path = os.path.join(RAW_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] Raw text saved to: {save_path}")
    return save_path


def save_processed_data(data: dict, filename: str = "processed_data.json") -> str:
    """
    Saves processed NLP data (tokens, lemmas, etc.) as a JSON file.

    We use JSON because:
    - It's human-readable (you can open it in any text editor)
    - It stores dictionaries and lists natively
    - Easy to load back with one line of code

    Args:
        data (dict): A dictionary containing processed results
                     Example: {"tokens": [...], "lemmas": [...], "clean_text": "..."}
        filename (str): Name for the saved JSON file

    Returns:
        str: Path where the file was saved
    """
    ensure_directories()
    save_path = os.path.join(PROCESSED_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        # indent=2 makes it nicely formatted (readable)
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Processed data saved to: {save_path}")
    return save_path


def load_processed_data(filename: str = "processed_data.json") -> dict:
    """
    Loads previously saved processed data from disk.

    Args:
        filename (str): The JSON file to load

    Returns:
        dict: The loaded data, or empty dict if file doesn't exist
    """
    load_path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(load_path):
        print(f"[WARNING] No processed data found at: {load_path}")
        return {}

    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded processed data from: {load_path}")
    return data


def save_text_chunks(chunks: list, filename: str = "chunks.json") -> str:
    """
    Saves a list of text chunks (used later in RAG module).

    When we split a large document into smaller pieces for FAISS,
    we save those pieces here.

    Args:
        chunks (list): List of text strings (chunks)
        filename (str): Name of the file to save

    Returns:
        str: Path to saved file
    """
    ensure_directories()
    save_path = os.path.join(PROCESSED_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[INFO] {len(chunks)} text chunks saved to: {save_path}")
    return save_path


def load_text_chunks(filename: str = "chunks.json") -> list:
    """
    Loads previously saved text chunks.

    Returns:
        list: List of text chunk strings
    """
    load_path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(load_path):
        print(f"[WARNING] No chunks found at: {load_path}")
        return []

    with open(load_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[INFO] Loaded {len(chunks)} chunks from: {load_path}")
    return chunks