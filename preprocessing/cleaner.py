# preprocessing/cleaner.py

"""
Text Cleaner
------------
Removes noise from raw text before NLP processing.

Why clean text?
- "Machine Learning!!!" and "machine learning" should be treated the same
- HTML tags like <p> are meaningless to a language model
- Extra spaces and newlines confuse tokenizers
"""

import re  # Regular expressions — powerful text pattern matching


def clean_text(text: str, remove_numbers: bool = False) -> str:
    """
    Cleans raw text by removing noise.

    Args:
        text (str): Raw input text
        remove_numbers (bool): If True, removes digits too (default: False)
                               Keep numbers for study notes (e.g., "3 laws of motion")

    Returns:
        str: Cleaned text
    """

    if not text or not isinstance(text, str):
        return ""

    # Step 1: Lowercase everything
    # "Machine" and "machine" are the same word — unify them
    text = text.lower()

    # Step 2: Remove HTML tags (e.g., <p>, <b>, <br/>)
    # Pattern explanation: <.*?> matches anything between < and >
    text = re.sub(r"<.*?>", " ", text)

    # Step 3: Remove URLs (http://... or www....)
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Step 4: Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Step 5: Optionally remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # Step 6: Remove special characters but KEEP periods and commas
    # We keep sentence structure for summarization later
    text = re.sub(r"[^a-z0-9\s.,!?]", " ", text)

    # Step 7: Remove extra whitespace (multiple spaces → single space)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_sentences(text: str) -> list:
    """
    Splits text into individual sentences.

    Why do we need this?
    - Summarization works sentence by sentence
    - Quiz questions are generated from individual sentences
    - Classification models work on single sentences

    Args:
        text (str): Full cleaned text

    Returns:
        list: List of sentence strings
    """
    import nltk

    # Download punkt if not already present (sentence tokenizer)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # sent_tokenize splits on ".", "!", "?" smartly
    # e.g., "Dr. Smith went home. He was tired." → 2 sentences (not 3!)
    sentences = nltk.sent_tokenize(text)

    # Filter out very short sentences (less than 4 words — likely garbage)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 4]

    return sentences


def remove_duplicate_sentences(sentences: list) -> list:
    """
    Removes duplicate or near-duplicate sentences.

    Study notes often have repeated headings or copied lines.

    Args:
        sentences (list): List of sentence strings

    Returns:
        list: Deduplicated list
    """
    seen = set()
    unique = []

    for sentence in sentences:
        # Normalize before checking (strip spaces, lowercase)
        normalized = sentence.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(sentence)

    return unique
