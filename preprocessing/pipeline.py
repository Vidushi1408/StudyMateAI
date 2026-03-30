"""
preprocessing/pipeline.py — fast NLP pipeline.

Key fix: Step 1 (clean/tokenize) should NOT call embed_sentences.
Embeddings happen in Step 3 (index_document) separately.
This means the UI progress bar actually advances between steps.
"""
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lazy_nltk():
    """Import NLTK tools lazily — avoids slow import at module load time."""
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet",
                "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)
    return nltk


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", "", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    nltk = _lazy_nltk()
    return nltk.word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english"))
    return [t for t in tokens if t.lower() not in stops and t.isalpha()]


def lemmatize(tokens: list[str]) -> list[str]:
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t.lower()) for t in tokens]


def extract_keywords(tokens: list[str], top_n: int = 20) -> list[str]:
    from collections import Counter
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]


def split_sentences(text: str) -> list[str]:
    nltk = _lazy_nltk()
    sents = nltk.sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip()) > 20]


def run_preprocessing_pipeline(text: str) -> dict:
    """
    Steps 1 & 2 of the UI pipeline.
    Deliberately does NOT call embed_sentences — that's Step 3 (index_document).
    This keeps the UI progress bar responsive.
    """
    print("[PIPELINE] Step 1: Cleaning & tokenizing...")
    cleaned  = clean_text(text)
    tokens   = tokenize(cleaned)
    sentences = split_sentences(cleaned)

    print("[PIPELINE] Step 2: Stopwords, lemmatize, keywords...")
    filtered = remove_stopwords(tokens)
    lemmas   = lemmatize(filtered)
    keywords = extract_keywords(lemmas, top_n=25)

    result = {
        "cleaned_text"  : cleaned,
        "tokens"        : tokens,
        "sentences"     : sentences,
        "keywords"      : keywords,
        "word_count"    : len(tokens),
        "sentence_count": len(sentences),
    }
    print(f"[PIPELINE] Done — {len(tokens)} tokens, {len(sentences)} sentences, {len(keywords)} keywords")
    return result