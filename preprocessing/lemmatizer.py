# preprocessing/lemmatizer.py

"""
Lemmatizer
----------
Converts words to their base dictionary form (lemma).

Lemmatization vs Stemming:
┌─────────────────┬──────────────┬───────────────┐
│ Original Word   │ Stemming     │ Lemmatization │
├─────────────────┼──────────────┼───────────────┤
│ running         │ run          │ run           │
│ studies         │ studi ❌     │ study ✅      │
│ better          │ better       │ good ✅       │
│ flies           │ fli ❌       │ fly ✅        │
└─────────────────┴──────────────┴───────────────┘

Always prefer Lemmatization for NLP tasks — cleaner results!
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def get_wordnet_pos(word: str) -> str:
    """
    Maps NLTK POS tags to WordNet POS tags.

    Why do we need this?
    The lemmatizer needs to know the PART OF SPEECH of a word
    to lemmatize it correctly:
    - "better" as ADJECTIVE → "good"
    - "better" as VERB      → "better" (different meaning!)

    Args:
        word (str): A single word token

    Returns:
        str: WordNet POS constant (ADJ, VERB, NOUN, or ADV)
    """

    # Get POS tag (e.g., "NN"=noun, "VB"=verb, "JJ"=adjective)
    tag = nltk.pos_tag([word])[0][1][0].upper()

    # Map to WordNet format
    tag_map = {
        "J": wordnet.ADJ,    # Adjective
        "V": wordnet.VERB,   # Verb
        "N": wordnet.NOUN,   # Noun (default)
        "R": wordnet.ADV     # Adverb
    }

    # Default to NOUN if tag not found in map
    return tag_map.get(tag, wordnet.NOUN)


def lemmatize_tokens(tokens: list) -> list:
    """
    Converts each token to its lemma (base form).

    Args:
        tokens (list): List of word tokens (already lowercased)

    Returns:
        list: List of lemmatized tokens

    Example:
        ["neural", "networks", "are", "learning"] 
        → ["neural", "network", "be", "learn"]
    """

    lemmatizer = WordNetLemmatizer()
    lemmatized = []

    for token in tokens:
        # Get the correct part of speech for better lemmatization
        pos = get_wordnet_pos(token)
        lemma = lemmatizer.lemmatize(token, pos=pos)
        lemmatized.append(lemma)

    return lemmatized
