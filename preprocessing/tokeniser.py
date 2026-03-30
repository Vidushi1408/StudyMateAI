import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (only downloads if not already present)
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)  # newer NLTK versions need this
nltk.download("stopwords",    quiet=True)


def tokenize(text: str) -> list:
    """
    Splits text into a list of word tokens.

    Args:
        text (str): Cleaned text string

    Returns:
        list: List of word tokens (strings)

    Example:
        "deep learning is powerful" → ["deep", "learning", "is", "powerful"]
    """

    if not text:
        return []

    # word_tokenize handles punctuation correctly:
    # "don't" → ["do", "n't"]  (smarter than just splitting on spaces)
    tokens = word_tokenize(text)

    # Keep only alphabetic tokens (remove punctuation tokens like ".", ",")
    tokens = [token for token in tokens if token.isalpha()]

    return tokens


def remove_stopwords(tokens: list, extra_stopwords: list = None) -> list:
    """
    Removes common English stopwords from a token list.

    Args:
        tokens (list): List of word tokens
        extra_stopwords (list): Optional additional words to remove
                                e.g., ["also", "however", "therefore"]

    Returns:
        list: Filtered tokens with stopwords removed
    """

    # Load English stopwords: ["i", "me", "my", "the", "is", "are", ...]
    stop_words = set(stopwords.words("english"))

    # Add any custom stopwords the user wants to remove
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    # Keep only tokens that are NOT in the stopword list
    filtered = [token for token in tokens if token not in stop_words]

    return filtered


def get_word_frequency(tokens: list) -> dict:
    """
    Counts how often each word appears.

    Useful for:
    - Identifying key concepts (most frequent = most important)
    - Building word clouds
    - Basic keyword extraction

    Args:
        tokens (list): List of word tokens

    Returns:
        dict: {word: count} sorted by frequency (highest first)

    Example:
        ["cell", "cell", "mitochondria"] → {"cell": 2, "mitochondria": 1}
    """

    frequency = {}
    for word in tokens:
        frequency[word] = frequency.get(word, 0) + 1

    # Sort by frequency: highest count first
    sorted_freq = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))

    return sorted_freq


def extract_keywords(tokens: list, top_n: int = 15) -> list:
    """
    Returns the top N most frequent words — these are likely key concepts.

    Args:
        tokens (list): Filtered, lemmatized tokens
        top_n (int): How many keywords to return

    Returns:
        list: Top N keyword strings
    """
    freq = get_word_frequency(tokens)
    keywords = list(freq.keys())[:top_n]
    return keywords
