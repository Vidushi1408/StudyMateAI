# preprocessing/test_preprocessing.py
# Run: python -m preprocessing.test_preprocessing

from preprocessing.pipeline import run_preprocessing_pipeline

sample_text = """
Machine Learning is a subset of Artificial Intelligence.
It allows computers to learn from data without being explicitly programmed.
Neural networks are computing systems inspired by the human brain.
Deep learning uses multiple layers of neural networks to learn representations.
The backpropagation algorithm is used to train neural networks efficiently.
Supervised learning requires labeled training data for model development.
"""

result = run_preprocessing_pipeline(sample_text)

print("\n===== PIPELINE RESULTS =====")
print(f"Word Count      : {result['word_count']}")
print(f"Sentence Count  : {result['sentence_count']}")
print(f"\nClean Text      :\n{result['clean_text'][:200]}...")
print(f"\nSentences       :\n{result['sentences'][:3]}")
print(f"\nTokens (first 10) : {result['tokens'][:10]}")
print(f"\nFiltered Tokens   : {result['filtered_tokens'][:10]}")
print(f"\nLemmas            : {result['lemmas'][:10]}")
print(f"\nTop Keywords      : {result['keywords']}")
