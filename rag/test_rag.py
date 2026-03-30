# rag/test_rag.py
# Run: python -m rag.test_rag

"""
Full RAG Pipeline Test
-----------------------
Tests: indexing → retrieval → question answering
"""

from rag.indexer   import index_document
from rag.retriever import retrieve_relevant_chunks, is_query_answerable
from rag.qa_chain  import answer_question, format_qa_response

# ── Sample study notes ──────────────────────────────────────────
SAMPLE_NOTES = """
Machine learning is a subset of artificial intelligence that enables systems
to learn from data and improve over time without being explicitly programmed.
The three main types of machine learning are supervised learning,
unsupervised learning, and reinforcement learning.
In supervised learning, the model is trained on labeled data where
each input has a known correct output.
Neural networks are computing systems loosely inspired by biological
neural networks in the human brain.
A neural network consists of an input layer, one or more hidden layers,
and an output layer.
Backpropagation is the algorithm used to train neural networks by computing
the gradient of the loss function with respect to each weight.
The learning rate determines how much the model's weights are updated
during each step of training.
Overfitting occurs when a model learns the training data too well and
fails to generalize to new, unseen data.
Dropout is a regularization technique where random neurons are temporarily
disabled during training to prevent overfitting.
The transformer architecture was introduced in the paper
"Attention is All You Need" and revolutionized natural language processing.
BERT is a transformer-based model pre-trained on large text corpora
using masked language modeling and next sentence prediction.
"""

print("="*60)
print("STEP 1: Indexing document into FAISS")
print("="*60)
index, chunks = index_document(SAMPLE_NOTES, chunk_size=3, overlap=1)
print(f"Total chunks indexed: {len(chunks)}")


print("\n" + "="*60)
print("STEP 2: Testing retrieval")
print("="*60)
query    = "How does backpropagation work?"
results  = retrieve_relevant_chunks(query, index, chunks, top_k=3)

print(f"\nQuery: '{query}'")
print(f"Answerable: {is_query_answerable(results)}")
for i, (chunk, score) in enumerate(results, 1):
    print(f"\n  Result {i} (score={score}):\n  '{chunk[:120]}...'")


print("\n" + "="*60)
print("STEP 3: Full Q&A test")
print("="*60)

# Questions answerable from notes
questions = [
    "What is backpropagation?",
    "What are the types of machine learning?",
    "What is dropout and why is it used?",
    "Who invented the transformer architecture?",
]

# Question NOT in notes (should gracefully say so)
out_of_scope = "What is the capital of France?"

all_questions = questions + [out_of_scope]

for question in all_questions:
    result = answer_question(question, index, chunks)
    print(format_qa_response(result))
    print("-" * 50)

print("\n✅ RAG pipeline test complete!")