# generative/test_generative.py
# Run: python -m generative.test_generative

from generative.summarizer    import summarize_text
from generative.quiz_generator import generate_quiz, format_quiz_for_display
from generative.explainer      import explain_concept, generate_study_tips

SAMPLE_NOTES = """
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed.
Neural networks are computing systems inspired by biological neural networks
in the human brain. They consist of layers of interconnected nodes.
Deep learning uses multiple layers of neural networks to learn
hierarchical representations of data.
Backpropagation is the algorithm used to train neural networks by computing
gradients and updating weights to minimize the loss function.
Supervised learning requires labeled training data where the model learns
to map inputs to known outputs. Unsupervised learning finds patterns
in data without labels.
The transformer architecture introduced self-attention mechanisms that
allow models to process entire sequences simultaneously rather than
step by step like RNNs.
"""

# ── Test 1: Summarization ────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: SUMMARIZATION")
print("="*60)

summary = summarize_text(SAMPLE_NOTES, style="concise")
print(f"\n📋 Concise Summary:\n{summary}")

# ── Test 2: Quiz Generation ──────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: QUIZ GENERATION")
print("="*60)

sentences = [s.strip() for s in SAMPLE_NOTES.split(".") if len(s.strip()) > 20]
quiz      = generate_quiz(sentences, num_questions=3)
print(format_quiz_for_display(quiz))

# ── Test 3: Concept Explanation ──────────────────────────────────
print("\n" + "="*60)
print("TEST 3: CONCEPT EXPLANATION")
print("="*60)

for style in ["simple", "example"]:
    print(f"\n🔍 Explaining 'backpropagation' [{style} style]:")
    explanation = explain_concept("backpropagation", style=style)
    print(explanation)

# ── Test 4: Study Tips ───────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4: STUDY TIPS")
print("="*60)
tips = generate_study_tips("neural networks")
print(f"\n📚 Study Tips for Neural Networks:\n{tips}")

print("\n✅ All generative AI tests complete!")