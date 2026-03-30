# generative/explainer.py
"""
Concept Explainer — Groq API (llama-3.2-3b-preview)
Gives deep, structured concept explanations grounded in the student's notes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Groq config ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.2-3b-preview"

_notes_sentences = []


def set_notes_context(sentences: list):
    global _notes_sentences
    _notes_sentences = sentences


def _call_ollama(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str | None:
    """
    Calls Groq API — drop-in replacement for Ollama.
    Function name kept as _call_ollama so nothing else changes.
    """
    if not GROQ_API_KEY:
        print("[EXPLAINER] ❌ No GROQ_API_KEY found. Add it to Streamlit secrets.")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model       = GROQ_MODEL,
            max_tokens  = max_tokens,
            temperature = 0.3,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[EXPLAINER] Groq call failed: {e}")
        return None


SYSTEM_PROMPT = (
    "You are an expert tutor explaining concepts to a student. "
    "Use ONLY the provided context from their notes. "
    "If the concept isn't fully covered in the notes, use what IS there "
    "and note any gaps. Use markdown formatting throughout."
)

STYLE_INSTRUCTIONS = {
    "simple": """Explain this concept simply for a student who is new to the topic.
Structure your answer as:
- **What is it?** (plain-language definition)
- **In simple terms** (analogy or everyday comparison)
- **Why it matters** (practical importance)
- **From the notes** (most relevant fact from context)
- **💡 Remember** (one memorable sentence)
Use simple language, no jargon.""",

    "technical": """Give a detailed technical explanation.
Structure your answer as:
- **Definition** (precise technical definition)
- **How it works** (mechanism, process, or formula)
- **Components / Types** (if applicable)
- **Mathematical or Logical Basis** (if relevant)
- **From the notes** (relevant facts from context)
- **💡 Key Insight** (most important technical takeaway)
Be precise and thorough.""",

    "example": """Explain this concept through examples and analogies.
Structure your answer as:
- **Core Idea** (one-sentence definition)
- **Real-World Analogy** (everyday comparison that clicks)
- **Concrete Example** (step-by-step worked example)
- **From the notes** (example or application mentioned in context)
- **Counter-example** (what it is NOT, to sharpen understanding)
- **💡 Remember** (memorable takeaway)
Make it vivid and concrete.""",
}


def _find_relevant(concept: str, sentences: list, max_results: int = 8) -> list:
    """Find sentences most relevant to the concept using keyword matching."""
    concept_lower = concept.lower().strip()
    concept_words = concept_lower.split()
    scored = []
    for sentence in sentences:
        sl    = sentence.lower()
        score = 0
        if concept_lower in sl:
            score += 3
        elif all(w in sl for w in concept_words):
            score += 2
        elif any(w in sl for w in concept_words if len(w) > 3):
            score += 1
        if score > 0:
            scored.append((sentence, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:max_results]]


def explain_concept(concept: str, style: str = "simple") -> str:
    if not concept.strip():
        return "Please enter a concept to explain."

    relevant = _find_relevant(concept, _notes_sentences, max_results=8)

    if not relevant:
        return (
            f"The concept **'{concept}'** was not found in your uploaded notes. "
            "Try a different keyword or check your document covers this topic."
        )

    context  = "\n".join(f"• {s}" for s in relevant)
    instr    = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["simple"])
    user_msg = (
        f"Concept to explain: **{concept}**\n\n"
        f"Context from student's notes:\n{context}\n\n"
        f"Instructions:\n{instr}"
    )

    result = _call_ollama(SYSTEM_PROMPT, user_msg, max_tokens=1000)
    if result:
        return result

    # Fallback — return relevant sentences formatted nicely
    lines = [f"### 📖 {concept.title()}", ""]
    for s in relevant[:3]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("> 💡 *Groq API key not found. Add GROQ_API_KEY to Streamlit secrets.*")
    return "\n".join(lines)


def explain_keywords(keywords: list, style: str = "simple") -> dict:
    return {kw: explain_concept(kw, style) for kw in keywords}


def generate_study_tips(topic: str) -> str:
    tips = [
        f"1. **Active Recall** — After reading about {topic}, close your notes and write everything you remember.",
        f"2. **Spaced Repetition** — Review {topic} after 1 day, 3 days, then 1 week.",
        f"3. **Teach It** — Explain {topic} out loud as if teaching a classmate.",
    ]
    return "\n\n".join(tips)
