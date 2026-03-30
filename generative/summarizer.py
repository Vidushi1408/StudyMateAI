# generative/summarizer.py
"""
Ollama (llama3.2:3b) powered Summarizer — structured output
"""
import os, sys, math, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus   import stopwords
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Ollama config ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"


def _call_ollama(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str | None:
    """Call local Ollama model. Returns text or None."""
    payload = {
        "model"   : OLLAMA_MODEL,
        "stream"  : False,
        "options" : {"num_predict": max_tokens, "temperature": 0.3},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        print(f"[SUMMARIZER] Ollama error {resp.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        print("[SUMMARIZER] ❌ Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        print(f"[SUMMARIZER] Failed: {e}")
        return None


SYSTEM_PROMPT = (
    "You are an expert academic summarizer helping students revise effectively. "
    "Follow the formatting instructions EXACTLY — use the exact headers, bold text, "
    "bullet points, and blockquotes as specified. "
    "Base your summary ONLY on the provided notes. "
    "Use markdown formatting throughout."
)

STYLE_PROMPTS = {

"concise": """Create a CONCISE structured summary of the study notes below.

Use this EXACT format (do not skip any section):

## 📌 Overview
Write 2-3 sentences capturing the main theme and scope of the notes.

## 🔑 Key Points
- **Point 1**: One clear sentence
- **Point 2**: One clear sentence
- **Point 3**: One clear sentence
- **Point 4**: One clear sentence
- **Point 5**: One clear sentence

## 🧠 Core Concepts
**Concept 1 Name**: One-line explanation of what it is and why it matters.
**Concept 2 Name**: One-line explanation.
**Concept 3 Name**: One-line explanation.

## 💡 Key Takeaway
> One powerful sentence that a student should remember from this material.

Notes to summarize:
""",

"detailed": """Create a DETAILED structured summary of the study notes below.
Cover everything important. A student should be able to study from this alone.

Use this EXACT format:

## 📌 Overview
3-4 sentences on the main theme, scope, and why this material matters.

## 📚 Topics Covered
List every major topic with a 1-2 sentence explanation:
### Topic Name
Explanation of this topic from the notes.

## 🔑 Key Definitions
**Term 1**: Precise definition as stated in the notes.
**Term 2**: Precise definition.
**Term 3**: Precise definition.

## 🔗 How Concepts Connect
Explain in 3-5 sentences how the main concepts relate to each other.

## ⚠️ Important Points to Remember
- Critical point 1
- Critical point 2
- Critical point 3

## 💡 Key Takeaway
> A 2-3 sentence synthesis of the most important ideas from this material.

Notes to summarize:
""",

"bullets": """Create a BULLET-POINT summary of the study notes below.
Maximum scannability — perfect for quick revision.

Use this EXACT format:

## 📋 Quick Revision Notes

**Main Theme:** One sentence describing what this is about.

---

### 🔵 Core Facts
- Fact 1
- Fact 2
- Fact 3
- Fact 4
- Fact 5

### 🟣 Key Definitions
- **Term**: Definition
- **Term**: Definition
- **Term**: Definition

### 🟡 Important Relationships
- Concept A → leads to → Concept B
- X is different from Y because...

### 🔴 Don't Forget
- Critical point 1
- Critical point 2

---
💡 **One-Line Summary:** Single sentence capturing the entire topic.

Notes to summarize:
""",
}


def _extractive_fallback(text: str, style: str) -> str:
    """Used when Ollama is unavailable."""
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    if not sentences:
        return text[:500]
    ratio    = 0.20 if style == "concise" else 0.40
    num_pick = max(3, min(int(len(sentences) * ratio), 10))
    stop_w   = set(stopwords.words("english"))

    def score(s):
        words   = word_tokenize(s.lower())
        freq    = sum(1 for w in words if w.isalpha() and w not in stop_w)
        penalty = 0.3 if len(words) < 6 else (0.7 if len(words) > 40 else 1.0)
        return freq * penalty

    scored  = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    top     = sorted(scored[:num_pick], key=lambda x: x[0])
    bullets = "\n".join(f"- {s}" for _, s in top)
    return (
        f"## 📌 Key Points\n\n{bullets}\n\n"
        f"> 💡 **Note:** Ollama is not running. Start it with `ollama serve` for AI-powered summaries."
    )


def summarize_text(text: str, style: str = "concise") -> str:
    if not text or len(text.strip()) < 50:
        return "Text is too short to summarize."
    prompt   = STYLE_PROMPTS.get(style, STYLE_PROMPTS["concise"])
    user_msg = f"{prompt}\n{text[:4000]}"
    result   = _call_ollama(SYSTEM_PROMPT, user_msg, max_tokens=1500)
    return result if result else _extractive_fallback(text, style)


def summarize_by_section(sentences: list, section_size: int = 5) -> list:
    summaries = []
    for i in range(math.ceil(len(sentences) / section_size)):
        section  = " ".join(sentences[i * section_size:(i + 1) * section_size])
        if len(section.split()) < 10:
            summaries.append(f"**Section {i+1}:** {section}")
            continue
        prompt   = STYLE_PROMPTS["bullets"]
        user_msg = f"{prompt}\n{section}"
        result   = _call_ollama(SYSTEM_PROMPT, user_msg, max_tokens=600)
        summaries.append(result or f"**Section {i+1}:** {section[:200]}")
    return summaries