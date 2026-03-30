# generative/quiz_generator.py
"""
Quiz Generator — Groq API (llama-3.2-3b-preview)
Generates exam-style MCQ questions with plausible distractors and full explanations.
"""
import os, sys, re, json, random, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Groq config ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.2-3b-preview"

# ── Visual topics (for Wikipedia image fetch) ─────────────────
VISUAL_TOPICS = {
    "neural network", "deep learning", "cnn", "convolution", "lstm",
    "transformer", "attention mechanism", "backpropagation", "gradient descent",
    "decision tree", "random forest", "clustering", "k-means", "pca",
    "rnn", "autoencoder", "gan", "architecture", "diagram", "graph",
    "cell", "dna", "chromosome", "mitosis", "photosynthesis", "atom",
    "circuit", "processor", "memory", "cpu", "gpu", "network topology",
    "osi model", "tcp ip", "sorting", "binary tree", "hash table",
    "data structure", "algorithm", "flowchart", "uml", "er diagram",
}

QUIZ_SYSTEM_PROMPT = """You are a professor creating high-quality exam questions.

Generate MCQ questions that test DEEP UNDERSTANDING — not memorization.

QUESTION TYPES (vary across questions):
- "application": Apply the concept to a new scenario
- "scenario": "A student notices X... what is happening?"
- "cause_effect": "What happens when Y occurs?"
- "conceptual": "Which statement CORRECTLY describes X?"
- "comparison": "What is the KEY difference between X and Y?"
- "error_spotting": "Which statement about X is INCORRECT?"

RULES:
- Base questions ONLY on the provided notes
- All 4 options must be plausible — use common misconceptions as distractors
- Explanation must say WHY correct is right AND why others are wrong
- Set needs_image true ONLY for highly visual topics (diagrams, biology, circuits)

CRITICAL: Output ONLY a valid JSON array. No markdown. No extra text.

[
  {
    "question": "question text",
    "question_type": "conceptual",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": "B",
    "explanation": "B is correct because... others are wrong because...",
    "needs_image": false,
    "image_topic": "",
    "topic": "main concept"
  }
]"""


def _call_ollama(system_prompt: str, user_message: str, max_tokens: int = 3000) -> str | None:
    """
    Calls Groq API — drop-in replacement for Ollama.
    Function name kept as _call_ollama so nothing else changes.
    """
    if not GROQ_API_KEY:
        print("[QUIZ] ❌ No GROQ_API_KEY found. Add it to Streamlit secrets.")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model       = GROQ_MODEL,
            max_tokens  = max_tokens,
            temperature = 0.4,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[QUIZ] Groq call failed: {e}")
        return None


def _parse_json_from_response(raw: str) -> list:
    """
    Robustly extract a JSON array from the model response.
    Models sometimes wrap output in markdown fences.
    """
    if not raw:
        return []
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$",          "", raw.strip())
    raw = raw.strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    # Find JSON array anywhere in the text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass

    print(f"[QUIZ] Could not parse JSON. Raw snippet: {raw[:300]}")
    return []


def _needs_visual(topic: str) -> bool:
    if not topic:
        return False
    tl = topic.lower().strip()
    if any(vt in tl for vt in VISUAL_TOPICS):
        return True
    visual_keywords = ["architecture", "structure", "diagram", "topology",
                       "layer", "cycle", "process", "flow", "model"]
    return any(kw in tl for kw in visual_keywords)


def _fetch_wikipedia_image(topic: str) -> str | None:
    """Fetch a Wikipedia thumbnail. Returns URL or None."""
    try:
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}",
            timeout=5,
            headers={"User-Agent": "StudyMateAI/1.0"},
        )
        if resp.status_code == 200:
            data  = resp.json()
            thumb = data.get("thumbnail", {}).get("source")
            width = data.get("thumbnail", {}).get("width", 0)
            if thumb and width >= 100:
                return thumb
    except Exception:
        pass
    return None


def generate_quiz(sentences: list, num_questions: int = 5) -> list:
    """Generate quiz questions from note sentences using Groq."""
    if not sentences:
        return []

    selected    = sentences[:60]
    notes_chunk = " ".join(selected)
    if len(notes_chunk) > 3500:
        notes_chunk = notes_chunk[:3500]

    print(f"[QUIZ] Requesting {num_questions} questions from Groq ({GROQ_MODEL})...")

    user_msg = (
        f"Generate exactly {num_questions} exam MCQ questions from these notes. "
        f"Mix question types. Make distractors plausible.\n\n"
        f"Output ONLY the JSON array — no other text.\n\n"
        f"Notes:\n{notes_chunk}"
    )
    raw           = _call_ollama(QUIZ_SYSTEM_PROMPT, user_msg, max_tokens=3000)
    raw_questions = _parse_json_from_response(raw)

    if not raw_questions:
        print("[QUIZ] Groq failed — using rule-based fallback")
        return _rule_based_fallback(sentences, num_questions)

    # Post-process: validate + fetch images
    questions = []
    for q in raw_questions[:num_questions]:
        q.setdefault("explanation",   "")
        q.setdefault("question_type", "conceptual")
        q.setdefault("needs_image",   False)
        q.setdefault("image_topic",   "")
        q.setdefault("topic",         "")

        topic     = q.get("image_topic") or q.get("topic", "")
        use_image = _needs_visual(topic) and bool(topic)
        q["use_image"] = use_image
        q["image_url"] = None

        if use_image:
            img = _fetch_wikipedia_image(topic)
            if img:
                q["image_url"] = img
                print(f"[QUIZ] 🖼  Image: {topic}")
            else:
                q["use_image"] = False

        questions.append(q)

    img_count = sum(1 for q in questions if q.get("use_image"))
    print(f"[QUIZ] ✅ {len(questions)} questions ready ({img_count} with images)")
    return questions


def _rule_based_fallback(sentences: list, num: int) -> list:
    """Simple fallback when Groq API is unavailable."""
    import nltk
    from nltk.corpus import stopwords
    nltk.download("punkt",     quiet=True)
    nltk.download("stopwords", quiet=True)
    stop_w    = set(stopwords.words("english"))
    good      = [s for s in sentences if len(s.split()) >= 10 and "?" not in s]
    random.shuffle(good)
    all_words = list({
        w for s in sentences
        for w in s.split()
        if w.isalpha() and w.lower() not in stop_w and len(w) > 4
    })
    results = []
    for sentence in good[:num]:
        words = [w for w in sentence.split()
                 if w.isalpha() and w.lower() not in stop_w and len(w) > 4]
        if not words:
            continue
        key   = random.choice(words[:3])
        wrong = [w for w in all_words if w.lower() != key.lower()][:3]
        while len(wrong) < 3:
            wrong.append(f"option {len(wrong)+1}")
        opts    = [key] + wrong
        random.shuffle(opts)
        letters = ["A", "B", "C", "D"]
        opt_map = {letters[i]: opts[i] for i in range(4)}
        correct = next(l for l, v in opt_map.items() if v == key)
        results.append({
            "question"     : f"Which term correctly completes: '{sentence[:80].replace(key, '___')}'?",
            "question_type": "conceptual",
            "options"      : opt_map,
            "answer"       : correct,
            "explanation"  : f"'{key}' is correct based on: {sentence}",
            "topic"        : key,
            "use_image"    : False,
            "image_url"    : None,
        })
    return results


def format_quiz_for_display(quiz: list) -> str:
    if not quiz:
        return "No quiz questions generated."
    type_labels = {
        "application"  : "Application",
        "scenario"     : "Scenario",
        "cause_effect" : "Cause & Effect",
        "conceptual"   : "Conceptual",
        "comparison"   : "Comparison",
        "error_spotting": "Error Spotting",
    }
    lines = [f"Quiz — {len(quiz)} Questions\n", "=" * 60]
    for i, q in enumerate(quiz, 1):
        qtype = type_labels.get(q.get("question_type", ""), "MCQ")
        lines.append(f"\nQ{i}. [{qtype}] {q['question']}")
        for l, t in q["options"].items():
            lines.append(f"   {l}) {t}")
        lines.append(f"   ✓ Answer: {q['answer']}")
        if q.get("explanation"):
            lines.append(f"   💡 {q['explanation']}")
        lines.append("")
    return "\n".join(lines)
