# rag/agent.py
"""
Multi-Tool Study Agent — Ollama (llama3.2:3b) powered
Works synchronously — fully compatible with Streamlit.
No API key needed. Runs locally via Ollama.
"""
import os, sys, json, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import retrieve_relevant_chunks, build_context_string, is_query_answerable
from rag.indexer   import load_index

# ── Ollama config ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"


def _call_ollama(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str | None:
    """
    Calls the local Ollama /api/chat endpoint.
    Returns the assistant's reply text, or None on failure.
    """
    payload = {
        "model"   : OLLAMA_MODEL,
        "stream"  : False,
        "options" : {"num_predict": max_tokens, "temperature": 0.3},
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        print(f"[AGENT] Ollama error {resp.status_code}: {resp.text[:200]}")
        return None
    except requests.exceptions.ConnectionError:
        print("[AGENT] ❌ Ollama not running. Start it with: ollama serve")
        return None
    except Exception as e:
        print(f"[AGENT] Ollama call failed: {e}")
        return None


# ── Prompts ───────────────────────────────────────────────────

ANSWER_SYSTEM = """You are an expert academic tutor. Give structured, precise answers using ONLY the context provided.

FORMAT RULES — pick the right format based on the question:

For DEFINITION questions (What is X?):
**[Term]** — [precise one-line definition].

### How it works
[mechanism in 2-3 sentences]

### Key Facts
- **Fact 1**: detail
- **Fact 2**: detail
- **Fact 3**: detail

> 💡 **Remember:** [one memorable sentence]

---

For PROCESS questions (How does X work? / Steps?):
**[Process]** involves these steps:

1. **[Step name]** — explanation
2. **[Step name]** — explanation
3. **[Step name]** — explanation

### Key Insight
[why this matters]

> 💡 **Remember:** [one memorable sentence]

---

For COMPARISON questions (Difference between X and Y?):
### X vs Y

| Aspect | **X** | **Y** |
|--------|-------|-------|
| Definition | ... | ... |
| How it works | ... | ... |
| Best used for | ... | ... |

### Bottom Line
[one sentence on when to use each]

> 💡 **Remember:** [one memorable sentence]

---

For GENERAL questions:
**Direct Answer:** [one clear sentence]

### Explanation
[3-4 sentences expanding the answer]

### Key Points
- **Point 1**: detail
- **Point 2**: detail
- **Point 3**: detail

> 💡 **Remember:** [one memorable sentence]

STRICT RULES:
- Answer ONLY from the provided context — never add outside knowledge
- Bold all technical terms on first use
- If topic is not in context: say "**Not in your notes.** This topic isn't covered in your document."
"""


def _detect_format(question: str) -> str:
    """Detect the best answer format from the question text."""
    q = question.lower()
    if any(w in q for w in ["what is", "define", "definition of", "meaning of"]):
        return "definition"
    if any(w in q for w in ["how does", "how do", "steps", "process", "explain how"]):
        return "process"
    if any(w in q for w in ["difference", "compare", "vs", "versus", "distinguish"]):
        return "comparison"
    return "general"


def _search_notes(query: str, index, chunks: list, top_k: int = 5) -> str:
    """Retrieve top-k relevant chunks from FAISS and return as formatted string."""
    results = retrieve_relevant_chunks(query, index, chunks, top_k=top_k)
    if not results:
        return "No relevant passages found."
    return "\n\n".join(f"[score:{s:.2f}] {c}" for c, s in results)


def _generate_answer(question: str, context: str) -> str:
    """Call Ollama to generate a structured answer from retrieved context."""
    fmt = _detect_format(question)
    user_msg = (
        f"Context from student's notes:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n"
        f"Format hint: {fmt}\n\n"
        f"Generate a structured answer using ONLY the context above."
    )
    result = _call_ollama(ANSWER_SYSTEM, user_msg, max_tokens=1000)
    return result or "**Could not generate answer.** Ollama may not be running — try `ollama serve`."


def run_agent(question: str, index=None, chunks: list = None) -> dict:
    """
    Main agent function. Runs synchronously.
    Steps:
      1. Quick answerable check
      2. Search notes (primary query)
      3. If comparison question, search again with second concept
      4. Generate structured answer from retrieved context
    Returns dict: answer, tool_calls, answerable, question
    """
    if index is None or chunks is None:
        index, chunks = load_index()

    if index is None:
        return {
            "answer"    : "**No document indexed.**\n\nUpload and process your notes first.",
            "tool_calls": [], "answerable": False, "question": question,
        }

    # ── Step 1: Quick relevance check ────────────────────────
    quick = retrieve_relevant_chunks(question, index, chunks, top_k=3)
    if not quick or not is_query_answerable(quick, threshold=0.18):
        return {
            "answer"    : (
                "**Not found in your notes.**\n\n"
                "This topic doesn't appear to be covered in your uploaded document. "
                "Try rephrasing or check if this topic is in your notes."
            ),
            "tool_calls": [], "answerable": False, "question": question,
        }

    tool_log = []

    # ── Step 2: Primary search ────────────────────────────────
    print(f"[AGENT] 🔍 search_notes: '{question[:60]}'")
    context1 = _search_notes(question, index, chunks, top_k=5)
    tool_log.append({
        "tool"  : "search_notes",
        "input" : {"query": question},
        "result": context1[:200] + "...",
    })

    all_context = context1

    # ── Step 3: Second search for comparison questions ────────
    q_lower       = question.lower()
    is_comparison = any(w in q_lower for w in ["difference", "compare", "vs", "versus", "distinguish"])
    if is_comparison:
        words        = [w for w in question.split() if len(w) > 4]
        second_query = " ".join(words[-3:]) if len(words) > 3 else question
        print(f"[AGENT] 🔍 search_notes_again: '{second_query[:60]}'")
        context2 = _search_notes(second_query, index, chunks, top_k=4)
        tool_log.append({
            "tool"  : "search_notes_again",
            "input" : {"query": second_query},
            "result": context2[:200] + "...",
        })
        all_context = context1 + "\n\n" + context2

    # ── Step 4: Generate answer ───────────────────────────────
    print(f"[AGENT] ✍️ Generating answer via Ollama ({OLLAMA_MODEL})...")
    final_answer = _generate_answer(question, all_context[:2500])
    tool_log.append({
        "tool"  : "ask_ollama",
        "input" : {"question": question, "context_length": len(all_context)},
        "result": final_answer[:200] + "...",
    })

    print(f"[AGENT] ✅ Done ({len(tool_log)} steps)")
    return {
        "answer"    : final_answer,
        "tool_calls": tool_log,
        "answerable": True,
        "question"  : question,
    }


def answer_question(question: str, index=None, chunks: list = None, **kwargs) -> dict:
    """Compatibility wrapper — same interface used in app.py."""
    return run_agent(question, index, chunks)
