# rag/agent.py
"""
Multi-Tool Study Agent — Groq API (llama-3.2-3b-preview)
Drop-in replacement for Ollama. Fully compatible with Streamlit.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import retrieve_relevant_chunks, build_context_string, is_query_answerable
from rag.indexer   import load_index

# ── Groq config ───────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.2-3b-preview"


def _call_ollama(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str | None:
    """
    Calls Groq API — drop-in replacement for Ollama.
    Function name kept as _call_ollama so nothing else in the codebase changes.
    """
    if not GROQ_API_KEY:
        print("[AGENT] ❌ No GROQ_API_KEY found. Add it to Streamlit secrets.")
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
        print(f"[AGENT] Groq call failed: {e}")
        return None


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
    q = question.lower()
    if any(w in q for w in ["what is", "define", "definition of", "meaning of"]):
        return "definition"
    if any(w in q for w in ["how does", "how do", "steps", "process", "explain how"]):
        return "process"
    if any(w in q for w in ["difference", "compare", "vs", "versus", "distinguish"]):
        return "comparison"
    return "general"


def _search_notes(query: str, index, chunks: list, top_k: int = 5) -> str:
    results = retrieve_relevant_chunks(query, index, chunks, top_k=top_k)
    if not results:
        return "No relevant passages found."
    return "\n\n".join(f"[score:{s:.2f}] {c}" for c, s in results)


def _generate_answer(question: str, context: str) -> str:
    fmt = _detect_format(question)
    user_msg = (
        f"Context from student's notes:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n"
        f"Format hint: {fmt}\n\n"
        f"Generate a structured answer using ONLY the context above."
    )
    result = _call_ollama(ANSWER_SYSTEM, user_msg, max_tokens=1000)
    return result or "**Could not generate answer.** Check your GROQ_API_KEY in Streamlit secrets."


def run_agent(question: str, index=None, chunks: list = None) -> dict:
    if index is None or chunks is None:
        index, chunks = load_index()

    if index is None:
        return {
            "answer"    : "**No document indexed.**\n\nUpload and process your notes first.",
            "tool_calls": [], "answerable": False, "question": question,
        }

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

    print(f"[AGENT] 🔍 search_notes: '{question[:60]}'")
    context1 = _search_notes(question, index, chunks, top_k=5)
    tool_log.append({
        "tool"  : "search_notes",
        "input" : {"query": question},
        "result": context1[:200] + "...",
    })

    all_context   = context1
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

    print(f"[AGENT] ✍️ Generating answer via Groq ({GROQ_MODEL})...")
    final_answer = _generate_answer(question, all_context[:2500])
    tool_log.append({
        "tool"  : "ask_groq",
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
    return run_agent(question, index, chunks)
