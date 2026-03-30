# rag/qa_chain.py
"""
RAG QA Chain — Reliable Claude-powered answers
No threading, simple agentic loop, robust fallback
"""
import os, sys, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import retrieve_relevant_chunks, build_context_string, is_query_answerable
from rag.indexer   import load_index

SYSTEM_PROMPT = """You are an expert academic tutor. Answer ONLY from the provided context.

STRICT RULES:
1. Use ONLY information from the context below.
2. If the answer is not in the context, say: "**Not covered in your notes.** This topic doesn't appear in your uploaded document."
3. Always use proper markdown formatting.

ANSWER FORMAT — pick the right one:

For DEFINITION questions (What is X?):
**[Term]** — [precise definition from notes].

### How it works
[mechanism or process]

### Why it matters
[significance]

### Key Facts
- Fact 1
- Fact 2
- Fact 3

> 💡 **Remember:** [one memorable line]

For PROCESS questions (How does X work? / Steps?):
**[Process]** works as follows:

1. **Step 1** — explanation
2. **Step 2** — explanation
3. **Step 3** — explanation

> 💡 **Remember:** [one memorable line]

For COMPARISON questions (Difference between X and Y?):
| Aspect | X | Y |
|--------|---|---|
| Definition | ... | ... |
| How it works | ... | ... |
| Use case | ... | ... |

> 💡 **Remember:** [one memorable line]

For GENERAL questions:
**Direct Answer:** [one clear sentence]

### Explanation
[2–4 sentences from the notes]

### Key Points
- **Point 1**: detail
- **Point 2**: detail
- **Point 3**: detail

> 💡 **Remember:** [one memorable line]

Bold all technical terms. Keep it tight — no padding or repetition."""


def _call_claude(question: str, context: str) -> str | None:
    """Simple synchronous Claude API call."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model"      : "claude-haiku-4-5-20251001",
                "max_tokens" : 1200,
                "system"     : SYSTEM_PROMPT,
                "messages"   : [{
                    "role"   : "user",
                    "content": f"Context from notes:\n\n{context}\n\n---\n\nQuestion: {question}",
                }],
            },
            timeout=40,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"].strip()
        print(f"[QA] API error {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"[QA] API call failed: {e}")
        return None


def answer_question(question: str, index=None,
                    chunks: list = None, top_k: int = 5) -> dict:
    """Main entry point — retrieve context then generate answer."""
    if index is None or chunks is None:
        index, chunks = load_index()

    if index is None:
        return {
            "answer"    : "**No document indexed.**\n\nPlease upload and process your notes first.",
            "sources"   : [], "answerable": False, "question": question,
            "tool_calls": [],
        }

    print(f"[QA] Retrieving context for: '{question}'")
    retrieved = retrieve_relevant_chunks(question, index, chunks, top_k=top_k)

    if not retrieved or not is_query_answerable(retrieved, threshold=0.20):
        return {
            "answer"    : (
                "**Not covered in your notes.**\n\n"
                "This topic doesn't appear in your uploaded document. "
                "Try rephrasing or check if this topic is in your notes."
            ),
            "sources"   : retrieved, "answerable": False,
            "question"  : question, "tool_calls": [],
        }

    context = build_context_string(retrieved, max_chars=2500)
    print(f"[QA] Calling Claude with {len(retrieved)} context chunks...")
    answer  = _call_claude(question, context)

    if not answer:
        # Fallback: format best chunks as structured answer
        best    = retrieved[0][0].strip()
        answer  = f"**From your notes:**\n\n{best}"
        if len(retrieved) > 1:
            answer += f"\n\n{retrieved[1][0].strip()}"

    print(f"[QA] ✅ Answer generated ({len(answer)} chars)")
    return {
        "answer"    : answer,
        "sources"   : retrieved,
        "answerable": True,
        "question"  : question,
        "tool_calls": [{"tool": "search_notes", "input": {"query": question},
                        "result": f"{len(retrieved)} chunks retrieved"}],
    }