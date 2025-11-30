from typing import List, Dict

from core.models import generate_text
from core.prompts import SEARCHER_SYSTEM_PROMPT
from retrieval.vector_store import LocalVectorStore


def run_searcher_agent(
    question: str,
    index_name: str = "default_index",
    top_k: int = 5,
) -> Dict:
    """
    SEARCHER agent:
    - Uses the vector store to get relevant chunks.
    - Summarizes them with the LLM.
    Returns a dict with:
    {
        "summary": str,
        "retrieved_chunks": List[Dict]
    }
    """
    store = LocalVectorStore(index_name=index_name)
    retrieved = store.similarity_search(question, top_k=top_k)

    if not retrieved:
        return {
            "summary": "No relevant context found in the index.",
            "retrieved_chunks": [],
        }

    context_blocks = []
    for r in retrieved:
        src = r["metadata"].get("source", "unknown")
        cid = r["metadata"].get("chunk_id", -1)
        context_blocks.append(
            f"[Source: {src}, Chunk: {cid}]\n{r['text']}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context from vector store:\n{context_text}\n\n"
        "Now produce the requested summary."
    )

    summary = generate_text(
        system_prompt=SEARCHER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=600,
    )

    return {
        "summary": summary,
        "retrieved_chunks": retrieved,
    }
