import os
from typing import List

from core.models import generate_text
from retrieval.pdf_loader import load_pdf_text
from retrieval.chunker import chunk_text
from retrieval.vector_store import LocalVectorStore
from agents.searcher import run_searcher_agent
from agents.critic import run_critic_agent
from agents.writer import run_writer_agent


def build_index_from_pdfs(
    pdf_paths: List[str],
    index_name: str = "default_index",
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> None:
    """
    Build (or extend) a vector index from a list of PDF files.
    If the index already has data, new chunks are appended.
    """
    store = LocalVectorStore(index_name=index_name)

    for pdf_path in pdf_paths:
        print(f"[INDEX] Loading PDF: {pdf_path}")
        doc = load_pdf_text(pdf_path)
        chunks = chunk_text(doc["full_text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"[INDEX] {os.path.basename(pdf_path)} -> {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": os.path.basename(pdf_path),
                "chunk_id": c["chunk_id"],
            }
            for c in chunks
        ]

        store.add_texts(texts, metadatas)

    print("[INDEX] Index building completed.")


def answer_question_with_rag(
    question: str,
    index_name: str = "default_index",
    top_k: int = 5,
) -> str:
    """
    RAG pipeline:
    - Retrieve top_k relevant chunks from the vector store
    - Pass them with the question to the local LLM
    - Return the generated answer
    """
    store = LocalVectorStore(index_name=index_name)

    results = store.similarity_search(question, top_k=top_k)
    if not results:
        return "I could not find any relevant information in the current index."

    # Build context from retrieved chunks
    context_blocks = []
    for r in results:
        source = r["metadata"].get("source", "unknown")
        chunk_id = r["metadata"].get("chunk_id", -1)
        context_blocks.append(
            f"[Source: {source}, Chunk: {chunk_id}]\n{r['text']}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a research assistant. Use ONLY the provided context from papers. "
        "If something is not supported by the context, say you are not sure. "
        "Cite the sources and chunk IDs when you answer."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context from papers:\n{context_text}\n\n"
        "Now provide a clear, concise answer based only on this context."
    )

    answer = generate_text(system_prompt, user_prompt, temperature=0.2, max_tokens=600)
    return answer
def multi_agent_answer(
    question: str,
    index_name: str = "default_index",
    top_k: int = 5,
) -> dict:
    """
    Multi-agent pipeline:
    1) SEARCHER agent retrieves and summarizes context.
    2) CRITIC agent analyses the summary.
    3) WRITER agent generates final structured answer.

    Returns a dict with:
    {
        "question": str,
        "searcher_summary": str,
        "critic_feedback": str,
        "final_answer": str,
    }
    """
    # 1) Searcher
    searcher_output = run_searcher_agent(
        question=question,
        index_name=index_name,
        top_k=top_k,
    )
    searcher_summary = searcher_output["summary"]

    # 2) Critic
    critic_feedback = run_critic_agent(
        question=question,
        searcher_summary=searcher_summary,
    )

    # 3) Writer
    final_answer = run_writer_agent(
        question=question,
        searcher_summary=searcher_summary,
        critic_feedback=critic_feedback,
    )

    return {
        "question": question,
        "searcher_summary": searcher_summary,
        "critic_feedback": critic_feedback,
        "final_answer": final_answer,
    }

