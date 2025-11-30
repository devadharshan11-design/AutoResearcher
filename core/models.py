import os
import sys
from typing import List

import numpy as np
import torch
import ollama
from sentence_transformers import SentenceTransformer

# Ensure we can import config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME  # noqa: E402

# ---------- LLM CLIENT (Ollama) ----------

def generate_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """
    Call the local Ollama model with a system + user prompt.
    Returns the generated text content.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=messages,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    return response["message"]["content"]


# ---------- EMBEDDING MODEL (local, GPU) ----------

_device = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)
    return _embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    Returns a NumPy array of shape (n_texts, dim).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings
