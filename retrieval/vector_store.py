import os
import json
from typing import List, Dict, Optional, Tuple

import numpy as np

from core.models import embed_texts
from config import VECTOR_DB_DIR


class LocalVectorStore:
    """
    A simple local vector store:
    - Stores texts, embeddings, and metadata.
    - Uses cosine similarity for retrieval.
    - Persists to disk as .npy + .json files.
    """

    def __init__(self, index_name: str = "default_index"):
        self.index_name = index_name
        self.index_dir = os.path.join(VECTOR_DB_DIR, index_name)
        os.makedirs(self.index_dir, exist_ok=True)

        self.embeddings_path = os.path.join(self.index_dir, "embeddings.npy")
        self.texts_path = os.path.join(self.index_dir, "texts.json")
        self.metadata_path = os.path.join(self.index_dir, "metadata.json")

        self.embeddings: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

        self._load()

    # ---------- Persistence ----------

    def _load(self) -> None:
        """
        Load existing index from disk if available.
        """
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)

        if os.path.exists(self.texts_path):
            with open(self.texts_path, "r", encoding="utf-8") as f:
                self.texts = json.load(f)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)

        # Sanity check
        if self.embeddings is not None:
            if len(self.texts) != self.embeddings.shape[0]:
                print("[WARN] Texts and embeddings count mismatch. Resetting index.")
                self.embeddings = None
                self.texts = []
                self.metadatas = []

    def _save(self) -> None:
        """
        Save current index to disk.
        """
        if self.embeddings is not None:
            np.save(self.embeddings_path, self.embeddings)

        with open(self.texts_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    # ---------- Indexing ----------

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        Add a batch of texts with optional metadata to the index.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        # Compute embeddings
        new_embeddings = embed_texts(texts)  # shape: (n, dim)

        # Append
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

        # Save to disk
        self._save()

    # ---------- Retrieval ----------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between each row in a and b.
        a: (n, d), b: (m, d)
        Returns: (n, m)
        """
        # Normalize
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)

        # Similarity = dot product of normalized vectors
        return np.dot(a_norm, b_norm.T)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Given a query string, returns top_k most similar documents.

        Each result is:
        {
            "text": str,
            "metadata": dict,
            "score": float
        }
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []

        # Embed query
        query_emb = embed_texts([query])  # shape: (1, dim)

        # Compute cosine similarity with all embeddings
        sims = self._cosine_similarity(query_emb, self.embeddings)  # (1, n)
        sims = sims[0]  # shape: (n,)

        # Get top_k indices
        top_k = min(top_k, len(self.texts))
        top_indices = np.argsort(sims)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_indices:
            results.append(
                {
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(sims[idx]),
                }
            )

        return results
