from typing import List, Dict


def simple_text_clean(text: str) -> str:
    """
    Basic cleanup: remove extra spaces and weird chars if needed.
    You can expand this later.
    """
    # Replace multiple spaces/newlines with single spaces
    text = text.replace("\r", " ")
    return text


def chunk_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split a long text into overlapping chunks.

    Returns a list of dicts:
    [
        {"chunk_id": int, "start": int, "end": int, "text": str},
        ...
    ]

    chunk_size and chunk_overlap are in number of words.
    """
    text = simple_text_clean(text)
    words = text.split()
    n = len(words)
    chunks: List[Dict] = []

    if n == 0:
        return chunks

    chunk_id = 0
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words).strip()

        if chunk_text_str:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end,
                    "text": chunk_text_str,
                }
            )
            chunk_id += 1

        if end == n:
            break

        # move start with overlap
        start = max(0, end - chunk_overlap)

    return chunks
