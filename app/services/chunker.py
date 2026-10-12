
from typing import Any


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks.

    chunk_size: maximum characters per chunk.
    overlap: characters shared between consecutive chunks.
    """

    # Validate the configuration.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    # Reject text that has no useful content.
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(text):
        # Find the end position of this chunk.
        end = min(start + chunk_size, len(text))

        # Extract the current piece of text.
        chunk = text[start:end]

        # Avoid storing chunks that contain only whitespace.
        if chunk.strip():
            chunks.append({
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "text": chunk,
            })
            chunk_index += 1

        # Move forward while retaining the configured overlap.
        start += step

    return chunks
