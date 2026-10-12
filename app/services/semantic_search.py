
import numpy as np

from app.services.embedding_service import create_embeddings


def semantic_search(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Return the text chunks most semantically similar to a query."""

    # Return no results when there is no query or no document content.
    if not query.strip() or not chunks:
        return []

    # Convert the user's question into a numerical vector.
    query_vector = np.array(
        create_embeddings([query])[0],
        dtype=np.float32,
    )

    # Convert every stored chunk into a numerical vector.
    chunk_vectors = np.array(
        create_embeddings([chunk["text"] for chunk in chunks]),
        dtype=np.float32,
    )

    # Because the vectors are normalized, their dot product
    # calculates cosine similarity.
    scores = chunk_vectors @ query_vector

    # Rank chunk indexes from highest similarity to lowest.
    ranked_indexes = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        # Keep the original chunk information and add its score.
        result = chunks[int(index)].copy()
        result["similarity_score"] = float(scores[index])
        results.append(result)

    return results
