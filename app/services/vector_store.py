
from pathlib import Path

import chromadb


# Project root:
# C:\Users\pc\Downloads\ai_engineering_practical
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="document_chunks",
    metadata={"hnsw:space": "cosine"},
)


def store_chunks(
    document_id: str,
    filename: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Save document chunks and their embeddings persistently."""

    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have the same length.")

    ids = [
        f"{document_id}_{index}"
        for index in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ],
    )


def search_chunks(
    document_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Find relevant chunks belonging to a specific document."""

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    ids = results.get("ids") or []
    if not ids or not ids[0]:
        return matches

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    for index, chunk_id in enumerate(ids[0]):
        metadata = metadatas[0][index]
        distance = distances[0][index]

        matches.append({
            "id": chunk_id,
            "text": documents[0][index],
            "filename": metadata.get("filename"),
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index", 0),
            "similarity_score": 1.0 - float(distance),
        })

    return matches


def get_document_chunks(document_id: str) -> list[dict]:
    """Retrieve all chunks for a document, in their original order."""

    results = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    chunks = []

    for text, metadata in zip(documents, metadatas):
        chunks.append({
            "text": text,
            "filename": metadata.get("filename"),
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index", 0),
        })

    chunks.sort(key=lambda chunk: chunk["chunk_index"])

    return chunks
