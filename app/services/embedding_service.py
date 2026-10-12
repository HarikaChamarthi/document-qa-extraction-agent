
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the pretrained embedding model.
# The first load may download the model files.
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert text passages into numerical vectors."""

    if not texts:
        return []

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vectors.tolist()
