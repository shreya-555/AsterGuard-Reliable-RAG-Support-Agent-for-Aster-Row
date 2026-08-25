from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Single embedding adapter used for both indexing and queries."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
