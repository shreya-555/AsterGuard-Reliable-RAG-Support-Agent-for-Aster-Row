from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import MIN_RELEVANCE_SCORE
from app.rag.index import VectorIndex


@dataclass
class RetrievedChunk:
    """One customer-eligible chunk returned by retrieval."""

    content: str
    metadata: dict[str, Any]
    score: float

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "")

    @property
    def heading(self) -> str:
        return self.metadata.get("heading", "")

    @property
    def document_id(self) -> str:
        return self.metadata.get("document_id", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def status(self) -> str:
        return self.metadata.get("status", "")

    @property
    def audience(self) -> str:
        return self.metadata.get("audience", "")

    @property
    def policy_authority(self) -> str:
        return self.metadata.get("policy_authority", "")


class Retriever:
    """Metadata-aware customer-evidence retrieval on top of FAISS."""

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.vector_index = vector_index
        self.embedder = embedder
        self.min_relevance_score = min_relevance_score

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        candidates = self.vector_index.search(
            query_embedding,
            top_k=candidate_k,
        )

        retrieved: list[RetrievedChunk] = []

        for item in candidates:
            # Canonical generated indexes now use `content`.
            # `text` remains supported for indexes created by an older build.
            chunk_text = item.get(
                "content",
                item.get("text", ""),
            )

            if not chunk_text:
                continue

            chunk = RetrievedChunk(
                content=chunk_text,
                metadata=item.get("metadata", {}),
                score=float(item.get("similarity", 0.0)),
            )

            if not self._is_usable(chunk):
                continue

            if chunk.score < self.min_relevance_score:
                continue

            retrieved.append(chunk)

        retrieved.sort(
            key=self._ranking_key,
            reverse=True,
        )

        return retrieved[:top_k]

    @staticmethod
    def _is_usable(chunk: RetrievedChunk) -> bool:
        """Only active, official, customer-answerable content is evidence."""

        metadata = chunk.metadata

        if metadata.get("status") != "active":
            return False

        if metadata.get("policy_authority") != "official":
            return False

        if metadata.get("audience") != "customer":
            return False

        if metadata.get("customer_answering") is False:
            return False

        return True

    @staticmethod
    def _ranking_key(chunk: RetrievedChunk) -> tuple:
        # All returned chunks already passed strict authority filters.
        # Similarity is therefore the deciding ranking signal.
        return (chunk.score,)
