import json
from pathlib import Path

try:
    import faiss
except ImportError:  # allows lightweight unit tests with fake indexes
    faiss = None

import numpy as np


class VectorIndex:
    """Thin FAISS persistence/search layer with no policy knowledge."""

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.faiss_path = self.index_path / "faiss.index"
        self.metadata_path = self.index_path / "metadata.json"
        self.index = None
        self.chunks = []

    @staticmethod
    def _require_faiss():
        if faiss is None:
            raise RuntimeError("faiss-cpu is required. Install requirements.txt first.")

    def build(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
    ):
        self._require_faiss()
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2 or not len(embeddings):
            raise ValueError("Embeddings must be a non-empty 2D array.")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = chunks

    def save(self):
        if self.index is None:
            raise RuntimeError("Cannot save an index before build/load.")

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.faiss_path))

        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                self.chunks,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def load(self):
        self._require_faiss()
        if not self.faiss_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run: python -m ingestion.build_index"
            )

        self.index = faiss.read_index(str(self.faiss_path))

        with self.metadata_path.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "FAISS vector count does not match metadata chunk count. "
                "Rebuild the index."
            )

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ):
        if self.index is None:
            raise RuntimeError("Index is not loaded.")

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            result = self.chunks[index].copy()
            result["similarity"] = float(score)
            results.append(result)

        return results
