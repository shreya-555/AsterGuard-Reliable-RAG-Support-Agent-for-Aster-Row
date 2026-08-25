from pathlib import Path

from app.config import (
    INDEX_DIR,
    KNOWLEDGE_BASE_DIR,
)

from app.rag.parser import (
    load_knowledge_base,
)

from app.rag.chunker import (
    create_all_chunks,
)

from app.rag.embeddings import (
    EmbeddingModel,
)

from app.rag.index import (
    VectorIndex,
)


def main():

    print("Loading knowledge base...")

    documents = load_knowledge_base(
        KNOWLEDGE_BASE_DIR
    )

    print(
        f"Loaded {len(documents)} documents."
    )

    print("Creating chunks...")

    chunks = create_all_chunks(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [
        (
            chunk["metadata"]["title"]
            + "\n"
            + chunk["metadata"]["heading"]
            + "\n"
            + chunk["content"]
        )
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    print("Building FAISS index...")

    vector_index = VectorIndex(
        INDEX_DIR
    )

    vector_index.build(
        chunks,
        embeddings
    )

    vector_index.save()

    print(
        f"Index saved to {Path(INDEX_DIR).absolute()}"
    )


if __name__ == "__main__":
    main()