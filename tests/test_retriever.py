import numpy as np

from app.rag.retriever import Retriever


class FakeEmbedder:
    """Small deterministic embedder for unit tests."""

    def encode(self, text, normalize_embeddings=True):
        return np.array([1.0, 0.0], dtype="float32")


class FakeVectorIndex:
    """Fake FAISS layer returning controlled candidates."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query_embedding, top_k=5):
        return self.chunks[:top_k]


def make_chunk(
    filename,
    content,
    similarity,
    status="active",
    audience="customer",
    policy_authority="official",
    customer_answering=True,
):
    return {
        "content": content,
        "metadata": {
            "filename": filename,
            "heading": "Test Heading",
            "document_id": filename,
            "title": "Test Document",
            "status": status,
            "audience": audience,
            "policy_authority": policy_authority,
            "customer_answering": customer_answering,
        },
        "similarity": similarity,
    }


def test_retriever_filters_unsafe_documents():
    chunks = [
        make_chunk(
            "14-internal-content-migration-notes.md",
            "Unapproved internal content",
            0.99,
            status="draft",
            audience="internal",
            policy_authority="none",
            customer_answering=False,
        ),
        make_chunk(
            "05-domestic-shipping.md",
            "Standard domestic shipping information",
            0.80,
        ),
    ]

    index = FakeVectorIndex(chunks)

    retriever = Retriever(
        vector_index=index,
        embedder=FakeEmbedder(),
    )

    results = retriever.search(
        "Do you ship within the US?"
    )

    assert len(results) == 1
    assert (
        results[0].filename
        == "05-domestic-shipping.md"
    )


def test_official_active_document_beats_lower_quality_document():
    chunks = [
        make_chunk(
            "legacy.md",
            "Old return information",
            0.99,
            status="superseded",
            policy_authority="official",
        ),
        make_chunk(
            "current.md",
            "Current return information",
            0.80,
            status="active",
            policy_authority="official",
        ),
    ]

    index = FakeVectorIndex(chunks)

    retriever = Retriever(
        vector_index=index,
        embedder=FakeEmbedder(),
    )

    results = retriever.search(
        "What is the return policy?"
    )

    assert results[0].filename == "current.md"


def test_empty_query_returns_no_results():
    index = FakeVectorIndex([])

    retriever = Retriever(
        vector_index=index,
        embedder=FakeEmbedder(),
    )

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_similarity_is_preserved():
    chunks = [
        make_chunk(
            "05-domestic-shipping.md",
            "Shipping information",
            0.87,
        )
    ]

    index = FakeVectorIndex(chunks)

    retriever = Retriever(
        vector_index=index,
        embedder=FakeEmbedder(),
    )

    results = retriever.search("shipping")

    assert len(results) == 1
    assert results[0].score == 0.87


def test_metadata_is_preserved():
    chunks = [
        make_chunk(
            "06-international-shipping.md",
            "Canada shipping information",
            0.91,
        )
    ]

    index = FakeVectorIndex(chunks)

    retriever = Retriever(
        vector_index=index,
        embedder=FakeEmbedder(),
    )

    results = retriever.search(
        "Do you ship to Canada?"
    )

    result = results[0]

    assert result.filename == (
        "06-international-shipping.md"
    )
    assert result.document_id == (
        "06-international-shipping.md"
    )
    assert result.heading == "Test Heading"
    assert result.status == "active"
    assert result.audience == "customer"
    assert result.policy_authority == "official"

def test_retriever_accepts_real_chunk_schema():

    chunks = [
        {
            "chunk_id": "RET-2026-01-chunk-1",
            "text": (
                "Customers on the standard plan may request "
                "a return within 30 calendar days of delivery."
            ),
            "metadata": {
                "filename": "01-returns-policy-current.md",
                "heading": "Standard return window",
                "document_id": "RET-2026-01",
                "title": "Returns Policy",
                "status": "active",
                "audience": "customer",
                "policy_authority": "official",
            },
            "similarity": 0.92,
        }
    ]

    index = FakeVectorIndex(chunks)

    retriever = Retriever(
        vector_index=index,
        embedder=FakeEmbedder(),
    )

    results = retriever.search(
        "What is the return window?"
    )

    assert len(results) == 1
    assert "30 calendar days" in results[0].content