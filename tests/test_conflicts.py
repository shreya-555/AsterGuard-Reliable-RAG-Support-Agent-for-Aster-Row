from app.rag.conflicts import ConflictDetector
from app.rag.retriever import RetrievedChunk


def make_chunk(
    filename: str,
    heading: str,
    content: str,
):
    return RetrievedChunk(
        content=content,
        score=0.9,
        metadata={
            "filename": filename,
            "heading": heading,
            "document_id": filename,
            "title": filename,
            "status": "active",
            "audience": "customer",
            "policy_authority": "official",
        },
    )


def test_breeze_tumbler_cleaning_conflict():

    detector = ConflictDetector()

    care = make_chunk(
        "11-product-care.md",
        "Breeze Tumbler",
        (
            "The stainless-steel body of the Breeze "
            "Tumbler should be hand-washed."
        ),
    )

    product = make_chunk(
        "12-breeze-tumbler-product-card.md",
        "Cleaning",
        (
            "The product card states that all "
            "components are dishwasher safe."
        ),
    )

    result = detector.detect(
        "Can I put my Breeze Tumbler in the dishwasher?",
        [care, product],
    )

    assert result.has_conflict is True

    assert len(
        result.conflicting_chunks
    ) == 2


def test_single_source_is_not_conflict():

    detector = ConflictDetector()

    chunk = make_chunk(
        "05-domestic-shipping.md",
        "Processing time",
        "Most orders require 1–2 business days.",
    )

    result = detector.detect(
        "How long does processing take?",
        [chunk],
    )

    assert result.has_conflict is False


def test_non_authoritative_document_does_not_create_conflict():

    detector = ConflictDetector()

    official = make_chunk(
        "11-product-care.md",
        "Breeze Tumbler",
        "The tumbler should be hand-washed.",
    )

    draft = RetrievedChunk(
        content="All components are dishwasher safe.",
        score=0.99,
        metadata={
            "filename": "draft.md",
            "heading": "Cleaning",
            "document_id": "DRAFT",
            "title": "Draft",
            "status": "draft",
            "audience": "internal",
            "policy_authority": "none",
        },
    )

    result = detector.detect(
        "Can I wash the tumbler in a dishwasher?",
        [official, draft],
    )

    assert result.has_conflict is False

def test_breeze_conflict_does_not_select_unrelated_handwash_chunk():
    detector = ConflictDetector()

    product = make_chunk(
        "12-breeze-tumbler-product-card.md",
        "Cleaning",
        "The product card states that all components are dishwasher safe.",
    )

    care = make_chunk(
        "11-product-care.md",
        "Breeze Tumbler",
        "The stainless-steel body of the Breeze Tumbler should be hand-washed.",
    )

    unrelated = make_chunk(
        "11-product-care.md",
        "Packing cubes",
        "Packing cubes may be hand-washed in cool water with mild detergent.",
    )

    result = detector.detect(
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        [product, care, unrelated],
    )

    assert result.has_conflict is True
    headings = {chunk.heading for chunk in result.conflicting_chunks or []}
    assert "Breeze Tumbler" in headings
    assert "Cleaning" in headings
    assert "Packing cubes" not in headings
