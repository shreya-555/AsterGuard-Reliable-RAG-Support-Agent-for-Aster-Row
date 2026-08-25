import json

from app.rag.chunker import create_chunks, split_into_sections
from app.rag.parser import parse_front_matter


def test_front_matter_dates_are_json_serializable_strings():
    content = """---
document_id: TEST-1
status: active
effective_date: 2026-05-01
---
# Test

## Rule
Text.
"""
    metadata, body = parse_front_matter(content)
    assert metadata["effective_date"] == "2026-05-01"
    json.dumps(metadata)
    assert "## Rule" in body


def test_heading_aware_chunk_schema_uses_content_key():
    document = {
        "metadata": {
            "document_id": "TEST-1",
            "title": "Test",
            "status": "active",
        },
        "content": "# Test\n\n## First\nAlpha\n\n## Second\nBeta",
    }
    chunks = create_chunks(document)
    assert len(chunks) == 2
    assert chunks[0]["content"] == "Alpha"
    assert chunks[0]["metadata"]["heading"] == "First"
    assert "text" not in chunks[0]


def test_split_into_sections_preserves_relevant_headings():
    sections = split_into_sections(
        "# Shipping\n\n## Canada\n5-9 days\n\n## Duties\nNot prepaid"
    )
    assert [s["heading"] for s in sections] == ["Canada", "Duties"]
