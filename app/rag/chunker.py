import re
from typing import Any


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_into_sections(markdown: str) -> list[dict[str, str]]:
    """Split Markdown into heading-aware sections."""

    lines = markdown.splitlines()
    sections = []
    current_heading = None
    current_level = None
    current_content = []

    for line in lines:
        match = HEADING_PATTERN.match(line)

        if match:
            if current_heading is not None:
                sections.append(
                    {
                        "heading": current_heading,
                        "level": current_level,
                        "content": "\n".join(current_content).strip(),
                    }
                )

            current_level = len(match.group(1))
            current_heading = match.group(2)
            current_content = []
        else:
            current_content.append(line)

    if current_heading is not None:
        sections.append(
            {
                "heading": current_heading,
                "level": current_level,
                "content": "\n".join(current_content).strip(),
            }
        )

    return [section for section in sections if section["content"]]


def create_chunks(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert one parsed document into heading-aware chunks."""

    metadata = document["metadata"]
    sections = split_into_sections(document["content"])
    chunks = []

    for index, section in enumerate(sections):
        chunk_metadata = metadata.copy()
        chunk_metadata.update(
            {
                "heading": section["heading"],
                "heading_level": section["level"],
            }
        )

        chunks.append(
            {
                "chunk_id": (
                    f"{metadata.get('document_id', 'DOC')}-chunk-{index}"
                ),
                "content": section["content"],
                "metadata": chunk_metadata,
            }
        )

    return chunks


def create_all_chunks(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks = []
    for document in documents:
        chunks.extend(create_chunks(document))
    return chunks
