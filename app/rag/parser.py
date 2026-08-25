from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


def normalize_metadata_value(value):
    """
    Convert YAML-specific Python objects into
    JSON-serializable values.
    """

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            key: normalize_metadata_value(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            normalize_metadata_value(item)
            for item in value
        ]

    return value


def parse_front_matter(
    content: str,
) -> tuple[dict[str, Any], str]:

    content = content.strip()

    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)

    if len(parts) != 3:
        return {}, content

    _, front_matter, markdown_body = parts

    try:
        metadata = yaml.safe_load(
            front_matter
        ) or {}

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML front matter: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(
            "Front matter must contain a YAML object."
        )

    metadata = normalize_metadata_value(
        metadata
    )

    return metadata, markdown_body.strip()


def parse_markdown_file(
    file_path: str | Path,
) -> dict[str, Any]:

    file_path = Path(file_path)

    content = file_path.read_text(
        encoding="utf-8"
    )

    metadata, markdown = parse_front_matter(
        content
    )

    metadata["filename"] = file_path.name

    return {
        "metadata": metadata,
        "content": markdown,
    }


def load_knowledge_base(
    knowledge_base_dir: str | Path,
) -> list[dict[str, Any]]:

    knowledge_base_dir = Path(
        knowledge_base_dir
    )

    documents = []

    for file_path in sorted(
        knowledge_base_dir.glob("*.md")
    ):
        document = parse_markdown_file(
            file_path
        )

        documents.append(document)

    return documents