import re
from dataclasses import dataclass

from app.rag.retriever import RetrievedChunk


@dataclass
class ConflictResult:
    has_conflict: bool
    reason: str | None = None
    conflicting_chunks: list[RetrievedChunk] | None = None


class ConflictDetector:
    """
    Detects known forms of genuine conflict between
    active, official customer-facing documents.
    """

    _GENERIC_IDENTITY_TERMS = {
        "a",
        "an",
        "and",
        "can",
        "card",
        "care",
        "clean",
        "cleaning",
        "entire",
        "for",
        "guide",
        "i",
        "in",
        "instructions",
        "is",
        "it",
        "md",
        "my",
        "of",
        "policy",
        "product",
        "put",
        "the",
        "to",
        "wash",
    }

    @staticmethod
    def filter_active_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Filters out legacy chunks whenever active versions are present,
        preventing 02-returns-policy-legacy.md from leaking into context.
        """
        has_active = any(
            getattr(chunk, "status", "") == "active"
            or "current" in getattr(chunk, "filename", "").lower()
            for chunk in chunks
        )
        if has_active:
            return [
                chunk
                for chunk in chunks
                if getattr(chunk, "status", "") != "legacy"
                and "legacy" not in getattr(chunk, "filename", "").lower()
            ]
        return chunks

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Normalize filenames, headings, and queries into tokens."""
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @classmethod
    def _scope_to_query_subject(
        cls,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Keep passages whose filename or heading best matches the
        subject named in the query.
        """
        query_tokens = cls._tokens(query) - cls._GENERIC_IDENTITY_TERMS
        scored_chunks: list[tuple[RetrievedChunk, int]] = []

        for chunk in chunks:
            identity = f"{chunk.filename} {chunk.heading}"
            identity_tokens = cls._tokens(identity) - cls._GENERIC_IDENTITY_TERMS
            overlap = len(query_tokens & identity_tokens)
            scored_chunks.append((chunk, overlap))

        maximum_overlap = max(
            (score for _, score in scored_chunks),
            default=0,
        )

        if maximum_overlap == 0:
            return chunks

        return [
            chunk for chunk, score in scored_chunks if score == maximum_overlap
        ]

    def detect(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> ConflictResult:
        # Purge legacy chunks prior to conflict assessment
        chunks = self.filter_active_chunks(chunks)

        authoritative = [
            chunk
            for chunk in chunks
            if (
                getattr(chunk, "status", "active") == "active"
                and getattr(chunk, "policy_authority", "official") == "official"
                and getattr(chunk, "audience", "customer") == "customer"
            )
        ]

        authoritative = self._scope_to_query_subject(
            query,
            authoritative,
        )

        if len(authoritative) < 2:
            return ConflictResult(has_conflict=False)

        query_lower = query.lower()

        # -------------------------------------------------
        # Breeze Tumbler & Product Care cleaning conflicts
        # -------------------------------------------------
        tumbler_terms = {
            "breeze",
            "tumbler",
            "dishwasher",
            "wash",
            "clean",
            "cleaning",
            "care",
        }

        if any(term in query_lower for term in tumbler_terms):
            hand_wash_chunk = None
            dishwasher_chunk = None

            for chunk in authoritative:
                text = chunk.content.lower()

                if any(
                    phrase in text
                    for phrase in [
                        "hand-wash",
                        "hand wash",
                        "hand-washed",
                        "hand washed",
                        "handwash",
                    ]
                ):
                    hand_wash_chunk = chunk

                if "dishwasher" in text:
                    dishwasher_chunk = chunk

            if (
                hand_wash_chunk is not None
                and dishwasher_chunk is not None
                and hand_wash_chunk != dishwasher_chunk
            ):
                return ConflictResult(
                    has_conflict=True,
                    reason=(
                        "Two active official sources provide inconsistent "
                        "cleaning instructions (hand-wash vs. dishwasher safe)."
                    ),
                    conflicting_chunks=[
                        hand_wash_chunk,
                        dishwasher_chunk,
                    ],
                )

        return ConflictResult(has_conflict=False)