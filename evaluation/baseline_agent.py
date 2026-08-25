import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.agent.agent import AgentResponse
from app.agent.state import AgentState
from app.rag.retriever import RetrievedChunk


class BaselineRetriever:
    """Naive semantic-only retriever used only to reproduce the early baseline."""

    def __init__(self, vector_index, embedder):
        self.vector_index = vector_index
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5):
        embedding = np.asarray(
            self.embedder.encode(query, normalize_embeddings=True),
            dtype="float32",
        )
        raw = self.vector_index.search(embedding, top_k=top_k)
        results = []
        for item in raw:
            content = item.get("content", item.get("text", ""))
            if not content:
                continue
            results.append(
                RetrievedChunk(
                    content=content,
                    metadata=item.get("metadata", {}),
                    score=float(item.get("similarity", 0.0)),
                )
            )
        return results


class BaselineSupportAgent:
    """Simple RAG + lookup reference representing the pre-hardening approach."""

    def __init__(self, retriever, order_lookup, llm):
        self.retriever = retriever
        self.order_lookup = order_lookup
        self.llm = llm

    def handle_message(self, message: str, state: AgentState) -> AgentResponse:
        order_id = self._extract_order_id(message)

        if order_id:
            result = self.order_lookup.lookup(order_id)
            if not result["success"]:
                return AgentResponse(
                    answer=result["message"],
                    sources=[],
                    tool_used="order_lookup",
                    tool_arguments={"order_id": order_id},
                )
            order = result["order"]
            return AgentResponse(
                answer=(
                    f"Order {order_id}: "
                    f"{order.get('customer_safe_message') or order.get('status')}"
                ),
                sources=[],
                tool_used="order_lookup",
                tool_arguments={"order_id": order_id},
            )

        if "order" in message.lower() and any(
            token in message.lower()
            for token in ("where", "track", "status", "arrive")
        ):
            return AgentResponse(
                answer="Please provide your order ID.",
                sources=[],
            )

        chunks = self.retriever.search(message, top_k=5)
        if not chunks:
            return AgentResponse(
                answer="I don't know based on the available information.",
                sources=[],
            )

        context = "\n\n".join(
            f"{c.filename} — {c.heading}\n{c.content}"
            for c in chunks
        )
        prompt = (
            "Answer the customer using the retrieved Aster & Row text. "
            "Include useful source filenames.\n\n"
            f"Retrieved text:\n{context}\n\nQuestion: {message}"
        )
        answer = self.llm.generate(
            system_prompt="You are an ecommerce support assistant.",
            user_prompt=prompt,
        )

        return AgentResponse(
            answer=answer,
            sources=[
                {
                    "filename": c.filename,
                    "heading": c.heading,
                    "document_id": c.document_id,
                    "score": c.score,
                }
                for c in chunks
            ],
            handoff=False,
        )

    @staticmethod
    def _extract_order_id(message: str) -> str | None:
        match = re.search(r"\bORD-\d{4}\b", message.upper())
        return match.group(0) if match else None
