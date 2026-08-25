from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str


@dataclass
class AgentState:
    """Small per-session memory; intentionally bounded and topic-aware."""

    history: list[ConversationTurn] = field(default_factory=list)
    current_order_id: str | None = None
    last_topic: str | None = None
    last_knowledge_query: str | None = None
    last_retrieved_sources: list[dict[str, Any]] = field(
        default_factory=list
    )

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        self.history.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
            )
        )

        # Do not retain unrelated conversation indefinitely.
        if len(self.history) > 8:
            self.history = self.history[-8:]

    def recent_history(
        self,
        limit: int = 6,
    ) -> list[ConversationTurn]:
        return self.history[-limit:]
