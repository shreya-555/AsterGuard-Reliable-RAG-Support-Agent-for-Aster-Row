from pathlib import Path

import pytest

from app.agent.agent import SupportAgent
from app.agent.state import AgentState
from app.tools.order_lookup import OrderLookup


ORDERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "orders.json"
)


class FakeRetriever:
    """Minimal fake retriever for routing tests."""

    def search(
        self,
        query,
        top_k=5,
        candidate_k=20,
    ):
        return []


def make_agent():
    return SupportAgent(
        retriever=FakeRetriever(),
        order_lookup=OrderLookup(
            ORDERS_PATH
        ),
    )


@pytest.mark.parametrize(
    "message",
    [
        "Where is ord-1005?",
        "Where is ORD - 1005?",
        "Where is ord - 1005?",
        "Where is ORD 1005?",
        "Where is ORD_1005?",
    ],
)
def test_order_id_variations_are_detected(message):
    agent = make_agent()

    assert (
        agent._extract_order_id(message)
        == "ORD-1005"
    )



def test_order_lookup_is_used():
    agent = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "Where is ORD-1007?",
        state,
    )

    assert response.tool_used == "order_lookup"
    assert response.handoff is False
    assert "ORD-1007" in response.answer
    assert state.current_order_id == "ORD-1007"


def test_spaced_order_id_uses_order_lookup():
    agent = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "Where is ORD - 1005?",
        state,
    )

    assert response.tool_used == "order_lookup"
    assert "ORD-1005" in response.answer
    assert state.current_order_id == "ORD-1005"


def test_order_follow_up_uses_previous_order():
    agent = make_agent()

    state = AgentState()

    first = agent.handle_message(
        "Where is ORD - 1007?",
        state,
    )

    second = agent.handle_message(
        "When will it arrive?",
        state,
    )

    assert first.tool_used == "order_lookup"
    assert second.tool_used == "order_lookup"
    assert "August 22" in second.answer


def test_unknown_order_does_not_invent_information():
    agent = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "Where is ORD-9999?",
        state,
    )

    assert response.tool_used == "order_lookup"
    assert "couldn't find" in response.answer.lower()


def test_missing_order_follow_up_asks_for_id():
    agent = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "When will it arrive?",
        state,
    )

    assert "order ID" in response.answer


def test_empty_message_is_handled():
    agent = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "   ",
        state,
    )

    assert response.answer