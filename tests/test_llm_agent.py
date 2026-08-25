from pathlib import Path

from app.agent.agent import SupportAgent
from app.agent.state import AgentState
from app.rag.retriever import RetrievedChunk
from app.tools.order_lookup import OrderLookup


ORDERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "orders.json"
)


class FakeLLM:
    def __init__(self):
        self.system_prompt = None
        self.user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return (
            "Standard returns are allowed within "
            "30 calendar days of delivery.\n\n"
            "Source:\n"
            "- 01-returns-policy-current.md — "
            "Standard return window"
        )


class FakeRetriever:

    def search(
        self,
        query,
        top_k=5,
        candidate_k=20,
    ):
        return [
            RetrievedChunk(
                content=(
                    "Customers on the standard plan may "
                    "request a return within 30 calendar "
                    "days of delivery."
                ),
                score=0.92,
                metadata={
                    "filename":
                        "01-returns-policy-current.md",
                    "heading":
                        "Standard return window",
                    "document_id":
                        "RET-2026-01",
                    "title":
                        "Returns Policy",
                    "status":
                        "active",
                    "audience":
                        "customer",
                    "policy_authority":
                        "official",
                },
            )
        ]


def make_agent():
    llm = FakeLLM()

    agent = SupportAgent(
        retriever=FakeRetriever(),
        order_lookup=OrderLookup(
            ORDERS_PATH
        ),
        llm=llm,
    )

    return agent, llm


def test_llm_receives_grounded_context():

    agent, llm = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "What is the standard return window?",
        state,
    )

    assert (
        "30 calendar days"
        in response.answer
    )

    assert (
        "01-returns-policy-current.md"
        in response.answer
    )

    assert (
        "RET-2026-01"
        in llm.user_prompt
    )


def test_system_prompt_is_separate_from_retrieved_data():

    agent, llm = make_agent()

    state = AgentState()

    agent.handle_message(
        "What is the return policy?",
        state,
    )

    assert llm.system_prompt is not None

    assert (
        "Retrieved knowledge-base passages"
        in llm.system_prompt
    )

    assert (
        "BEGIN UNTRUSTED KNOWLEDGE"
        in llm.user_prompt
    )


def test_customer_evidence_source_is_returned():

    agent, _ = make_agent()

    state = AgentState()

    response = agent.handle_message(
        "What is the standard return window?",
        state,
    )

    assert len(response.sources) >= 1

    assert (
        response.sources[0]["filename"]
        == "01-returns-policy-current.md"
    )

    assert (
        response.sources[0]["heading"]
        == "Standard return window"
    )