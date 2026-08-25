from app.agent.agent import SupportAgent
from app.config import DEBUG_TRACE, INDEX_DIR
from app.llm.groq_llm import GroqLLM
from app.observability import TraceLogger
from app.rag.embeddings import EmbeddingModel
from app.rag.index import VectorIndex
from app.rag.retriever import Retriever
from app.tools.order_lookup import OrderLookup


def create_agent(
    debug: bool | None = None,
    use_llm: bool = True,
) -> SupportAgent:
    """Assemble the real application from small testable components."""

    debug_enabled = DEBUG_TRACE if debug is None else debug

    embedder = EmbeddingModel()

    vector_index = VectorIndex(INDEX_DIR)
    vector_index.load()

    retriever = Retriever(
        vector_index=vector_index,
        embedder=embedder,
    )

    order_lookup = OrderLookup("data/orders.json")

    llm = GroqLLM() if use_llm else None
    tracer = TraceLogger(
        enabled=debug_enabled,
        echo=debug_enabled,
    )

    return SupportAgent(
        retriever=retriever,
        order_lookup=order_lookup,
        llm=llm,
        tracer=tracer,
    )
