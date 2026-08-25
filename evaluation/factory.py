from app.bootstrap import create_agent
from app.config import INDEX_DIR
from app.llm.groq_llm import GroqLLM
from app.rag.embeddings import EmbeddingModel
from app.rag.index import VectorIndex
from app.tools.order_lookup import OrderLookup
from evaluation.baseline_agent import BaselineRetriever, BaselineSupportAgent


def create_evaluation_agent(kind: str = "final", debug: bool = False):
    if kind == "final":
        return create_agent(debug=debug)

    embedder = EmbeddingModel()
    vector_index = VectorIndex(INDEX_DIR)
    vector_index.load()
    retriever = BaselineRetriever(vector_index, embedder)

    return BaselineSupportAgent(
        retriever=retriever,
        order_lookup=OrderLookup("data/orders.json"),
        llm=GroqLLM(),
    )
