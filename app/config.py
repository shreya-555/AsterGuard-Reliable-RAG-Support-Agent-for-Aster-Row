import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gpt-5-mini",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

KNOWLEDGE_BASE_DIR = os.getenv(
    "KNOWLEDGE_BASE_DIR",
    "knowledge-base",
)

INDEX_DIR = os.getenv(
    "INDEX_DIR",
    "index",
)

TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.30")
)

DEBUG_TRACE = os.getenv(
    "DEBUG_TRACE",
    "false",
).lower() in {"1", "true", "yes", "on"}
