"""
MemoryAgent — stores and retrieves user preferences/facts using Mem0.
LLM calls use llm.call_groq() directly.
"""
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

from llm import call_groq
from utils import with_retry

load_dotenv()
logger = logging.getLogger("jarvis")

DEFAULT_USER_ID = "jarvis_user"

_MEM0_CONFIG = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY", ""),
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},
    },
    "vector_store": {
        "provider": "chroma",
        "config": {"collection_name": "jarvis_memory", "path": "./memory_store"},
    },
}


@lru_cache(maxsize=1)
def get_memory():
    try:
        from mem0 import Memory
        return Memory.from_config(_MEM0_CONFIG)
    except Exception as exc:
        logger.error("Failed to initialize Mem0: %s", exc)
        return None


def remember(text: str, user_id: str = DEFAULT_USER_ID) -> bool:
    mem = get_memory()
    if not mem:
        return False
    try:
        mem.add(text, user_id=user_id)
        return True
    except Exception as exc:
        logger.warning("Memory add failed: %s", exc)
        return False


def recall(query: str, user_id: str = DEFAULT_USER_ID) -> list[dict]:
    mem = get_memory()
    if not mem:
        return []
    try:
        result = mem.search(query, user_id=user_id)
        return result.get("results", [])
    except Exception as exc:
        logger.warning("Memory search failed: %s", exc)
        return []


@with_retry(max_attempts=3, agent_name="MemoryAgent")
def run_memory_agent(query: str = "", text_to_remember: str = "") -> str:
    if text_to_remember:
        ok = remember(text_to_remember)
        return f"Remembered: '{text_to_remember}'" if ok else "Memory storage failed."

    memories = recall(query)
    if not memories:
        return "I don't have any stored memories relevant to that yet."

    memory_text = "\n".join(
        f"- {m['memory']} (relevance: {m.get('score', 0):.2f})"
        for m in memories[:10]
    )

    system = (
        "You are JARVIS's memory module. Answer the user's question using only "
        "the stored facts and preferences provided."
    )
    user = f"Question: {query}\n\nStored memories:\n{memory_text}"
    return call_groq(system=system, user=user)
