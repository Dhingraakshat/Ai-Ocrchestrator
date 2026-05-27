"""
Direct Groq SDK wrapper — bypasses litellm and crewai's LLM infrastructure
entirely, which injects an unsupported 'cache_breakpoint' field into messages
that Groq's API rejects with a 400 Bad Request.
"""
import logging
import os

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()
logger = logging.getLogger("jarvis")

_PRIMARY_MODEL  = "llama-3.3-70b-versatile"
_FALLBACK_MODEL = "llama-3.1-8b-instant"   # separate rate-limit bucket on Groq free tier
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _chat(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _call(messages: list[dict], temperature: float = 0.1, max_tokens: int = 2048) -> str:
    try:
        return _chat(messages, _PRIMARY_MODEL, temperature, max_tokens)
    except RateLimitError:
        logger.warning("Primary model rate-limited — falling back to %s", _FALLBACK_MODEL)
        try:
            return _chat(messages, _FALLBACK_MODEL, temperature, max_tokens)
        except RateLimitError:
            return (
                "I'm sorry, sir — Groq's daily token limit has been reached for both models. "
                "Please wait ~10 minutes or upgrade at https://console.groq.com/settings/billing"
            )


def call_groq(
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    return _call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature,
        max_tokens,
    )


def call_groq_with_history(
    system: str,
    history: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    return _call(
        [{"role": "system", "content": system}] + history,
        temperature,
        max_tokens,
    )
