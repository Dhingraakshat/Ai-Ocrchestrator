"""
NewsAgent — searches Serper API for daily news on configured topics.

Search tool adapted from _refs/crewAI-examples/crews/instagram_post/tools/search_tools.py
LLM calls use llm.call_groq() directly.
"""
import json
import logging
import os

import requests
from dotenv import load_dotenv

from llm import call_groq
from utils import with_retry

load_dotenv()
logger = logging.getLogger("jarvis")

TOPICS = ["ai", "startups", "crypto"]
_SERPER_URL = "https://google.serper.dev/search"


def parse_serper_results(response_json: dict, topic: str) -> list[dict]:
    items = []
    for r in response_json.get("organic", []):
        try:
            items.append({
                "title": r["title"],
                "link": r["link"],
                "snippet": r.get("snippet", ""),
                "topic": topic,
            })
        except KeyError:
            continue
    return items


@with_retry(max_attempts=3, agent_name="NewsAgent")
def search_topic(query: str, n_results: int = 5) -> list[dict]:
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        logger.warning("SERPER_API_KEY not set — skipping news fetch")
        return []
    try:
        payload = json.dumps({"q": query, "num": n_results})
        headers = {"X-API-KEY": api_key, "content-type": "application/json"}
        response = requests.request("POST", _SERPER_URL, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        return parse_serper_results(response.json(), topic=query)
    except Exception as exc:
        logger.error("Serper search failed for '%s': %s", query, exc)
        return []


def run_news_agent(topics: list[str] | None = None, query: str = "") -> list[dict] | str:
    topics = topics or TOPICS
    all_articles: list[dict] = []
    for topic in topics:
        all_articles.extend(search_topic(f"{topic} news today" if not query else query))

    if not all_articles:
        return "No news fetched. Check your SERPER_API_KEY." if query else []

    articles_text = "\n".join(
        f"[{a['topic'].upper()}] {a['title']}: {a['snippet'][:100]}"
        for a in all_articles[:15]
    )

    system = (
        "You are JARVIS, a personal AI assistant. Address the user as 'sir'. "
        "Give a 2-3 sentence news summary — headline and one-line takeaway only. No filler."
    )
    user = (
        f"Summarize the most important news from these articles:\n\n{articles_text}"
        + (f"\n\nUser focus: {query}" if query else "")
    )

    summary = call_groq(system=system, user=user, temperature=0.3)
    return summary if query else all_articles
