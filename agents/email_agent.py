"""
EmailAgent — reads Gmail, extracts deadlines/urgency/action items.

Auth adapted from _refs/google-api-python-client/samples/calendar_api/calendar_sample.py
LLM calls use llm.call_groq() directly (bypasses litellm cache_breakpoint injection).
"""
import logging
import os

from dotenv import load_dotenv
from googleapiclient.discovery import build

from google_auth import get_google_creds
from llm import call_groq
from utils import with_retry

load_dotenv()
logger = logging.getLogger("jarvis")

PRIORITY_KEYWORDS = {
    "HIGH": ["urgent", "asap", "immediately", "deadline", "critical", "action required", "due today"],
    "MEDIUM": ["please review", "follow up", "reminder", "by end of week", "by eod"],
}


def parse_gmail_message(msg: dict) -> dict:
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    subject = headers.get("Subject", "(no subject)")
    sender = headers.get("From", "unknown")
    date_str = headers.get("Date", "")
    snippet = msg.get("snippet", "")

    text = (subject + " " + snippet).lower()
    priority = "LOW"
    for level, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            priority = level
            break

    return {
        "id": msg.get("id", ""),
        "subject": subject,
        "sender": sender,
        "date": date_str,
        "snippet": snippet,
        "priority": priority,
    }


@with_retry(max_attempts=3, agent_name="EmailAgent")
def fetch_raw_emails(max_results: int = 50) -> list[dict]:
    creds = get_google_creds()
    if not creds:
        logger.warning("Google credentials not configured — skipping Gmail fetch")
        return []

    service = build("gmail", "v1", credentials=creds)
    response = service.users().messages().list(
        userId="me", maxResults=max_results, q="is:unread"
    ).execute()

    raw_msgs = []
    for item in response.get("messages", []):
        msg = service.users().messages().get(
            userId="me", id=item["id"], format="full"
        ).execute()
        raw_msgs.append(msg)

    return raw_msgs


def run_email_agent(query: str = "") -> list[dict] | str:
    raw_msgs = fetch_raw_emails()
    if not raw_msgs:
        return "No unread emails found, or Google credentials are not configured." if query else []

    parsed = [parse_gmail_message(m) for m in raw_msgs]
    parsed.sort(key=lambda e: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[e["priority"]])

    email_summary = "\n".join(
        f"[{e['priority']}] From: {e['sender']} | Subject: {e['subject']} | {e['snippet'][:80]}"
        for e in parsed[:10]
    )

    system = (
        "You are JARVIS, a personal AI assistant. Address the user as 'sir'. "
        "Be concise — one or two sentences. Only flag genuinely urgent items. "
        "If action is needed, ask: 'Shall I help with that, sir?'"
    )
    user = (
        f"Analyze these emails and extract action items and deadlines:\n\n{email_summary}"
        if not query
        else f"User query: {query}\n\nEmails:\n{email_summary}"
    )

    ai_analysis = call_groq(system=system, user=user)

    if query:
        return ai_analysis

    for e in parsed[:3]:
        e["ai_analysis"] = ai_analysis
    return parsed
