"""
CalendarAgent — reads Google Calendar, detects conflicts, warns about prep time.

Auth adapted from _refs/google-api-python-client/samples/calendar_api/calendar_sample.py
LLM calls use llm.call_groq() directly.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from googleapiclient.discovery import build

from google_auth import get_google_creds
from llm import call_groq
from utils import with_retry

load_dotenv()
logger = logging.getLogger("jarvis")

TIMEZONE = os.getenv("TIMEZONE", "Europe/Riga")


def parse_event(event: dict) -> dict:
    start_raw = event.get("start", {})
    end_raw = event.get("end", {})

    def parse_dt(raw: dict) -> datetime | None:
        if "dateTime" in raw:
            try:
                return datetime.fromisoformat(raw["dateTime"].replace("Z", "+00:00"))
            except ValueError:
                return None
        if "date" in raw:
            return datetime.fromisoformat(raw["date"]).replace(tzinfo=timezone.utc)
        return None

    start_dt = parse_dt(start_raw)
    end_dt = parse_dt(end_raw)

    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(no title)"),
        "start": start_dt.isoformat() if start_dt else "",
        "end": end_dt.isoformat() if end_dt else "",
        "start_dt": start_dt,
        "end_dt": end_dt,
        "location": event.get("location", ""),
        "description": event.get("description", "")[:200],
        "all_day": "date" in start_raw,
    }


def detect_conflicts(events: list[dict]) -> list[str]:
    conflicts = []
    timed = [e for e in events if e.get("start_dt") and e.get("end_dt") and not e.get("all_day")]
    timed.sort(key=lambda e: e["start_dt"])
    for i in range(len(timed) - 1):
        a, b = timed[i], timed[i + 1]
        if a["end_dt"] > b["start_dt"]:
            conflicts.append(f"⚠ Conflict: '{a['title']}' overlaps with '{b['title']}'")
    return conflicts


@with_retry(max_attempts=3, agent_name="CalendarAgent")
def fetch_raw_events(days_ahead: int = 4) -> list[dict]:
    creds = get_google_creds()
    if not creds:
        logger.warning("Google credentials not configured — skipping Calendar fetch")
        return []

    service = build("calendar", "v3", credentials=creds)
    now = datetime.now(timezone.utc)
    result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=(now + timedelta(days=days_ahead)).isoformat(),
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    return result.get("items", [])


def create_calendar_event(description: str) -> str:
    """
    Parse a natural-language event description and add it to Google Calendar.
    Example: "add 1 pm haircut tomorrow"
    """
    from datetime import date
    import json as _json, re as _re

    creds = get_google_creds()
    if not creds:
        return "Google credentials not configured. Please set up credentials.json first."

    today = date.today().isoformat()

    system = (
        "You are a calendar event parser. Extract event details from natural language. "
        f"Today is {today}. If no date is mentioned, assume today. "
        "Return ONLY valid JSON with these fields: "
        '{"title": "...", "date": "YYYY-MM-DD", "start_time": "HH:MM", "duration_minutes": 60}'
    )
    raw = call_groq(system=system, user=description)

    try:
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        data = _json.loads(match.group()) if match else {}
        if not data.get("title") or not data.get("date") or not data.get("start_time"):
            return f"Couldn't parse event details from: '{description}'. Try: 'add haircut at 1pm tomorrow'"

        start_iso = f"{data['date']}T{data['start_time']}:00"
        duration = int(data.get("duration_minutes", 60))
        from datetime import datetime, timedelta
        end_dt = datetime.fromisoformat(start_iso) + timedelta(minutes=duration)
        end_iso = end_dt.isoformat()

        service = build("calendar", "v3", credentials=creds)
        event_body = {
            "summary": data["title"],
            "start": {"dateTime": start_iso, "timeZone": TIMEZONE},
            "end":   {"dateTime": end_iso,   "timeZone": TIMEZONE},
        }
        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return (
            f"✅ Added **{data['title']}** to your calendar on {data['date']} at {data['start_time']}. "
            f"[Open in Google Calendar]({created.get('htmlLink', '')})"
        )
    except Exception as exc:
        logger.error("Calendar create failed: %s | raw: %s", exc, raw[:200])
        return f"Failed to create event: {exc}"


def run_calendar_agent(query: str = "") -> list[dict] | str:
    raw_events = fetch_raw_events()
    if not raw_events:
        return "No calendar events found for the next 4 days, or credentials are not configured." if query else []

    parsed = [parse_event(e) for e in raw_events]
    conflicts = detect_conflicts(parsed)

    event_summary = "\n".join(
        f"- {e['title']} @ {e['start']} → {e['end']}"
        + (f" [{e['location']}]" if e["location"] else "")
        for e in parsed
    )
    conflict_text = "\n".join(conflicts) if conflicts else "No conflicts."

    system = (
        "You are JARVIS, a personal AI assistant. Address the user as 'sir' without inserting a comma before it. "
        "Answer in one or two sentences. Flag conflicts only if genuinely important. "
        "If something needs scheduling, ask: 'Shall I add that to your calendar sir?'"
    )
    user = (
        f"Schedule for the next 4 days:\n{event_summary}\n\nConflicts:\n{conflict_text}"
        + (f"\n\nUser question: {query}" if query else "")
    )

    ai_analysis = call_groq(system=system, user=user)

    for e in parsed:
        e.pop("start_dt", None)
        e.pop("end_dt", None)

    if query:
        return ai_analysis

    return [*parsed, *({"type": "conflict", "message": c} for c in conflicts)]
