"""
TDD tests for all JARVIS agents.
All external services (Google, Groq, Serper, GitHub) are mocked.
Run with: pytest tests/test_agents.py -v
"""
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_with_retry_succeeds_on_first_attempt():
    from utils import with_retry
    call_count = {"n": 0}

    @with_retry(max_attempts=3, agent_name="Test")
    def flaky():
        call_count["n"] += 1
        return "ok"

    assert flaky() == "ok"
    assert call_count["n"] == 1


def test_with_retry_retries_and_raises():
    from utils import with_retry
    call_count = {"n": 0}

    @with_retry(max_attempts=3, agent_name="Test")
    def always_fails():
        call_count["n"] += 1
        raise ValueError("boom")

    with patch("utils.send_notification"):
        with patch("time.sleep"):
            with pytest.raises(ValueError, match="boom"):
                always_fails()

    assert call_count["n"] == 3


def test_send_notification_does_not_raise_when_plyer_missing():
    with patch.dict("sys.modules", {"plyer": None, "plyer.notification": None}):
        from utils import send_notification
        send_notification("Title", "Body")  # must not raise


# ---------------------------------------------------------------------------
# email_agent
# ---------------------------------------------------------------------------

FAKE_GMAIL_MSG = {
    "id": "abc123",
    "snippet": "Please review the PR by EOD",
    "payload": {
        "headers": [
            {"name": "Subject", "value": "Urgent: PR Review"},
            {"name": "From", "value": "boss@example.com"},
            {"name": "Date", "value": "Sat, 24 May 2025 09:00:00 +0000"},
        ]
    },
}


def test_email_agent_parses_message():
    from agents.email_agent import parse_gmail_message
    result = parse_gmail_message(FAKE_GMAIL_MSG)
    assert result["subject"] == "Urgent: PR Review"
    assert result["sender"] == "boss@example.com"
    assert "snippet" in result


def test_email_agent_graceful_without_creds():
    with patch("google_auth.get_google_creds", return_value=None):
        from agents.email_agent import fetch_raw_emails
        msgs = fetch_raw_emails()
    assert msgs == []


def test_email_agent_fetches_from_gmail():
    mock_creds = MagicMock()
    mock_service = MagicMock()
    mock_service.users().messages().list().execute.return_value = {
        "messages": [{"id": "abc123"}]
    }
    mock_service.users().messages().get().execute.return_value = FAKE_GMAIL_MSG

    with patch("google_auth.get_google_creds", return_value=mock_creds):
        with patch("googleapiclient.discovery.build", return_value=mock_service):
            from agents.email_agent import fetch_raw_emails
            msgs = fetch_raw_emails()
    assert len(msgs) == 1
    assert msgs[0]["id"] == "abc123"


# ---------------------------------------------------------------------------
# calendar_agent
# ---------------------------------------------------------------------------

FAKE_EVENT = {
    "id": "evt1",
    "summary": "Team Standup",
    "start": {"dateTime": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"},
    "end": {"dateTime": (datetime.utcnow() + timedelta(hours=2)).isoformat() + "Z"},
    "location": "Zoom",
}


def test_calendar_agent_parses_event():
    from agents.calendar_agent import parse_event
    result = parse_event(FAKE_EVENT)
    assert result["title"] == "Team Standup"
    assert result["location"] == "Zoom"


def test_calendar_agent_graceful_without_creds():
    with patch("google_auth.get_google_creds", return_value=None):
        from agents.calendar_agent import fetch_raw_events
        events = fetch_raw_events()
    assert events == []


def test_calendar_detects_conflict():
    from agents.calendar_agent import detect_conflicts
    now = datetime.utcnow()
    events = [
        {"start": now, "end": now + timedelta(hours=2), "title": "Meeting A"},
        {"start": now + timedelta(hours=1), "end": now + timedelta(hours=3), "title": "Meeting B"},
        {"start": now + timedelta(hours=5), "end": now + timedelta(hours=6), "title": "Meeting C"},
    ]
    conflicts = detect_conflicts(events)
    assert len(conflicts) == 1
    assert "Meeting A" in conflicts[0] or "Meeting B" in conflicts[0]


# ---------------------------------------------------------------------------
# news_agent
# ---------------------------------------------------------------------------

FAKE_SERPER_RESPONSE = {
    "organic": [
        {"title": "AI startup raises $100M", "link": "https://example.com/1", "snippet": "A new AI startup..."},
        {"title": "Bitcoin hits new high", "link": "https://example.com/2", "snippet": "Crypto markets..."},
    ]
}


def test_news_agent_parses_serper():
    from agents.news_agent import parse_serper_results
    results = parse_serper_results(FAKE_SERPER_RESPONSE, topic="ai")
    assert len(results) == 2
    assert results[0]["title"] == "AI startup raises $100M"
    assert results[0]["topic"] == "ai"


def test_news_agent_calls_serper():
    with patch("requests.request") as mock_req:
        mock_req.return_value.json.return_value = FAKE_SERPER_RESPONSE
        from agents.news_agent import search_topic
        results = search_topic("ai startups")
    assert len(results) == 2


def test_news_agent_graceful_on_api_error():
    with patch("requests.request", side_effect=ConnectionError("network down")):
        with patch("utils.send_notification"):
            from agents.news_agent import search_topic
            results = search_topic("ai")
    assert results == []


# ---------------------------------------------------------------------------
# task_agent
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "jarvis_test.db")
    return db_path


def test_task_agent_creates_and_lists_tasks(tmp_db):
    from agents.task_agent import TaskDB
    db = TaskDB(db_path=tmp_db)
    task_id = db.create_task("Review PR #42", source="email", priority="HIGH")
    assert task_id > 0
    tasks = db.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Review PR #42"


def test_task_agent_complete_task(tmp_db):
    from agents.task_agent import TaskDB
    db = TaskDB(db_path=tmp_db)
    task_id = db.create_task("Write tests", source="manual", priority="MEDIUM")
    db.complete_task(task_id)
    tasks = db.list_tasks(include_done=True)
    assert tasks[0]["done"] is True


def test_task_agent_priority_filter(tmp_db):
    from agents.task_agent import TaskDB
    db = TaskDB(db_path=tmp_db)
    db.create_task("Urgent thing", source="email", priority="HIGH")
    db.create_task("Later thing", source="email", priority="LOW")
    high = db.list_tasks(priority="HIGH")
    assert len(high) == 1
    assert high[0]["priority"] == "HIGH"


# ---------------------------------------------------------------------------
# memory_agent
# ---------------------------------------------------------------------------

def test_memory_agent_add_and_search():
    mock_memory = MagicMock()
    mock_memory.search.return_value = {
        "results": [{"memory": "User prefers dark mode", "score": 0.95}]
    }
    with patch("agents.memory_agent.get_memory", return_value=mock_memory):
        from agents.memory_agent import remember, recall
        remember("User prefers dark mode", user_id="test_user")
        results = recall("preferences", user_id="test_user")
    assert "dark mode" in results[0]["memory"]


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_factory_requires_confirm_deploy():
    """Factory must return a pending state, not deploy immediately."""
    mock_crew_result = MagicMock()
    mock_crew_result.__str__ = lambda self: "class WeatherAgent: ..."

    with patch("agents.factory.factory_crew._run_planner", return_value="spec: WeatherAgent fetches weather"):
        with patch("agents.factory.factory_crew._run_coder", return_value="class WeatherAgent: ..."):
            with patch("agents.factory.factory_crew._run_tester", return_value=True):
                from agents.factory.factory_crew import plan_new_agent
                result = plan_new_agent("Build a weather agent that fetches daily forecasts")

    assert result["type"] == "factory_pending"
    assert "code" in result
    assert "preview" in result


def test_deployer_commits_file():
    mock_repo = MagicMock()
    mock_gh = MagicMock()
    mock_gh.get_repo.return_value = mock_repo
    mock_repo.get_contents.side_effect = Exception("not found")

    with patch("github.Github", return_value=mock_gh):
        with patch("utils.send_notification"):
            from agents.factory.deployer import deploy_agent_code
            result = deploy_agent_code({
                "filename": "agents/weather_agent.py",
                "code": "class WeatherAgent: pass",
                "agent_name": "WeatherAgent",
            })
    assert "WeatherAgent" in result
