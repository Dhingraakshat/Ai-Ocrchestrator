"""
JARVIS Scheduler — all timed jobs using the 'schedule' library.

Adapted from _refs/schedule/README.rst (schedule.every().day.at().do() pattern)

Runs in a daemon thread started from main.py startup.
Uses asyncio.run_coroutine_threadsafe to broadcast from thread to WebSocket.
"""
import asyncio
import json
import logging
import time

import schedule
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("jarvis")


def _fetch_github_deploys(state: dict) -> None:
    """Pull recent GitHub Actions workflow runs for the deploys panel."""
    import os
    import requests

    token = os.getenv("GITHUB_TOKEN", "")
    repo = os.getenv("GITHUB_REPO", "")
    if not token or not repo:
        return

    try:
        url = f"https://api.github.com/repos/{repo}/actions/runs?per_page=10"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        runs = resp.json().get("workflow_runs", [])
        state["deploys"] = [
            {
                "id": r["id"],
                "name": r["name"],
                "status": r["status"],
                "conclusion": r.get("conclusion", ""),
                "branch": r["head_branch"],
                "commit": r["head_commit"]["message"][:60] if r.get("head_commit") else "",
                "created_at": r["created_at"],
                "html_url": r["html_url"],
            }
            for r in runs
        ]
    except Exception as exc:
        logger.warning("GitHub deploys fetch failed: %s", exc)


def _build_morning_digest(state: dict, loop: asyncio.AbstractEventLoop, manager) -> None:
    """08:00 daily: run Email + Calendar + News → update dashboard."""
    logger.info("Running morning digest...")

    async def _run():
        from orchestrator import run_agent
        for agent in ("EmailAgent", "CalendarAgent", "NewsAgent"):
            await run_agent(agent, manager, state)
        _fetch_github_deploys(state)
        await manager.broadcast(json.dumps({"type": "state_update", "data": state}))
        from utils import send_notification
        send_notification("JARVIS Morning Digest", "Email, Calendar & News updated!")

    asyncio.run_coroutine_threadsafe(_run(), loop)


def _refresh_emails(state: dict, loop: asyncio.AbstractEventLoop, manager) -> None:
    """Every 60 min: check for new urgent emails."""
    logger.info("Scheduled email refresh...")

    async def _run():
        from orchestrator import run_agent
        prev_count = len(state.get("emails", []))
        await run_agent("EmailAgent", manager, state)
        new_emails = [e for e in state.get("emails", []) if e.get("priority") == "HIGH"]
        if new_emails and len(state.get("emails", [])) != prev_count:
            from utils import send_notification
            send_notification("JARVIS — Urgent Email", f"{len(new_emails)} high-priority email(s)")

    asyncio.run_coroutine_threadsafe(_run(), loop)


def _refresh_deploys(state: dict, loop: asyncio.AbstractEventLoop, manager) -> None:
    """Every 15 min: refresh GitHub deploy history."""

    async def _run():
        _fetch_github_deploys(state)
        await manager.broadcast(json.dumps({"type": "state_update", "data": state}))

    asyncio.run_coroutine_threadsafe(_run(), loop)


def start_scheduler(loop: asyncio.AbstractEventLoop, manager, state: dict) -> None:
    """
    Entry point called from main.py startup in a daemon thread.
    Sets up all jobs and runs the schedule loop.
    """
    # 08:00 daily morning digest
    schedule.every().day.at("08:00").do(
        _build_morning_digest, state=state, loop=loop, manager=manager
    )

    # Every 60 minutes: email refresh
    schedule.every(60).minutes.do(
        _refresh_emails, state=state, loop=loop, manager=manager
    )

    # Every 15 minutes: GitHub deploy history
    schedule.every(15).minutes.do(
        _refresh_deploys, state=state, loop=loop, manager=manager
    )

    logger.info("Scheduler started. Jobs: %s", [str(j) for j in schedule.jobs])

    while True:
        schedule.run_pending()
        time.sleep(30)
