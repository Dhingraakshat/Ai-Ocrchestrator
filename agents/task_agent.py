"""
TaskAgent — creates and manages tasks in a local SQLite DB.
LLM calls use llm.call_groq() directly.
"""
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm import call_groq
from utils import with_retry

load_dotenv()
logger = logging.getLogger("jarvis")

DEFAULT_DB = str(Path(__file__).parent.parent / "jarvis_tasks.db")


class TaskDB:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    title    TEXT NOT NULL,
                    source   TEXT DEFAULT 'manual',
                    priority TEXT DEFAULT 'MEDIUM',
                    done     INTEGER DEFAULT 0,
                    created  TEXT NOT NULL,
                    deadline TEXT
                )"""
            )

    def create_task(self, title: str, source: str = "manual", priority: str = "MEDIUM", deadline: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO tasks (title, source, priority, done, created, deadline) VALUES (?, ?, ?, 0, ?, ?)",
                (title, source, priority, datetime.now().isoformat(timespec="seconds"), deadline),
            )
            return cur.lastrowid

    def complete_task(self, task_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))

    def list_tasks(self, include_done: bool = False, priority: str | None = None) -> list[dict]:
        clauses = [] if include_done else ["done = 0"]
        params: list = []
        if priority:
            clauses.append("priority = ?")
            params.append(priority)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY CASE priority WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END, created DESC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]


_db = TaskDB()


@with_retry(max_attempts=3, agent_name="TaskAgent")
def run_task_agent(query: str = "", email_text: str = "") -> str | list[dict]:
    if query.lower().startswith(("list", "show")):
        tasks = _db.list_tasks()
        return tasks if tasks else "No pending tasks."

    source_text = email_text or query
    if not source_text:
        return _db.list_tasks()

    system = (
        "You are a task extraction specialist. Extract concrete action items from text. "
        'Return ONLY a JSON array: [{"title": "...", "priority": "HIGH|MEDIUM|LOW", "deadline": null}]'
    )
    raw = call_groq(system=system, user=f"Extract tasks from:\n{source_text}")

    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        items = json.loads(match.group()) if match else []
        created = []
        for item in items:
            task_id = _db.create_task(
                title=item.get("title", "Unnamed task"),
                source="email" if email_text else "chat",
                priority=item.get("priority", "MEDIUM"),
                deadline=item.get("deadline"),
            )
            created.append(f"[{item.get('priority','MEDIUM')}] #{task_id}: {item.get('title')}")
        return "Created tasks:\n" + "\n".join(created) if created else "No action items found."
    except Exception as exc:
        logger.warning("Task parsing failed: %s", exc)
        return raw
