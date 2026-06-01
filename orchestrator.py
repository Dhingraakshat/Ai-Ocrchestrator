"""
OrchestratorAgent — routes user chat messages to the right agent.
LLM fallback uses llm.call_groq() directly (no litellm cache_breakpoint).
"""
import asyncio
import json
import logging
import os
import re

from dotenv import load_dotenv

from llm import call_groq, call_groq_with_history
from utils import agent_status_entry, send_notification

load_dotenv()
logger = logging.getLogger("jarvis")

_EMAIL_WORDS = {"email", "gmail", "inbox", "mail", "unread", "message"}
_CALENDAR_READ_WORDS = {"calendar", "meeting", "event", "appointment", "schedule"}
_CALENDAR_CREATE_WORDS = {"add", "create", "schedule", "book", "set up", "remind"}
_NEWS_WORDS = {"news", "latest", "headline", "article", "crypto", "startup", "startups"}
_TASK_WORDS = {"task", "todo", "list tasks", "show tasks", "complete"}
_MEMORY_WORDS = {"remember", "memory", "recall", "preference", "forget"}
_FACTORY_WORDS = {"build agent", "create agent", "new agent", "make agent", "deploy agent"}
_RUN_ALL_PHRASES = {"run all", "start all", "wake up", "launch all", "trigger all", "all agents"}

# Time pattern: "1 pm", "2:30", "noon", "midnight", "tomorrow", "monday" etc.
_TIME_RE = re.compile(r'\b(\d{1,2}(:\d{2})?\s*(am|pm)|noon|midnight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I)


def _matches(text: str, keywords: set[str]) -> bool:
    """Word-boundary aware matching — prevents 'ai' matching inside 'haircut'."""
    tl = text.lower()
    word_set = set(re.findall(r'\b\w+\b', tl))
    for kw in keywords:
        if ' ' in kw:          # phrase — substring match
            if kw in tl:
                return True
        else:                  # single word — whole-word match only
            if kw in word_set:
                return True
    return False


def _is_calendar_create(text: str) -> bool:
    """Detect intent to ADD something to calendar (vs just reading it)."""
    return _matches(text, _CALENDAR_CREATE_WORDS) and bool(_TIME_RE.search(text))


def route_message(message: str, client_id: str, state: dict, force_agent: str = "Auto", history: list | None = None) -> str | dict:
    def _update(name: str, status: str, preview: str = ""):
        state["agents"][name] = agent_status_entry(status, preview)

    try:
        # ── Run all agents ──
        if _matches(message, _RUN_ALL_PHRASES):
            _update("EmailAgent", "running")
            _update("CalendarAgent", "running")
            _update("NewsAgent", "running")
            _update("TaskAgent", "running")
            import threading
            def _run_all():
                from agents.email_agent import run_email_agent
                from agents.calendar_agent import run_calendar_agent
                from agents.news_agent import run_news_agent
                from agents.task_agent import _db
                try:
                    result = run_email_agent()
                    if isinstance(result, list): state["emails"] = result
                    state["agents"]["EmailAgent"] = agent_status_entry("done", f"{len(result) if isinstance(result, list) else 0} emails")
                except Exception as e:
                    state["agents"]["EmailAgent"] = agent_status_entry("error", str(e)[:60])
                try:
                    result = run_calendar_agent()
                    if isinstance(result, list): state["calendar"] = result
                    state["agents"]["CalendarAgent"] = agent_status_entry("done", f"{len(result) if isinstance(result, list) else 0} events")
                except Exception as e:
                    state["agents"]["CalendarAgent"] = agent_status_entry("error", str(e)[:60])
                try:
                    result = run_news_agent()
                    if isinstance(result, list): state["news"] = result
                    state["agents"]["NewsAgent"] = agent_status_entry("done", f"{len(result) if isinstance(result, list) else 0} articles")
                except Exception as e:
                    state["agents"]["NewsAgent"] = agent_status_entry("error", str(e)[:60])
                try:
                    tasks = _db.list_tasks()
                    state["tasks"] = tasks
                    state["agents"]["TaskAgent"] = agent_status_entry("done", f"{len(tasks)} tasks")
                except Exception as e:
                    state["agents"]["TaskAgent"] = agent_status_entry("error", str(e)[:60])
            threading.Thread(target=_run_all, daemon=True).start()
            return "Right away sir. All agents are now running — the dashboard will update shortly."

        # ── Force-route to a specific agent (overrides keyword detection) ──
        if force_agent and force_agent != "Auto":
            if force_agent == "EmailAgent":
                _update("EmailAgent", "running")
                from agents.email_agent import run_email_agent
                result = run_email_agent(query=message)
                if isinstance(result, list):
                    state["emails"] = result
                    _update("EmailAgent", "done", f"{len(result)} emails")
                    return f"Fetched {len(result)} emails. Dashboard updated."
                _update("EmailAgent", "done", str(result)[:80])
                return str(result)

            if force_agent == "CalendarAgent":
                _update("CalendarAgent", "running")
                if _is_calendar_create(message):
                    from agents.calendar_agent import create_calendar_event
                    result = create_calendar_event(description=message)
                else:
                    from agents.calendar_agent import run_calendar_agent
                    result = run_calendar_agent(query=message)
                    if isinstance(result, list):
                        state["calendar"] = result
                        _update("CalendarAgent", "done", f"{len(result)} events")
                        return f"Fetched {len(result)} calendar items. Dashboard updated."
                _update("CalendarAgent", "done", str(result)[:80])
                return str(result)

            if force_agent == "NewsAgent":
                _update("NewsAgent", "running")
                from agents.news_agent import run_news_agent
                result = run_news_agent(query=message)
                if isinstance(result, list):
                    state["news"] = result
                    _update("NewsAgent", "done", f"{len(result)} articles")
                    return f"Fetched {len(result)} news articles. Dashboard updated."
                _update("NewsAgent", "done", str(result)[:80])
                return str(result)

            if force_agent == "TaskAgent":
                _update("TaskAgent", "running")
                from agents.task_agent import run_task_agent
                result = run_task_agent(query=message)
                if isinstance(result, list):
                    state["tasks"] = result
                    _update("TaskAgent", "done", f"{len(result)} tasks")
                    return f"Found {len(result)} tasks. Dashboard updated."
                _update("TaskAgent", "done", str(result)[:80])
                return str(result)

            if force_agent == "MemoryAgent":
                _update("MemoryAgent", "running")
                from agents.memory_agent import run_memory_agent
                lower = message.lower()
                if lower.startswith("remember ") or "remember that" in lower:
                    text = message.split(" ", 1)[1] if " " in message else message
                    result = run_memory_agent(text_to_remember=text)
                else:
                    result = run_memory_agent(query=message)
                _update("MemoryAgent", "done", str(result)[:80])
                return str(result)

            if force_agent == "AgentFactory":
                _update("AgentFactory", "running")
                from agents.factory.factory_crew import plan_new_agent
                result = plan_new_agent(description=message)
                _update("AgentFactory", "done", result.get("agent_name", ""))
                return result

        if _matches(message, _FACTORY_WORDS):
            _update("AgentFactory", "running")
            from agents.factory.factory_crew import plan_new_agent
            result = plan_new_agent(description=message)
            _update("AgentFactory", "done", result.get("agent_name", ""))
            return result

        if _matches(message, _EMAIL_WORDS):
            _update("EmailAgent", "running")
            from agents.email_agent import run_email_agent
            result = run_email_agent(query=message)
            if isinstance(result, list):
                state["emails"] = result
                _update("EmailAgent", "done", f"{len(result)} emails")
                return f"Fetched {len(result)} emails. Dashboard updated."
            _update("EmailAgent", "done", str(result)[:80])
            return str(result)

        if _is_calendar_create(message):
            _update("CalendarAgent", "running")
            from agents.calendar_agent import create_calendar_event
            result = create_calendar_event(description=message)
            _update("CalendarAgent", "done", result[:80])
            return result

        if _matches(message, _CALENDAR_READ_WORDS):
            _update("CalendarAgent", "running")
            from agents.calendar_agent import run_calendar_agent
            result = run_calendar_agent(query=message)
            if isinstance(result, list):
                state["calendar"] = result
                _update("CalendarAgent", "done", f"{len(result)} events")
                return f"Fetched {len(result)} calendar items. Dashboard updated."
            _update("CalendarAgent", "done", str(result)[:80])
            return str(result)

        if _matches(message, _NEWS_WORDS):
            _update("NewsAgent", "running")
            from agents.news_agent import run_news_agent
            result = run_news_agent(query=message)
            if isinstance(result, list):
                state["news"] = result
                _update("NewsAgent", "done", f"{len(result)} articles")
                return f"Fetched {len(result)} news articles. Dashboard updated."
            _update("NewsAgent", "done", str(result)[:80])
            return str(result)

        if _matches(message, _TASK_WORDS):
            _update("TaskAgent", "running")
            from agents.task_agent import run_task_agent
            result = run_task_agent(query=message)
            if isinstance(result, list):
                state["tasks"] = result
                _update("TaskAgent", "done", f"{len(result)} tasks")
                return f"Found {len(result)} tasks. Dashboard updated."
            _update("TaskAgent", "done", str(result)[:80])
            return str(result)

        if _matches(message, _MEMORY_WORDS):
            _update("MemoryAgent", "running")
            from agents.memory_agent import run_memory_agent
            lower = message.lower()
            if lower.startswith("remember ") or "remember that" in lower:
                text = message.split(" ", 1)[1] if " " in message else message
                result = run_memory_agent(text_to_remember=text)
            else:
                result = run_memory_agent(query=message)
            _update("MemoryAgent", "done", str(result)[:80])
            return str(result)

        # General fallback via direct Groq call
        _update("OrchestratorAgent", "running")

        _AGENT_DESCRIPTIONS = {
            "EmailAgent":        "reads Gmail, extracts urgent emails and action items",
            "CalendarAgent":     "reads and creates Google Calendar events, detects conflicts",
            "NewsAgent":         "fetches latest tech/AI/crypto/startup news via Serper",
            "TaskAgent":         "manages a to-do list stored in SQLite",
            "MemoryAgent":       "stores and recalls user preferences and facts using Mem0",
            "OrchestratorAgent": "that's you — JARVIS core, handles general chat and routing",
            "AgentFactory":      "builds and deploys new AI agents from plain-English descriptions",
        }

        agents_info = state.get("agents", {})
        agent_status_lines = []
        for name, desc in _AGENT_DESCRIPTIONS.items():
            info = agents_info.get(name, {})
            status = info.get("status", "idle")
            preview = info.get("preview", "")
            last_run = info.get("last_run", "")
            line = f"- {name} ({desc}): {status}"
            if preview:
                line += f" — {preview}"
            if last_run:
                line += f" [last run: {last_run}]"
            agent_status_lines.append(line)

        agent_context = "\n".join(agent_status_lines)

        system = (
            "You are JARVIS, a world-class personal AI assistant modelled after Tony Stark's JARVIS. "
            "Always address the user as 'sir' without inserting a comma before it. Use forms like 'Yes sir' or 'Shall I do that sir?' rather than 'Yes, sir'. "
            "Be sharp, witty, and confident — never robotic or generic. "
            "Keep answers to one or two sentences unless detail is explicitly requested. "
            "Before taking any significant action, ask permission: 'Shall I do that sir?' "
            "You have the following agents at your disposal:\n"
            f"{agent_context}\n\n"
            "Never refer to yourself as 'OrchestratorAgent'. You are JARVIS."
        )
        # Use full conversation history so JARVIS has context across messages
        conv = list(history) if history else [{"role": "user", "content": message}]
        result = call_groq_with_history(system=system, history=conv, temperature=0.3)
        _update("OrchestratorAgent", "done")
        return result

    except Exception as exc:
        logger.error("Orchestrator error: %s", exc)
        send_notification("JARVIS Error", str(exc)[:150])
        return f"Sorry, I encountered an error: {exc}"


async def run_agent(agent_name: str, manager, state: dict) -> None:
    state["agents"][agent_name] = agent_status_entry("running")
    await manager.broadcast(json.dumps({"type": "state_update", "data": state}))

    loop = asyncio.get_event_loop()
    try:
        if agent_name == "EmailAgent":
            from agents.email_agent import run_email_agent
            result = await loop.run_in_executor(None, run_email_agent)
            if isinstance(result, list):
                state["emails"] = result
            state["agents"][agent_name] = agent_status_entry("done", f"{len(result) if isinstance(result, list) else 0} emails")

        elif agent_name == "CalendarAgent":
            from agents.calendar_agent import run_calendar_agent
            result = await loop.run_in_executor(None, run_calendar_agent)
            if isinstance(result, list):
                state["calendar"] = result
            state["agents"][agent_name] = agent_status_entry("done")

        elif agent_name == "NewsAgent":
            from agents.news_agent import run_news_agent
            result = await loop.run_in_executor(None, run_news_agent)
            if isinstance(result, list):
                state["news"] = result
            state["agents"][agent_name] = agent_status_entry("done", f"{len(result) if isinstance(result, list) else 0} articles")

        elif agent_name == "TaskAgent":
            from agents.task_agent import _db
            tasks = _db.list_tasks()
            state["tasks"] = tasks
            state["agents"][agent_name] = agent_status_entry("done", f"{len(tasks)} tasks")

        elif agent_name == "MemoryAgent":
            state["agents"][agent_name] = agent_status_entry("idle", "Use chat to remember/recall")

        elif agent_name == "AgentFactory":
            state["agents"][agent_name] = agent_status_entry("idle", "Use chat: 'build agent …'")

        else:
            state["agents"].setdefault(agent_name, {})["status"] = "idle"

    except Exception as exc:
        logger.error("Agent %s failed: %s", agent_name, exc)
        state["agents"][agent_name] = agent_status_entry("error", str(exc)[:80])
        send_notification(f"JARVIS — {agent_name} Failed", str(exc)[:150])

    await manager.broadcast(json.dumps({"type": "state_update", "data": state}))
