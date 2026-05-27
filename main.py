"""
JARVIS — FastAPI server + WebSocket handler.

ConnectionManager adapted from _refs/fastapi/docs_src/websockets_/tutorial003_py310.py
"""
import asyncio
import json
import logging
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*on_event is deprecated.*")
logging.getLogger("botocore").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("jarvis")

app = FastAPI(title="JARVIS", description="Autonomous Personal AI Assistant")

# ---------------------------------------------------------------------------
# Shared state — all agents read/write this dict
# ---------------------------------------------------------------------------
jarvis_state: dict = {
    "emails": [],
    "calendar": [],
    "news": [],
    "tasks": [],
    "agents": {
        "EmailAgent": {"status": "idle", "last_run": "", "preview": ""},
        "CalendarAgent": {"status": "idle", "last_run": "", "preview": ""},
        "NewsAgent": {"status": "idle", "last_run": "", "preview": ""},
        "TaskAgent": {"status": "idle", "last_run": "", "preview": ""},
        "MemoryAgent": {"status": "idle", "last_run": "", "preview": ""},
        "OrchestratorAgent": {"status": "idle", "last_run": "", "preview": ""},
        "AgentFactory": {"status": "idle", "last_run": "", "preview": ""},
    },
    "deploys": [],
}

# factory_pending: client_id -> payload dict from factory_crew
factory_pending: dict[str, dict] = {}

# chat_histories: client_id -> list of {role, content} dicts (last 20 messages)
chat_histories: dict[str, list] = {}


# ---------------------------------------------------------------------------
# ConnectionManager (from ref: websockets_/tutorial003_py310.py)
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        await websocket.send_text(message)

    async def broadcast(self, message: str) -> None:
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_text(message)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.active_connections.remove(d)


manager = ConnectionManager()
_main_loop: asyncio.AbstractEventLoop | None = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup() -> None:
    global _main_loop
    _main_loop = asyncio.get_event_loop()

    # Start scheduler in a daemon thread
    from scheduler import start_scheduler
    t = threading.Thread(
        target=start_scheduler,
        kwargs={"loop": _main_loop, "manager": manager, "state": jarvis_state},
        daemon=True,
        name="jarvis-scheduler",
    )
    t.start()
    logger.info("JARVIS started. Dashboard: http://localhost:8000")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#0d1117"/>'
    '<text x="16" y="23" font-size="20" text-anchor="middle" font-family="monospace" fill="#58a6ff">J</text>'
    '</svg>'
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/state")
async def get_state() -> dict:
    return jarvis_state


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "agents": len(jarvis_state["agents"])}


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def devtools_json() -> dict:
    return {}


@app.post("/api/trigger/all")
async def trigger_all_agents() -> dict:
    agents = ["EmailAgent", "CalendarAgent", "NewsAgent", "TaskAgent"]
    try:
        from orchestrator import run_agent
        for name in agents:
            asyncio.create_task(run_agent(name, manager, jarvis_state))
        return {"status": "triggered", "agents": agents}
    except Exception as exc:
        logger.error("Failed to trigger all agents: %s", exc)
        return {"status": "error", "detail": str(exc)}


@app.post("/api/trigger/{agent_name}")
async def trigger_agent(agent_name: str) -> dict:
    try:
        from orchestrator import run_agent
    except Exception as exc:
        logger.error("Import error in orchestrator: %s", exc)
        return {"status": "error", "agent": agent_name, "detail": str(exc)}
    try:
        asyncio.create_task(run_agent(agent_name, manager, jarvis_state))
        return {"status": "triggered", "agent": agent_name}
    except Exception as exc:
        logger.error("Failed to schedule %s: %s", agent_name, exc)
        return {"status": "error", "agent": agent_name, "detail": str(exc)}


# ---------------------------------------------------------------------------
# WebSocket chat endpoint (adapted from ref: tutorial003_py310.py)
# ---------------------------------------------------------------------------
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await manager.connect(websocket)
    logger.info("Client connected: %s", client_id)
    history = chat_histories.setdefault(client_id, [])

    # Send current state immediately on connect
    await manager.send_personal_message(
        json.dumps({"type": "state_update", "data": jarvis_state}), websocket
    )

    try:
        while True:
            raw = await websocket.receive_text()
            # Support both plain text and JSON {"text": "...", "agent": "..."}
            try:
                msg_json = json.loads(raw)
                data = msg_json.get("text", "").strip()
                force_agent = msg_json.get("agent", "Auto")
            except (json.JSONDecodeError, TypeError):
                data = raw.strip()
                force_agent = "Auto"

            # --- confirm deploy flow ---
            if data.lower() == "confirm deploy" and client_id in factory_pending:
                payload = factory_pending.pop(client_id)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "chat",
                        "role": "assistant",
                        "content": f"Deploying **{payload.get('agent_name')}** to GitHub...",
                    }),
                    websocket,
                )
                from agents.factory.deployer import deploy_agent_code
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, deploy_agent_code, payload)

                # Register the new agent in state
                agent_name = payload.get("agent_name", "NewAgent")
                jarvis_state["agents"][agent_name] = {
                    "status": "deployed",
                    "last_run": "",
                    "preview": f"Deployed via AgentFactory",
                }
                await manager.send_personal_message(
                    json.dumps({"type": "chat", "role": "assistant", "content": result}),
                    websocket,
                )
                await manager.broadcast(json.dumps({"type": "state_update", "data": jarvis_state}))
                continue

            # --- cancel factory pending ---
            if client_id in factory_pending and data.lower() != "confirm deploy":
                factory_pending.pop(client_id, None)

            # --- echo user message ---
            await manager.send_personal_message(
                json.dumps({"type": "chat", "role": "user", "content": data}),
                websocket,
            )

            # --- append to history ---
            history.append({"role": "user", "content": data})

            # --- route through orchestrator ---
            from orchestrator import route_message
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, route_message, data, client_id, jarvis_state, force_agent, history
            )

            if isinstance(response, dict) and response.get("type") == "factory_pending":
                factory_pending[client_id] = response
                content = response["preview"] + "\n\nType **confirm deploy** to commit to GitHub, or send any other message to cancel."
            else:
                content = str(response)

            # --- append assistant reply to history (cap at 20 messages) ---
            history.append({"role": "assistant", "content": content})
            if len(history) > 20:
                del history[:-20]

            await manager.send_personal_message(
                json.dumps({"type": "chat", "role": "assistant", "content": content}),
                websocket,
            )
            # Broadcast updated state to all connected clients
            await manager.broadcast(json.dumps({"type": "state_update", "data": jarvis_state}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        chat_histories.pop(client_id, None)
        logger.info("Client disconnected: %s", client_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
