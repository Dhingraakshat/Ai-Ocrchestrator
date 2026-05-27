"""
AgentFactory — builds new agents from plain-English descriptions.

Pipeline:
  1. Planner: call_groq → structured spec
  2. Coder:   call_groq → Python code
  3. Returns pending payload — user must type "confirm deploy" to commit via Deployer
"""
import logging
import re

from dotenv import load_dotenv

from llm import call_groq
from utils import send_notification, with_retry

_KNOWN_APIS = {
    "OPENAI_API_KEY":    ("OpenAI", "https://platform.openai.com/api-keys"),
    "ANTHROPIC_API_KEY": ("Anthropic Claude", "https://console.anthropic.com/"),
    "TWITTER_API_KEY":   ("Twitter/X Developer", "https://developer.twitter.com/"),
    "TWITTER_BEARER_TOKEN": ("Twitter/X Developer", "https://developer.twitter.com/"),
    "REDDIT_CLIENT_ID":  ("Reddit API", "https://www.reddit.com/prefs/apps"),
    "REDDIT_CLIENT_SECRET": ("Reddit API", "https://www.reddit.com/prefs/apps"),
    "SLACK_BOT_TOKEN":   ("Slack API", "https://api.slack.com/apps"),
    "TELEGRAM_BOT_TOKEN": ("Telegram BotFather", "https://t.me/BotFather"),
    "NOTION_API_KEY":    ("Notion Integrations", "https://www.notion.so/my-integrations"),
    "SPOTIFY_CLIENT_ID": ("Spotify Developer", "https://developer.spotify.com/dashboard"),
    "SPOTIFY_CLIENT_SECRET": ("Spotify Developer", "https://developer.spotify.com/dashboard"),
    "WEATHER_API_KEY":   ("OpenWeatherMap", "https://home.openweathermap.org/api_keys"),
    "ALPHA_VANTAGE_KEY": ("Alpha Vantage (stocks)", "https://www.alphavantage.co/support/#api-key"),
    "NEWS_API_KEY":      ("NewsAPI", "https://newsapi.org/register"),
    "AIRTABLE_API_KEY":  ("Airtable", "https://airtable.com/account"),
    "DISCORD_BOT_TOKEN": ("Discord Developer Portal", "https://discord.com/developers/applications"),
    "SENDGRID_API_KEY":  ("SendGrid Email", "https://app.sendgrid.com/settings/api_keys"),
    "STRIPE_API_KEY":    ("Stripe Dashboard", "https://dashboard.stripe.com/apikeys"),
    "GITHUB_TOKEN":      ("GitHub Personal Access Token", "https://github.com/settings/tokens"),
}

load_dotenv()
logger = logging.getLogger("jarvis")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_")


def _plan(description: str) -> str:
    system = (
        "You are a senior AI engineer who designs Python agent specifications. "
        "Given a requirement, output a structured spec with these fields:\n"
        "AgentName: <PascalCase name>\n"
        "Role: <one line>\n"
        "Goal: <one line>\n"
        "Backstory: <2-3 sentences>\n"
        "Tools: <comma-separated list or 'none'>\n"
        "InputFormat: <description>\n"
        "OutputFormat: <description>\n"
        "Filename: agents/<snake_case_name>.py"
    )
    return call_groq(system=system, user=f"Design an agent for: {description}", temperature=0.2)


def _code(spec: str) -> str:
    system = (
        "You are an expert Python developer writing a JARVIS agent module. "
        "Follow these conventions exactly:\n"
        "- Use 'from llm import call_groq' for all LLM calls\n"
        "- Use 'from utils import with_retry' for retry logic\n"
        "- Use python-dotenv load_dotenv()\n"
        "- Expose a run_<agent_name>_agent(query='') entry-point function\n"
        "- Use logging.getLogger('jarvis')\n"
        "- Output ONLY valid Python code, no markdown fences, no explanations"
    )
    user = f"Write the complete Python agent module for this spec:\n\n{spec}"
    raw = call_groq(system=system, user=user, temperature=0.1, max_tokens=3000)
    return raw.replace("```python", "").replace("```", "").strip()


@with_retry(max_attempts=2, agent_name="AgentFactory")
def plan_new_agent(description: str) -> dict:
    logger.info("AgentFactory: planning agent for: %s", description[:80])

    spec = _plan(description)
    logger.info("Spec generated (%d chars)", len(spec))

    code = _code(spec)
    logger.info("Code generated (%d chars)", len(code))

    name_match = re.search(r"AgentName[:\s]+(\w+)", spec)
    agent_name = name_match.group(1) if name_match else "GeneratedAgent"
    filename = f"agents/{_slugify(agent_name)}.py"

    # Detect any API keys the generated code needs
    needed_keys = [key for key in _KNOWN_APIS if key in code]
    import os as _os
    missing_keys = [k for k in needed_keys if not _os.getenv(k)]

    api_guidance = ""
    if missing_keys:
        lines = ["**Required API keys not found in your .env:**\n"]
        for key in missing_keys:
            name, url = _KNOWN_APIS[key]
            lines.append(f"- `{key}` — get it at {url}")
        lines.append(f"\nAdd them to your `.env` file as:\n```\n" + "\n".join(f"{k}=your_key_here" for k in missing_keys) + "\n```")
        api_guidance = "\n\n" + "\n".join(lines)

    preview = (
        f"**{agent_name}** — ready to deploy\n\n"
        f"File: `{filename}`{api_guidance}\n\n"
        f"```python\n{code[:800]}{'...' if len(code) > 800 else ''}\n```"
    )

    send_notification("JARVIS — AgentFactory", f"Agent {agent_name} ready for review")

    return {
        "type": "factory_pending",
        "agent_name": agent_name,
        "filename": filename,
        "code": code,
        "test_code": "",
        "spec": spec,
        "tests_passed": False,
        "preview": preview,
    }
