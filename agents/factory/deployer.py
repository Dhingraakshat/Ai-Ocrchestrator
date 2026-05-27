"""
Factory Deployer sub-agent.
Commits generated agent code to GitHub via PyGithub and sends desktop notification.
"""
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from utils import send_notification

load_dotenv()
logger = logging.getLogger("jarvis")


def deploy_agent_code(payload: dict) -> str:
    """
    Commit agent code to GitHub.

    payload keys:
      - filename: str  (e.g. "agents/weather_agent.py")
      - code: str
      - agent_name: str
      - test_code: str (optional)
    """
    import github

    token = os.getenv("GITHUB_TOKEN", "")
    repo_name = os.getenv("GITHUB_REPO", "")

    if not token or not repo_name:
        return "GITHUB_TOKEN or GITHUB_REPO not configured. Set them in .env."

    filename = payload.get("filename", f"agents/{payload.get('agent_name', 'new_agent').lower()}.py")
    code = payload.get("code", "")
    agent_name = payload.get("agent_name", "NewAgent")
    test_code = payload.get("test_code", "")

    try:
        gh = github.Github(token)
        repo = gh.get_repo(repo_name)

        commit_msg = f"feat: add {agent_name} via AgentFactory [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"

        # Commit agent file
        try:
            existing = repo.get_contents(filename)
            repo.update_file(filename, commit_msg, code, existing.sha)
        except Exception:
            repo.create_file(filename, commit_msg, code)

        # Commit test file if provided
        if test_code:
            test_filename = f"tests/test_{filename.split('/')[-1]}"
            try:
                existing_test = repo.get_contents(test_filename)
                repo.update_file(test_filename, commit_msg, test_code, existing_test.sha)
            except Exception:
                repo.create_file(test_filename, commit_msg, test_code)

        send_notification(
            "JARVIS — Agent Deployed!",
            f"{agent_name} committed to {repo_name} and CI/CD triggered.",
        )

        return (
            f"✅ **{agent_name}** deployed to `{repo_name}/{filename}`.\n"
            f"GitHub Actions CI/CD pipeline has been triggered."
        )

    except Exception as exc:
        logger.error("Deployment failed: %s", exc)
        send_notification("JARVIS — Deploy Failed", str(exc)[:150])
        return f"❌ Deployment failed: {exc}"
