"""
Factory Tester sub-agent.
Writes pytest tests and runs them with auto-fix on failure (max 3 retries).
Adapted from _refs/crewAI-examples/crews/instagram_post/agents.py
"""
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from llm import get_llm

load_dotenv()
logger = logging.getLogger("jarvis")

_llm = get_llm(temperature=0.1)

MAX_TEST_RETRIES = 3


def make_tester_agent() -> Agent:
    return Agent(
        role="QA Engineer",
        goal="Write thorough pytest tests for CrewAI agents and ensure they pass",
        backstory=(
            "You are a senior QA engineer who writes pytest test suites. "
            "You mock all external APIs (Groq, Google, Serper). "
            "You output ONLY valid Python pytest code, no markdown fences."
        ),
        llm=_llm,
        allow_delegation=False,
        verbose=False,
    )


def make_fixer_agent() -> Agent:
    return Agent(
        role="Bug Fixer",
        goal="Fix Python code based on test failure output",
        backstory=(
            "You are an expert Python debugger. Given code and pytest failure output, "
            "you return the corrected Python code. Output ONLY valid Python, no markdown."
        ),
        llm=_llm,
        allow_delegation=False,
        verbose=False,
    )


def _run_pytest(test_code: str, agent_code: str, agent_filename: str) -> tuple[bool, str]:
    """Write code + test to temp dir and run pytest. Returns (passed, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / agent_filename).write_text(agent_code)
        test_file = tmp_path / f"test_{agent_filename}"
        test_file.write_text(test_code)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=60,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output


def write_and_run_tests(
    agent_code: str,
    agent_filename: str,
    agent_spec: str,
) -> tuple[bool, str, str]:
    """
    Write tests, run them, auto-fix up to MAX_TEST_RETRIES times.
    Returns (success, final_agent_code, test_code).
    """
    tester = make_tester_agent()
    fixer = make_fixer_agent()

    # Generate initial tests
    test_task = Task(
        description=(
            f"Write pytest tests for this agent spec:\n{agent_spec}\n\n"
            f"Agent code:\n{agent_code}\n\n"
            "Mock all external calls. Test the main run_ function."
        ),
        expected_output="Complete pytest test file as plain Python code",
        agent=tester,
    )
    crew = Crew(agents=[tester], tasks=[test_task], verbose=False)
    test_code = str(crew.kickoff()).strip()
    test_code = test_code.replace("```python", "").replace("```", "").strip()

    current_code = agent_code

    for attempt in range(MAX_TEST_RETRIES):
        passed, output = _run_pytest(test_code, current_code, agent_filename)
        if passed:
            logger.info("Tests passed on attempt %d", attempt + 1)
            return True, current_code, test_code

        logger.warning("Tests failed (attempt %d/%d):\n%s", attempt + 1, MAX_TEST_RETRIES, output[:500])

        if attempt < MAX_TEST_RETRIES - 1:
            fix_task = Task(
                description=(
                    f"Fix this Python agent code so the tests pass.\n\n"
                    f"Failing code:\n{current_code}\n\n"
                    f"Test output (failures):\n{output[:1000]}\n\n"
                    "Return ONLY the corrected Python code."
                ),
                expected_output="Corrected Python code that passes the tests",
                agent=fixer,
            )
            fix_crew = Crew(agents=[fixer], tasks=[fix_task], verbose=False)
            current_code = str(fix_crew.kickoff()).strip()
            current_code = current_code.replace("```python", "").replace("```", "").strip()

    return False, current_code, test_code
