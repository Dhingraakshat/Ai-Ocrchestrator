"""
Factory Coder sub-agent.
Writes full Python CrewAI code for a new agent given its spec.
Adapted from _refs/crewAI-examples/crews/instagram_post/agents.py
"""
import os

from crewai import Agent
from dotenv import load_dotenv

from llm import get_llm

load_dotenv()

_llm = get_llm(temperature=0.1)


def make_coder_agent() -> Agent:
    return Agent(
        role="Python Agent Developer",
        goal="Write complete, production-quality Python CrewAI agent code",
        backstory=(
            "You are an expert Python developer who writes clean, well-structured "
            "CrewAI agents. You follow the project's coding conventions: "
            "use utils.with_retry for retry logic, crewai.LLM with groq model, "
            "load_dotenv(), and a run_<name>_agent() entry-point function. "
            "You output ONLY valid Python code, no markdown fences."
        ),
        llm=_llm,
        allow_delegation=False,
        verbose=False,
    )
