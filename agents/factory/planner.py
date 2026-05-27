"""
Factory Planner sub-agent.
Given a plain-English description, outputs a structured agent spec.
Adapted from _refs/crewAI-examples/crews/instagram_post/agents.py
"""
import os

from crewai import Agent
from dotenv import load_dotenv

from llm import get_llm

load_dotenv()

_llm = get_llm(temperature=0.2)


def make_planner_agent() -> Agent:
    return Agent(
        role="Agent Architect",
        goal="Design a precise CrewAI agent specification from a plain-English description",
        backstory=(
            "You are a senior AI engineer who specializes in designing CrewAI agents. "
            "Given a requirement, you produce a detailed spec: agent role, goal, backstory, "
            "tools needed, inputs, outputs, and file name."
        ),
        llm=_llm,
        allow_delegation=False,
        verbose=False,
    )
