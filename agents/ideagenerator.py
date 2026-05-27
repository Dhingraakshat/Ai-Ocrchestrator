import json
import logging
from llm import call_groq
from utils import with_retry
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger('jarvis')

def generate_agent_concept(prompt):
    try:
        response = with_retry(call_groq, prompt)
        return response
    except Exception as e:
        logger.error(f"Error generating agent concept: {e}")
        return None

def create_agent_specification(concept):
    agent_spec = {
        "AgentName": concept["name"],
        "Role": concept["role"],
        "Goal": concept["goal"],
        "Backstory": concept["backstory"],
        "Tools": concept["tools"],
        "InputFormat": concept["input_format"],
        "OutputFormat": concept["output_format"]
    }
    return agent_spec

def run_idea_generator_agent(query=''):
    try:
        prompt = f"Generate a novel agent concept based on the following prompt: {query}"
        concept = generate_agent_concept(prompt)
        if concept:
            agent_spec = create_agent_specification(concept)
            return json.dumps(agent_spec, indent=4)
        else:
            return None
    except Exception as e:
        logger.error(f"Error running IdeaGenerator agent: {e}")
        return None

if __name__ == "__main__":
    query = os.getenv("QUERY", "")
    result = run_idea_generator_agent(query)
    if result:
        print(result)