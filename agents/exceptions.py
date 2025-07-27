from crewai import Agent
from crewai.tools import BaseTool
import json

class PriorityTool(BaseTool):
    name: str = "Exception Prioritization"
    description: str = "Prioritize exceptions by rules."

    def _run(self, exceptions_json: str) -> str:
        exceptions = json.loads(exceptions_json)
        for ex in exceptions:
            ex['priority'] = 'High' if ex['amount'] > 10000 else 'Low'
        return json.dumps(exceptions)

def create_exception_management_agent():
    return Agent(
        role="Exception Management Agent",
        goal="Prioritize and route exceptions",
        backstory="Use predefined rules for sorting.",
        llm=None,  # LLM-free
        tools=[PriorityTool()],
        verbose=True,
        max_iter=1
    )
# Export the agent for use in main crew
exception_management_agent = create_exception_management_agent()
