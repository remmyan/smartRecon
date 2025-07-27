from crewai import Agent
from config import Config

def create_orchestration_agent():
    return Agent(
        role="Orchestration Agent",
        goal="Coordinate agents and delegate tasks",
        backstory="Manage workflow with efficient delegation.",
        llm=Config.get_llm(model="gpt-4", temperature=0.4),  # Keep GPT-4 for complexity
        verbose=True,
        max_iter=3
    )
# Export the agent for use in main crew
orchestration_agent = create_orchestration_agent()
