from crewai import Agent
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_orchestration_agent():
    """Create the CrewAI Orchestration Agent for hierarchical workflow management (no tools, uses delegation)."""
    from config import Config
    
    orchestration_agent = Agent(
        role="Orchestration Agent",
        goal="Coordinate the AP reconciliation workflow by delegating tasks, managing dependencies, and monitoring progress.",
        backstory="""You are the central coordinator for the AP reconciliation system. 
        You delegate data ingestion, matching, anomaly detection, exception handling, and learning tasks 
        to specialized agents, ensuring smooth execution and handling any failures via re-delegation.""",
        tools=[],  # No tools for manager agent in hierarchical process
        llm=Config.OPENAI_MODEL,
        verbose=True,
        memory=True,
        max_iter=5,
        allow_delegation=True  # Enable delegation for task assignment
    )
    
    return orchestration_agent

# Export the agent for use in main crew
orchestration_agent = create_orchestration_agent()
