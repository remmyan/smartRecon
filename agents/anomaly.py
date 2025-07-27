from crewai import Agent
from crewai.tools import BaseTool
import json
import pandas as pd

class AnomalyTool(BaseTool):
    name: str = "Anomaly Detection"
    description: str = "Detect duplicates and fraud risks."

    def _run(self, data_json: str) -> str:
        df = pd.DataFrame(json.loads(data_json))
        duplicates = df[df.duplicated(subset=['amount', 'date'], keep=False)].to_dict('records')
        high_value = df[df['amount'] > 10000].to_dict('records')  # Fraud threshold
        return json.dumps({'duplicates': duplicates, 'high_value': high_value})

def create_anomaly_detection_agent():
    return Agent(
        role="Anomaly Detection Agent",
        goal="Identify anomalies via rules",
        backstory="Rule-based detection for efficiency.",
        llm=None,  # LLM-free
        tools=[AnomalyTool()],
        verbose=True,
        max_iter=1
    )
# Export the agent for use in main crew
anomaly_detection_agent = create_anomaly_detection_agent()
