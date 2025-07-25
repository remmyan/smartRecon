from crewai import Agent
from crewai.tools import BaseTool
from typing import List, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = Config.OPENAI_MODEL

class PrioritizeExceptionsTool(BaseTool):
    name: str = "Prioritize Exceptions"
    description: str = "Prioritize exceptions based on amount, confidence, and type."

    def _run(self, exceptions_json: str) -> str:
        try:
            exceptions = json.loads(exceptions_json)
            prioritized_results = []

            for ex in exceptions:
                priority = 1  # Default: Low
                risk_score = 0
                tags = []

                amount = float(str(ex.get("amount", "0")).replace("$", "").replace(",", ""))
                if amount > 10000:
                    risk_score += 3
                    tags.append("High Amount")
                elif amount > 1000:
                    risk_score += 2
                else:
                    risk_score += 1

                confidence = float(ex.get("confidence", 1.0))
                if confidence < 0.5:
                    risk_score += 2
                    tags.append("Low Confidence")
                elif confidence < 0.7:
                    risk_score += 1

                ex_type = ex.get("type", "").lower()
                if "fraud" in ex_type or "duplicate" in ex_type:
                    risk_score += 3
                    tags.append("Critical Type")

                if risk_score >= 6:
                    priority = 4  # Critical
                elif risk_score >= 4:
                    priority = 3  # High
                elif risk_score >= 2:
                    priority = 2  # Medium

                prioritized_results.append({
                    "exception_id": ex.get("id", f"EX_{datetime.now().timestamp()}"),
                    "priority_level": priority,
                    "tags": tags,
                    "risk_score": risk_score,
                    "amount": amount,
                    "confidence": confidence,
                    "type": ex_type
                })

            return json.dumps({
                "prioritized_exceptions": prioritized_results,
                "summary": {
                    "total": len(prioritized_results),
                    "critical": len([p for p in prioritized_results if p["priority_level"] == 4]),
                    "high": len([p for p in prioritized_results if p["priority_level"] == 3]),
                    "medium": len([p for p in prioritized_results if p["priority_level"] == 2]),
                    "low": len([p for p in prioritized_results if p["priority_level"] == 1])
                }
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
class RouteWorkflowTool(BaseTool):
    name: str = "Route Exceptions to Review"
    description: str = "Route prioritized exceptions to appropriate resolution workflows based on type and priority."

    def _run(self, prioritized_json: str) -> str:
        try:
            data = json.loads(prioritized_json)
            exceptions = data.get("prioritized_exceptions", [])
            assignments = []

            for ex in exceptions:
                priority = ex.get("priority_level", 1)
                ex_type = ex.get("type", "")
                assignee = "reviewer_general"
                routing = "standard_review"

                if "duplicate" in ex_type:
                    assignee = "auto_duplicate_handler"
                    routing = "auto_resolve"

                elif "fraud" in ex_type:
                    assignee = "fraud_specialist"
                    routing = "escalate_to_fraud"

                elif priority == 4:
                    assignee = "senior_manager"
                    routing = "urgent_review"

                elif priority == 3:
                    assignee = "reviewer_senior"
                    routing = "senior_review"

                assignments.append({
                    "exception_id": ex.get("exception_id"),
                    "assignee": assignee,
                    "routing_method": routing,
                    "priority": priority,
                    "status": "assigned",
                    "assigned_at": datetime.now().isoformat()
                })

            return json.dumps({"assignments": assignments}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
class TrackExceptionStatusTool(BaseTool):
    name: str = "Track Exception Status"
    description: str = "Update and track status of exceptions."

    def _run(self, status_update_json: str) -> str:
        try:
            updates = json.loads(status_update_json)
            result = []

            for update in updates:
                record = {
                    "exception_id": update.get("exception_id"),
                    "previous_status": update.get("previous_status", "pending"),
                    "new_status": update.get("new_status"),
                    "updated_at": datetime.now().isoformat(),
                    "reviewer": update.get("reviewer", "system"),
                    "notes": update.get("notes", "")
                }
                result.append(record)

            return json.dumps({"status_updates": result}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
class TrackExceptionStatusTool(BaseTool):
    name: str = "Track Exception Status"
    description: str = "Update and track status of exceptions."

    def _run(self, status_update_json: str) -> str:
        try:
            updates = json.loads(status_update_json)
            result = []

            for update in updates:
                record = {
                    "exception_id": update.get("exception_id"),
                    "previous_status": update.get("previous_status", "pending"),
                    "new_status": update.get("new_status"),
                    "updated_at": datetime.now().isoformat(),
                    "reviewer": update.get("reviewer", "system"),
                    "notes": update.get("notes", "")
                }
                result.append(record)

            return json.dumps({"status_updates": result}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
def create_exception_management_agent():

    return Agent(
        role="Exception Management Agent",
        goal="Prioritize, route, and track high-risk financial exceptions for reconciliation review.",
        backstory="""You are responsible for managing financial exceptions. You check each flagged item, 
        score them for priority, assign to the correct reviewer (or auto-resolve), and ensure follow-up action is tracked.""",
        tools=[
            PrioritizeExceptionsTool(),
            RouteWorkflowTool(),
            TrackExceptionStatusTool()
        ],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False
    )

exception_management_agent = create_exception_management_agent()
            

