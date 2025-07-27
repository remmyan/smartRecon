import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from pydantic import Field
from crewai import Agent
from crewai.tools import BaseTool

from config import Config  # Ensure Config.get_llm() is implemented correctly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticMatchingTool(BaseTool):
    name: str = "Semantic Matching"
    description: str = "Match data using learned patterns."
    patterns: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        # Proper initialization for Pydantic models
        object.__setattr__(self, "patterns", self.load_patterns())

    def load_patterns(self) -> Dict[str, Any]:
        """Load patterns from file or use defaults."""
        patterns_file = Path("./learned_patterns.json")
        if patterns_file.exists():
            try:
                return json.loads(patterns_file.read_text())
            except Exception as e:
                logger.error(f"Error reading patterns file: {e}")
        return {
            "fuzzy_threshold": 0.7,
            "amount_tolerance": 0.02,
            "date_tolerance_days": 3
        }

    def _run(self, erp_json: str, bank_json: str) -> str:
        """Perform semantic matching between ERP and bank data."""
        try:
            erp: List[Dict[str, Any]] = json.loads(erp_json)
            bank: List[Dict[str, Any]] = json.loads(bank_json)
            matches = []

            for e in erp:
                for b in bank:
                    try:
                        amount_diff = abs(e['amount'] - b['amount']) / max(e['amount'], b['amount'])
                        date_diff = abs(
                            (datetime.strptime(e['date'], '%Y-%m-%d') -
                             datetime.strptime(b['date'], '%Y-%m-%d')).days
                        )

                        if (
                            amount_diff <= self.patterns['amount_tolerance']
                            and date_diff <= self.patterns['date_tolerance_days']
                        ):
                            confidence = (
                                1.0
                                if e.get('vendor') == b.get('payee')
                                else self.patterns['fuzzy_threshold']
                            )
                            matches.append(
                                {
                                    'erp_id': e['id'],
                                    'bank_id': b['id'],
                                    'confidence': confidence
                                }
                            )
                    except KeyError as ke:
                        logger.warning(f"Missing key in record: {ke}")

            return json.dumps(matches)

        except Exception as e:
            logger.error(f"Error in SemanticMatchingTool._run: {e}")
            return json.dumps({"error": str(e)})


def create_matching_agent() -> Agent:
    """Create an Agent that uses the SemanticMatchingTool."""
    return Agent(
        role="Matching Agent",
        goal="Perform semantic matching using learned patterns",
        backstory="You apply rules from corrections to improve accuracy.",
        llm=Config.get_llm(model="gpt-3.5-turbo"),  # Ensure Config.get_llm exists
        tools=[SemanticMatchingTool()],
        verbose=True,
        max_iter=1
    )


# Export agent
matching_agent = create_matching_agent()
