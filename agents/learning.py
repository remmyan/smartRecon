from crewai import Agent
from crewai.tools import BaseTool
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import json
import uuid
from typing import Any, Dict, List
from datetime import datetime, timedelta
import logging
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = Config.OPENAI_MODEL

# Initialize ChromaDB with persistence
client = chromadb.PersistentClient(path="./learning_agent_chroma_db", settings=Settings(anonymized_telemetry=False))

embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="corrections",
    embedding_function=embedding_function
)
class StoreCorrectionTool(BaseTool):
    name: str = "Store Correction"
    description: str = "Store human correction data with embeddings in ChromaDB."

    def _run(self, correction_json: str) -> str:
        try:
            data = json.loads(correction_json)
            correction_id = str(uuid.uuid4())
            text = (
                f"Transaction ID: {data.get('transaction_id')} | "
                f"Original Match: {data.get('original_match')} | "
                f"Human Decision: {data.get('human_decision')} | "
                f"Notes: {data.get('notes')} | "
                f"Vendor: {data.get('vendor')} | "
                f"Amount: {data.get('amount')} | "
                f"Date: {data.get('date')}"
            )
            metadata = {
                "correction_id": correction_id,
                "timestamp": datetime.now().isoformat(),
                "transaction_id": data.get("transaction_id"),
                "human_decision": data.get("human_decision"),
                "amount": str(data.get("amount")),
                "vendor": data.get("vendor"),
                "reason": data.get("reason", "N/A"),
                "created_by": data.get("user", "auditor")
            }

            collection.add(
                documents=[text],
                ids=[correction_id],
                metadatas=[metadata]
            )

            return json.dumps({"status": "success", "correction_id": correction_id})
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)})

class RetrieveSimilarCorrectionsTool(BaseTool):
    name: str = "Retrieve Similar Corrections"
    description: str = "Query the embedding DB to find similar past human corrections."

    def _run(self, query_json: str) -> str:
        try:
            query = json.loads(query_json)
            query_text = (
                f"{query.get('vendor', '')} | {query.get('description', '')} | "
                f"{query.get('amount', '')} | {query.get('date', '')}"
            )

            results = collection.query(
                query_texts=[query_text],
                n_results=5,
                include=["documents", "metadatas"]
            )

            simplified = [
                {
                    "match_score": round(1 - dist, 2),
                    "text": doc,
                    "metadata": meta,
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]

            return json.dumps({"similar_corrections": simplified}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

class CorrectionAnalyticsTool(BaseTool):
    name: str = "Correction Analytics Tool"
    description: str = "Analyze all corrections in the vector DB to derive patterns and statistics."

    def _run(self, _: str = "") -> str:
        try:
            results = collection.get(include=["metadatas"])
            metadata_list = results.get("metadatas", [])
            if not metadata_list:
                return json.dumps({"message": "No data found."})

            totals = len(metadata_list)
            by_user = {}
            decision_counts = {}
            for meta in metadata_list:
                reviewer = meta.get("created_by", "unknown")
                decision = meta.get("human_decision", "N/A")
                by_user[reviewer] = by_user.get(reviewer, 0) + 1
                decision_counts[decision] = decision_counts.get(decision, 0) + 1

            return json.dumps({
                "total_corrections": totals,
                "by_user": by_user,
                "decision_counts": decision_counts
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

class UpdateMatchingPatternsTool(BaseTool):
    name: str = "Update Matching Patterns Tool"
    description: str = "Analyze recent corrections and propose new matching rules (mock logic)."

    def _run(self, _input: str = "") -> str:
        try:
            cutoff = datetime.now() - timedelta(days=7)
            results = collection.get(include=["metadatas"])

            recent = [
                meta for meta in results.get("metadatas", [])
                if datetime.fromisoformat(meta["timestamp"]) >= cutoff
            ]

            if not recent:
                return json.dumps({"message": "No recent patterns to learn from."})

            common_vendors = {}
            decisions = {}
            for meta in recent:
                vendor = meta.get("vendor", "unknown")
                decision = meta.get("human_decision", "N/A")
                key = f"{vendor}_{decision}"
                common_vendors[key] = common_vendors.get(key, 0) + 1
                decisions[decision] = decisions.get(decision, 0) + 1

            suggestions = [
                {
                    "pattern": k,
                    "recommendation": f"Auto-{v.lower()}" if v.lower().startswith("approve") else "Review"
                }
                for k, v in decisions.items()
            ]

            return json.dumps({
                "recent_corrections_count": len(recent),
                "suggestions": suggestions
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

def create_learning_agent():

    return Agent(
        role="Learning Agent",
        goal="Continuously learn from human corrections to improve semantic matching and reconciliation accuracy.",
        backstory="""You monitor and learn from human corrections. Your job is to retain past mismatch patterns, suggest improvements,
        and update reconciliation logic to increase automation accuracy over time.""",
        tools=[
            StoreCorrectionTool(),
            RetrieveSimilarCorrectionsTool(),
            CorrectionAnalyticsTool(),
            UpdateMatchingPatternsTool()
        ],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False
    )

learning_agent = create_learning_agent()

