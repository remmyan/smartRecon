from crewai import Agent
from crewai.tools import BaseTool
from config import Config
import chromadb
import json
from datetime import datetime
from pathlib import Path

# Simple ChromaDB setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("corrections")

class LearningTool(BaseTool):
    name: str = "Learning Tool"
    description: str = "Store corrections and retrieve similar patterns for learning."

    def _run(self, action: str, data: str = "") -> str:
        try:
            if action == "store":
                # Store correction
                correction = json.loads(data)
                doc = f"Exception {correction.get('exception_id')}: {correction.get('resolution')} - {correction.get('notes', '')}"
                
                collection.add(
                    documents=[doc],
                    ids=[f"correction_{correction.get('exception_id', datetime.now().isoformat())}"],
                    metadatas=[correction]
                )
                return json.dumps({"status": "stored"})
                
            elif action == "retrieve":
                # Get similar patterns
                results = collection.query(query_texts=[data], n_results=3)
                return json.dumps(results['documents'][0] if results['documents'] else [])
                
            elif action == "update_rules":
                # Update matching patterns file
                patterns = {
                    "fuzzy_threshold": 0.8,
                    "amount_tolerance": 0.02,
                    "date_tolerance_days": 3,
                    "last_updated": datetime.now().isoformat()
                }
                
                Path("./learned_patterns.json").write_text(json.dumps(patterns, indent=2))
                return json.dumps({"status": "rules_updated"})
                
        except Exception as e:
            return json.dumps({"error": str(e)})

def create_learning_agent():
    return Agent(
        role="Learning Agent",
        goal="Learn from human corrections to improve matching accuracy.",
        backstory="You store corrections, find similar patterns, and update rules to reduce future exceptions.",
        tools=[LearningTool()],
        llm=Config.get_llm(model="gpt-3.5-turbo", temperature=0.3),
        verbose=True,
        max_iter=2
    )

learning_agent = create_learning_agent()
