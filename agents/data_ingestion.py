from crewai import Agent
from crewai.tools import BaseTool
import logging
import json
import pandas as pd
from pathlib import Path
import tempfile
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadDataTool(BaseTool):
    name: str = "Load Data"
    description: str = "Load data from CSV sources like ERP, bank, or invoices."

    def _run(self, source: str, file_path: str) -> str:
        try:
            if isinstance(file_path, list):  # For multiple files
                dfs = [pd.read_csv(file) for file in file_path if file.name.endswith('.csv')]
                combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            else:
                combined_df = pd.read_csv(file_path) if file_path.name.endswith('.csv') else pd.DataFrame()

            # Normalize columns
            column_map = {
                'id': next((col for col in combined_df.columns if 'id' in col.lower()), None),
                'amount': next((col for col in combined_df.columns if 'amount' in col.lower()), None),
                'date': next((col for col in combined_df.columns if 'date' in col.lower()), None),
                'description': next((col for col in combined_df.columns if 'desc' in col.lower()), None),
                'vendor': next((col for col in combined_df.columns if 'vendor' in col.lower() or 'payee' in col.lower()), None)
            }
            combined_df = combined_df.rename(columns=column_map)
            data = combined_df.to_dict(orient='records')
            logger.info(f"Loaded {len(data)} records from {source}")
            return json.dumps(data)
        except Exception as e:
            return json.dumps({"error": str(e)})

class ValidateDataTool(BaseTool):
    name: str = "Validate Data"
    description: str = "Validate loaded data for completeness."

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            validation_results = []
            for record in data:
                issues = []
                if 'amount' not in record or not isinstance(record['amount'], (int, float)):
                    issues.append("Invalid amount")
                if 'date' not in record:
                    issues.append("Missing date")
                validation_results.append({'valid': len(issues) == 0, 'issues': issues})
            summary = {'total': len(data), 'valid': sum(r['valid'] for r in validation_results)}
            return json.dumps(summary)
        except Exception as e:
            return json.dumps({"error": str(e)})

def create_data_ingestion_agent():
    return Agent(
        role="Data Ingestion Agent",
        goal="Load and validate CSV data",
        backstory="You process structured files without AI.",
        llm=None,  # LLM-free
        tools=[LoadDataTool(), ValidateDataTool()],
        verbose=True,
        max_iter=1
    )
# Export the agent for use in main crew
data_ingestion_agent = create_data_ingestion_agent()
