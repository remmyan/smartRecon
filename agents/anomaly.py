from crewai import Agent
from crewai.tools import BaseTool
import pandas as pd
import logging
import json
from typing import Any, Dict, List
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = Config.OPENAI_MODEL

class DuplicateDetectionTool(BaseTool):
    name: str = "Duplicate Detection Tool"
    description: str = "Detects duplicate transactions based on date, amount, and vendor."

    def _run(self, transactions_json: str) -> str:
        try:
            transactions = json.loads(transactions_json)
            df = pd.DataFrame(transactions)

            df['normalized_amount'] = df['amount'].astype(str).str.replace("[$,]", "", regex=True).astype(float)
            df['normalized_date'] = pd.to_datetime(df['date'], errors='coerce')
            df['vendor'] = df['vendor'].str.lower().str.strip()

            duplicates = df[df.duplicated(subset=['normalized_amount', 'normalized_date', 'vendor'], keep=False)]
            logger.info(f"Found {len(duplicates)} duplicate transactions")

            result = {
                "duplicate_count": len(duplicates),
                "duplicates": duplicates.to_dict(orient="records")
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
class UnusualPatternTool(BaseTool):
    name: str = "Unusual Pattern Detection Tool"
    description: str = "Detects unusual transaction patterns using clustering (DBSCAN)."

    def _run(self, transactions_json: str) -> str:
        try:
            transactions = json.loads(transactions_json)
            df = pd.DataFrame(transactions)

            df['amount'] = df['amount'].astype(str).str.replace("[$,]", "", regex=True).astype(float)
            df['weekday'] = pd.to_datetime(df['date'], errors='coerce').dt.dayofweek
            df['description_length'] = df['description'].fillna("").str.len()

            features = df[['amount', 'weekday', 'description_length']].fillna(0)
            scaled = StandardScaler().fit_transform(features)

            db = DBSCAN(eps=1.0, min_samples=3).fit(scaled)
            df['cluster'] = db.labels_

            outliers = df[df['cluster'] == -1]
            logger.info(f"Found {len(outliers)} outlier transactions")

            result = {
                "anomalies_count": len(outliers),
                "anomalies": outliers.to_dict(orient="records")
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
class FraudAssessmentTool(BaseTool):
    name: str = "Fraud Risk Assessment Tool"
    description: str = "Assesses fraud risk by scanning for high-value or suspicious patterns."

    def _run(self, transactions_json: str) -> str:
        try:
            transactions = json.loads(transactions_json)
            suspicious = []

            for txn in transactions:
                risk_score = 0
                notes = []

                amount = float(str(txn.get("amount", 0)).replace("$", "").replace(",", ""))
                if amount > 10000:
                    risk_score += 2
                    notes.append("High amount")

                if any(x in txn.get("vendor", "").lower() for x in ['unknown', 'test', 'misc']):
                    risk_score += 1
                    notes.append("Suspicious vendor")

                txn_time = datetime.fromisoformat(txn.get("date")) if txn.get("date") else None
                if txn_time and txn_time.hour < 6 or txn_time.hour > 20:
                    risk_score += 1
                    notes.append("Unusual transaction time")

                if risk_score >= 2:
                    suspicious.append({
                        'transaction': txn,
                        'risk_score': risk_score,
                        'flags': notes
                    })

            response = {
                "fraud_suspicions": len(suspicious),
                "high_risk_transactions": suspicious
            }
            return json.dumps(response, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
def create_anomaly_detection_agent():
    return Agent(
        role="Anomaly Detection Agent",
        backstory="""You are a forensic analyst specializing in transaction monitoring. 
        Your job is to surface duplicates, detect outliers, and raise fraud flags 
        with high accuracy and explainability.""",
        goal="Identify financial anomalies for review (e.g. duplicates, fraud, unusual patterns)",
        tools=[
            DuplicateDetectionTool(),
            UnusualPatternTool(),
            FraudAssessmentTool()
        ],
        llm=llm,
        verbose=True,
        memory=True,
        allow_delegation=False
    )

anomaly_detection_agent = create_anomaly_detection_agent()
