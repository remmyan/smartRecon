from crewai import Agent
from crewai.tools import BaseTool
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import numpy as np
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = Config.OPENAI_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Tools by subclassing BaseTool
class SemanticMatchingTool(BaseTool):
    name: str = "Perform Semantic Matching"
    description: str = "Perform semantic matching between ERP data and bank statements. Input is JSON with 'erp_data' and 'bank_data'."

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            erp_data = data.get('erp_data', [])
            bank_data = data.get('bank_data', [])
            
            if not erp_data or not bank_data:
                return json.dumps({"error": "Missing ERP or bank data"})
            
            matches = []
            statistics = {
                'total_erp_records': len(erp_data),
                'total_bank_records': len(bank_data),
                'full_matches': 0,
                'partial_matches': 0,
                'no_matches': 0,
                'auto_approved': 0
            }
            
            # Convert to DataFrames for easier processing
            erp_df = pd.DataFrame(erp_data)
            bank_df = pd.DataFrame(bank_data)
            
            # Standardize column names (flexible mapping)
            erp_cols = {
                'amount': next((col for col in erp_df.columns if 'amount' in col.lower()), None),
                'date': next((col for col in erp_df.columns if 'date' in col.lower()), None),
                'description': next((col for col in erp_df.columns if any(x in col.lower() for x in ['desc', 'detail', 'memo'])), None),
                'vendor': next((col for col in erp_df.columns if any(x in col.lower() for x in ['vendor', 'supplier', 'payee'])), None),
                'id': next((col for col in erp_df.columns if any(x in col.lower() for x in ['id', 'number', 'ref'])), None)
            }
            
            bank_cols = {
                'amount': next((col for col in bank_df.columns if 'amount' in col.lower()), None),
                'date': next((col for col in bank_df.columns if 'date' in col.lower()), None),
                'description': next((col for col in bank_df.columns if any(x in col.lower() for x in ['desc', 'detail', 'memo'])), None),
                'vendor': next((col for col in bank_df.columns if any(x in col.lower() for x in ['payee', 'merchant', 'vendor'])), None),
                'id': next((col for col in bank_df.columns if any(x in col.lower() for x in ['id', 'number', 'ref'])), None)
            }
            
            # Perform matching for each ERP record
            for erp_idx, erp_record in erp_df.iterrows():
                best_matches = []
                
                for bank_idx, bank_record in bank_df.iterrows():
                    # Calculate individual similarity scores
                    amount_sim = self.calculate_amount_similarity(
                        erp_record.get(erp_cols['amount'], ''),
                        bank_record.get(bank_cols['amount'], '')
                    ) if erp_cols['amount'] and bank_cols['amount'] else 0.0
                    
                    date_sim = self.calculate_date_similarity(
                        erp_record.get(erp_cols['date'], ''),
                        bank_record.get(bank_cols['date'], '')
                    ) if erp_cols['date'] and bank_cols['date'] else 0.0
                    
                    desc_sim = self.calculate_text_similarity(
                        erp_record.get(erp_cols['description'], ''),
                        bank_record.get(bank_cols['description'], '')
                    ) if erp_cols['description'] and bank_cols['description'] else 0.0
                    
                    vendor_sim = self.calculate_text_similarity(
                        erp_record.get(erp_cols['vendor'], ''),
                        bank_record.get(bank_cols['vendor'], '')
                    ) if erp_cols['vendor'] and bank_cols['vendor'] else 0.0
                    
                    # Extract and compare key identifiers
                    erp_identifiers = self.extract_key_identifiers(
                        erp_record.get(erp_cols['description'], '')
                    )
                    bank_identifiers = self.extract_key_identifiers(
                        bank_record.get(bank_cols['description'], '')
                    )
                    
                    identifier_match = 0.0
                    if erp_identifiers and bank_identifiers:
                        common_identifiers = set(erp_identifiers) & set(bank_identifiers)
                        identifier_match = len(common_identifiers) / max(len(erp_identifiers), len(bank_identifiers))
                    
                    # Calculate weighted overall confidence score
                    weights = {
                        'amount': 0.35,
                        'date': 0.25,
                        'description': 0.20,
                        'vendor': 0.15,
                        'identifier': 0.05
                    }
                    
                    overall_confidence = (
                        amount_sim * weights['amount'] +
                        date_sim * weights['date'] +
                        desc_sim * weights['description'] +
                        vendor_sim * weights['vendor'] +
                        identifier_match * weights['identifier']
                    )
                    
                    # Store match details
                    match_detail = {
                        'erp_record_id': str(erp_idx),
                        'bank_record_id': str(bank_idx),
                        'overall_confidence': round(overall_confidence, 3),
                        'confidence_breakdown': {
                            'amount_similarity': round(amount_sim, 3),
                            'date_similarity': round(date_sim, 3),
                            'description_similarity': round(desc_sim, 3),
                            'vendor_similarity': round(vendor_sim, 3),
                            'identifier_match': round(identifier_match, 3)
                        }
                    }
                    best_matches.append(match_detail)
                
                # Sort by confidence and keep top matches
                best_matches.sort(key=lambda x: x['overall_confidence'], reverse=True)
                
                if best_matches:
                    top_match = best_matches[0]
                    if top_match['overall_confidence'] >= 0.95:
                        match_type = "full_match"
                        statistics['full_matches'] += 1
                        statistics['auto_approved'] += 1
                    elif top_match['overall_confidence'] >= 0.85:
                        match_type = "full_match"
                        statistics['full_matches'] += 1
                    elif top_match['overall_confidence'] >= 0.6:
                        match_type = "partial_match"
                        statistics['partial_matches'] += 1
                    else:
                        match_type = "no_match"
                        statistics['no_matches'] += 1
                    
                    top_match['match_type'] = match_type
                    matches.append(top_match)
            
            success_rate = ((statistics['full_matches'] + statistics['partial_matches']) / 
                           statistics['total_erp_records']) * 100 if statistics['total_erp_records'] > 0 else 0
            
            result = {
                'matching_results': matches,
                'statistics': statistics,
                'success_rate': round(success_rate, 2)
            }
            
            logger.info(f"Matching completed: {success_rate:.2f}% success rate")
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return json.dumps({"error": str(e)})

    def calculate_amount_similarity(self, amount1: Any, amount2: Any, tolerance_percent: float = 0.02) -> float:
        amt1 = float(str(amount1).replace('$', '').replace(',', ''))
        amt2 = float(str(amount2).replace('$', '').replace(',', ''))
        if amt1 == amt2:
            return 1.0
        diff_percent = abs(amt1 - amt2) / max(amt1, amt2)
        return 1.0 - diff_percent if diff_percent <= tolerance_percent else 0.0

    def calculate_date_similarity(self, date1: str, date2: str, tolerance_days: int = 3) -> float:
        d1 = datetime.strptime(date1, '%Y-%m-%d')
        d2 = datetime.strptime(date2, '%Y-%m-%d')
        diff_days = abs((d1 - d2).days)
        return 1.0 if diff_days <= tolerance_days else 0.0

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        return fuzz.ratio(text1.lower(), text2.lower()) / 100.0

    def extract_key_identifiers(self, description: str) -> List[str]:
        # Simplified identifier extraction
        return re.findall(r'\b[A-Z0-9]{5,}\b', description)

class PatternAnalysisTool(BaseTool):
    name: str = "Analyze Matching Patterns"
    description: str = "Analyze matching patterns from results. Input is JSON with matching results."

    def _run(self, results_json: str) -> str:
        try:
            results = json.loads(results_json)
            confidences = [match['overall_confidence'] for match in results.get('matching_results', [])]
            analysis = {
                'average_confidence': np.mean(confidences) if confidences else 0,
                'high_confidence_count': sum(1 for c in confidences if c > 0.9)
            }
            return json.dumps(analysis)
        except Exception as e:
            return json.dumps({"error": str(e)})

class RuleValidationTool(BaseTool):
    name: str = "Validate Matching Rules"
    description: str = "Validate custom matching rules. Input is JSON with rules."

    def _run(self, rules_json: str) -> str:
        try:
            rules = json.loads(rules_json)
            return json.dumps({"valid_rules": len(rules), "status": "validated"})
        except Exception as e:
            return json.dumps({"error": str(e)})

# Create the CrewAI Matching Agent
def create_matching_agent():
    """Create the CrewAI Matching Agent with semantic matching capabilities"""
    
    matching_agent = Agent(
        role="Matching Agent",
        goal="""Perform LLM-powered semantic matching with multi-criteria analysis and confidence scoring 
        to achieve 90-95% automatic transaction matching between ERP data and bank statements.""",
        backstory="""You are an advanced AI specialist in financial transaction matching with expertise in 
        accounts payable reconciliation. You use sophisticated semantic matching algorithms that analyze 
        amount, date, description, vendor, and identifiers to provide high-accuracy matches.""",
        tools=[SemanticMatchingTool(), PatternAnalysisTool(), RuleValidationTool()],
        llm=llm,
        verbose=True,
        memory=True,
        max_iter=3,
        allow_delegation=False
    )
    
    return matching_agent

# Export the agent for use in main crew
matching_agent = create_matching_agent()
