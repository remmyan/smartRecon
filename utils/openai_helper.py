import json
from typing import Dict, List, Any

from groq import Groq  # Groq SDK (pip install groq)

from config import Config
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recon_log.txt"),  # Logs to file for persistence
        logging.StreamHandler()  # Logs to console
    ]
)

class GroqHelper:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL  # e.g., "llama3-8b-8192" or as defined in config
        self.max_tokens = Config.OPENAI_MAX_TOKENS  # Reuse if same; adjust if needed for Groq
        self.logger = logging.getLogger("GroqHelper")

    def semantic_match_analysis(self, record1, record2, similar_patterns=None):

        prompt = self._build_matching_prompt(record1, record2, similar_patterns)
        self.logger.info("Semantic matching prompt: %s", prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    #{"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=1,  # Low temperature for consistent results
                response_format={"type": "json_object"}
            )
            print(f"Groq response: {response}" )
            result = json.loads(response.choices[0].message.content)
            return self._process_matching_response(result)
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return {
                'is_match': False,
                'confidence': 0,
                'reasoning': f"API error: {str(e)}",
                'match_type': 'error'
            }

    def batch_semantic_matching(self, source_records: List[Dict],
                                target_records: List[Dict]) -> List[Dict]:
        """Perform batch semantic matching using Groq"""
        matches = []
        for i, source_record in enumerate(source_records):
            best_match = None
            best_confidence = 0
            comparison_limit = min(len(target_records), 10)  # Limit comparisons to avoid API rate limits
            for j, target_record in enumerate(target_records[:comparison_limit]):
                match_result = self.semantic_match_analysis(source_record, target_record)
                if (match_result['is_match'] and
                    match_result['confidence'] > best_confidence):
                    best_confidence = match_result['confidence']
                    best_match = {
                        'source_index': i,
                        'target_index': j,
                        'source_record': source_record,
                        'target_record': target_record,
                        'match_result': match_result
                    }
            if best_match:
                matches.append(best_match)
        return matches

    def analyze_transaction_anomaly(self, transaction: Dict,
                                    context_transactions: List[Dict]) -> Dict[str, Any]:
        """Use Groq to analyze if a transaction is anomalous"""
        prompt = self._build_anomaly_prompt(transaction, context_transactions)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_anomaly_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return self._process_anomaly_response(result)
        except Exception as e:
            return {
                'is_anomaly': False,
                'risk_score': 0,
                'reasoning': f"API error: {str(e)}",
                'anomaly_types': []
            }

    def generate_exception_resolution(self, exception_data: Dict) -> Dict[str, Any]:
        """Generate resolution suggestions for exceptions using Groq"""
        prompt = self._build_exception_prompt(exception_data)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_exception_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            return {
                'suggested_actions': ['Manual review required'],
                'priority_level': 'Medium',
                'estimated_effort': 'Unknown',
                'reasoning': f"API error: {str(e)}"
            }

    # The following methods remain unchanged from the original (prompt builders, system prompts, etc.)
    def _build_matching_prompt(self, record1: Dict, record2: Dict, similar_patterns=None) -> str:
        """Build prompt for semantic matching analysis"""
        patterns_text = ""
        corrected_amount = 0
        user_feedback = ""
        if similar_patterns and len(similar_patterns) > 0:
            for i, pat in enumerate(similar_patterns[:5]):  # limit to 5 patterns
                metadata = pat.get('pattern_metadata', {})
                corrected_amount = metadata.get('amount')
                user_feedback = metadata.get('user_feedback', '')
                # Build a formatted string of all metadata fields and values
                metadata_parts = []
                for key, value in metadata.items():
                    metadata_parts.append(f"{key}: {value}")
                metadata_str = "; ".join(metadata_parts)
                
                patterns_text += f"\nPattern {i+1} metadata: {metadata_str}"
        patterns_exist = similar_patterns and len(similar_patterns) > 0   
        if patterns_exist:
            return f"""
            You are a financial reconciliation assistant. Your task is to determine if two financial records represent the same transaction.

            correction patterns: {patterns_text if patterns_text else ' None.'}

            Analyze the two records below considering:
            - Amount similarity:   
            Ideally, amounts should match exactly.
            If amounts differ, check if either record’s amount matches the corrected amount from the correction patterns — in this case, {corrected_amount}. 
            At least one amount should match {corrected_amount} for a valid match and confidence score 100. 
            Use {user_feedback} for reasoning.
            - Date proximity (payments can occur days after invoice dates)
            - Vendor name variations and possible abbreviations
            - Description similarities

            Reply only with a JSON matching this schema:

            {{
            "is_match": true or false,
            "confidence": integer 0-100,
            "reasoning": {{
                "date_proximity": {{"score": 0-100, "reason": "explanation"}},
                "vendor_name_variations": {{...}},
                "amount_similarity": {{...}}
            }}
            }}

            Do not write anything else. Your entire answer MUST be valid JSON.

            Record 1:
            - Amount: {record1.get('amount', 'N/A')}
            - Date: {record1.get('date', 'N/A')}
            - Vendor: {record1.get('vendor', 'N/A')}
            - Description: {record1.get('description', 'N/A')}
            - Reference: {record1.get('reference', 'N/A')}

            Record 2:
            - Amount: {record2.get('amount', 'N/A')}
            - Date: {record2.get('date', 'N/A')}
            - Vendor: {record2.get('vendor', 'N/A')}
            - Description: {record2.get('description', 'N/A')}
            - Reference: {record2.get('reference', 'N/A')}
            """  
        else:
            return f"""    
            You are a financial reconciliation assistant. Your task is to determine if two financial records represent the same transaction.

            Please analyze the two records below considering:
            - Amount similarity (should ideally match exactly)
            - Date proximity (payments can occur days after invoice dates)
            - Vendor name variations and possible abbreviations
            - Description similarities

            If any field differs, provide clear reasoning.

            Reply only with a JSON matching this schema:

            {{
            "is_match": true or false,
            "confidence": integer 0-100,
            "reasoning": {{
                "date_proximity": {{"score": 0-100, "reason": "explanation"}},
                "vendor_name_variations": {{...}},
                "amount_similarity": {{...}}
            }}
            }}

            Do not write anything else. Your entire answer MUST be valid JSON.

            Record 1:
            - Amount: {record1.get('amount', 'N/A')}
            - Date: {record1.get('date', 'N/A')}
            - Vendor: {record1.get('vendor', 'N/A')}
            - Description: {record1.get('description', 'N/A')}
            - Reference: {record1.get('reference', 'N/A')}

            Record 2:
            - Amount: {record2.get('amount', 'N/A')}
            - Date: {record2.get('date', 'N/A')}
            - Vendor: {record2.get('vendor', 'N/A')}
            - Description: {record2.get('description', 'N/A')}
            - Reference: {record2.get('reference', 'N/A')}
            """
        

    def _build_anomaly_prompt(self, transaction: Dict, context: List[Dict]) -> str:
        """Build prompt for anomaly detection"""
        context_summary = self._summarize_context_transactions(context)
        return f"""
        Analyze this transaction for potential anomalies:

        Transaction:
        - Amount: {transaction.get('amount', 'N/A')}
        - Date: {transaction.get('date', 'N/A')}
        - Vendor: {transaction.get('vendor', 'N/A')}
        - Description: {transaction.get('description', 'N/A')}

        Context (similar transactions):
        {context_summary}

        Look for:
        1. Unusual amounts (significantly higher/lower than normal)
        2. Suspicious timing (weekends, holidays, after hours)
        3. Vendor anomalies (new vendors, name variations)
        4. Description patterns that seem suspicious
        5. Round number bias
        6. Potential duplicate indicators

        Provide analysis in JSON format with risk score (0-100) and specific anomaly types.
        """

    def _build_exception_prompt(self, exception_data: Dict) -> str:
        """Build prompt for exception resolution"""
        return f"""
        Provide resolution guidance for this accounts payable exception:

        Exception Details:
        - Type: {exception_data.get('type', 'Unknown')}
        - Description: {exception_data.get('description', 'N/A')}
        - Amount: {exception_data.get('amount', 'N/A')}
        - Age: {exception_data.get('age_days', 0)} days
        - Risk Score: {exception_data.get('risk_score', 0)}

        Additional Context:
        {json.dumps(exception_data.get('metadata', {}), indent=2)}

        Provide specific, actionable resolution steps, priority level, and estimated effort.
        Consider standard AP procedures and compliance requirements.
        """

    def _get_system_prompt(self) -> str:
        """System prompt for matching analysis"""
        return """
        You are an expert financial analyst specializing in accounts payable reconciliation.
        Your task is to analyze financial records and determine if they represent matching transactions.
        Be thorough but practical - consider real-world scenarios where:
        - Amounts may differ slightly due to fees, discounts, or currency conversion
        - Dates may differ due to processing delays
        - Vendor names may have variations
        - Descriptions may use different terminology
        Always provide structured JSON responses with clear reasoning.
        """

    def _get_anomaly_system_prompt(self) -> str:
        """System prompt for anomaly detection"""
        return """
        You are a fraud detection specialist for accounts payable systems.
        Analyze transactions for potential anomalies, fraud indicators, or unusual patterns.
        Consider both statistical anomalies and business logic violations.
        Be thorough but avoid false positives - focus on genuinely suspicious patterns.
        Provide risk scores based on severity and likelihood of actual problems.
        """

    def _get_exception_system_prompt(self) -> str:
        """System prompt for exception resolution"""
        return """
        You are an accounts payable manager with expertise in exception resolution.
        Provide practical, actionable guidance for resolving AP exceptions.
        Consider:
        - Standard AP procedures
        - Compliance requirements
        - Resource constraints
        - Risk management
        Prioritize solutions that maintain audit trails and financial controls.
        """

    def _summarize_context_transactions(self, context: List[Dict]) -> str:
        """Summarize context transactions for anomaly analysis"""
        if not context:
            return "No context transactions available"
        amounts = [t.get('amount', 0) for t in context if t.get('amount')]
        vendors = list(set(t.get('vendor', '') for t in context if t.get('vendor')))
        summary = f"""
        Context Summary:
        - Number of similar transactions: {len(context)}
        - Amount range: ${min(amounts):.2f} - ${max(amounts):.2f}
        - Average amount: ${sum(amounts)/len(amounts):.2f}
        - Common vendors: {', '.join(vendors[:5])}
        """
        return summary

    def _process_matching_response(self, response: Dict) -> Dict[str, Any]:
        """
        Process and validate matching response from Groq.
        The response has nested keys as per your example:
        {
        "analysis": {
            "same_transaction": true,
            "confidence_score": 90
        },
        "reasoning": { ... }
        }
        """
        analysis = response.get('analysis', {})
        reasoning = response.get('reasoning', {})
        
        is_match = response.get('is_match', False)
        confidence = response.get('confidence', 0)
        # Optionally clamp confidence between 0-100 as you do
        confidence = max(0, min(100, confidence))
        
        match_type = response.get('match_type', 'unknown')  # fallback if no explicit field
        
        return {
            'is_match': is_match,
            'confidence': confidence,
            'reasoning': reasoning,
            'match_type': match_type,
            'factors': response.get('factors', {}),
            'api_response': response
        }


    def _process_anomaly_response(self, response: Dict) -> Dict[str, Any]:
        """Process and validate anomaly response from Groq"""
        return {
            'is_anomaly': response.get('is_anomaly', False),
            'risk_score': max(0, min(100, response.get('risk_score', 0))),
            'reasoning': response.get('reasoning', 'No reasoning provided'),
            'anomaly_types': response.get('anomaly_types', []),
            'recommendations': response.get('recommendations', []),
            'api_response': response
        }

# Global instance
groq_helper = GroqHelper()
