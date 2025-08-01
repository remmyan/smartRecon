import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from typing import Tuple, List, Dict, Any
import re
from utils.openai_helper import GroqHelper
from config import Config
from agents.learning import LearningAgent  # Assuming LearningAgent is defined in agents/learning.py
from scipy.optimize import linear_sum_assignment

class MatchingAgent:
    def __init__(self):
        self.fuzzy_threshold = Config.FUZZY_THRESHOLD
        self.amount_tolerance = Config.AMOUNT_TOLERANCE
        self.date_window = Config.DATE_WINDOW
        self.use_llm = True  # Enable LLM matching
        self.groq_helper = GroqHelper()
        self.learning_agent = LearningAgent()  # Initialize Learning Agent for pattern retrieval
        
    def perform_matching(self, invoices: pd.DataFrame, ledger: pd.DataFrame, 
                        bank_statements: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Perform comprehensive matching with ChatGPT integration"""
        
        results = {
            'exact_matches': pd.DataFrame(),
            'llm_matches': pd.DataFrame(),
            'fuzzy_matches': pd.DataFrame(),
            'partial_matches': pd.DataFrame(),
            'unmatched': pd.DataFrame()
        }
        
        # Step 1: Exact matching (fast, rule-based)
        exact_matches = self._exact_match(bank_statements, ledger)
        results['exact_matches'] = exact_matches
        
        # Remove matched records
        unmatched_bank_statements = bank_statements[~bank_statements.index.isin(exact_matches.index)]
        unmatched_ledger = ledger[~ledger.index.isin(exact_matches.index)]
        
        # Step 2: LLM-powered semantic matching on remaining records
        if self.use_llm and not unmatched_bank_statements.empty and not unmatched_ledger.empty:
            llm_matches, llm_unmatched = self._llm_semantic_match(unmatched_bank_statements, unmatched_ledger)
            results['llm_matches'] = llm_matches
            
            # Remove LLM matched records
            unmatched_bank_statements = unmatched_bank_statements[~unmatched_bank_statements.index.isin(llm_matches.index)]
            unmatched_ledger = unmatched_ledger[~unmatched_ledger.index.isin(llm_matches.index)]
        
        # Step 3: Fuzzy matching on remaining records
        if not unmatched_bank_statements.empty and not unmatched_bank_statements.empty:
            fuzzy_matches = self._fuzzy_match(unmatched_bank_statements, unmatched_ledger)
            results['fuzzy_matches'] = fuzzy_matches
        
        # # Step 4: Bank statement matching
        # bank_matches = self._bank_matching(ledger, bank_statements)
        # results['bank_matches'] = bank_matches
        
        # Step 5: Collect unmatched items
        all_matched_ids = set()
        for match_type in ['exact_matches', 'llm_matches', 'fuzzy_matches']:
            if not results[match_type].empty:
                all_matched_ids.update(results[match_type].index)
        
        results['unmatched'] = llm_unmatched
        
        return results

    def extract_reason(self, reasoning: dict) -> str:
        candidates = ['date_proximity', 'vendor_name_variations', 'amount_similarity']
        best_reason = None
        best_score = 101  # Init above max 100
        print(f"Extracting reason : {reasoning}")
        for factor in candidates:
            score_info = reasoning.get(factor, None)
            print(f"Factor: {factor}, Score Info: {score_info}")
            if score_info:
                score = score_info.get('score', 100)
                reason = score_info.get('reason', '')
                if score < best_score and score < 100:
                    best_score = score
                best_reason = reason
        
        if best_reason:
            return best_reason
        else:
            return "No matching record found above confidence threshold."    
    
    def _llm_semantic_match(self, df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """LLM matching for remaining, collaborating with Learning Agent for patterns."""
        matches = []
        unmatched = []
        records1 = df1.to_dict('records')
        records2 = df2.to_dict('records')

        for i, record1 in enumerate(records1):
            # Collaborate: Query Learning Agent for patterns
            similar_patterns = self.learning_agent.retrieve_similar_patterns(record1)
            print(f"Record {i}: Retrieved {similar_patterns} similar patterns for LLM matching.")
            #pattern_adjustment = self._apply_pattern_adjustment(record1, similar_patterns)

            # Use Groq LLM for semantic matching
            best_match = None
            un_match = None
            best_confidence = 0
            for j, record2 in enumerate(records2):
                match_result = self.groq_helper.semantic_match_analysis(record1, record2, similar_patterns)  # Using Groq helper
                #adjusted_confidence = match_result['confidence'] + (len(similar_patterns) * 0.1)  # Boost from patterns
                if match_result['is_match'] :
                    best_match = {
                        'record': record1,
                        'matched_with': j,
                        'confidence': match_result['confidence'],
                        'reasoning': match_result['reasoning']
                    }
                else:
                    un_match = {
                        'record': record1,
                        'matched_with': None,
                        'confidence': match_result['confidence'],
                        'reasoning': match_result['reasoning']
                    }    

            if best_match:
                match_record = pd.Series(best_match['record'])
                match_record['match_type'] = 'llm'
                match_record['match_confidence'] = best_match['confidence']
                match_record['matched_with'] = best_match['matched_with']
                match_record['match_reasoning'] = self.extract_reason(best_match['reasoning'])
                matches.append(match_record)
            else:
                unmatched_record = pd.Series(un_match['record'])
                unmatched_record['match_type'] = 'llm'
                unmatched_record['match_confidence'] = un_match['confidence']
                unmatched_record['matched_with'] = None   
                unmatched_record['match_reasoning'] = self.extract_reason(un_match['reasoning']) 
                unmatched.append(unmatched_record)
        
        matched_df = pd.DataFrame(matches) if matches else pd.DataFrame()
        unmatched_df = pd.DataFrame(unmatched) if unmatched else pd.DataFrame()
    
        return matched_df, unmatched_df

    def compute_llm_score_matrix(records1, records2):
        n, m = len(records1), len(records2)
        score_matrix = np.zeros((n, m))
        metadata_matrix = np.empty((n, m), dtype=object)
        for i, rec1 in enumerate(records1):
            similar_patterns = self.learning_agent.retrieve_similar_patterns(rec1)
            for j, rec2 in enumerate(records2):
                match_result = self.groq_helper.semantic_match_analysis(rec1, rec2, similar_patterns)
                score_matrix[i, j] = match_result["confidence"]
                metadata_matrix[i, j] = match_result
        return score_matrix, metadata_matrix

    def optimal_llm_matching(records1, records2, confidence_threshold=60):
        score_matrix, meta_matrix = self.compute_llm_score_matrix(records1, records2)
        # Convert score to "cost" for Hungarian: maximize confidence -> minimize negative confidence
        cost_matrix = -score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for i, j in zip(row_ind, col_ind):
            if score_matrix[i, j] >= confidence_threshold:
                match_info = dict(records1[i])
                match_info.update({
                    "matched_with": j,
                    "match_type": "llm",
                    "match_confidence": score_matrix[i, j],
                    "match_reasoning": meta_matrix[i, j].get("reasoning", ""),
                })
                matches.append(match_info)
        return pd.DataFrame(matches)    

    def _apply_pattern_adjustment(self, record: Dict[str, Any], patterns: List[Dict[str, Any]]) -> str:
        """Apply learnings from patterns (from your history)."""
        adjustment = ""
        for pattern in patterns:
            if pattern['similarity'] > 0.8:
                # Example adjustment: Correct amount based on pattern
                record['amount'] = pattern['metadata'].get('corrected_amount', record['amount'])
                adjustment += f"Applied correction from pattern ID {pattern['id']}\n"
        return adjustment

    def _bank_matching(self, ledger: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
        """Bank matching from your original code (kept as is)."""
        # (Paste your _bank_matching code here from the attachment)
        pass  # Replace with your implementation     
    
    def _exact_match(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Perform exact matching on key fields"""
        
        matches = []
        print(f"Performing exact match between {df1} records and {df2} records")
        for idx1, row1 in df1.iterrows():
            for idx2, row2 in df2.iterrows():
                if (abs(row1['amount'] - row2['amount']) < 0.01 and
                    row1['vendor'].lower() == row2['vendor'].lower() and
                    abs((pd.to_datetime(row1['date']) - pd.to_datetime(row2['date'])).days) <= self.date_window):
                    
                    match_record = row1.copy()
                    match_record['match_type'] = 'exact'
                    match_record['match_confidence'] = 100
                    match_record['matched_with'] = idx2
                    match_record['match_reasoning'] = 'Exact match on amount, vendor, and date'
                    matches.append(match_record)
                    break
        
        return pd.DataFrame(matches)
    
    def _fuzzy_match(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Perform fuzzy matching using rapidfuzz"""
        
        matches = []
        
        for idx1, row1 in df1.iterrows():
            best_match = None
            best_score = 0
            
            for idx2, row2 in df2.iterrows():
                # Amount similarity (within tolerance)
                amount_diff = abs(row1['amount'] - row2['amount']) / max(row1['amount'], row2['amount'])
                if amount_diff > self.amount_tolerance:
                    continue
                
                # Vendor name similarity
                vendor_score = fuzz.token_set_ratio(
                    row1['vendor'].lower(), 
                    row2['vendor'].lower()
                )
                
                # Description similarity
                desc_score = fuzz.token_set_ratio(
                    str(row1.get('description', '')).lower(),
                    str(row2.get('description', '')).lower()
                )
                
                # Date proximity score
                date_diff = abs((pd.to_datetime(row1['date']) - pd.to_datetime(row2['date'])).days)
                date_score = max(0, 100 - (date_diff * 10))
                
                # Combined score
                combined_score = (vendor_score * 0.4 + desc_score * 0.3 + date_score * 0.3)
                
                if combined_score > best_score and combined_score >= self.fuzzy_threshold:
                    best_score = combined_score
                    best_match = {
                        'record': row1.copy(),
                        'matched_with': idx2,
                        'confidence': combined_score,
                        'vendor_score': vendor_score,
                        'desc_score': desc_score,
                        'date_score': date_score
                    }
            
            if best_match:
                match_record = best_match['record']
                match_record['match_type'] = 'fuzzy'
                match_record['match_confidence'] = best_match['confidence']
                match_record['matched_with'] = best_match['matched_with']
                match_record['match_reasoning'] = f"Fuzzy match: vendor={best_match['vendor_score']:.1f}%, desc={best_match['desc_score']:.1f}%, date={best_match['date_score']:.1f}%"
                matches.append(match_record)
        
        return pd.DataFrame(matches)
    
    def _bank_matching(self, ledger: pd.DataFrame, bank: pd.DataFrame) -> pd.DataFrame:
        """Match ledger entries with bank statements"""
        
        matches = []
        
        for idx1, ledger_row in ledger.iterrows():
            for idx2, bank_row in bank.iterrows():
                # Amount matching (exact for bank statements)
                if abs(ledger_row['amount'] - bank_row['amount']) < 0.01:
                    # Date matching (within window)
                    date_diff = abs((pd.to_datetime(ledger_row['date']) - pd.to_datetime(bank_row['date'])).days)
                    if date_diff <= self.date_window:
                        match_record = ledger_row.copy()
                        match_record['match_type'] = 'bank_exact'
                        match_record['match_confidence'] = 100 - (date_diff * 5)
                        match_record['bank_ref'] = bank_row.get('transaction_id', '')
                        match_record['match_reasoning'] = f'Bank statement match with {date_diff} day(s) difference'
                        matches.append(match_record)
                        break
        
        return pd.DataFrame(matches)
    
    def calculate_match_statistics(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate matching performance statistics"""
        
        total_records = sum(len(df) for df in results.values() if not df.empty)
        if total_records == 0:
            return {}
        
        stats = {
            'total_transactions': total_records,
            'exact_matches': len(results.get('exact_matches', [])),
            'llm_matches': len(results.get('llm_matches', [])),
            'fuzzy_matches': len(results.get('fuzzy_matches', [])),
            'bank_matches': len(results.get('bank_matches', [])),
            'unmatched': len(results.get('unmatched', [])),
        }
        
        matched_total = (stats['exact_matches'] + stats['llm_matches'] + 
                        stats['fuzzy_matches'])
        stats['match_rate'] = (matched_total / total_records * 100) if total_records > 0 else 0
        
        # LLM specific statistics
        if stats['llm_matches'] > 0:
            llm_df = results.get('llm_matches', pd.DataFrame())
            if not llm_df.empty and 'match_confidence' in llm_df.columns:
                stats['llm_avg_confidence'] = llm_df['match_confidence'].mean()
                stats['llm_min_confidence'] = llm_df['match_confidence'].min()
                stats['llm_max_confidence'] = llm_df['match_confidence'].max()
        
        return stats
    
    def get_match_explanations(self, results: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Get detailed explanations for LLM matches"""
        
        explanations = []
        
        llm_matches = results.get('llm_matches', pd.DataFrame())
        if not llm_matches.empty:
            for idx, match in llm_matches.iterrows():
                explanations.append({
                    'record_id': idx,
                    'match_type': match.get('match_type', 'unknown'),
                    'confidence': match.get('match_confidence', 0),
                    'reasoning': match.get('match_reasoning', 'No reasoning available'),
                    'vendor': match.get('vendor', 'Unknown'),
                    'amount': match.get('amount', 0),
                    'factors': match.get('match_factors', '{}')
                })
        
        return explanations
