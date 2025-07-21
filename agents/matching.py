import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from typing import Tuple, List, Dict, Any
import re
from utils.openai_helper import openai_helper
from config import Config

class MatchingAgent:
    def __init__(self):
        self.fuzzy_threshold = Config.FUZZY_THRESHOLD
        self.amount_tolerance = Config.AMOUNT_TOLERANCE
        self.date_window = Config.DATE_WINDOW
        self.use_llm = True  # Enable LLM matching
        
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
        exact_matches = self._exact_match(invoices, ledger)
        results['exact_matches'] = exact_matches
        
        # Remove matched records
        unmatched_invoices = invoices[~invoices.index.isin(exact_matches.index)]
        unmatched_ledger = ledger[~ledger.index.isin(exact_matches.index)]
        
        # Step 2: LLM-powered semantic matching on remaining records
        if self.use_llm and not unmatched_invoices.empty and not unmatched_ledger.empty:
            llm_matches = self._llm_semantic_match(unmatched_invoices, unmatched_ledger)
            results['llm_matches'] = llm_matches
            
            # Remove LLM matched records
            unmatched_invoices = unmatched_invoices[~unmatched_invoices.index.isin(llm_matches.index)]
            unmatched_ledger = unmatched_ledger[~unmatched_ledger.index.isin(llm_matches.index)]
        
        # Step 3: Fuzzy matching on remaining records
        if not unmatched_invoices.empty and not unmatched_ledger.empty:
            fuzzy_matches = self._fuzzy_match(unmatched_invoices, unmatched_ledger)
            results['fuzzy_matches'] = fuzzy_matches
        
        # Step 4: Bank statement matching
        bank_matches = self._bank_matching(ledger, bank_statements)
        results['bank_matches'] = bank_matches
        
        # Step 5: Collect unmatched items
        all_matched_ids = set()
        for match_type in ['exact_matches', 'llm_matches', 'fuzzy_matches']:
            if not results[match_type].empty:
                all_matched_ids.update(results[match_type].index)
        
        results['unmatched'] = invoices[~invoices.index.isin(all_matched_ids)]
        
        return results
    
    def _llm_semantic_match(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Perform LLM-powered semantic matching using ChatGPT"""
        
        matches = []
        
        # Convert DataFrames to list of dictionaries for API calls
        records1 = df1.to_dict('records')
        records2 = df2.to_dict('records')
        
        # Limit the number of comparisons to manage API costs
        max_comparisons = min(len(records1), 20)  # Process max 20 records
        
        for i, record1 in enumerate(records1[:max_comparisons]):
            best_match = None
            best_confidence = 0
            
            # Compare with up to 10 target records to control API usage
            comparison_limit = min(len(records2), 10)
            
            for j, record2 in enumerate(records2[:comparison_limit]):
                # Use ChatGPT for semantic analysis
                match_result = openai_helper.semantic_match_analysis(record1, record2)
                
                if (match_result['is_match'] and 
                    match_result['confidence'] > best_confidence and
                    match_result['confidence'] >= 70):  # Minimum confidence threshold
                    
                    best_confidence = match_result['confidence']
                    best_match = {
                        'record': df1.iloc[i].copy(),
                        'matched_with': j,
                        'confidence': match_result['confidence'],
                        'reasoning': match_result['reasoning'],
                        'match_factors': match_result.get('factors', {}),
                        'match_type': 'llm_semantic'
                    }
            
            if best_match:
                match_record = best_match['record']
                match_record['match_type'] = best_match['match_type']
                match_record['match_confidence'] = best_match['confidence']
                match_record['match_reasoning'] = best_match['reasoning']
                match_record['matched_with'] = best_match['matched_with']
                match_record['match_factors'] = str(best_match['match_factors'])
                matches.append(match_record)
        
        return pd.DataFrame(matches) if matches else pd.DataFrame()
    
    def _exact_match(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Perform exact matching on key fields"""
        
        matches = []
        
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
