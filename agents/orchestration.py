import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any
from .data_ingestion import DataIngestionAgent
from .matching import MatchingAgent
from .anomaly import AnomalyDetectionAgent
from .exceptions import ExceptionManagementAgent
from .learning import learning_agent

class OrchestrationAgent:
    def __init__(self):
        self.data_ingestion = DataIngestionAgent()
        self.matching = MatchingAgent()
        self.anomaly = AnomalyDetectionAgent()
        self.exceptions = ExceptionManagementAgent()
        self.learning = learning_agent  # Use the ChromaDB instance
        
        self.processing_log = []
        
        # Store data between operations
        self.current_data = {
            'invoices': pd.DataFrame(),
            'ledger': pd.DataFrame(),
            'bank_statements': pd.DataFrame(),
            'purchase_orders': pd.DataFrame()
        }
    
    def run_ai_reconciliation(self) -> Dict[str, Any]:
        """Execute complete reconciliation workflow with REAL data processing"""
        
        start_time = time.time()
        
        try:
            # Step 1: Load actual data
            self._log_action("Orchestration", "Loading transaction data")
            self._load_current_data()
            
            # Check if we have data to process
            if self.current_data['invoices'].empty:
                return self._return_no_data_results(start_time)
            
            # Step 2: Perform actual matching
            self._log_action("Matching Agent", "Starting semantic matching")
            matching_results = self.matching.perform_matching(
                self.current_data['invoices'],
                self.current_data['ledger'],
                self.current_data['bank_statements']
            )
            
            # Step 3: Real anomaly detection
            self._log_action("Anomaly Detection", "Scanning for anomalies")
            anomalies = self.anomaly.detect_anomalies(matching_results)
            
            # Step 4: Process actual exceptions
            self._log_action("Exception Management", "Processing exceptions")
            exceptions_result = self.exceptions.process_exceptions(matching_results, anomalies)
            
            # Step 5: Update learning with real patterns
            self._log_action("Learning Agent", "Updating ChromaDB knowledge base")
            learning_updates = self.learning.update_patterns(matching_results)
            
            # Step 6: Calculate real statistics
            match_stats = self.matching.calculate_match_statistics(matching_results)
            
            # Calculate final results from actual data
            processing_time = time.time() - start_time
            
            results = {
                'total_transactions': match_stats.get('total_transactions', 0),
                'exact_matches': match_stats.get('exact_matches', 0),
                'llm_matches': match_stats.get('llm_matches', 0),
                'fuzzy_matches': match_stats.get('fuzzy_matches', 0),
                'bank_matches': match_stats.get('bank_matches', 0),
                'exceptions': len(exceptions_result.get('prioritized_exceptions', [])),
                'match_rate': match_stats.get('match_rate', 0),
                'processing_time': processing_time,
                'anomalies_detected': sum(len(anomaly_list) for anomaly_list in anomalies.values()),
                'learning_updates': learning_updates.get('new_patterns', 0),
                'avg_llm_confidence': match_stats.get('llm_avg_confidence', 0),
                'llm_explanations': self.matching.get_match_explanations(matching_results)
            }
            
            self._log_action("Orchestration", f"Reconciliation completed in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            self._log_action("Orchestration", f"Error during reconciliation: {str(e)}", "ERROR")
            return self._return_error_results(start_time, str(e))
    
    def _load_current_data(self):
        """Load current data from session state or database"""
        
        # Try to load from Streamlit session state first
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                self.current_data['invoices'] = getattr(st.session_state, 'invoices', pd.DataFrame())
                self.current_data['ledger'] = getattr(st.session_state, 'ledger', pd.DataFrame())
                self.current_data['bank_statements'] = getattr(st.session_state, 'bank_statements', pd.DataFrame())
                self.current_data['purchase_orders'] = getattr(st.session_state, 'purchase_orders', pd.DataFrame())
        except ImportError:
            pass
        
        # If no data in session state, try to load sample data
        if self.current_data['invoices'].empty:
            self.current_data['invoices'] = self._load_sample_invoices()
            self.current_data['ledger'] = self._load_sample_ledger()
            self.current_data['bank_statements'] = self._load_sample_bank()
    
    def _return_no_data_results(self, start_time: float) -> Dict[str, Any]:
        """Return results when no data is available"""
        return {
            'total_transactions': 0,
            'exact_matches': 0,
            'llm_matches': 0,
            'fuzzy_matches': 0,
            'bank_matches': 0,
            'exceptions': 0,
            'match_rate': 0,
            'processing_time': time.time() - start_time,
            'anomalies_detected': 0,
            'learning_updates': 0,
            'avg_llm_confidence': 0,
            'llm_explanations': [],
            'error': 'No data available for processing'
        }
    
    def _return_error_results(self, start_time: float, error_msg: str) -> Dict[str, Any]:
        """Return results when an error occurs"""
        return {
            'total_transactions': 0,
            'exact_matches': 0,
            'llm_matches': 0,
            'fuzzy_matches': 0,
            'bank_matches': 0,
            'exceptions': 0,
            'match_rate': 0,
            'processing_time': time.time() - start_time,
            'anomalies_detected': 0,
            'learning_updates': 0,
            'avg_llm_confidence': 0,
            'llm_explanations': [],
            'error': error_msg
        }
    
    def process_uploads(self, uploaded_files: Dict[str, Any]) -> Dict[str, Any]:
        """Process uploaded files and store in current_data"""
        
        start_time = time.time()
        self._log_action("Data Ingestion", "Starting file processing")
        
        try:
            # Process files through data ingestion agent
            results = self.data_ingestion.process_files(uploaded_files)
            
            # Store processed data for later use
            for file_type, files in uploaded_files.items():
                if files and file_type in self.current_data:
                    # Process and store the actual data
                    processed_data = []
                    for file in files:
                        if file.name.endswith('.pdf'):
                            data = self.data_ingestion._extract_pdf_data(file)
                        elif file.name.endswith(('.xlsx', '.xls')):
                            data = pd.read_excel(file)
                        else:
                            data = pd.read_csv(file)
                        
                        data = self.data_ingestion._standardize_data(data, file_type)
                        processed_data.append(data)
                    
                    if processed_data:
                        self.current_data[file_type] = pd.concat(processed_data, ignore_index=True)
            
            processing_time = time.time() - start_time
            self._log_action("Data Ingestion", f"Processed {results['total_records']} records in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            self._log_action("Data Ingestion", f"Error: {str(e)}", "ERROR")
            raise
    
    # Keep the sample data methods for fallback
    def _load_sample_invoices(self) -> pd.DataFrame:
        """Load sample invoice data as fallback"""
        from utils.sample_data import generate_sample_data
        
        # Check if sample data files exist
        if not os.path.exists('data/invoices.csv'):
            generate_sample_data()
        
        try:
            return pd.read_csv('data/invoices.csv')
        except:
            # Generate minimal sample data if file doesn't exist
            return pd.DataFrame({
                'invoice_id': [f'INV-{i:04d}' for i in range(1, 21)],
                'vendor': [f'Vendor-{i%5}' for i in range(1, 21)],
                'amount': [100 + i*10 for i in range(1, 21)],
                'date': pd.date_range('2024-01-01', periods=20, freq='D'),
                'description': [f'Sample transaction {i}' for i in range(1, 21)]
            })
    
    def _load_sample_ledger(self) -> pd.DataFrame:
        """Load sample ledger data as fallback"""
        try:
            return pd.read_csv('data/general_ledger.csv')
        except:
            return pd.DataFrame({
                'gl_id': [f'GL-{i:04d}' for i in range(1, 19)],
                'invoice_id': [f'INV-{i:04d}' for i in range(1, 19)],
                'vendor': [f'Vendor-{i%5}' for i in range(1, 19)],
                'amount': [100 + i*10 for i in range(1, 19)],
                'date': pd.date_range('2024-01-01', periods=18, freq='D')
            })
    
    def _load_sample_bank(self) -> pd.DataFrame:
        """Load sample bank statement data as fallback"""
        try:
            return pd.read_csv('data/bank_statements.csv')
        except:
            return pd.DataFrame({
                'transaction_id': [f'TXN-{i:04d}' for i in range(1, 16)],
                'amount': [100 + i*10 for i in range(1, 16)],
                'date': pd.date_range('2024-01-02', periods=15, freq='D'),
                'description': [f'Payment to vendor {i%5}' for i in range(1, 16)]
            })
    
    def _log_action(self, agent: str, action: str, level: str = "INFO"):
        """Log agent actions for audit trail"""
        log_entry = {
            'timestamp': datetime.now(),
            'agent': agent,
            'action': action,
            'level': level
        }
        self.processing_log.append(log_entry)
    
    def get_processing_log(self) -> pd.DataFrame:
        """Return processing log as DataFrame"""
        return pd.DataFrame(self.processing_log)
