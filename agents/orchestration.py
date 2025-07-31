from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
from agents.data_ingestion import DataIngestionAgent
from agents.matching import MatchingAgent
from agents.anomaly import AnomalyDetectionAgent
from agents.exceptions import ExceptionManagementAgent
from agents.learning import LearningAgent  # Imported but used only for overrides
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recon_log.txt"),  # Logs to file for persistence
        logging.StreamHandler()  # Logs to console
    ]
)

class Orchestrator:
    def __init__(self):
        self.data_ingestion = DataIngestionAgent()
        self.anomaly = AnomalyDetectionAgent()
        self.exceptions = ExceptionManagementAgent()
        self.processing_log = []
        self.matching_agent = MatchingAgent()
        self.learning_agent = LearningAgent()  # Imported but used only for overrides
        self.logger = logging.getLogger("Orchestrator")

    def run_ai_reconciliation(self, uploaded_files: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        self._log_action("Orchestration", "Starting reconciliation (no learning in recon)")
        self.logger.info(uploaded_files)

        # Step 1: Data Ingestion
        ingestion_results = self.data_ingestion.process_files(uploaded_files)
        raw_invoices = pd.DataFrame(ingestion_results.get('invoices', []))
        raw_ledger = pd.DataFrame(ingestion_results.get('ledger', []))
        raw_bank = pd.DataFrame(ingestion_results.get('bank_statements', []))
        self.logger.info("results from data ingestion: %s", ingestion_results)

        # Step 1: Pre-matching anomaly detection (deduplication)
        invoices, invoice_dupes = self.anomaly.remove_duplicates(raw_invoices)
        ledger, ledger_dupes = self.anomaly.remove_duplicates(raw_ledger)
        bank, bank_dupes = self.anomaly.remove_duplicates(raw_bank)
        self.logger.info("Deduplication results: Invoices: %d, Ledger: %d, Bank: %d",
                         len(invoices), len(ledger), len(bank))
        
        # Step 2: Matching with optional learning collaboration (queries if available, but no storage during recon)
        self.logger.info("Matching", "Performing hybrid matching")
        matching_results = self.matching_agent.perform_matching(invoices, ledger, bank)
        self.logger.info("Matching results: %s", matching_results)

        # Step 3: Anomaly Detection
        self._log_action("Anomaly Detection", "Scanning for anomalies")
       # anomalies = self.anomaly.detect_anomalies(matching_results)

        # Step 4: Exception Management
        self._log_action("Exceptions", "Processing exceptions")
       # exceptions_result = self.exceptions.process_exceptions(matching_results, anomalies)

        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            'matching_results': matching_results,
#            'anomalies': anomalies,
#            'exceptions': exceptions_result,
            'processing_time': processing_time,
            'total_records': ingestion_results.get('total_records', 0),
            'message': "Recon complete. Use handle_user_overrides() for any corrections to invoke learning.",
            'invoices': len(invoices),
            'ledger': len(ledger),
            'bank': len(bank)
        }

    def handle_user_overrides(self, exceptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle user overrides post-recon and invoke Learning Agent to store corrections."""
        corrected = []
        for exception in exceptions:
            # Simulate or get user input (e.g., from Streamlit)
            original = exception['record']
            corrected_record = original.copy()  # Example correction
            corrected_record['vendor'] = "User Corrected Vendor"  # Replace with actual user input
            corrected_record['correction_reason'] = "User override for mismatch"

            # Invoke Learning Agent to store
            self.learning_agent.store_correction(original, corrected_record)
            corrected.append(corrected_record)

        return corrected

    def _log_action(self, agent: str, action: str):
        log_entry = {'timestamp': datetime.now().isoformat(), 'agent': agent, 'action': action}
        self.processing_log.append(log_entry)

# Global instance
orchestrator = Orchestrator()
