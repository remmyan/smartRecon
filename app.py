import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import time
import json
from datetime import datetime
import threading
import queue
import plotly.express as px
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

# Import configuration and agents
from agents.data_ingestion import data_ingestion_agent
from agents.matching import matching_agent
from agents.anomaly import anomaly_detection_agent
from agents.exceptions import exception_management_agent
from agents.learning import learning_agent
from agents.orchestration import orchestration_agent
from tasks.reconciliation_tasks import (
    ingest_data_task, match_transactions_task, detect_anomalies_task,
    manage_exceptions_task, learn_from_corrections_task
)
from crewai import Crew, Process
from config import Config

llm = Config.OPENAI_MODEL

# Page configuration
st.set_page_config(
    page_title="AP Reconciliation AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.agent-status {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.status-running { background-color: #fff3cd; border-left: 4px solid #ffc107; }
.status-completed { background-color: #d4edda; border-left: 4px solid #28a745; }
.status-error { background-color: #f8d7da; border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"
if 'agent_outputs' not in st.session_state:
    st.session_state.agent_outputs = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'crew_results' not in st.session_state:
    st.session_state.crew_results = None
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = None
if 'processing_thread' not in st.session_state:
    st.session_state.processing_thread = None
if 'exceptions' not in st.session_state:
    st.session_state.exceptions = pd.DataFrame(columns=['id', 'type', 'amount', 'confidence', 'status'])

def run_crew_async(crew, result_queue):
    try:
        result = crew.kickoff()
        result_queue.put(("success", result))
    except litellm.RateLimitError as e:
        wait_time = 6  # Extract from error if possible
        time.sleep(wait_time)
        result = crew.kickoff()  # Retry    
    except Exception as e:
        result_queue.put(("error", str(e)))

def process_uploaded_files(uploaded_files):
    processed = {}
    # ERP processing
    if 'erp' in uploaded_files:
        file = uploaded_files['erp']
        try:
            # Read CSV with error handling: skip bad lines
            processed['erp'] = pd.read_csv(file, on_bad_lines='skip', engine='python')  # 'python' engine handles complex parsing
            st.success(f"ERP file loaded: {len(processed['erp'])} rows")
        except Exception as e:
            st.error(f"Error loading ERP file: {str(e)}. Check for inconsistent columns.")
            processed['erp'] = pd.DataFrame()  # Empty DF as fallback
    # Bank processing
    if 'bank' in uploaded_files:
        file = uploaded_files['bank']
        try:
            # Read CSV with error handling: skip bad lines
            processed['bank'] = pd.read_csv(file, on_bad_lines='skip', engine='python')  # 'python' engine handles complex parsing
            st.success(f"Bank statement loaded: {len(processed['erp'])} rows")
        except Exception as e:
            st.error(f"Error loading bank statements: {str(e)}. Check for inconsistent columns.")
            processed['bank'] = pd.DataFrame()  # Empty DF as fallback
    # Invoices (OCR simulation for demo)
    if 'invoices' in uploaded_files:
        processed['invoices'] = [f"OCR text for {file.name}" for file in uploaded_files['invoices']]
    return processed

def start_reconciliation():
    logger.info("Starting reconciliation - initializing Crew")
    st.session_state.processing_status = "running"
    st.session_state.agent_outputs = []
    
    # Process files
    processed_data = process_uploaded_files(st.session_state.uploaded_files)
    logger.info(f"Processed data: {processed_data}")
    
    # Create crew
    reconciliation_crew = Crew(
        agents=[
            data_ingestion_agent, matching_agent, anomaly_detection_agent,
            exception_management_agent, learning_agent
        ],
        tasks=[
            ingest_data_task, match_transactions_task, detect_anomalies_task,
            manage_exceptions_task, learn_from_corrections_task
        ],
        manager_agent=orchestration_agent,
        process=Process.hierarchical,
        verbose=True
    )
    
    # Run asynchronously
    result_queue = queue.Queue()
    thread = threading.Thread(target=run_crew_async, args=(reconciliation_crew, result_queue))
    thread.start()
    
    st.session_state.result_queue = result_queue
    st.session_state.processing_thread = thread
    logger.info("Reconciliation thread started")
        

def check_processing_status():
    if st.session_state.processing_status == "running" and st.session_state.result_queue:
        try:
            result_type, result = st.session_state.result_queue.get_nowait()
            if result_type == "success":
                st.session_state.crew_results = result
                st.session_state.processing_status = "completed"
                # Mock exceptions for demo (replace with actual from exception_management_agent)
                st.session_state.exceptions = pd.DataFrame([
                    {'id': 'EX001', 'type': 'Amount Mismatch', 'amount': 1250.0, 'confidence': 0.45, 'status': 'Pending'},
                    {'id': 'EX002', 'type': 'Duplicate', 'amount': 850.75, 'confidence': 0.62, 'status': 'Pending'}
                ])
            else:
                st.session_state.processing_status = "error"
        except queue.Empty:
            pass

# Main app
st.markdown('<h1 class="main-header">AP Reconciliation AI System</h1>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    page = st.selectbox("Navigate", ["Dashboard", "Upload Documents", "Processing Monitor", "Results & Analytics"])

check_processing_status()

if page == "Dashboard":
    st.header("Dashboard Overview")
    if st.button("Test LLM Directly"):
        try:
            from config import Config
            llm = Config.OPENAI_MODEL
            logger.info("Testing LLM invoke in app")
            response = llm.invoke("Test prompt: What is 2+2?")
            st.success(f"LLM Response: {response}")
            logger.info("LLM test successful")
        except Exception as e:
            st.error(f"LLM test failed: {str(e)}")
            logger.error(f"LLM test error: {str(e)}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Processing Status", st.session_state.processing_status.upper())
    col2.metric("Uploaded Files", len(st.session_state.uploaded_files))
    col3.metric("Exceptions", len(st.session_state.exceptions))

elif page == "Upload Documents":
    st.header("Upload Documents")
    erp_file = st.file_uploader("ERP Data (CSV)", type="csv")
    bank_file = st.file_uploader("Bank Statements (CSV)", type="csv")
    invoice_files = st.file_uploader("Invoices (PDF/JPG)", type=["pdf", "jpg"], accept_multiple_files=True)
    
    if erp_file:
        st.session_state.uploaded_files['erp'] = erp_file
    if bank_file:
        st.session_state.uploaded_files['bank'] = bank_file
    if invoice_files:
        st.session_state.uploaded_files['invoices'] = invoice_files
    
    if st.button("Start Reconciliation") and st.session_state.uploaded_files:
        start_reconciliation()

elif page == "Processing Monitor":
    st.header("Processing Monitor")
    if st.session_state.processing_status == "running":
        st.progress(0.5)  # Simulated progress
    st.write("Agent Outputs:")
    for output in st.session_state.agent_outputs:
        st.write(output)

elif page == "Results & Analytics":
    st.header("Results & Analytics")
    if st.session_state.crew_results:
        # Sample results visualization
        st.subheader("Matching Success Rate")
        fig = px.pie(names=['Matched', 'Unmatched'], values=[95, 5])
        st.plotly_chart(fig)
        
        # Human Correction Interface
        st.subheader("Human Correction Interface")
        if not st.session_state.exceptions.empty:
            selected_id = st.selectbox("Select Exception to Review", st.session_state.exceptions['id'])
            if selected_id:
                exception = st.session_state.exceptions[st.session_state.exceptions['id'] == selected_id].iloc[0]
                st.write(f"Type: {exception['type']}, Amount: {exception['amount']}, Confidence: {exception['confidence']}")
                
                resolution = st.selectbox("Resolution", ["Approve", "Reject", "Manual Match"])
                notes = st.text_area("Notes")
                if st.button("Submit Correction"):
                    # Simulate sending to Learning Agent
                    correction = {
                        'exception_id': selected_id,
                        'resolution': resolution,
                        'notes': notes,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.success("Correction submitted!")
                    # Update status
                    st.session_state.exceptions.loc[st.session_state.exceptions['id'] == selected_id, 'status'] = 'Resolved'
        else:
            st.info("No exceptions to review.")
    else:
        st.info("Run reconciliation to see results.")
