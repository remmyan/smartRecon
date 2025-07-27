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

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

# Import configuration and agents
from config import Config
from agents.data_ingestion import create_data_ingestion_agent
from agents.matching import create_matching_agent
from agents.anomaly import create_anomaly_detection_agent
from agents.exceptions import create_exception_management_agent
from agents.learning import create_learning_agent
from agents.orchestration import create_orchestration_agent
from tasks.reconciliation_tasks import (
    ingest_data_task, match_transactions_task, detect_anomalies_task,
    manage_exceptions_task, learn_from_corrections_task
)
from crewai import Crew, Process, Task

llm = Config.get_llm()

# Page configuration
st.set_page_config(
    page_title="AP Reconciliation AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (from previous version - unchanged)
st.markdown("""
<style>
    .main { background-color: #f8f9fa; padding: 1rem; }
    .stButton > button { background-color: #007bff; color: white; border-radius: 5px; }
    .stButton > button:hover { background-color: #0056b3; }
    .main-header { font-size: 2.5rem; color: #007bff; text-align: center; margin-bottom: 1.5rem; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #343a40; margin-bottom: 1rem; font-weight: 600; }
    .card { background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1.5rem; }
    .metric-card { background-color: #e9ecef; border-left: 5px solid #007bff; padding: 1rem; border-radius: 5px; margin: 0.5rem; }
    .stDataFrame table { border-collapse: collapse; width: 100%; }
    .stDataFrame th, .stDataFrame td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }
    .stDataFrame tr:nth-child(even) { background-color: #f8f9fa; }
    .stDataFrame tr:hover { background-color: #e9ecef; }
    .selected-row { background-color: #d1e7ff !important; }
    .sidebar .stSelectbox { margin-bottom: 1rem; }
    .sidebar .active { font-weight: bold; color: #007bff; }
    .footer { text-align: center; color: #6c757d; font-size: 0.875rem; margin-top: 2rem; padding: 1rem; border-top: 1px solid #dee2e6; }
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
    except Exception as e:
        result_queue.put(("error", str(e)))

def process_uploaded_files(uploaded_files):
    processed = {}
    # ERP processing
    if 'erp' in uploaded_files:
        file = uploaded_files['erp']
        if file.name.endswith('.csv'):
            processed['erp'] = pd.read_csv(file)
    # Bank processing
    if 'bank' in uploaded_files:
        file = uploaded_files['bank']
        if file.name.endswith('.csv'):
            processed['bank'] = pd.read_csv(file)
    # Invoices processing (multiple CSVs)
    if 'invoices' in uploaded_files:
        invoice_dfs = []
        for file in uploaded_files['invoices']:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                invoice_dfs.append(df)
        if invoice_dfs:
            processed['invoices'] = pd.concat(invoice_dfs, ignore_index=True)
        else:
            processed['invoices'] = pd.DataFrame()
    return processed

def start_reconciliation():
    st.session_state.processing_status = "running"
    st.session_state.agent_outputs = []
    
    # Process files
    processed_data = process_uploaded_files(st.session_state.uploaded_files)
    
    # Create agents dynamically
    data_ingestion_agent = create_data_ingestion_agent()
    matching_agent = create_matching_agent()
    anomaly_detection_agent = create_anomaly_detection_agent()
    exception_management_agent = create_exception_management_agent()
    learning_agent = create_learning_agent()
    orchestration_agent = create_orchestration_agent()
    
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

def check_processing_status():
    if st.session_state.processing_status == "running" and st.session_state.result_queue:
        try:
            result_type, result = st.session_state.result_queue.get_nowait()
            if result_type == "success":
                st.session_state.crew_results = result
                st.session_state.processing_status = "completed"
                # Mock exceptions for demo (in production, pull from exception_management_agent output)
                st.session_state.exceptions = pd.DataFrame([
                    {'id': 'EX001', 'type': 'Amount Mismatch', 'amount': 1250.0, 'confidence': 0.45, 'status': 'Pending'},
                    {'id': 'EX002', 'type': 'Duplicate', 'amount': 850.75, 'confidence': 0.62, 'status': 'Pending'}
                ])
            else:
                st.session_state.processing_status = "error"
                st.error(f"Processing failed: {result}")
        except queue.Empty:
            pass

def process_correction(correction_data):
    """Helper to create and run a task for the Learning Agent."""
    try:
        learning_agent = create_learning_agent()
        
        # Create a simple task for processing the correction
        learn_task = Task(
            description=f"Process this human correction: {json.dumps(correction_data)}",
            expected_output="Updated patterns and analytics",
            agent=learning_agent
        )
        
        # Run a mini-crew for just this task
        learn_crew = Crew(
            agents=[learning_agent],
            tasks=[learn_task],
            verbose=2
        )
        
        result = learn_crew.kickoff()
        return result
    except Exception as e:
        return f"Error processing correction: {str(e)}"

# Main app
st.markdown('<h1 class="main-header">AP Reconciliation AI System</h1>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    pages = ["Dashboard", "Upload Documents", "Processing Monitor", "Results & Analytics"]
    page = st.radio("Navigate", pages, label_visibility="collapsed")
    st.markdown("---")
    st.info("Select a page to get started.")

check_processing_status()

if page == "Dashboard":
    st.markdown('<div class="sub-header">Dashboard Overview</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">**Processing Status**<br>' + st.session_state.processing_status.upper() + '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">**Uploaded Files**<br>' + str(len(st.session_state.uploaded_files)) + '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">**Exceptions**<br>' + str(len(st.session_state.exceptions)) + '</div>', unsafe_allow_html=True)

elif page == "Upload Documents":
    st.markdown('<div class="sub-header">Upload Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    erp_file = st.file_uploader("ERP Data (CSV)", type="csv")
    bank_file = st.file_uploader("Bank Statements (CSV)", type="csv")
    invoice_files = st.file_uploader("Invoices (CSV)", type="csv", accept_multiple_files=True)
    
    if erp_file:
        st.session_state.uploaded_files['erp'] = erp_file
    if bank_file:
        st.session_state.uploaded_files['bank'] = bank_file
    if invoice_files:
        st.session_state.uploaded_files['invoices'] = invoice_files
    
    if st.button("Start Reconciliation") and st.session_state.uploaded_files:
        start_reconciliation()
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Processing Monitor":
    st.markdown('<div class="sub-header">Processing Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.processing_status == "running":
        st.progress(0.5)  # Simulated progress
    st.write("Agent Outputs:")
    for output in st.session_state.agent_outputs:
        st.write(output)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Results & Analytics":
    st.markdown('<div class="sub-header">Results & Analytics</div>', unsafe_allow_html=True)
    if st.session_state.crew_results:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Matching Success Rate")
        fig = px.pie(names=['Matched', 'Unmatched'], values=[95, 5], title="Matching Distribution", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
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
                    correction_data = {
                        'exception_id': selected_id,
                        'resolution': resolution,
                        'notes': notes,
                        'timestamp': datetime.now().isoformat(),
                        'amount': exception['amount'],
                        'type': exception['type']
                    }
                    
                    with st.spinner("Processing correction..."):
                        learn_result = process_correction(correction_data)
                    
                    if "error" in learn_result.lower():
                        st.error(learn_result)
                    else:
                        st.success("Correction processed and learned from successfully!")
                        st.session_state.exceptions.loc[st.session_state.exceptions['id'] == selected_id, 'status'] = 'Resolved'
                
                # Display exceptions table with highlighting
                styled_df = st.session_state.exceptions.style.apply(lambda x: ['background-color: #d1e7ff' if x['id'] == selected_id else '' for _ in x], axis=1)
                st.dataframe(styled_df)
        else:
            st.info("No exceptions to review.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run reconciliation to see results.")

# Footer
st.markdown('<div class="footer">AP Reconciliation AI v1.0 | Powered by CrewAI & OpenAI | © 2025</div>', unsafe_allow_html=True)
