import os
import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from agents.orchestration import OrchestrationAgent
from utils.sample_data import generate_sample_data
from utils.database import init_database, get_connection
from config import Config

# Page configuration
st.set_page_config(
    page_title="AP Reconciliation System with ChatGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check AI configurations
if not Config.OPENAI_API_KEY and not Config.GROK_API_KEY:
    st.error("⚠️ No AI API Key configured. Please set either OPENAI_API_KEY or GROK_API_KEY in your .env file.")
    st.stop()

# Initialize database and agents
try:
    init_database()
    orchestrator = OrchestrationAgent()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .llm-card {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 AI-Powered AP Reconciliation System")
    st.markdown("*Automated accounts payable reconciliation with ChatGPT semantic matching and ChromaDB learning*")
    
    # Initialize session state for data persistence
    if 'reconciliation_results' not in st.session_state:
        st.session_state.reconciliation_results = None
    if 'uploaded_data_stats' not in st.session_state:
        st.session_state.uploaded_data_stats = None
    
    # API Status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        active_model = "Grok-3" if Config.GROK_API_KEY else "GPT-4"
        active_api = "Grok AI" if Config.GROK_API_KEY else "ChatGPT"
        st.markdown(f"""
        <div class="llm-card">
            <h4>🤖 {active_api} + ChromaDB</h4>
            <p>Model: {active_model} | Vector DB: Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Dashboard", 
        "Data Upload", 
        "AI Reconciliation", 
        "Match Analysis",
        "Exception Management",
        "Learning Analytics",
        "System Configuration"
    ])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "AI Reconciliation":
        show_ai_reconciliation()
    elif page == "Match Analysis":
        show_match_analysis()
    elif page == "Exception Management":
        show_exceptions()
    elif page == "Learning Analytics":
        show_learning_analytics()
    elif page == "System Configuration":
        show_configuration()

def show_dashboard():
    st.header("📊 System Dashboard")
    
    # Generate sample data if needed
    if st.sidebar.button("Generate Sample Data"):
        with st.spinner("Generating sample transaction data..."):
            generate_sample_data()
            # Clear any existing results to force refresh
            st.session_state.reconciliation_results = None
        st.success("Sample data generated successfully!")
    
    # Get real statistics from last reconciliation or current data
    if st.session_state.reconciliation_results:
        results = st.session_state.reconciliation_results
        total_transactions = results.get('total_transactions', 0)
        match_rate = results.get('match_rate', 0)
        exceptions = results.get('exceptions', 0)
        total_amount = results.get('total_amount', 0)  # We'll add this calculation
    else:
        # Get basic stats from available data
        stats = get_current_data_stats()
        total_transactions = stats['total_records']
        match_rate = 0  # No reconciliation run yet
        exceptions = 0
        total_amount = stats['total_amount']
    
    # Key metrics with real data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_transactions:,}</h3>
            <p>Total Transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{match_rate:.1f}%</h3>
            <p>Auto-Match Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{exceptions}</h3>
            <p>Open Exceptions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${total_amount:,.0f}</h3>
            <p>Total Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts with dynamic data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Performance")
        if st.session_state.reconciliation_results:
            results = st.session_state.reconciliation_results
            
            # Create performance chart
            performance_data = {
                'Match Type': ['Exact Matches', 'AI Matches', 'Fuzzy Matches', 'Exceptions'],
                'Count': [
                    results.get('exact_matches', 0),
                    results.get('llm_matches', 0), 
                    results.get('fuzzy_matches', 0),
                    results.get('exceptions', 0)
                ],
                'Color': ['#28a745', '#17a2b8', '#ffc107', '#dc3545']
            }
            
            fig = px.bar(
                performance_data, 
                x='Match Type', 
                y='Count',
                color='Color',
                color_discrete_map={color: color for color in performance_data['Color']}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run AI Reconciliation to see performance data")
    
    with col2:
        st.subheader("Learning Progress")
        
        # Get learning insights from ChromaDB
        try:
            from agents.learning import learning_agent
            insights = learning_agent.get_learning_insights()
            
            pattern_data = insights.get('pattern_breakdown', {})
            if pattern_data and any(pattern_data.values()):
                fig = px.pie(
                    values=list(pattern_data.values()),
                    names=list(pattern_data.keys()),
                    title="Knowledge Base Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No learning patterns yet. Process some data to build knowledge base.")
                
        except Exception as e:
            st.error(f"Error loading learning insights: {e}")
    
    # Recent activity with real data
    st.subheader("Recent System Activity")
    
    if hasattr(orchestrator, 'processing_log') and orchestrator.processing_log:
        # Show actual processing log
        log_df = orchestrator.get_processing_log()
        if not log_df.empty:
            # Show last 5 entries
            recent_log = log_df.tail(5).copy()
            recent_log['timestamp'] = pd.to_datetime(recent_log['timestamp'])
            st.dataframe(recent_log, use_container_width=True)
        else:
            show_placeholder_activity()
    else:
        show_placeholder_activity()

def show_placeholder_activity():
    """Show placeholder activity when no real log exists"""
    activity_data = {
        'Timestamp': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(hours=1),
        ],
        'Agent': ['System', 'Data Ingestion', 'Learning Agent'],
        'Action': [
            'System initialized with ChromaDB',
            'Ready to process uploaded files',
            'Knowledge base ready for pattern learning'
        ],
        'Level': ['INFO', 'INFO', 'INFO']
    }
    
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, use_container_width=True)

def get_current_data_stats():
    """Get statistics from currently available data"""
    stats = {
        'total_records': 0,
        'total_amount': 0,
        'data_sources': 0
    }
    
    try:
        # Check session state for uploaded data
        data_sources = ['invoices', 'ledger', 'bank_statements', 'purchase_orders']
        
        for source in data_sources:
            if hasattr(st.session_state, source):
                data = getattr(st.session_state, source)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    stats['total_records'] += len(data)
                    stats['data_sources'] += 1
                    
                    # Calculate total amount if amount column exists
                    if 'amount' in data.columns:
                        stats['total_amount'] += data['amount'].sum()
        
        # Fallback to sample data if no uploaded data
        if stats['total_records'] == 0:
            try:
                if os.path.exists('data/invoices.csv'):
                    sample_invoices = pd.read_csv('data/invoices.csv')
                    stats['total_records'] = len(sample_invoices)
                    if 'amount' in sample_invoices.columns:
                        stats['total_amount'] = sample_invoices['amount'].sum()
                else:
                    # Default values for fresh installation
                    stats['total_records'] = 200
                    stats['total_amount'] = 250000
            except Exception:
                pass
                
    except Exception as e:
        print(f"Error getting data stats: {e}")
    
    return stats

def show_data_upload():
    st.header("📁 Data Upload & Ingestion")
    
    tab1, tab2 = st.tabs(["File Upload", "Database Integration"])
    
    with tab1:
        st.subheader("Upload Financial Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_invoices = st.file_uploader(
                "Upload Invoices", 
                type=['csv', 'xlsx', 'pdf'],
                accept_multiple_files=True,
                key="invoices"
            )
            
            uploaded_pos = st.file_uploader(
                "Upload Purchase Orders",
                type=['csv', 'xlsx', 'pdf'],
                accept_multiple_files=True,
                key="pos"
            )
        
        with col2:
            uploaded_bank = st.file_uploader(
                "Upload Bank Statements",
                type=['csv', 'xlsx', 'pdf'],
                accept_multiple_files=True,
                key="bank"
            )
            
            uploaded_ledger = st.file_uploader(
                "Upload General Ledger",
                type=['csv', 'xlsx', 'pdf'],
                accept_multiple_files=True,
                key="ledger"
            )
        
        if st.button("Process Uploaded Files"):
            if any([uploaded_invoices, uploaded_pos, uploaded_bank, uploaded_ledger]):
                with st.spinner("Processing files with Data Ingestion Agent..."):
                    try:
                        results = orchestrator.process_uploads({
                            'invoices': uploaded_invoices,
                            'purchase_orders': uploaded_pos,
                            'bank_statements': uploaded_bank,
                            'general_ledger': uploaded_ledger
                        })
                        
                        # Store results in session state
                        st.session_state.uploaded_data_stats = results
                        
                        st.success(f"✅ Successfully processed {results['total_records']} records")
                        
                        # Show processing summary with real data
                        st.subheader("Processing Summary")
                        summary_data = []
                        
                        for doc_type, files in [
                            ("Invoices", uploaded_invoices),
                            ("Purchase Orders", uploaded_pos), 
                            ("Bank Statements", uploaded_bank),
                            ("General Ledger", uploaded_ledger)
                        ]:
                            record_count = results.get(doc_type.lower().replace(' ', '_'), 0)
                            status = "✅ Complete" if record_count > 0 else "⚪ No data"
                            summary_data.append({
                                "Document Type": doc_type,
                                "Records": record_count,
                                "Status": status
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing files: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least one file to process")
    
    with tab2:
        st.subheader("ERP System Integration")
        st.info("Direct ERP integration capabilities - connect to SAP, NetSuite, Oracle, etc.")
        
        col1, col2 = st.columns(2)
        with col1:
            erp_system = st.selectbox("ERP System", ["SAP", "NetSuite", "Oracle", "QuickBooks", "Other"])
            connection_string = st.text_input("Connection String", type="password")
        
        with col2:
            sync_frequency = st.selectbox("Sync Frequency", ["Real-time", "Hourly", "Daily", "Weekly"])
            last_sync = st.text_input("Last Sync", value="Not configured", disabled=True)
        
        if st.button("Test Connection"):
            st.info("🔧 ERP integration is planned for future release")

def show_ai_reconciliation():
    st.header("🤖 AI-Powered Reconciliation Process")
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        use_llm = st.checkbox("Enable ChatGPT Semantic Matching", value=True)
        llm_confidence_threshold = st.slider("LLM Confidence Threshold", 60, 95, 70)
    
    with col2:
        max_llm_comparisons = st.slider("Max LLM Comparisons (cost control)", 5, 50, 20)
        enable_fuzzy_fallback = st.checkbox("Enable Fuzzy Matching Fallback", value=True)
    
    if st.button("🚀 Start AI Reconciliation"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update orchestrator settings
        orchestrator.matching.use_llm = use_llm
        
        steps = [
            ("Data Ingestion Agent", "Loading and validating data sources..."),
            ("ChatGPT Matching Agent", "Performing AI semantic analysis..."),
            ("Fuzzy Matching Agent", "Applying traditional fuzzy matching..."),
            ("Anomaly Detection Agent", "AI-powered fraud detection..."),
            ("Exception Management Agent", "Intelligent exception categorization..."),
            ("Learning Agent", "Updating ChromaDB knowledge base..."),
            ("Orchestration Agent", "Finalizing results...")
        ]
        
        try:
            # Execute real reconciliation
            results = orchestrator.run_ai_reconciliation()
            
            # Update progress
            for i, (agent, message) in enumerate(steps):
                status_text.text(f"Running {agent}: {message}")
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.5)  # Reduced sleep time
            
            # Store results in session state
            st.session_state.reconciliation_results = results
            
            if results.get('error'):
                st.error(f"❌ Error during reconciliation: {results['error']}")
            else:
                st.success("🎉 AI Reconciliation completed successfully!")
                
                # Display real results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Processed", f"{results['total_transactions']:,}")
                    st.metric("Exact Matches", f"{results['exact_matches']:,}")
                
                with col2:
                    st.metric("🤖 AI Matches", f"{results['llm_matches']:,}")
                    st.metric("Fuzzy Matches", f"{results['fuzzy_matches']:,}")
                
                with col3:
                    st.metric("Exceptions", f"{results['exceptions']:,}")
                    st.metric("Match Rate", f"{results['match_rate']:.1f}%")
                
                with col4:
                    st.metric("AI Confidence", f"{results.get('avg_llm_confidence', 0):.1f}%")
                    st.metric("Processing Time", f"{results['processing_time']:.1f}s")
                
                # Show AI insights if available
                if results.get('llm_explanations'):
                    st.subheader("🧠 AI Matching Insights")
                    explanations_df = pd.DataFrame(results['llm_explanations'])
                    st.dataframe(explanations_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Reconciliation failed: {str(e)}")
            status_text.text("Error occurred during processing")
            progress_bar.progress(0)

def show_match_analysis():
    st.header("🔍 AI Match Analysis")
    
    if not st.session_state.reconciliation_results:
        st.info("🔄 Please run AI Reconciliation first to see match analysis")
        return
    
    results = st.session_state.reconciliation_results
    
    # Show match summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Matches", 
                 results['exact_matches'] + results['llm_matches'] + results['fuzzy_matches'])
    
    with col2:
        st.metric("AI Success Rate", 
                 f"{(results['llm_matches'] / max(results['total_transactions'], 1) * 100):.1f}%")
    
    with col3:
        st.metric("Overall Accuracy", f"{results['match_rate']:.1f}%")
    
    # Match breakdown chart
    st.subheader("Match Type Breakdown")
    
    match_data = {
        'Match Type': ['🤖 AI Semantic', '⚡ Exact', '🔍 Fuzzy', '❌ Unmatched'],
        'Count': [
            results['llm_matches'],
            results['exact_matches'],
            results['fuzzy_matches'],
            results['exceptions']
        ],
        'Percentage': [
            (results['llm_matches'] / max(results['total_transactions'], 1)) * 100,
            (results['exact_matches'] / max(results['total_transactions'], 1)) * 100,
            (results['fuzzy_matches'] / max(results['total_transactions'], 1)) * 100,
            (results['exceptions'] / max(results['total_transactions'], 1)) * 100
        ]
    }
    
    match_df = pd.DataFrame(match_data)
    
    # Show as both chart and table
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(match_df, values='Count', names='Match Type', 
                    title="Match Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(match_df, use_container_width=True)
    
    # AI explanations if available
    if results.get('llm_explanations'):
        st.subheader("🤖 AI Matching Explanations")
        explanations_df = pd.DataFrame(results['llm_explanations'])
        
        # Add filtering
        if not explanations_df.empty:
            min_confidence = st.slider("Minimum Confidence", 0, 100, 70)
            filtered_df = explanations_df[explanations_df['confidence'] >= min_confidence]
            st.dataframe(filtered_df, use_container_width=True)

def show_exceptions():
    st.header("⚠️ Exception Management")
    
    if not st.session_state.reconciliation_results:
        st.info("🔄 Please run AI Reconciliation first to see exceptions")
        return
    
    results = st.session_state.reconciliation_results
    
    st.metric("Total Exceptions", results['exceptions'])
    
    if results['exceptions'] > 0:
        st.info("🚧 Exception details and management interface will be enhanced in the next update")
        
        # Placeholder exception data for demonstration
        exception_data = {
            'Exception ID': [f'EXC-{i:03d}' for i in range(1, min(results['exceptions'] + 1, 6))],
            'Type': ['Amount Mismatch', 'Missing PO', 'Duplicate', 'Unusual Amount', 'Date Variance'][:results['exceptions']],
            'Priority': ['High', 'Medium', 'High', 'Critical', 'Low'][:results['exceptions']],
            'Status': ['Open'] * min(results['exceptions'], 5)
        }
        
        exception_df = pd.DataFrame(exception_data)
        st.dataframe(exception_df, use_container_width=True)
    else:
        st.success("🎉 No exceptions found! All transactions matched successfully.")

def show_learning_analytics():
    st.header("📊 AI Learning Analytics with ChromaDB")
    
    try:
        from agents.learning import learning_agent
        insights = learning_agent.get_learning_insights()
        
        # Learning statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patterns", insights['total_patterns'])
        
        with col2:
            pattern_breakdown = insights.get('pattern_breakdown', {})
            total_collections = sum(pattern_breakdown.values())
            st.metric("Knowledge Collections", len(pattern_breakdown))
        
        with col3:
            learning_stats = insights.get('learning_stats', {})
            accuracy_improvement = learning_stats.get('accuracy_improvement', 0)
            st.metric("Accuracy Improvement", f"+{accuracy_improvement:.1f}%")
        
        # Pattern breakdown chart
        if pattern_breakdown and any(pattern_breakdown.values()):
            st.subheader("Knowledge Base Distribution")
            
            fig = px.bar(
                x=list(pattern_breakdown.keys()),
                y=list(pattern_breakdown.values()),
                title="Patterns by Collection Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top vendors
        top_vendors = insights.get('top_vendors', [])
        if top_vendors:
            st.subheader("Top Vendors by Pattern Count")
            vendor_df = pd.DataFrame(top_vendors)
            st.dataframe(vendor_df, use_container_width=True)
        
        # Recent patterns
        recent_patterns = insights.get('recent_patterns', [])
        if recent_patterns:
            st.subheader("Recent Learning Patterns")
            st.info(f"Showing {len(recent_patterns)} most recent patterns learned by the system")
            
            for i, pattern in enumerate(recent_patterns[:5]):  # Show top 5
                with st.expander(f"Pattern {i+1}: {pattern['pattern'].get('match_type', 'Unknown')}"):
                    st.json(pattern['pattern'])
        
        # ChromaDB statistics
        st.subheader("ChromaDB Performance")
        
        chroma_stats = {
            'Metric': ['Vector Collections', 'Total Embeddings', 'Storage Type', 'Similarity Search'],
            'Value': [
                len(pattern_breakdown),
                insights['total_patterns'],
                'ChromaDB Persistent',
                'Cosine Similarity'
            ]
        }
        
        st.dataframe(pd.DataFrame(chroma_stats), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading learning analytics: {e}")
        st.info("Make sure ChromaDB is properly initialized and the learning agent is working")

def show_configuration():
    st.header("⚙️ System Configuration")
    
    tab1, tab2, tab3 = st.tabs(["Agent Settings", "Business Rules", "ChromaDB Settings"])
    
    with tab1:
        st.subheader("Agent Configuration")
        
        # Matching Agent settings
        with st.expander("Matching Agent Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                fuzzy_threshold = st.slider("Fuzzy Match Threshold", 60, 100, Config.FUZZY_THRESHOLD)
                active_api = "Grok" if Config.GROK_API_KEY else "OpenAI"
                semantic_model = st.selectbox(
                    "Semantic Model", 
                    ["Grok-3", "GPT-4", "GPT-3.5", "Local BERT"],
                    index=0 if active_api == "Grok" else 1
                )
                max_processing_time = st.number_input("Max Processing Time (seconds)", 1, 300, 30)
            
            with col2:
                confidence_threshold = st.slider("Auto-Approval Confidence", 70, 100, 90)
                if active_api == "Grok":
                    temperature = st.slider("Temperature", 0.0, 1.0, Config.GROK_TEMPERATURE)
                    max_tokens = st.number_input("Max Tokens", 100, 4000, Config.GROK_MAX_TOKENS)
                else:
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
                    max_tokens = st.number_input("Max Tokens", 100, 4000, Config.OPENAI_MAX_TOKENS)
                enable_learning = st.checkbox("Enable Continuous Learning", True)
                parallel_processing = st.checkbox("Parallel Processing", True)
        
        # ChromaDB settings
        with st.expander("ChromaDB Learning Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                rag_top_k = st.slider("RAG Retrieval Count", 3, 20, Config.RAG_TOP_K)
                similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, Config.RAG_SIMILARITY_THRESHOLD)
            
            with col2:
                embedding_model = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
                vector_distance = st.selectbox("Distance Metric", ["cosine", "euclidean", "manhattan"])
    
    with tab2:
        st.subheader("Business Rules Configuration")
        
        rules_data = {
            'Rule Name': ['Amount Tolerance', 'Date Window', 'Vendor Matching', 'Currency Handling'],
            'Current Setting': [f'±{Config.AMOUNT_TOLERANCE*100}%', f'±{Config.DATE_WINDOW} days', 'Fuzzy + Exact', 'Auto-convert'],
            'Status': ['Active', 'Active', 'Active', 'Active'],
            'Last Modified': ['2024-01-10', '2024-01-08', '2024-01-12', '2024-01-05']
        }
        
        rules_df = pd.DataFrame(rules_data)
        edited_rules = st.data_editor(rules_df, use_container_width=True)
        
        if st.button("Save Business Rules"):
            st.success("✅ Business rules would be updated (feature coming soon)")
    
    with tab3:
        st.subheader("ChromaDB Vector Database Settings")
        
        try:
            from agents.learning import learning_agent
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Database Path", Config.CHROMA_DB_PATH, disabled=True)
                st.text_input("Collection Name", Config.CHROMA_COLLECTION_NAME, disabled=True)
                
            with col2:
                st.text_input("Embedding Model", Config.EMBEDDING_MODEL, disabled=True)
                
                if st.button("Test ChromaDB Connection"):
                    try:
                        insights = learning_agent.get_learning_insights()
                        st.success(f"✅ Connected! {insights['total_patterns']} patterns stored")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
            
            # ChromaDB management
            st.subheader("Database Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export Patterns"):
                    try:
                        filename = learning_agent.export_patterns_to_json()
                        if filename:
                            st.success(f"✅ Exported to {filename}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col2:
                if st.button("View Collections"):
                    insights = learning_agent.get_learning_insights()
                    st.json(insights.get('pattern_breakdown', {}))
            
            with col3:
                if st.button("⚠️ Clear All Data", type="secondary"):
                    st.warning("This will delete all learned patterns!")
                    
        except Exception as e:
            st.error(f"ChromaDB configuration error: {e}")

if __name__ == "__main__":
    main()
