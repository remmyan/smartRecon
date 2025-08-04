import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from agents.orchestration import Orchestrator
from agents.learning import LearningAgent
from utils.sample_data import generate_sample_data
from utils.database import init_database, get_connection
from config import Config
import os
import traceback

# Page configuration
st.set_page_config(
    page_title="AP Reconciliation System with ChatGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check OpenAI configuration
if not Config.GROQ_API_KEY:
    st.error("⚠️ GROQ API Key not configured. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Initialize database and agents
try:
    init_database()
    orchestrator = Orchestrator()
    learning_agent = LearningAgent()
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
    st.title("🤖 SmartRecon - AI-Powered App-Reconciliation System")
    st.markdown("*Automated accounts payable reconciliation with LLM semantic matching and ChromaDB learning*")
    
    # Initialize session state for data persistence
    if 'reconciliation_results' not in st.session_state:
        st.session_state.reconciliation_results = None
    if 'uploaded_data_stats' not in st.session_state:
        st.session_state.uploaded_data_stats = None
    
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Dashboard", 
        "Reconciliation", 
        "Exception Management"
    ])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Reconciliation":
        show_data_upload()
    elif page == "Exception Management":
        show_matching_review()
    elif page == "Learning Analytics":
        show_learning_analytics()

def show_dashboard():
    st.header("📊 System Dashboard")
    
    # Generate sample data if needed
    # if st.sidebar.button("Generate Sample Data"):
    #     with st.spinner("Generating sample transaction data..."):
    #         generate_sample_data()
    #         # Clear any existing results to force refresh
    #         st.session_state.reconciliation_results = None
    #     st.success("Sample data generated successfully!")
    
    # Get real statistics from last reconciliation or current data
    if st.session_state.reconciliation_results:
        results = st.session_state.reconciliation_results
        if results:
            exact = len(results['matching_results'].get('exact_matches', pd.DataFrame()))
            llm = len(results['matching_results'].get('llm_matches', pd.DataFrame()))
            unmatched = len(results['matching_results'].get('unmatched', pd.DataFrame()))
            total = exact + llm + unmatched
            match_rate = (exact + llm) / total * 100 if total else 0
            # st.metric("Auto-Match Rate", f"{match_rate:.1f}%")
            # st.metric("Open Exceptions", len(results['exceptions'].get('prioritized_exceptions', [])))
            # st.metric("Total Transactions", total)
            # st.metric("Processing Time (s)", f"{results.get('processing_time', 0):.2f}")
            
            total_transactions = total
            match_rate = f"{match_rate:.1f}%"
            #exceptions = len(results['exceptions'].get('prioritized_exceptions', []))
            total_amount = results.get('total_amount', 0)  # We'll add this calculation
    else:
        # Get basic stats from available data
        stats = get_current_data_stats()
        total_transactions = stats['total_records']
        match_rate = 0  # No reconciliation run yet
        exceptions = 0
        total_amount = stats['total_amount']
        unmatched = 0
    
    # Key metrics with real data
    col1, col2, col3 = st.columns(3)

    
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
            <h3>{match_rate}</h3>
            <p>Auto-Match Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unmatched}</h3>
            <p>Open Exceptions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts with dynamic data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Processing Performance")
        # if st.session_state.reconciliation_results:
        #     results = st.session_state.reconciliation_results
            
        #     # Prepare performance data as a DataFrame
        #     performance_data = {
        #         'Match Type': ['Exact Matches', 'AI Matches', 'Fuzzy Matches', 'Exceptions'],
        #         'Count': [
        #             results.get('exact_matches', 0),
        #             results.get('llm_matches', 0), 
        #             results.get('fuzzy_matches', 0),
        #             results.get('exceptions', 0)
        #         ],
        #         'Color': ['#28a745', '#17a2b8', '#ffc107', '#dc3545']
        #     }
            
        #     fig = px.bar(
        #         performance_data, 
        #         x='Match Type', 
        #         y='Count',
        #         color='Color',
        #         color_discrete_map={color: color for color in performance_data['Color']}
        #     )
        #     fig.update_layout(showlegend=False)
        #     st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.reconciliation_results:
            results = st.session_state.reconciliation_results
            
            # Prepare performance data as a DataFrame
            performance_data = pd.DataFrame({
                'Match Type': ['Exact Matches', 'AI Matches', 'Fuzzy Matches', 'Exceptions'],
                'Count': [
                    len(results.get('matching_results', {}).get('exact_matches', pd.DataFrame())),
                    len(results.get('matching_results', {}).get('llm_matches', pd.DataFrame())),
                    len(results.get('matching_results', {}).get('fuzzy_matches', pd.DataFrame())),
                    len(results.get('matching_results', {}).get('exceptions', pd.DataFrame()))
                ],
                'Color': ['#28a745', '#17a2b8', '#ffc107', '#dc3545']
            })

            # Create bar chart
            fig = px.bar(
                performance_data,
                x='Match Type',
                y='Count',
                color='Match Type',
                color_discrete_map=dict(zip(performance_data['Match Type'], performance_data['Color']))
            )

            fig.update_layout(
                showlegend=False,
                title='🔍 Performance Poll Chart',
                xaxis_title='Match Type',
                yaxis_title='Number of Matches',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run AI Reconciliation to see performance data")
    
    # with col2:
    #     st.subheader("Learning Progress")
        
    #     # Get learning insights from ChromaDB
    #     try:
    #         insights = learning_agent.get_learning_stats()
    #         st.info(f"Total Patterns Learned: {insights}")
    #         pattern_data = insights.get('pattern_breakdown', {})
    #         if pattern_data and any(pattern_data.values()):
    #             fig = px.pie(
    #                 values=list(pattern_data.values()),
    #                 names=list(pattern_data.keys()),
    #                 title="Knowledge Base Distribution"
    #             )
    #             st.plotly_chart(fig, use_container_width=True)
    #         else:
    #             st.info("No learning patterns yet. Process some data to build knowledge base.")
                
    #     except Exception as e:
    #         st.error(f"Error loading learning insights: {e}")
    
    # Recent activity with real data
    # st.subheader("Recent System Activity")
    
    # if hasattr(orchestrator, 'processing_log') and orchestrator.processing_log:
    #     # Show actual processing log
    #     log_df = orchestrator.get_processing_log()
    #     if not log_df.empty:
    #         # Show last 5 entries
    #         recent_log = log_df.tail(5).copy()
    #         recent_log['timestamp'] = pd.to_datetime(recent_log['timestamp'])
    #         st.dataframe(recent_log, use_container_width=True)
    #     else:
    #         show_placeholder_activity()
    # else:
    #     show_placeholder_activity()

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
        
    except Exception as e:
        print(f"Error getting data stats: {e}")
    
    return stats

def show_data_upload():
    st.header("Reconciliation")
    
    (tab1,) = st.tabs(["File Upload"])
    
    with tab1:
        st.subheader("Upload Financial Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_bank = st.file_uploader(
                "Upload Bank Statements",
                type=['csv', 'xlsx', 'pdf'],
                accept_multiple_files=True,
                key="bank"
            )
            
        # Ledger upload
        with col2:
            uploaded_ledger = st.file_uploader(
                "Upload Purchase Order",
                type=['csv', 'xlsx', 'pdf'],
                accept_multiple_files=True,
                key="ledger"
            )

        if st.button("Start Reconciliation"):
            if any([uploaded_bank, uploaded_ledger]):
                with st.spinner("Reconsiliation in progress..."):
                    try:
                        results = orchestrator.run_ai_reconciliation({
                            'bank_statements': uploaded_bank,
                            'ledger': uploaded_ledger
                        })
                        
                        # Store results in session state
                        st.session_state.uploaded_data_stats = results
                        st.session_state.reconciliation_results = results
                        
                        st.success(f"✅ Successfully processed {results['total_records']} records")
                        
                        # Show processing summary with real data
                        st.subheader("Processing Summary")
                        summary_data = []
                        
                        for doc_type, file_type, files in [
                            ("Bank Statements", "bank_statements", uploaded_bank),
                            ("General Ledger", "ledger", uploaded_ledger)
                        ]:
                            record_count = results.get(file_type, 0)
                            status = "✅ Complete" if record_count > 0 else "⚪ No data"
                            summary_data.append({
                                "Document Type": doc_type,
                                "Records": record_count,
                                "Status": status
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        # Set row index to start at 1
                        summary_df.index = range(1, len(summary_df) + 1)
                        st.dataframe(summary_df, use_container_width=True)
                        
                    except Exception as e:
                        full_traceback = traceback.format_exc()
                        print(f"Full exception: {full_traceback}")
                        st.error(f"❌ Error processing files: {str(e)}")
            else:
                st.warning("⚠️ Missing required files: Upload Bank Statement and Purchase Order to proceed.")

    #with tab2:
        # st.subheader("ERP System Integration")
        # st.info("Direct ERP integration capabilities - connect to SAP, NetSuite, Oracle, etc.")
        
        # col1, col2 = st.columns(2)
        # with col1:
        #     erp_system = st.selectbox("ERP System", ["SAP", "NetSuite", "Oracle", "QuickBooks", "Other"])
        #     connection_string = st.text_input("Connection String", type="password")
        
        # with col2:
        #     sync_frequency = st.selectbox("Sync Frequency", ["Real-time", "Hourly", "Daily", "Weekly"])
        #     last_sync = st.text_input("Last Sync", value="Not configured", disabled=True)
        
        # if st.button("Test Connection"):
        #     st.info("🔧 ERP integration is planned for future release")

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

def show_matching_review():
    st.title("🔍 Review Matching Results")
    results = st.session_state.reconciliation_results
    if not results:
        st.warning("Run reconciliation first.")
        return
    matching_results = results.get('matching_results', {})    
    # Combine matched and unmatched records into a single DataFrame for display
    # Safely get each DataFrame, defaulting to empty DataFrame if the key is missing
    llm_matches = matching_results.get('llm_matches', pd.DataFrame())
    exact_matches = matching_results.get('exact_matches', pd.DataFrame())
    fuzzy_matches = matching_results.get('fuzzy_matches', pd.DataFrame())

    # Combine them into one DataFrame (rows from all three)
    matched = pd.concat([llm_matches, exact_matches, fuzzy_matches], ignore_index=True)
    unmatched = matching_results.get('unmatched', pd.DataFrame())

    if matched.empty and unmatched.empty:
        st.info("No matching results to review.")
        return

    # Prepare matched DataFrame with uniform columns
    if not matched.empty:
        matched_display = matched.copy()
        matched_display['status'] = 'Matched'
        # Ensure reasoning is string (if dict, convert to JSON for display)
        matched_display['reasoning_display'] = matched_display['match_reasoning'].apply(
            lambda x: x if isinstance(x, str) else st.json(x, expanded=False) if x else "No reasoning provided."
        )
    else:
        matched_display = pd.DataFrame()

    # Prepare unmatched DataFrame with uniform columns and default reason placeholder
    if not unmatched.empty:
        unmatched_display = unmatched.copy()
        unmatched_display['status'] = 'Unmatched'
        # Fill missing reasoning with default str
        unmatched_display['reasoning_display'] = unmatched_display['match_reasoning']
    else:
        unmatched_display = pd.DataFrame()

    # Combine for display
    combined_df = pd.concat([matched_display, unmatched_display], ignore_index=True)

    # Show combined results in a table (without editing yet)
    st.subheader("All Matching Results")
    # Set row index to start at 1
    combined_df.index = range(1, len(combined_df) + 1)
    st.dataframe(combined_df.drop(columns=['match_reasoning'], errors='ignore'))

    st.markdown("---")
    st.subheader("Review Unmatched Records")

    # Filter unmatched for interactive review
    unmatched_for_review = unmatched_display.copy()

    if unmatched_for_review.empty:
        st.info("No unmatched records to review.")
        return

    # Initialize session state storage for user feedback if not present
    if 'review_feedback' not in st.session_state:
        st.session_state['review_feedback'] = {}

    # Display each unmatched record with option to mark as matched + mandatory reason
    for idx, row in unmatched_for_review.iterrows():
        st.markdown(f"### Unmatched Record: {row.get('transaction_id', idx)}")

        # Display record fields (customize as needed)
        # st.json(row.drop(labels=['status', 'reasoning_display'], errors='ignore').to_dict())

        # Show reasoning in expander
        with st.expander("LLM Reasoning / Explanation"):
            st.write(row['reasoning_display'])

        # Checkbox to mark as matched
        mark_matched = st.checkbox("Mark as matched", key=f"mark_matched_{idx}")

        # Text input for mandatory reason if marked
        reason_input = ""
        if mark_matched:
            reason_input = st.text_area(
                "Please provide a reason for marking as matched (required):", 
                key=f"match_reason_{idx}"
            )
            if not reason_input.strip():
                st.error("A reason is required when marking as matched.")

        # Store user inputs in session state
        st.session_state['review_feedback'][idx] = {
            "mark_matched": mark_matched,
            "reason": reason_input.strip() if mark_matched else None,
            "original_record": row.to_dict(),
        }

    
    # Submit button to process feedback
    if st.button("Submit Reviewed Matches"):
        feedback = st.session_state.get('review_feedback', {})
        learned = []

        for idx, entry in feedback.items():
            if entry.get("mark_matched") and entry.get("reason"):
                original = entry['original_record']
                corrected = original.copy()
                corrected['user_confirmed_match'] = True
                corrected['user_feedback_reason'] = entry['reason']
                #corrected['review_timestamp'] = datetime.now().isoformat()

                # Store the correction with the Learning Agent
                learning_agent.store_correction(original, corrected)

                learned.append({'original': original, 'corrected': corrected})


        if learned:
            st.success(f"✅ Success! {len(learned)} user-reviewed match saved. This will improve future matching accuracy.")
            #st.json(learned)
            # Prepare data for the grid
            corrected_data = []
            for record in learned:
                corrected = record["corrected"]
                corrected_data.append({
                    "Transaction ID": corrected.get("transaction_id"),
                    "Amount": corrected.get("amount"),
                    "Date": corrected.get("date"),
                    "Vendor": corrected.get("vendor"),
                    "Original Match Reason": corrected.get("reasoning_display"),
                    "Corrected Reason": corrected.get("user_feedback_reason")
                })

            # Create and display DataFrame
            df = pd.DataFrame(corrected_data)
            df.index = range(1, len(df) + 1)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No corrections submitted or reasons missing for marked matches. Please provide reasons.")

def show_exceptions():
    st.title("⚡ Exception Management")
    results = st.session_state.reconciliation_results
    if not results:
        st.warning("Run reconciliation first.")
        return

    prioritized_ex = results['exceptions'].get('prioritized_exceptions', [])
    unmatched = results['matching_results'].get('unmatched', pd.DataFrame())

    # ... Display anomalies, exceptions, unmatched as before ...

    # Prepare correction targets: metadata from exceptions + unmatched invoices
    corrections_target = []
    for exc in prioritized_ex:
        rec = exc.get('metadata', {})
        if rec:
            # Add fields for reconciliation tracking - default False and empty comment
            rec['reconciled'] = False
            rec['comments'] = ''
            corrections_target.append(rec)
    corrections_target += unmatched.to_dict('records') if unmatched is not None else []

    if not corrections_target:
        st.info("No exceptions or unmatched records to correct.")
        return

    df = pd.DataFrame(corrections_target)

    # Add columns for reconciliation and comments
    if 'reconciled' not in df.columns:
        df['reconciled'] = False
    if 'comments' not in df.columns:
        df['comments'] = ''

    # Display editable table with reconciliation controls
    st.markdown("#### Mark Exceptions as Reconciled and Enter Comments")
    edited_df = st.data_editor(df, num_rows="dynamic", key="correction_editor", use_container_width=True)

    if st.button("Submit Reconciliations"):
        original_records = df.to_dict('records')
        edited_records = edited_df.to_dict('records')

        learned = []
        for old, new in zip(original_records, edited_records):
            # Only proceed if user marked reconciled
            if new.get('reconciled') and (old != new):
                corrected = new.copy()
                # Optionally record reconciliation timestamp
                corrected['reconciliation_timestamp'] = datetime.now().isoformat()

                # Store correction in Learning Agent
                learning_agent.store_correction(old, corrected)

                learned.append({'original': old, 'corrected': corrected})

        if learned:
            st.success(f"Stored {len(learned)} reconciliations. These will improve matching in future runs.")
            st.json(learned)
        else:
            st.info("No reconciliations submitted.")

def show_learning_analytics():
    st.header("📊 AI Learning Analytics with ChromaDB")
    
    try:
        from agents.learning import LearningAgent
        insights = learning_agent.get_learning_stats()
        st.info(f"learnings. {insights}")
        # Learning statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patterns", insights['total_corrections'])
        
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
                insights['total_corrections'],
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
                semantic_model = st.selectbox("Semantic Model", ["GPT-4", "GPT-3.5", "Local BERT"])
                max_processing_time = st.number_input("Max Processing Time (seconds)", 1, 300, 30)
            
            with col2:
                confidence_threshold = st.slider("Auto-Approval Confidence", 70, 100, 90)
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
                        insights = learning_agent.get_learning_stats()
                        st.success(f"✅ Connected! {insights['total_corrections']} patterns stored")
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
                    insights = learning_agent.get_learning_stats()
                    st.json(insights.get('pattern_breakdown', {}))
            
            with col3:
                if st.button("⚠️ Clear All Data", type="secondary"):
                    st.warning("This will delete all learned patterns!")
                    
        except Exception as e:
            st.error(f"ChromaDB configuration error: {e}")

if __name__ == "__main__":
    main()
