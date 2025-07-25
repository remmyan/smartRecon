from crewai import Task

# Import all agents (adjust paths based on your project structure)
from agents.data_ingestion import data_ingestion_agent
from agents.matching import matching_agent
from agents.anomaly import anomaly_detection_agent
from agents.exceptions import exception_management_agent
from agents.learning import learning_agent
from agents.orchestration import orchestration_agent

# Data Ingestion Task
ingest_data_task = Task(
    description="Ingest and process data from ERP systems, bank statements, and invoices using OCR. Validate and structure the data for downstream agents.",
    agent=data_ingestion_agent,
    expected_output="Structured JSON data containing ERP records, bank statements, and extracted invoice details, with validation summary."
)

# Matching Task
match_transactions_task = Task(
    description="Perform semantic matching on ingested data to identify matches between ERP and bank transactions, aiming for 90-95% automation with confidence scoring.",
    agent=matching_agent,
    expected_output="JSON with matched transactions, confidence scores, partial/full match classifications, and overall matching statistics."
)

# Anomaly Detection Task
detect_anomalies_task = Task(
    description="Analyze matched and unmatched transactions for duplicates, unusual patterns, and fraud risks. Flag anomalies with risk levels.",
    agent=anomaly_detection_agent,
    expected_output="JSON list of detected anomalies, including types (duplicate, unusual pattern, fraud suspect), risk levels, and affected transactions."
)

# Exception Management Task
manage_exceptions_task = Task(
    description="Prioritize flagged exceptions, route them to appropriate workflows or human reviewers, and track their resolution status.",
    agent=exception_management_agent,
    expected_output="JSON with prioritized exceptions, routing assignments, and status tracking updates."
)

# Learning Task
learn_from_corrections_task = Task(
    description="Store human corrections, retrieve similar patterns using RAG, generate analytics, and update matching rules for improved future performance.",
    agent=learning_agent,
    expected_output="JSON with stored corrections, retrieved similar cases, performance analytics, and updated matching patterns."
)

# Orchestration Task (Manager-level task to coordinate the workflow)
orchestrate_workflow_task = Task(
    description="Coordinate the entire reconciliation workflow: initialize tasks, manage dependencies, facilitate agent communication, prioritize dynamically, and monitor health.",
    agent=orchestration_agent,
    expected_output="JSON with workflow status, task coordination logs, communication records, prioritization updates, and health metrics."
)
