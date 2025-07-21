# Multi-Agent AP Reconciliation System with ChatGPT

An advanced accounts payable reconciliation system powered by six specialized AI agents, featuring ChatGPT integration for semantic matching and intelligent exception handling.

## 🚀 Key Features

- **ChatGPT Semantic Matching**: AI-powered transaction matching with 90%+ accuracy
- **Multi-Agent Architecture**: Six specialized agents handling different aspects of reconciliation
- **Real-time Processing**: Streamlit-based web interface for immediate results
- **Intelligent Exception Management**: AI-driven exception categorization and resolution
- **Continuous Learning**: RAG-based pattern recognition and improvement
- **Enterprise Integration**: Ready for ERP system integration

## 🏗️ Architecture

The system employs six specialized agents:

1. **Data Ingestion Agent**: Multi-format file processing with OCR
2. **Matching Agent**: ChatGPT-powered semantic matching
3. **Anomaly Detection Agent**: Fraud detection and duplicate identification
4. **Exception Management Agent**: Intelligent exception handling
5. **Learning Agent**: Pattern learning with RAG implementation
6. **Orchestration Agent**: Workflow coordination and management

## 🛠️ Installation

1. **Clone and setup:**

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env file with your OpenAI API key

streamlit run app.py

