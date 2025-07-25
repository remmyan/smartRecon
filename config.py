import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq

load_dotenv()

class Config:
    #OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_MODEL = ChatGroq(
    model="groq/llama3-8b-8192",  # Choose your model
    temperature=0.7,
    max_tokens=512,
    api_key=os.getenv("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")
)
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
    
    # Database Configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/reconciliation.db')
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', 'data/chroma_db')
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'transaction_patterns')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    # Application Settings
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    MAX_UPLOAD_SIZE = os.getenv('MAX_UPLOAD_SIZE', '10MB')
    
    # Matching Configuration
    FUZZY_THRESHOLD = 85
    AMOUNT_TOLERANCE = 0.02  # 2%
    DATE_WINDOW = 3  # days
    
    # Exception Management
    HIGH_RISK_AMOUNT = 10000
    CRITICAL_AGE_DAYS = 30
    
    # RAG Configuration
    RAG_TOP_K = int(os.getenv('RAG_TOP_K', '5'))
    RAG_SIMILARITY_THRESHOLD = float(os.getenv('RAG_SIMILARITY_THRESHOLD', '0.7'))

