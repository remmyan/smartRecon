import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = "llama3-8b-8192"  # Or "mixtral-8x7b-32768" for better reasoning
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence Transformers model
    CHROMA_DB_PATH = "./chroma_db"  # Persistent path for Chroma
    CHROMA_COLLECTION_NAME="transaction_patterns"
    RAG_TOP_K = 5  # Number of similar patterns to retrieve
    RAG_SIMILARITY_THRESHOLD = 0.7  # Min similarity for patterns
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

    # @staticmethod
    # def get_llm(temperature=0.3):
    #     return ChatGroq(
    #         model=Config.GROQ_MODEL,
    #         api_key=Config.GROQ_API_KEY,
    #         temperature=temperature
    #     )
