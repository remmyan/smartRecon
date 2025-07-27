import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class Config:
    # Load API key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))  # Auto-retry on rate limits
    def get_llm(model="gpt-4", temperature=0.5):
        return ChatOpenAI(
            model=model,  # "gpt-3.5-turbo" for lighter tasks to save tokens
            temperature=temperature,
            api_key=Config.OPENAI_API_KEY
        )
    # For backwards compatibility
    # Defaults to GPT-4
    # OpenAI Configuration
    #OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    # OPENAI_MODEL = ChatPerplexity(
    #     model="sonar",  # Or "sonar-large" for search-enhanced tasks
    #     temperature=0.5,     # Lower for more deterministic outputs in your analytical agents
    #     api_key=os.getenv("PERPLEXITY_API_KEY")
    # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # )
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

# Validate required configuration
if not Config.OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Please check .env file.")
