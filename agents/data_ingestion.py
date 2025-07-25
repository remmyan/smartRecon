from crewai import Agent
from crewai.tools import BaseTool
import os
import logging
import json
from typing import Any
from PIL import Image
import pytesseract
import pandas as pd
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm = Config.OPENAI_MODEL

# Custom Tools by subclassing BaseTool
class OCRExtractionTool(BaseTool):
    name: str = "OCR Extraction"
    description: str = "Extract text from an image using OCR. Provide the image_path as input."

    def _run(self, image_path: str) -> str:
        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            logger.info(f"OCR extraction successful for {image_path}")
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return f"Error: {str(e)}"

class LoadDataTool(BaseTool):
    name: str = "Load Data"
    description: str = "Load data from various sources like ERP CSV or bank statements. Provide source ('erp' or 'bank') and file_path."

    def _run(self, source: str, file_path: str) -> str:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Normalize columns (example: standardize to 'id', 'amount', 'date', 'description')
            column_map = {
                'amount': next((col for col in df.columns if 'amount' in col.lower()), None),
                'date': next((col for col in df.columns if 'date' in col.lower()), None),
                'description': next((col for col in df.columns if 'desc' in col.lower()), None)
            }
                issues = []
            df = df.rename(columns=column_map)
            
            data = df.to_dict(orient='records')
                    issues.append("Missing date")
            logger.info(f"Loaded {len(data)} records from {source} at {file_path}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Data loading failed for {source} at {file_path}: {e}")
            return json.dumps({"error": str(e)})

class ValidateDataTool(BaseTool):
    name: str = "Validate Data"
    description: str = "Validate ingested data for completeness and format. Provide data_json as input."

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            validation_results = []
            
        
            for record in data:
                if 'amount' not in record or not isinstance(record['amount'], (int, float)):
                    issues.append("Invalid or missing amount")
                if 'date' not in record:
                if 'description' not in record or not record['description']:
                    issues.append("Missing description")
                
                validation_results.append({
                    'record_id': record.get('id', 'unknown'),
                    'valid': len(issues) == 0,
                    'issues': issues
                })
            
            summary = {
                'total_records': len(data),
                'valid_records': len([r for r in validation_results if r['valid']]),
                'invalid_records': len([r for r in validation_results if not r['valid']])
            }
            
            logger.info(f"Validated {summary['total_records']} records: {summary['valid_records']} valid")
            return json.dumps({"summary": summary, "details": validation_results})
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return json.dumps({"error": str(e)})

# Create the CrewAI Data Ingestion Agent
def create_data_ingestion_agent():
    """Create the CrewAI Data Ingestion Agent with extraction and processing capabilities"""
    
    data_ingestion_agent = Agent(
        role="Data Ingestion Agent",
        goal="""Extract and ingest data from multiple sources including ERP systems, bank statements, 
        and documents via OCR. Ensure data is cleaned, validated, and structured for downstream agents.""",
        backstory="""You are an expert data extraction specialist for financial reconciliation systems. 
        Your capabilities include:
        1. **Data Loading**: Ingesting from various formats like CSV, Excel from ERP and bank sources.
        
        2. **OCR Processing**: Extracting text from invoice images and PDFs with high accuracy.
        
        3. **Data Validation**: Checking for completeness, format correctness, and basic quality 
           to ensure reliable input for matching and analysis.
        
        You prepare high-quality, structured data to support 90-95% automated reconciliation accuracy.""",
        tools=[OCRExtractionTool(), LoadDataTool(), ValidateDataTool()],  # Instantiate the tool classes
        llm=llm,
        verbose=True,
        memory=True,
        max_iter=3,
        allow_delegation=False
    )
    
    return data_ingestion_agent

# Export the agent for use in main crew
data_ingestion_agent = create_data_ingestion_agent()
