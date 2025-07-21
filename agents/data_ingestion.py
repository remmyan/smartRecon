import pandas as pd
import io
import pdfplumber
from typing import Dict, List, Any
import re
from datetime import datetime

class DataIngestionAgent:
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.pdf']
        
    def process_files(self, uploaded_files: Dict[str, List]) -> Dict[str, Any]:
        """Process uploaded files and extract structured data"""
        results = {'total_records': 0}
        
        for file_type, files in uploaded_files.items():
            if files:
                processed_data = []
                for file in files:
                    if file.name.endswith('.pdf'):
                        data = self._extract_pdf_data(file)
                    elif file.name.endswith(('.xlsx', '.xls')):
                        data = pd.read_excel(file)
                    else:
                        data = pd.read_csv(file)
                    
                    # Standardize column names and data types
                    data = self._standardize_data(data, file_type)
                    processed_data.append(data)
                
                # Combine all files of same type
                combined_data = pd.concat(processed_data, ignore_index=True)
                results[file_type] = len(combined_data)
                results['total_records'] += len(combined_data)
                
                # Store in database or session state
                self._store_data(combined_data, file_type)
        
        return results
    
    def _extract_pdf_data(self, pdf_file) -> pd.DataFrame:
        """Extract data from PDF using OCR and pattern recognition"""
        with pdfplumber.open(pdf_file) as pdf:
            all_data = []
            for page in pdf.pages:
                text = page.extract_text()
                # Use regex patterns to extract structured data
                transactions = self._parse_transaction_text(text)
                all_data.extend(transactions)
        
        return pd.DataFrame(all_data)
    
    def _parse_transaction_text(self, text: str) -> List[Dict]:
        """Parse transaction data from text using regex patterns"""
        transactions = []
        
        # Pattern for amounts: $1,234.56
        amount_pattern = r'\$[\d,]+\.?\d*'
        # Pattern for dates: MM/DD/YYYY or DD/MM/YYYY
        date_pattern = r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
        # Pattern for reference numbers
        ref_pattern = r'[A-Z]{2,4}[-\s]?\d{4,}'
        
        lines = text.split('\n')
        for line in lines:
            amounts = re.findall(amount_pattern, line)
            dates = re.findall(date_pattern, line)
            refs = re.findall(ref_pattern, line)
            
            if amounts and dates:
                transactions.append({
                    'amount': float(amounts[0].replace('$', '').replace(',', '')),
                    'date': dates[0],
                    'reference': refs[0] if refs else '',
                    'description': line.strip()
                })
        
        return transactions
    
    def _standardize_data(self, data: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Standardize column names and data types based on file type"""
        
        # Standard column mappings
        column_mappings = {
            'invoices': {
                'invoice_id': ['invoice_number', 'inv_no', 'invoice_id'],
                'vendor': ['vendor_name', 'supplier', 'vendor'],
                'amount': ['amount', 'total', 'invoice_amount'],
                'date': ['invoice_date', 'date', 'transaction_date'],
                'description': ['description', 'memo', 'details']
            },
            'bank_statements': {
                'transaction_id': ['txn_id', 'transaction_id', 'ref_no'],
                'amount': ['amount', 'debit', 'credit'],
                'date': ['date', 'transaction_date', 'posting_date'],
                'description': ['description', 'memo', 'details']
            }
        }
        
        # Apply column standardization
        if file_type in column_mappings:
            mapping = column_mappings[file_type]
            for std_col, possible_cols in mapping.items():
                for col in possible_cols:
                    if col.lower() in [c.lower() for c in data.columns]:
                        data = data.rename(columns={col: std_col})
                        break
        
        # Standardize data types
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
        
        if 'amount' in data.columns:
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
        
        return data
    
    def _store_data(self, data: pd.DataFrame, file_type: str):
        """Store processed data in database"""
        # Implementation would store in SQLite or session state
        pass
