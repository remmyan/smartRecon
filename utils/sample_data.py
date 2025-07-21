import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_sample_data():
    """Generate realistic sample data for testing"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate invoices
    invoices = generate_invoices(200)
    invoices.to_csv('data/invoices.csv', index=False)
    
    # Generate purchase orders
    purchase_orders = generate_purchase_orders(invoices)
    purchase_orders.to_csv('data/purchase_orders.csv', index=False)
    
    # Generate ledger entries
    ledger = generate_ledger(invoices)
    ledger.to_csv('data/general_ledger.csv', index=False)
    
    # Generate bank statements
    bank_statements = generate_bank_statements(ledger)
    bank_statements.to_csv('data/bank_statements.csv', index=False)

def generate_invoices(count: int) -> pd.DataFrame:
    """Generate sample invoice data"""
    
    vendors = [
        'Acme Corporation', 'Beta Systems Inc', 'Gamma Technologies', 'Delta Supply Co',
        'Echo Partners LLC', 'Foxtrot Industries', 'Golf Solutions', 'Hotel Services',
        'India Manufacturing', 'Juliet Consulting', 'Kilo Logistics', 'Lima Retail',
        'Mike Construction', 'November Software', 'Oscar Equipment', 'Papa Materials'
    ]
    
    descriptions = [
        'Office supplies and equipment', 'Software licensing fees', 'Consulting services',
        'Raw materials procurement', 'Equipment maintenance', 'Marketing services',
        'IT infrastructure setup', 'Legal consulting fees', 'Accounting services',
        'Construction materials', 'Vehicle maintenance', 'Utilities payment'
    ]
    
    data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(count):
        invoice_date = base_date + timedelta(days=random.randint(0, 90))
        
        # Generate realistic amounts
        amount_base = random.choice([100, 250, 500, 750, 1000, 1500, 2500, 5000])
        amount = amount_base + random.randint(-50, 200) + (random.random() * 100)
        
        data.append({
            'invoice_id': f'INV-2024-{i+1:04d}',
            'po_id': f'PO-2024-{i+1:04d}',
            'vendor': random.choice(vendors),
            'description': random.choice(descriptions),
            'date': invoice_date.strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'status': 'Pending' if random.random() < 0.15 else 'Approved'
        })
    
    return pd.DataFrame(data)

def generate_purchase_orders(invoices: pd.DataFrame) -> pd.DataFrame:
    """Generate purchase orders matching invoices"""
    
    pos = []
    
    for _, invoice in invoices.iterrows():
        # 95% of invoices have matching POs
        if random.random() < 0.95:
            po_date = pd.to_datetime(invoice['date']) - timedelta(days=random.randint(1, 30))
            
            # Sometimes PO amounts differ slightly from invoice
            po_amount = invoice['amount']
            if random.random() < 0.1:  # 10% have amount differences
                po_amount += random.randint(-50, 50)
            
            pos.append({
                'po_id': invoice['po_id'],
                'vendor': invoice['vendor'],
                'description': invoice['description'],
                'date': po_date.strftime('%Y-%m-%d'),
                'amount': round(po_amount, 2),
                'status': 'Complete'
            })
    
    return pd.DataFrame(pos)

def generate_ledger(invoices: pd.DataFrame) -> pd.DataFrame:
    """Generate general ledger entries"""
    
    ledger_entries = []
    
    for _, invoice in invoices.iterrows():
        # 90% of invoices make it to the ledger
        if random.random() < 0.90:
            # Ledger entries are usually posted a few days after invoice
            ledger_date = pd.to_datetime(invoice['date']) + timedelta(days=random.randint(1, 7))
            
            ledger_entries.append({
                'gl_id': f'GL-{len(ledger_entries)+1:05d}',
                'invoice_id': invoice['invoice_id'],
                'vendor': invoice['vendor'],
                'account_code': '2000-' + str(random.randint(100, 999)),
                'date': ledger_date.strftime('%Y-%m-%d'),
                'amount': invoice['amount'],
                'debit_credit': 'Credit'
            })
    
    return pd.DataFrame(ledger_entries)

def generate_bank_statements(ledger: pd.DataFrame) -> pd.DataFrame:
    """Generate bank statement entries"""
    
    bank_transactions = []
    
    for _, entry in ledger.iterrows():
        # 85% of ledger entries result in bank payments
        if random.random() < 0.85:
            # Payments are usually made 1-10 days after ledger posting
            payment_date = pd.to_datetime(entry['date']) + timedelta(days=random.randint(1, 10))
            
            # Sometimes bank amounts differ due to fees, discounts, etc.
            bank_amount = entry['amount']
            if random.random() < 0.05:  # 5% have differences
                bank_amount += random.randint(-25, 25)
            
            bank_transactions.append({
                'transaction_id': f'TXN-{len(bank_transactions)+1:06d}',
                'date': payment_date.strftime('%Y-%m-%d'),
                'description': f"Payment to {entry['vendor']} - {entry['invoice_id']}",
                'amount': round(bank_amount, 2),
                'balance': random.randint(50000, 150000),
                'reference': entry['invoice_id']
            })
    
    # Add some unmatched bank transactions
    for i in range(20):
        random_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        bank_transactions.append({
            'transaction_id': f'TXN-{len(bank_transactions)+1:06d}',
            'date': random_date.strftime('%Y-%m-%d'),
            'description': f"Misc payment - {random.choice(['Utilities', 'Rent', 'Insurance'])}",
            'amount': round(random.randint(200, 2000), 2),
            'balance': random.randint(50000, 150000),
            'reference': ''
        })
    
    return pd.DataFrame(bank_transactions)
