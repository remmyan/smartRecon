import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class AnomalyDetectionAgent:
    def __init__(self):
        self.duplicate_threshold = 0.95
        self.amount_variance_threshold = 2.5  # Standard deviations
        self.fraud_indicators = [
            'round_amounts',
            'weekend_transactions', 
            'unusual_vendors',
            'amount_patterns'
        ]
        
    def detect_anomalies(self, matching_results: Dict[str, pd.DataFrame]) -> Dict[str, List]:
        """Detect various types of anomalies in transaction data"""
        
        anomalies = {
            'duplicates': [],
            'unusual_amounts': [],
            'fraud_indicators': [],
            'data_quality_issues': []
        }
        
        # Combine all transaction data
        all_transactions = pd.DataFrame()
        for key, df in matching_results.items():
            if not df.empty:
                all_transactions = pd.concat([all_transactions, df], ignore_index=True)
        
        if all_transactions.empty:
            return anomalies
        
        # Detect duplicates
        anomalies['duplicates'] = self._detect_duplicates(all_transactions)
        
        # Detect unusual amounts
        anomalies['unusual_amounts'] = self._detect_unusual_amounts(all_transactions)
        
        # Detect fraud indicators
        anomalies['fraud_indicators'] = self._detect_fraud_indicators(all_transactions)
        
        # Detect data quality issues
        anomalies['data_quality_issues'] = self._detect_data_quality_issues(all_transactions)
        
        return anomalies
    
    def _detect_duplicates(self, df: pd.DataFrame) -> List[Dict]:
        """Detect duplicate transactions"""
        duplicates = []
        
        # Group by key fields to find potential duplicates
        if 'amount' in df.columns and 'vendor' in df.columns and 'date' in df.columns:
            grouped = df.groupby(['amount', 'vendor', 'date']).size()
            duplicate_groups = grouped[grouped > 1]
            
            for (amount, vendor, date), count in duplicate_groups.items():
                duplicate_records = df[
                    (df['amount'] == amount) & 
                    (df['vendor'] == vendor) & 
                    (df['date'] == date)
                ]
                
                duplicates.append({
                    'type': 'exact_duplicate',
                    'count': count,
                    'amount': amount,
                    'vendor': vendor,
                    'date': date,
                    'record_ids': duplicate_records.index.tolist(),
                    'risk_score': 90
                })
        
        # Near-duplicate detection (similar amounts, same vendor, close dates)
        near_duplicates = self._detect_near_duplicates(df)
        duplicates.extend(near_duplicates)
        
        return duplicates
    
    def _detect_near_duplicates(self, df: pd.DataFrame) -> List[Dict]:
        """Detect near-duplicate transactions"""
        near_duplicates = []
        
        if len(df) < 2:
            return near_duplicates
        
        # Sort by vendor and date for efficient comparison
        df_sorted = df.sort_values(['vendor', 'date']).reset_index()
        
        for i in range(len(df_sorted) - 1):
            for j in range(i + 1, min(i + 10, len(df_sorted))):  # Check next 10 records
                row1, row2 = df_sorted.iloc[i], df_sorted.iloc[j]
                
                # Check if same vendor
                if row1['vendor'] != row2['vendor']:
                    continue
                
                # Check if amounts are similar (within 5%)
                amount_diff = abs(row1['amount'] - row2['amount']) / max(row1['amount'], row2['amount'])
                if amount_diff > 0.05:
                    continue
                
                # Check if dates are close (within 7 days)
                date_diff = abs((pd.to_datetime(row1['date']) - pd.to_datetime(row2['date'])).days)
                if date_diff > 7:
                    continue
                
                # Calculate similarity score
                similarity_score = (1 - amount_diff) * (1 - date_diff/7) * 100
                
                if similarity_score > 80:
                    near_duplicates.append({
                        'type': 'near_duplicate',
                        'similarity_score': similarity_score,
                        'amount_diff': amount_diff,
                        'date_diff': date_diff,
                        'record_1': i,
                        'record_2': j,
                        'risk_score': min(similarity_score, 95)
                    })
        
        return near_duplicates
    
    def _detect_unusual_amounts(self, df: pd.DataFrame) -> List[Dict]:
        """Detect statistically unusual transaction amounts"""
        unusual_amounts = []
        
        if 'amount' not in df.columns or len(df) < 10:
            return unusual_amounts
        
        amounts = df['amount'].dropna()
        mean_amount = amounts.mean()
        std_amount = amounts.std()
        
        # Define thresholds
        upper_threshold = mean_amount + (self.amount_variance_threshold * std_amount)
        lower_threshold = max(0, mean_amount - (self.amount_variance_threshold * std_amount))
        
        # Find outliers
        outliers = df[
            (df['amount'] > upper_threshold) | 
            (df['amount'] < lower_threshold)
        ]
        
        for idx, row in outliers.iterrows():
            z_score = abs((row['amount'] - mean_amount) / std_amount)
            
            unusual_amounts.append({
                'type': 'statistical_outlier',
                'record_id': idx,
                'amount': row['amount'],
                'z_score': z_score,
                'mean_amount': mean_amount,
                'std_amount': std_amount,
                'risk_score': min(z_score * 20, 100)
            })
        
        # Detect round number bias (potential manual manipulation)
        round_amounts = df[df['amount'] % 100 == 0]
        if len(round_amounts) / len(df) > 0.3:  # More than 30% round amounts
            unusual_amounts.append({
                'type': 'round_amount_bias',
                'count': len(round_amounts),
                'percentage': len(round_amounts) / len(df) * 100,
                'risk_score': 70
            })
        
        return unusual_amounts
    
    def _detect_fraud_indicators(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential fraud indicators"""
        fraud_indicators = []
        
        # Weekend transactions (unusual for B2B)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            weekend_transactions = df[df['date'].dt.dayofweek.isin([5, 6])]
            
            if len(weekend_transactions) > 0:
                fraud_indicators.append({
                    'type': 'weekend_transactions',
                    'count': len(weekend_transactions),
                    'percentage': len(weekend_transactions) / len(df) * 100,
                    'record_ids': weekend_transactions.index.tolist(),
                    'risk_score': 60
                })
        
        # After-hours transactions (if timestamp available)
        # Velocity-based fraud (many transactions in short time)
        if 'date' in df.columns and len(df) > 5:
            df_sorted = df.sort_values('date')
            
            # Look for high transaction velocity (>5 transactions per day)
            daily_counts = df_sorted.groupby(df_sorted['date'].dt.date).size()
            high_velocity_days = daily_counts[daily_counts > 5]
            
            if len(high_velocity_days) > 0:
                fraud_indicators.append({
                    'type': 'high_transaction_velocity',
                    'days_affected': len(high_velocity_days),
                    'max_daily_transactions': high_velocity_days.max(),
                    'risk_score': min(high_velocity_days.max() * 10, 100)
                })
        
        # Vendor name variations (potential shell companies)
        if 'vendor' in df.columns:
            vendor_variations = self._detect_vendor_variations(df['vendor'])
            if vendor_variations:
                fraud_indicators.extend(vendor_variations)
        
        return fraud_indicators
    
    def _detect_vendor_variations(self, vendors: pd.Series) -> List[Dict]:
        """Detect similar vendor names that might indicate shell companies"""
        from rapidfuzz import fuzz
        
        variations = []
        unique_vendors = vendors.unique()
        
        for i, vendor1 in enumerate(unique_vendors):
            for vendor2 in unique_vendors[i+1:]:
                similarity = fuzz.ratio(vendor1.lower(), vendor2.lower())
                
                # High similarity but not exact match
                if 70 < similarity < 100:
                    variations.append({
                        'type': 'vendor_name_variation',
                        'vendor_1': vendor1,
                        'vendor_2': vendor2,
                        'similarity_score': similarity,
                        'risk_score': similarity - 20  # Reduce risk score
                    })
        
        return variations
    
    def _detect_data_quality_issues(self, df: pd.DataFrame) -> List[Dict]:
        """Detect data quality issues that might indicate problems"""
        quality_issues = []
        
        # Missing data
        missing_data = df.isnull().sum()
        critical_fields = ['amount', 'vendor', 'date']
        
        for field in critical_fields:
            if field in missing_data and missing_data[field] > 0:
                quality_issues.append({
                    'type': 'missing_critical_data',
                    'field': field,
                    'missing_count': missing_data[field],
                    'percentage': missing_data[field] / len(df) * 100,
                    'risk_score': missing_data[field] / len(df) * 100
                })
        
        # Invalid data formats
        if 'amount' in df.columns:
            negative_amounts = df[df['amount'] < 0]
            if len(negative_amounts) > 0:
                quality_issues.append({
                    'type': 'negative_amounts',
                    'count': len(negative_amounts),
                    'record_ids': negative_amounts.index.tolist(),
                    'risk_score': 50
                })
        
        # Future dates
        if 'date' in df.columns:
            future_dates = df[pd.to_datetime(df['date']) > datetime.now()]
            if len(future_dates) > 0:
                quality_issues.append({
                    'type': 'future_dates',
                    'count': len(future_dates),
                    'record_ids': future_dates.index.tolist(),
                    'risk_score': 80
                })
        
        return quality_issues
    
    def generate_anomaly_report(self, anomalies: Dict[str, List]) -> pd.DataFrame:
        """Generate a summary report of all detected anomalies"""
        
        report_data = []
        
        for category, items in anomalies.items():
            for item in items:
                report_data.append({
                    'category': category,
                    'type': item.get('type', 'unknown'),
                    'risk_score': item.get('risk_score', 0),
                    'count': item.get('count', 1),
                    'description': self._generate_description(item),
                    'recommended_action': self._recommend_action(item)
                })
        
        return pd.DataFrame(report_data).sort_values('risk_score', ascending=False)
    
    def _generate_description(self, anomaly: Dict) -> str:
        """Generate human-readable description of anomaly"""
        
        anomaly_type = anomaly.get('type', 'unknown')
        
        descriptions = {
            'exact_duplicate': f"Found {anomaly.get('count', 0)} identical transactions for {anomaly.get('vendor', 'unknown vendor')}",
            'near_duplicate': f"Similar transactions detected with {anomaly.get('similarity_score', 0):.1f}% similarity",
            'statistical_outlier': f"Amount ${anomaly.get('amount', 0):.2f} is {anomaly.get('z_score', 0):.1f} standard deviations from mean",
            'weekend_transactions': f"{anomaly.get('count', 0)} transactions processed on weekends",
            'high_transaction_velocity': f"Up to {anomaly.get('max_daily_transactions', 0)} transactions in a single day",
            'missing_critical_data': f"Missing {anomaly.get('field', 'data')} in {anomaly.get('count', 0)} records"
        }
        
        return descriptions.get(anomaly_type, f"Anomaly of type {anomaly_type} detected")
    
    def _recommend_action(self, anomaly: Dict) -> str:
        """Recommend appropriate action for anomaly"""
        
        risk_score = anomaly.get('risk_score', 0)
        anomaly_type = anomaly.get('type', 'unknown')
        
        if risk_score >= 90:
            return "Immediate review required - High risk"
        elif risk_score >= 70:
            return "Review recommended - Medium risk"
        elif risk_score >= 50:
            return "Monitor closely - Low risk"
        else:
            return "Informational - No immediate action needed"
