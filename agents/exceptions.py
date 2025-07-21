import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import json

class ExceptionManagementAgent:
    def __init__(self):
        self.priority_weights = {
            'amount': 0.4,
            'age': 0.3,
            'risk': 0.2,
            'complexity': 0.1
        }
        
    def process_exceptions(self, matching_results: Dict[str, pd.DataFrame], 
                          anomalies: Dict[str, List]) -> Dict[str, Any]:
        """Process and prioritize exceptions from matching and anomaly detection"""
        
        exceptions = {
            'unmatched_transactions': [],
            'low_confidence_matches': [],
            'anomaly_exceptions': [],
            'data_quality_exceptions': []
        }
        
        # Process unmatched transactions
        if 'unmatched' in matching_results and not matching_results['unmatched'].empty:
            exceptions['unmatched_transactions'] = self._process_unmatched(
                matching_results['unmatched']
            )
        
        # Process low confidence matches
        for match_type, matches in matching_results.items():
            if 'fuzzy' in match_type and not matches.empty:
                low_confidence = matches[matches.get('match_confidence', 100) < 70]
                if not low_confidence.empty:
                    exceptions['low_confidence_matches'].extend(
                        self._process_low_confidence(low_confidence)
                    )
        
        # Process anomaly-based exceptions
        exceptions['anomaly_exceptions'] = self._process_anomalies(anomalies)
        
        # Calculate priorities and assign
        all_exceptions = self._combine_and_prioritize(exceptions)
        assigned_exceptions = self._assign_exceptions(all_exceptions)
        
        return {
            'exceptions_by_type': exceptions,
            'prioritized_exceptions': all_exceptions,
            'assigned_exceptions': assigned_exceptions,
            'summary_stats': self._generate_summary_stats(exceptions)
        }
    
    def _process_unmatched(self, unmatched_df: pd.DataFrame) -> List[Dict]:
        """Process unmatched transactions into exceptions"""
        exceptions = []
        
        for idx, row in unmatched_df.iterrows():
            age_days = (datetime.now() - pd.to_datetime(row['date'])).days
            
            exception = {
                'exception_id': f'UNM-{idx:04d}',
                'type': 'unmatched_transaction',
                'record_id': idx,
                'description': f"No matching record found for {row.get('vendor', 'Unknown')} transaction",
                'amount': row.get('amount', 0),
                'vendor': row.get('vendor', 'Unknown'),
                'date': row.get('date'),
                'age_days': age_days,
                'priority_score': self._calculate_priority_score({
                    'amount': row.get('amount', 0),
                    'age_days': age_days,
                    'risk_factors': ['unmatched']
                }),
                'suggested_actions': self._suggest_actions('unmatched', row),
                'status': 'Open',
                'assigned_to': None,
                'created_at': datetime.now(),
                'metadata': row.to_dict()
            }
            
            exceptions.append(exception)
        
        return exceptions
    
    def _process_low_confidence(self, low_confidence_df: pd.DataFrame) -> List[Dict]:
        """Process low confidence matches into exceptions"""
        exceptions = []
        
        for idx, row in low_confidence_df.iterrows():
            confidence = row.get('match_confidence', 0)
            age_days = (datetime.now() - pd.to_datetime(row['date'])).days
            
            exception = {
                'exception_id': f'LCM-{idx:04d}',
                'type': 'low_confidence_match',
                'record_id': idx,
                'description': f"Low confidence match ({confidence:.1f}%) for {row.get('vendor', 'Unknown')}",
                'amount': row.get('amount', 0),
                'vendor': row.get('vendor', 'Unknown'),
                'date': row.get('date'),
                'confidence': confidence,
                'age_days': age_days,
                'priority_score': self._calculate_priority_score({
                    'amount': row.get('amount', 0),
                    'age_days': age_days,
                    'risk_factors': ['low_confidence'],
                    'confidence': confidence
                }),
                'suggested_actions': self._suggest_actions('low_confidence', row),
                'status': 'Open',
                'assigned_to': None,
                'created_at': datetime.now(),
                'metadata': row.to_dict()
            }
            
            exceptions.append(exception)
        
        return exceptions
    
    def _process_anomalies(self, anomalies: Dict[str, List]) -> List[Dict]:
        """Process anomalies into exceptions"""
        exceptions = []
        
        for category, anomaly_list in anomalies.items():
            for i, anomaly in enumerate(anomaly_list):
                exception_id = f'ANM-{category[:3].upper()}-{i:03d}'
                
                exception = {
                    'exception_id': exception_id,
                    'type': 'anomaly_detected',
                    'anomaly_category': category,
                    'anomaly_type': anomaly.get('type', 'unknown'),
                    'description': self._generate_anomaly_description(anomaly),
                    'risk_score': anomaly.get('risk_score', 0),
                    'priority_score': anomaly.get('risk_score', 0),  # Risk score as priority
                    'suggested_actions': self._suggest_anomaly_actions(anomaly),
                    'status': 'Open',
                    'assigned_to': None,
                    'created_at': datetime.now(),
                    'metadata': anomaly
                }
                
                exceptions.append(exception)
        
        return exceptions
    
    def _calculate_priority_score(self, factors: Dict[str, Any]) -> float:
        """Calculate priority score based on multiple factors"""
        
        # Amount factor (higher amounts = higher priority)
        amount = factors.get('amount', 0)
        amount_score = min(amount / 10000 * 100, 100)  # Normalize to 0-100
        
        # Age factor (older = higher priority)
        age_days = factors.get('age_days', 0)
        age_score = min(age_days * 5, 100)  # 5 points per day, max 100
        
        # Risk factor
        risk_factors = factors.get('risk_factors', [])
        risk_score = len(risk_factors) * 20  # 20 points per risk factor
        
        # Confidence factor (lower confidence = higher priority)
        confidence = factors.get('confidence', 100)
        confidence_score = 100 - confidence
        
        # Weighted calculation
        priority = (
            amount_score * self.priority_weights['amount'] +
            age_score * self.priority_weights['age'] +
            risk_score * self.priority_weights['risk'] +
            confidence_score * self.priority_weights['complexity']
        )
        
        return min(priority, 100)
    
    def _suggest_actions(self, exception_type: str, record: pd.Series) -> List[str]:
        """Suggest appropriate actions for different exception types"""
        
        actions = {
            'unmatched': [
                "Verify if transaction exists in source systems",
                "Check for timing differences in posting dates",
                "Review vendor name variations or misspellings",
                "Confirm if manual journal entry is required"
            ],
            'low_confidence': [
                "Review suggested match manually",
                "Verify vendor name and amount accuracy",
                "Check for partial payments or adjustments",
                "Update matching rules if pattern is correct"
            ],
            'duplicate': [
                "Investigate potential duplicate entry",
                "Verify if both transactions are legitimate",
                "Check for system processing errors",
                "Remove duplicate entry if confirmed"
            ]
        }
        
        return actions.get(exception_type, ["Manual review required"])
    
    def _suggest_anomaly_actions(self, anomaly: Dict[str, Any]) -> List[str]:
        """Suggest actions for anomaly-based exceptions"""
        
        anomaly_type = anomaly.get('type', 'unknown')
        risk_score = anomaly.get('risk_score', 0)
        
        base_actions = {
            'exact_duplicate': ["Verify if duplicate is legitimate", "Remove duplicate entry if confirmed"],
            'statistical_outlier': ["Verify unusual amount with supporting documentation", "Check authorization levels"],
            'weekend_transactions': ["Verify business justification for weekend processing"],
            'high_transaction_velocity': ["Review for potential fraud or system errors"]
        }
        
        actions = base_actions.get(anomaly_type, ["Investigate anomaly"])
        
        # Add risk-based actions
        if risk_score >= 90:
            actions.insert(0, "URGENT: Immediate investigation required")
        elif risk_score >= 70:
            actions.insert(0, "High priority review recommended")
        
        return actions
    
    def _combine_and_prioritize(self, exceptions: Dict[str, List]) -> List[Dict]:
        """Combine all exceptions and sort by priority"""
        
        all_exceptions = []
        for exception_type, exception_list in exceptions.items():
            all_exceptions.extend(exception_list)
        
        # Sort by priority score (descending)
        all_exceptions.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Add priority rankings
        for i, exception in enumerate(all_exceptions):
            if i < len(all_exceptions) * 0.2:  # Top 20%
                exception['priority_level'] = 'Critical'
            elif i < len(all_exceptions) * 0.5:  # Top 50%
                exception['priority_level'] = 'High'
            elif i < len(all_exceptions) * 0.8:  # Top 80%
                exception['priority_level'] = 'Medium'
            else:
                exception['priority_level'] = 'Low'
        
        return all_exceptions
    
    def _assign_exceptions(self, exceptions: List[Dict]) -> Dict[str, List]:
        """Assign exceptions to team members based on workload and expertise"""
        
        # Simulated team members with specialties
        team_members = {
            'John Smith': {'specialty': ['unmatched', 'anomaly'], 'workload': 0, 'max_workload': 10},
            'Sarah Connor': {'specialty': ['low_confidence', 'duplicate'], 'workload': 0, 'max_workload': 12},
            'Mike Johnson': {'specialty': ['anomaly', 'data_quality'], 'workload': 0, 'max_workload': 8}
        }
        
        assignments = {member: [] for member in team_members.keys()}
        assignments['Unassigned'] = []
        
        for exception in exceptions:
            assigned = False
            exception_type = exception.get('type', 'unknown')
            
            # Try to assign based on specialty and workload
            for member, info in team_members.items():
                if (any(spec in exception_type for spec in info['specialty']) and 
                    info['workload'] < info['max_workload']):
                    
                    assignments[member].append(exception)
                    exception['assigned_to'] = member
                    info['workload'] += 1
                    assigned = True
                    break
            
            # If not assigned by specialty, assign to least loaded member
            if not assigned:
                least_loaded = min(team_members.items(), key=lambda x: x[1]['workload'])
                if least_loaded[1]['workload'] < least_loaded[1]['max_workload']:
                    assignments[least_loaded[0]].append(exception)
                    exception['assigned_to'] = least_loaded[0]
                    least_loaded[1]['workload'] += 1
                else:
                    assignments['Unassigned'].append(exception)
                    exception['assigned_to'] = None
        
        return assignments
    
    def _generate_summary_stats(self, exceptions: Dict[str, List]) -> Dict[str, Any]:
        """Generate summary statistics for exceptions"""
        
        total_exceptions = sum(len(exc_list) for exc_list in exceptions.values())
        
        # Calculate totals by type
        by_type = {exc_type: len(exc_list) for exc_type, exc_list in exceptions.items()}
        
        # Calculate priority distribution (from all exceptions)
        all_exceptions = []
        for exc_list in exceptions.values():
            all_exceptions.extend(exc_list)
        
        priority_counts = {}
        for exception in all_exceptions:
            priority = exception.get('priority_level', 'Unknown')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Calculate amount statistics
        amounts = [exc.get('amount', 0) for exc in all_exceptions if exc.get('amount')]
        total_amount = sum(amounts)
        avg_amount = total_amount / len(amounts) if amounts else 0
        
        return {
            'total_exceptions': total_exceptions,
            'by_type': by_type,
            'by_priority': priority_counts,
            'total_amount': total_amount,
            'average_amount': avg_amount,
            'oldest_exception_days': max([exc.get('age_days', 0) for exc in all_exceptions], default=0)
        }
    
    def _generate_anomaly_description(self, anomaly: Dict[str, Any]) -> str:
        """Generate human-readable description for anomaly"""
        
        anomaly_type = anomaly.get('type', 'unknown')
        
        descriptions = {
            'exact_duplicate': f"Exact duplicate found: {anomaly.get('count', 0)} identical transactions",
            'statistical_outlier': f"Unusual amount: ${anomaly.get('amount', 0):.2f} (Z-score: {anomaly.get('z_score', 0):.1f})",
            'weekend_transactions': f"{anomaly.get('count', 0)} transactions processed on weekends",
            'high_transaction_velocity': f"High velocity: {anomaly.get('max_daily_transactions', 0)} transactions in one day"
        }
        
        return descriptions.get(anomaly_type, f"Anomaly detected: {anomaly_type}")
    
    def update_exception_status(self, exception_id: str, new_status: str, 
                               resolution_notes: str = None) -> Dict[str, Any]:
        """Update exception status and add resolution notes"""
        
        # This would typically update a database
        # For now, return a confirmation structure
        return {
            'exception_id': exception_id,
            'old_status': 'Open',  # Would be retrieved from database
            'new_status': new_status,
            'resolution_notes': resolution_notes,
            'updated_by': 'System',  # Would be current user
            'updated_at': datetime.now(),
            'success': True
        }
