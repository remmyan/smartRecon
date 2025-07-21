import pandas as pd
import numpy as np
import json
import os
import uuid
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

try:
    from ..config import Config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import Config

class LearningAgent:
    def __init__(self):
        """Initialize Learning Agent with ChromaDB vector storage"""
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Create collections for different types of patterns
        self.transaction_patterns = self._get_or_create_collection("transaction_patterns")
        self.vendor_patterns = self._get_or_create_collection("vendor_patterns")
        self.description_patterns = self._get_or_create_collection("description_patterns")
        self.amount_patterns = self._get_or_create_collection("amount_patterns")
        self.user_corrections = self._get_or_create_collection("user_corrections")
        
        # Statistics tracking
        self.learning_stats = {
            'total_patterns': 0,
            'accuracy_improvement': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        print("✅ Learning Agent initialized with ChromaDB")
        
    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one"""
        try:
            return self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception:
            return self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
    
    def update_patterns(self, matching_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Update learning patterns based on matching results using ChromaDB"""
        
        learning_updates = {
            'new_patterns': 0,
            'updated_patterns': 0,
            'total_patterns': 0,
            'accuracy_improvement': 0.0,
            'embedding_updates': 0
        }
        
        try:
            # Process successful matches
            new_patterns = 0
            
            for match_type, matches in matching_results.items():
                if not matches.empty:
                    if match_type == 'llm_matches':
                        new_patterns += self._store_llm_patterns(matches)
                    elif match_type == 'exact_matches':
                        new_patterns += self._store_exact_patterns(matches)
                    elif match_type == 'fuzzy_matches':
                        new_patterns += self._store_fuzzy_patterns(matches)
            
            learning_updates['new_patterns'] = new_patterns
            learning_updates['total_patterns'] = self._get_total_pattern_count()
            learning_updates['accuracy_improvement'] = self._calculate_improvement()
            
            # Update statistics
            self.learning_stats['total_patterns'] = learning_updates['total_patterns']
            self.learning_stats['accuracy_improvement'] = learning_updates['accuracy_improvement']
            self.learning_stats['last_updated'] = datetime.now().isoformat()
            
            print(f"📚 Updated knowledge base: {new_patterns} new patterns added")
            
        except Exception as e:
            print(f"❌ Error updating patterns: {e}")
            
        return learning_updates
    
    def _store_llm_patterns(self, llm_matches: pd.DataFrame) -> int:
        """Store patterns from LLM/ChatGPT matches"""
        
        patterns_added = 0
        
        for idx, match in llm_matches.iterrows():
            try:
                # Create pattern document
                pattern_doc = {
                    'match_type': 'llm_semantic',
                    'confidence': float(match.get('match_confidence', 0)),
                    'vendor': str(match.get('vendor', '')),
                    'amount': float(match.get('amount', 0)),
                    'description': str(match.get('description', '')),
                    'reasoning': str(match.get('match_reasoning', '')),
                    'timestamp': datetime.now().isoformat(),
                    'pattern_id': str(uuid.uuid4())
                }
                
                # Create searchable text for embeddings
                searchable_text = f"vendor: {pattern_doc['vendor']} description: {pattern_doc['description']} reasoning: {pattern_doc['reasoning']}"
                
                # Add to transaction patterns collection
                self.transaction_patterns.add(
                    documents=[searchable_text],
                    metadatas=[pattern_doc],
                    ids=[pattern_doc['pattern_id']]
                )
                
                # Store vendor-specific patterns
                if pattern_doc['vendor']:
                    self._store_vendor_pattern(pattern_doc)
                
                # Store description patterns
                if pattern_doc['description']:
                    self._store_description_pattern(pattern_doc)
                
                patterns_added += 1
                
            except Exception as e:
                print(f"Error storing LLM pattern: {e}")
                continue
        
        return patterns_added
    
    def _store_exact_patterns(self, exact_matches: pd.DataFrame) -> int:
        """Store patterns from exact matches"""
        
        patterns_added = 0
        
        for idx, match in exact_matches.iterrows():
            try:
                pattern_doc = {
                    'match_type': 'exact',
                    'confidence': 100.0,
                    'vendor': str(match.get('vendor', '')),
                    'amount': float(match.get('amount', 0)),
                    'description': str(match.get('description', '')),
                    'timestamp': datetime.now().isoformat(),
                    'pattern_id': str(uuid.uuid4())
                }
                
                searchable_text = f"exact match vendor: {pattern_doc['vendor']} amount: {pattern_doc['amount']} description: {pattern_doc['description']}"
                
                self.transaction_patterns.add(
                    documents=[searchable_text],
                    metadatas=[pattern_doc],
                    ids=[pattern_doc['pattern_id']]
                )
                
                patterns_added += 1
                
            except Exception as e:
                print(f"Error storing exact pattern: {e}")
                continue
        
        return patterns_added
    
    def _store_fuzzy_patterns(self, fuzzy_matches: pd.DataFrame) -> int:
        """Store patterns from fuzzy matches"""
        
        patterns_added = 0
        
        for idx, match in fuzzy_matches.iterrows():
            try:
                pattern_doc = {
                    'match_type': 'fuzzy',
                    'confidence': float(match.get('match_confidence', 0)),
                    'vendor': str(match.get('vendor', '')),
                    'amount': float(match.get('amount', 0)),
                    'description': str(match.get('description', '')),
                    'reasoning': str(match.get('match_reasoning', '')),
                    'timestamp': datetime.now().isoformat(),
                    'pattern_id': str(uuid.uuid4())
                }
                
                searchable_text = f"fuzzy match vendor: {pattern_doc['vendor']} description: {pattern_doc['description']}"
                
                self.transaction_patterns.add(
                    documents=[searchable_text],
                    metadatas=[pattern_doc],
                    ids=[pattern_doc['pattern_id']]
                )
                
                patterns_added += 1
                
            except Exception as e:
                print(f"Error storing fuzzy pattern: {e}")
                continue
        
        return patterns_added
    
    def _store_vendor_pattern(self, pattern_doc: Dict[str, Any]):
        """Store vendor-specific patterns"""
        
        try:
            vendor_text = f"vendor: {pattern_doc['vendor']} typical_amount: {pattern_doc['amount']} confidence: {pattern_doc['confidence']}"
            
            vendor_pattern = {
                'vendor': pattern_doc['vendor'],
                'typical_amount': pattern_doc['amount'],
                'confidence_score': pattern_doc['confidence'],
                'match_type': pattern_doc['match_type'],
                'timestamp': pattern_doc['timestamp'],
                'pattern_id': f"vendor_{uuid.uuid4()}"
            }
            
            self.vendor_patterns.add(
                documents=[vendor_text],
                metadatas=[vendor_pattern],
                ids=[vendor_pattern['pattern_id']]
            )
            
        except Exception as e:
            print(f"Error storing vendor pattern: {e}")
    
    def _store_description_pattern(self, pattern_doc: Dict[str, Any]):
        """Store description-specific patterns"""
        
        try:
            description_pattern = {
                'description': pattern_doc['description'],
                'vendor': pattern_doc['vendor'],
                'confidence_score': pattern_doc['confidence'],
                'match_type': pattern_doc['match_type'],
                'timestamp': pattern_doc['timestamp'],
                'pattern_id': f"desc_{uuid.uuid4()}"
            }
            
            self.description_patterns.add(
                documents=[pattern_doc['description']],
                metadatas=[description_pattern],
                ids=[description_pattern['pattern_id']]
            )
            
        except Exception as e:
            print(f"Error storing description pattern: {e}")
    
    def retrieve_similar_patterns(self, transaction: Dict[str, Any], top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve similar patterns from ChromaDB for RAG enhancement"""
        
        if top_k is None:
            top_k = Config.RAG_TOP_K
        
        try:
            # Create query text from transaction
            query_text = f"vendor: {transaction.get('vendor', '')} amount: {transaction.get('amount', 0)} description: {transaction.get('description', '')}"
            
            # Query ChromaDB for similar patterns
            results = self.transaction_patterns.query(
                query_texts=[query_text],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            similar_patterns = []
            
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    if similarity_score >= Config.RAG_SIMILARITY_THRESHOLD:
                        pattern = {
                            'metadata': metadata,
                            'document': results['documents'][0][i],
                            'similarity_score': similarity_score,
                            'distance': results['distances'][0][i]
                        }
                        similar_patterns.append(pattern)
            
            return similar_patterns
            
        except Exception as e:
            print(f"Error retrieving similar patterns: {e}")
            return []
    
    def retrieve_vendor_patterns(self, vendor: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve patterns for a specific vendor"""
        
        try:
            results = self.vendor_patterns.query(
                query_texts=[f"vendor: {vendor}"],
                n_results=top_k,
                where={"vendor": {"$eq": vendor}},
                include=['metadatas', 'documents', 'distances']
            )
            
            patterns = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    patterns.append({
                        'metadata': metadata,
                        'similarity_score': 1 - results['distances'][0][i]
                    })
            
            return patterns
            
        except Exception as e:
            print(f"Error retrieving vendor patterns: {e}")
            return []
    
    def store_user_correction(self, original_match: Dict[str, Any], corrected_match: Dict[str, Any]):
        """Store user corrections for continuous learning"""
        
        try:
            correction_doc = {
                'original_vendor': str(original_match.get('vendor', '')),
                'original_amount': float(original_match.get('amount', 0)),
                'original_description': str(original_match.get('description', '')),
                'corrected_vendor': str(corrected_match.get('vendor', '')),
                'corrected_amount': float(corrected_match.get('amount', 0)),
                'corrected_description': str(corrected_match.get('description', '')),
                'correction_type': 'user_manual',
                'timestamp': datetime.now().isoformat(),
                'correction_id': str(uuid.uuid4())
            }
            
            correction_text = f"correction from {correction_doc['original_vendor']} to {correction_doc['corrected_vendor']} amount {correction_doc['original_amount']} to {correction_doc['corrected_amount']}"
            
            self.user_corrections.add(
                documents=[correction_text],
                metadatas=[correction_doc],
                ids=[correction_doc['correction_id']]
            )
            
            print(f"📝 User correction stored: {correction_doc['original_vendor']} → {correction_doc['corrected_vendor']}")
            
        except Exception as e:
            print(f"Error storing user correction: {e}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        
        insights = {
            'total_patterns': self._get_total_pattern_count(),
            'pattern_breakdown': self._get_pattern_breakdown(),
            'top_vendors': self._get_top_vendors(),
            'recent_patterns': self._get_recent_patterns(),
            'learning_stats': self.learning_stats
        }
        
        return insights
    
    def _get_total_pattern_count(self) -> int:
        """Get total number of patterns stored"""
        
        try:
            return self.transaction_patterns.count()
        except Exception:
            return 0
    
    def _get_pattern_breakdown(self) -> Dict[str, int]:
        """Get breakdown of patterns by type"""
        
        breakdown = {
            'transaction_patterns': 0,
            'vendor_patterns': 0,
            'description_patterns': 0,
            'user_corrections': 0
        }
        
        try:
            breakdown['transaction_patterns'] = self.transaction_patterns.count()
            breakdown['vendor_patterns'] = self.vendor_patterns.count()
            breakdown['description_patterns'] = self.description_patterns.count()
            breakdown['user_corrections'] = self.user_corrections.count()
        except Exception as e:
            print(f"Error getting pattern breakdown: {e}")
        
        return breakdown
    
    def _get_top_vendors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top vendors by pattern count"""
        
        try:
            # This is a simplified version - ChromaDB doesn't have built-in aggregation
            # In a production system, you might want to maintain this separately
            results = self.vendor_patterns.get(
                include=['metadatas'],
                limit=limit
            )
            
            vendor_counts = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    vendor = metadata.get('vendor', 'Unknown')
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
            
            sorted_vendors = sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)
            return [{'vendor': vendor, 'pattern_count': count} for vendor, count in sorted_vendors[:limit]]
            
        except Exception as e:
            print(f"Error getting top vendors: {e}")
            return []
    
    def _get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently added patterns"""
        
        try:
            results = self.transaction_patterns.get(
                include=['metadatas', 'documents'],
                limit=limit
            )
            
            recent_patterns = []
            if results['metadatas'] and results['documents']:
                for i, metadata in enumerate(results['metadatas']):
                    recent_patterns.append({
                        'pattern': metadata,
                        'document': results['documents'][i]
                    })
            
            return recent_patterns
            
        except Exception as e:
            print(f"Error getting recent patterns: {e}")
            return []
    
    def _calculate_improvement(self) -> float:
        """Calculate accuracy improvement based on stored patterns"""
        
        try:
            total_patterns = self._get_total_pattern_count()
            
            # Simple improvement calculation based on pattern count
            # In production, this would be based on actual accuracy metrics
            if total_patterns < 100:
                return total_patterns * 0.1  # 0.1% per pattern
            elif total_patterns < 500:
                return 10.0 + (total_patterns - 100) * 0.05  # Diminishing returns
            else:
                return 30.0 + (total_patterns - 500) * 0.01  # Even more diminishing returns
            
        except Exception:
            return 0.0
    
    def clear_all_patterns(self):
        """Clear all stored patterns (use with caution!)"""
        
        try:
            self.chroma_client.delete_collection("transaction_patterns")
            self.chroma_client.delete_collection("vendor_patterns")
            self.chroma_client.delete_collection("description_patterns")
            self.chroma_client.delete_collection("user_corrections")
            
            # Recreate collections
            self.transaction_patterns = self._get_or_create_collection("transaction_patterns")
            self.vendor_patterns = self._get_or_create_collection("vendor_patterns")
            self.description_patterns = self._get_or_create_collection("description_patterns")
            self.user_corrections = self._get_or_create_collection("user_corrections")
            
            print("🗑️ All patterns cleared from ChromaDB")
            
        except Exception as e:
            print(f"Error clearing patterns: {e}")
    
    def export_patterns_to_json(self, filename: str = None) -> str:
        """Export all patterns to JSON file for backup"""
        
        if filename is None:
            filename = f"patterns_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'learning_stats': self.learning_stats,
                'patterns': {}
            }
            
            # Export transaction patterns
            results = self.transaction_patterns.get(include=['metadatas', 'documents'])
            export_data['patterns']['transaction_patterns'] = {
                'metadatas': results.get('metadatas', []),
                'documents': results.get('documents', [])
            }
            
            # Export vendor patterns
            results = self.vendor_patterns.get(include=['metadatas', 'documents'])
            export_data['patterns']['vendor_patterns'] = {
                'metadatas': results.get('metadatas', []),
                'documents': results.get('documents', [])
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📦 Patterns exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error exporting patterns: {e}")
            return ""

# Create global instance
learning_agent = LearningAgent()
