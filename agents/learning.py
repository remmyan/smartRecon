import json
import uuid
from typing import Dict, List, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import Config
import pandas as pd
import numpy as np

_embedder = None
_client = None
_collection = None

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    else:
        return obj

class LearningAgent:
    def __init__(self):
        # Initialize ChromaDB
        global _client, _embedder, _collection
        if _client is None:
            _client = chromadb.PersistentClient(
                path=Config.CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
        self.chroma_client = _client
        if _embedder is None:
            _embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.embedder = _embedder
        if _collection is None:
            try:
                _collection = self._get_or_create_collection("corrections")
            except Exception:
                _collection = self.chroma_client.create_collection("corrections")
        self.corrections_collection = _collection
        self.learning_stats = {'total_corrections': 0, 'last_updated': datetime.now().isoformat(), 'corrections_collection': self.corrections_collection}

    def _get_or_create_collection(self, name: str):
        try:
            return self.chroma_client.get_collection(name)
        except Exception:
            return self.chroma_client.create_collection(name, metadata={"hnsw:space": "cosine"})

    def _embedding_text(self, record: Dict[str, Any]) -> str:
        """
        Generate normalized embedding input string from the corrected record fields.
        This ensures consistent embedding generation for both storage and retrieval.
        """
        amount = record.get('amount', 0)
        vendor = record.get('vendor', '').strip().lower()
        description = record.get('description', '').strip().lower() if 'description' in record else ''
        return f"{amount} {vendor} {description}"

    def store_correction(self, original_record: Dict[str, Any], corrected_record: Dict[str, Any]) -> str:
        """
        Store user correction embedding into ChromaDB with consistent embedding input.
        """
        # Full correction text stored as document for reference
        correction_text = (
            f"Original: {json.dumps(sanitize_for_json(original_record))} | "
            f"Corrected: {json.dumps(sanitize_for_json(corrected_record))}"
        )

        # Embed using consistent normalized combined fields string
        embedding_text = self._embedding_text(corrected_record)
        embedding = self.embedder.encode(embedding_text, normalize_embeddings=True).tolist()

        correction_id = str(uuid.uuid4())
        metadata = {
            'vendor': corrected_record.get('vendor', ''),
            'amount': corrected_record.get('amount', 0),
            'correction_timestamp': datetime.now().isoformat(),
            'user_feedback': corrected_record.get('user_feedback_reason', ''),
            'user_confirmed_match': corrected_record.get('user_confirmed_match', False)
        }

        # Add to ChromaDB collection
        self.corrections_collection.add(
            documents=[correction_text],  # store full text for reference
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[correction_id]
        )

        self.learning_stats['total_corrections'] += 1
        self.learning_stats['last_updated'] = datetime.now().isoformat()
        print(f"Stored correction ID: {correction_id}")
        return correction_id

    def embed_record(self, record: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding of the record using the same consistent normalized text format.
        """
        text = self._embedding_text(record)
        embedding = self.embedder.encode(text, normalize_embeddings=True)
        return embedding

    def retrieve_similar_patterns(self, record: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB to retrieve patterns similar to the input record,
        filtering by similarity threshold and returning metadata.
        """
        query_embedding = self.embed_record(record)

        results = self.corrections_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['metadatas', 'distances']
        )

        matched_patterns = []
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            similarity = 1 - distance  # assuming cosine distance metric
            print(f"Candidate metadata: {metadata}, similarity score: {similarity:.4f}")
            if similarity >= Config.RAG_SIMILARITY_THRESHOLD:
                matched_patterns.append({
                    "pattern_metadata": metadata,
                    "similarity": similarity
                })

        print(f"Retrieved {len(matched_patterns)} similar patterns.")
        return matched_patterns

    def test_store_and_retrieve_same_pattern():
        example_correction = {
            'amount': 100.0,
            'vendor': 'Example Vendor',
            'description': 'Office supplies'
        }

        # Store correction
        self.store_correction(example_correction)

        # Now retrieve immediately
        retrieved = self.retrieve_similar_patterns(example_correction, top_k=10)
        print("Retrieved patterns after storing:", retrieved)
    def get_learning_stats(self) -> Dict[str, Any]:
        return self.learning_stats

# Global instance
if __name__ == "__main__":
    test_matching_flow()
