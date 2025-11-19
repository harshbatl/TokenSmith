import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np


class QueryCache:
    # Cache manager for storing and retrieving query results based on semantic similarity.

    
    def __init__(self, 
                 cache_dir: str = "cache",
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 100):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Cache storage paths
        self.entries_file = self.cache_dir / "cache_entries.pkl"
        self.embeddings_file = self.cache_dir / "cache_embeddings.npy"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Load existing cache
        self.entries: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Dict = {}
        
        self._load_cache()
    
    def _load_cache(self):
        # Loading the cache from the disk
        try:
            if self.entries_file.exists():
                with open(self.entries_file, 'rb') as f:
                    self.entries = pickle.load(f)
            
            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            print(f"[Cache] Loaded {len(self.entries)} cached queries")
        except Exception as e:
            print(f"[Cache] Error loading cache: {e}. Starting fresh.")
            self.entries = []
            self.embeddings = None
            self.metadata = {}
    
    def _save_cache(self):
        # Saving the cache to the disk
        try:
            with open(self.entries_file, 'wb') as f:
                pickle.dump(self.entries, f)
            
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"[Cache] Saved {len(self.entries)} queries to cache")
        except Exception as e:
            print(f"[Cache] Error saving cache: {e}")
    
    def _compute_similarity(self, query_embedding: np.ndarray) -> Tuple[int, float]:
        # Computing the similarity score
        if self.embeddings is None or len(self.embeddings) == 0:
            return -1, 0.0
        
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        cache_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(cache_norms, query_norm)
        
        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        return best_idx, best_score
    
    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Dict]:
        # Fetching from the cache itself
        if len(self.entries) == 0:
            return None
        
        best_idx, similarity = self._compute_similarity(query_embedding)
        
        if similarity >= self.similarity_threshold:
            entry = self.entries[best_idx]
            entry['similarity_score'] = similarity
            entry['cache_hit'] = True
            
            # Update access metadata
            self.metadata[entry['query_id']]['last_accessed'] = datetime.now().isoformat()
            self.metadata[entry['query_id']]['access_count'] = self.metadata[entry['query_id']].get('access_count', 0) + 1
            
            print(f"[Cache] HIT! Similarity: {similarity:.3f} | Cached query: '{entry['original_query']}'")
            return entry
        
        print(f"[Cache] MISS. Best match similarity: {similarity:.3f} (threshold: {self.similarity_threshold})")
        return None
    
    def add(self, 
            query: str, 
            query_embedding: np.ndarray,
            answer: str,
            chunks_info: Optional[List] = None,
            hyde_query: Optional[str] = None,
            citation_manager = None):
        
        # Generate unique ID
        query_id = hashlib.md5(query.encode()).hexdigest()[:16]
        
        # Create cache entry
        entry = {
            'query_id': query_id,
            'original_query': query,
            'answer': answer,
            'chunks_info': chunks_info,
            'hyde_query': hyde_query,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': False
        }
        
        # Store citation information if available
        if citation_manager:
            entry['citations'] = {
                'unique_sources': list(citation_manager.unique_sources),
                'used_chunks': citation_manager.used_chunks
            }
        
        # Add to entries
        self.entries.append(entry)
        
        # Add embedding
        if self.embeddings is None:
            self.embeddings = query_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, query_embedding.reshape(1, -1)])
        
        # Add metadata
        self.metadata[query_id] = {
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1,
            'query_length': len(query)
        }
        
        # Check cache size and cleanup if needed
        if len(self.entries) > self.max_cache_size:
            self._cleanup_cache()
        
        # Save to disk
        self._save_cache()
        
        print(f"[Cache] Added query '{query[:50]}...' to cache")
    
    def _cleanup_cache(self):
        # Remove least recently used (LRU Policy) entries when cache is full.
        print(f"[Cache] Cache full ({len(self.entries)} entries). Cleaning up...")
        
        # Sort by last accessed time
        sorted_ids = sorted(
            self.metadata.keys(),
            key=lambda qid: self.metadata[qid]['last_accessed']
        )
        
        # Remove oldest 20% of entries
        num_to_remove = len(self.entries) // 5
        ids_to_remove = set(sorted_ids[:num_to_remove])
        
        # Filter entries
        new_entries = []
        new_embeddings = []
        
        for i, entry in enumerate(self.entries):
            if entry['query_id'] not in ids_to_remove:
                new_entries.append(entry)
                new_embeddings.append(self.embeddings[i])
        
        # Update cache
        self.entries = new_entries
        self.embeddings = np.array(new_embeddings) if new_embeddings else None
        
        # Remove metadata
        for qid in ids_to_remove:
            del self.metadata[qid]
        
        print(f"[Cache] Removed {num_to_remove} entries. Cache size: {len(self.entries)}")
    
    def clear(self):
        # Clearing the entire cache
        self.entries = []
        self.embeddings = None
        self.metadata = {}
        
        # Remove cache files
        for file in [self.entries_file, self.embeddings_file, self.metadata_file]:
            if file.exists():
                file.unlink()
        
        print("[Cache] Cache cleared")
    
    def get_stats(self) -> Dict:
        # Getting the cache stats
        if not self.entries:
            return {"size": 0, "total_accesses": 0}
        
        total_accesses = sum(m.get('access_count', 0) for m in self.metadata.values())
        
        return {
            "size": len(self.entries),
            "total_accesses": total_accesses,
            "avg_accesses_per_query": total_accesses / len(self.entries) if self.entries else 0,
            "cache_dir": str(self.cache_dir),
            "similarity_threshold": self.similarity_threshold
        }