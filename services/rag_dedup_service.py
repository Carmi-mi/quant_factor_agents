"""
RAG Semantic Similarity Service
Provides semantic similarity computation for idea deduplication
"""
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class RAGDedupService:
    """
    RAG-based semantic similarity service.
    Only responsible for computing embeddings and checking similarity.
    Does NOT manage file storage or history.
    """

    # Default embedding model
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RAG semantic similarity service

        Args:
            config: Configuration dict with keys:
                - similarity_threshold: Threshold for considering duplicates (default: 0.85)
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)

        # Resolve model path (local or download)
        self.embedding_model_name = self._resolve_model_path()

        # Initialize embedding model (lazy loading)
        self._model = None

    def _resolve_model_path(self) -> str:
        """
        Resolve model path: use local if exists, otherwise use model name (will download)

        Returns:
            Resolved model path or name
        """
        model_name = self.DEFAULT_MODEL

        # Look in project_root/models
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(project_root, 'models', model_name)

        if os.path.exists(local_path):
            print(f"[RAGDedup] Using local model: {local_path}")
            return local_path

        # Fallback to model name (will download from HuggingFace on first use)
        print(f"[RAGDedup] Local model not found, will use/download: {model_name}")
        return model_name

    @property
    def model(self):
        """Lazy load embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[RAGDedup] Loading embedding model: {self.embedding_model_name}")
                self._model = SentenceTransformer(self.embedding_model_name)
                print(f"[RAGDedup] Model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for RAG deduplication. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode([text], convert_to_numpy=True)[0]

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> Tuple[float, int]:
        """
        Find most similar embedding from candidates

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings

        Returns:
            (similarity_score, index_of_most_similar)
        """
        if candidate_embeddings.size == 0:
            return 0.0, -1

        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return 0.0, -1
        query_normalized = query_embedding / query_norm

        # Normalize candidates
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        candidate_norms[candidate_norms == 0] = 1
        candidates_normalized = candidate_embeddings / candidate_norms[:, np.newaxis]

        # Compute cosine similarities
        similarities = np.dot(candidates_normalized, query_normalized)

        # Find max similarity
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]

        return float(max_similarity), int(max_idx)

    def filter_duplicates(
        self,
        ideas: List[Dict[str, Any]],
        existing_embeddings: np.ndarray,
        threshold: float = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Filter duplicate ideas based on semantic similarity

        Args:
            ideas: List of new ideas
            existing_embeddings: Embeddings of existing ideas
            threshold: Similarity threshold

        Returns:
            (unique_ideas, embeddings_of_unique_ideas)
        """
        if threshold is None:
            threshold = self.similarity_threshold

        unique_ideas = []
        unique_embeddings = []
        unique_embeddings_array = np.array([]) if existing_embeddings is None or existing_embeddings.size == 0 else existing_embeddings

        for idea in ideas:
            content = idea.get("content", "").strip()
            if not content:
                continue

            # Compute embedding once for this idea
            idea_embedding = self.compute_embedding(content)

            # Check if duplicate against existing history
            if unique_embeddings_array.size > 0:
                max_similarity, _ = self.find_most_similar(idea_embedding, unique_embeddings_array)
                if max_similarity >= threshold:
                    continue

            # Check if duplicate against other new ideas in this batch
            if unique_embeddings:
                new_embeddings_array = np.array(unique_embeddings)
                max_similarity_in_batch, _ = self.find_most_similar(idea_embedding, new_embeddings_array)
                if max_similarity_in_batch >= threshold:
                    continue

            unique_ideas.append(idea)
            unique_embeddings.append(idea_embedding)

        if len(unique_ideas) < len(ideas):
            print(f"[RAGDedup] Filtered {len(ideas) - len(unique_ideas)} duplicate ideas")

        # Convert to numpy array
        if unique_embeddings:
            unique_embeddings_array = np.array(unique_embeddings)
        else:
            unique_embeddings_array = np.array([])

        return unique_ideas, unique_embeddings_array
