"""
Base Cross-Encoder Re-ranker Class.

Provides the foundation for all cross-encoder reranker implementations
for Indonesian code search.
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossEncoderReranker(ABC):
    """
    Abstract base class for cross-encoder rerankers.
    
    Cross-encoders jointly encode query-document pairs to compute
    relevance scores, providing higher precision than bi-encoders
    at the cost of slower inference.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
        logger.info(f"Using device: {self.device}")
    
    @abstractmethod
    def load_model(self):
        """Load the cross-encoder model and tokenizer."""
        pass
    
    def score(
        self,
        query: str,
        documents: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Score query-document pairs.
        
        Args:
            query: The query string
            documents: List of document strings to score
            show_progress: Whether to show progress bar
            
        Returns:
            Array of relevance scores for each document
        """
        if not self._is_loaded:
            self.load_model()
        
        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Score in batches
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_inputs = {
                    key: val[i:i+self.batch_size] 
                    for key, val in inputs.items()
                }
                outputs = self.model(**batch_inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores.tolist())
        
        return np.array(scores)
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on cross-encoder scores.
        
        Args:
            query: The query string
            documents: List of document dicts with 'id' and 'text' keys
            top_k: Number of top documents to return
            show_progress: Whether to show progress bar
            
        Returns:
            Re-ranked list of documents with scores
        """
        if not self._is_loaded:
            self.load_model()
        
        # Extract text for scoring
        doc_texts = [doc.get('text', doc.get('content', '')) for doc in documents]
        
        # Score documents
        scores = self.score(query, doc_texts, show_progress=show_progress)
        
        # Add scores to documents and sort
        scored_docs = [
            {**doc, 'cross_encoder_score': float(score)}
            for doc, score in zip(documents, scores)
        ]
        scored_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        
        return scored_docs[:top_k]
    
    def compute_similarity(
        self,
        queries: List[str],
        documents: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix for batch queries and documents.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            
        Returns:
            Similarity matrix of shape (num_queries, num_documents)
        """
        if not self._is_loaded:
            self.load_model()
        
        pairs = [[q, d] for q in queries for d in documents]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Compute scores in batches
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_inputs = {
                    key: val[i:i+self.batch_size] 
                    for key, val in inputs.items()
                }
                outputs = self.model(**batch_inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                all_scores.extend(batch_scores.tolist())
        
        # Reshape to similarity matrix
        scores_matrix = np.array(all_scores).reshape(len(queries), len(documents))
        
        return scores_matrix
    
    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """
        Predict scores for query-document pairs.
        
        Args:
            pairs: List of [query, document] pairs
            
        Returns:
            Array of scores
        """
        if not self._is_loaded:
            self.load_model()
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        return scores
    
    def get_latency(self) -> float:
        """
        Get approximate latency per query-document pair in milliseconds.
        
        Returns:
            Estimated latency in ms
        """
        if not self._is_loaded:
            self.load_model()
        
        # Measure latency with a small batch
        test_pairs = [["test query", "test document"] * 10]
        
        import time
        start = time.time()
        for _ in range(10):
            self.score("test query", ["test doc"] * 10)
        elapsed = time.time() - start
        
        # Average per pair in ms
        avg_latency = (elapsed / 100) * 1000
        return avg_latency
    
    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded from memory")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"
