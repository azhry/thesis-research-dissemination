"""
MMMiniLMv2 Cross-Encoder Re-ranker Implementation.

This module implements cross-encoder re-ranking using the 
mmarco-mMiniLMv2-L12 model, which is recommended in the research plan
for its balance of performance and latency.
"""

import logging
from typing import List, Dict, Any, Optional
import torch

try:
    from .cross_encoder_base import CrossEncoderReranker
except ImportError:
    from cross_encoder_base import CrossEncoderReranker

logger = logging.getLogger(__name__)


class MMMiniReranker(CrossEncoderReranker):
    """
    Cross-Encoder Re-ranker using mmarco-mMiniLMv2-L12.
    
    This model is recommended in the research plan as it provides:
    - 33M parameters (efficient)
    - 100+ language support
    - ~65ms latency
    - Good relevance scoring
    """
    
    # Default model names to try
    DEFAULT_MODELS = [
        "sentence-transformers/ms-marco-MiniLM-L-12-v2-cross-encoder",
        "castorini/mmarco-mMiniLMv2-L12-H384-uncased",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize the MMMini cross-encoder reranker.
        
        Args:
            model_name: Specific model name or None to use default
            device: Device to use ('cpu', 'cuda', etc.)
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            **kwargs: Additional configuration
        """
        # Use default model if not specified
        if model_name is None:
            model_name = self.DEFAULT_MODELS[0]
        
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            **kwargs
        )
        
        logger.info(f"MMMiniReranker configured with model: {self.model_name}")
    
    def load_model(self):
        """Load the mmarco-mMiniLMv2 cross-encoder model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Try fallback models
            for fallback_model in self.DEFAULT_MODELS[1:]:
                if fallback_model == self.model_name:
                    continue
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        fallback_model
                    ).to(self.device)
                    self.model_name = fallback_model
                    self._is_loaded = True
                    logger.info(f"Fallback model loaded successfully")
                    break
                except Exception as e2:
                    logger.warning(f"Fallback model {fallback_model} also failed: {e2}")
                    continue
            else:
                raise RuntimeError(f"Could not load any model: {e}")
    
    def score_with_transform(
        self,
        query: str,
        documents: List[str],
        query_transform: str = "query: ",
        doc_transform: str = "passage: "
    ) -> List[float]:
        """
        Score documents with optional query/document transformations.
        
        Args:
            query: The query string
            documents: List of document strings
            query_transform: Prefix for query
            doc_transform: Prefix for document
            
        Returns:
            List of relevance scores
        """
        if not self._is_loaded:
            self.load_model()
        
        # Apply transformations
        pairs = [
            [query_transform + query, doc_transform + doc]
            for doc in documents
        ]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
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
        
        return scores
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Re-rank with a minimum score threshold.
        
        Args:
            query: The query string
            documents: List of document dicts
            top_k: Number of top documents to return
            score_threshold: Minimum score threshold
            
        Returns:
            Filtered and re-ranked documents
        """
        reranked = self.rerank(query, documents, top_k=top_k)
        
        # Filter by threshold
        filtered = [
            doc for doc in reranked 
            if doc.get('cross_encoder_score', 0) >= score_threshold
        ]
        
        return filtered


class MultilingualMMMiniReranker(MMMiniReranker):
    """
    Multilingual variant of MMMini reranker using models 
    trained on multilingual MSMARCO data.
    """
    
    DEFAULT_MODELS = [
        "cross-encoder/ms-marco-MultiBERT-L-12",
        "cross-encoder/mmarco-bert-base-multilingual-cased",
        "bert-base-multilingual-cased",  # Fallback to mBERT
    ]
    
    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-bert-base-multilingual-cased",
        device: Optional[str] = None,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            **kwargs
        )
