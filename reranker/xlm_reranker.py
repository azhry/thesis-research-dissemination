"""
XLM-RoBERTa Cross-Encoder Re-ranker Implementation.

This module implements cross-encoder re-ranking using XLM-RoBERTa,
which provides higher quality but slower inference than mmarco-mMiniLMv2.
"""

import logging
from typing import List, Dict, Any, Optional
import torch

from .cross_encoder_base import CrossEncoderReranker

logger = logging.getLogger(__name__)


class XLMReranker(CrossEncoderReranker):
    """
    Cross-Encoder Re-ranker using XLM-RoBERTa.
    
    This model provides:
    - 278M parameters
    - 100+ language support
    - ~250ms latency
    - Higher quality than smaller models
    
    The trade-off is higher latency, so this is better suited
    when accuracy is prioritized over speed.
    """
    
    DEFAULT_MODELS = [
        "xlm-roberta-base",
        "xlm-roberta-large",
        "xlm-roberta-base-finetuned-mmarco",
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 4,
        **kwargs
    ):
        """
        Initialize the XLM-RoBERTa cross-encoder reranker.
        
        Args:
            model_name: Specific model name or None to use default
            device: Device to use ('cpu', 'cuda', etc.)
            max_length: Maximum sequence length
            batch_size: Batch size for inference (smaller due to model size)
            **kwargs: Additional configuration
        """
        if model_name is None:
            model_name = self.DEFAULT_MODELS[0]
        
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            **kwargs
        )
        
        logger.info(f"XLMReranker configured with model: {self.model_name}")
    
    def load_model(self):
        """Load the XLM-RoBERTa cross-encoder model."""
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
    
    def rerank_with_cascade(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        first_stage_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Cascade re-ranking: score top candidates from first stage.
        
        This method first takes top_k candidates from documents
        then applies cross-encoder scoring.
        
        Args:
            query: The query string
            documents: List of document dicts (sorted by first-stage score)
            top_k: Number of top documents to return
            first_stage_k: Number of candidates to score with CE
            
        Returns:
            Re-ranked documents
        """
        # Take top candidates from first stage
        candidates = documents[:first_stage_k]
        
        # Re-rank with cross-encoder
        reranked = self.rerank(query, candidates, top_k=top_k)
        
        return reranked


class MBERTReranker(CrossEncoderReranker):
    """
    Cross-Encoder Re-ranker using mBERT (multilingual BERT).
    
    This provides a baseline comparison with XLM-RoBERTa.
    """
    
    DEFAULT_MODELS = [
        "bert-base-multilingual-cased",
        "bert-base-multilingual-uncased",
    ]
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 4,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            **kwargs
        )
    
    def load_model(self):
        """Load the mBERT cross-encoder model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # For base BERT, we need to add a classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1  # Regression task for scoring
            ).to(self.device)
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
