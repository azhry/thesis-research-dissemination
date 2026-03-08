"""
Cross-Encoder Re-ranker module for the full pipeline.

Uses sentence_transformers.CrossEncoder for robust scoring instead of
the custom AutoModelForSequenceClassification implementation.

The custom implementation had domain mismatch issues — MS MARCO cross-encoders
are trained on natural language passages, not code. This module uses the
sentence-transformers CrossEncoder class for correct inference and supports
multiple model choices.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Model registry with details
RERANKER_MODELS = {
    "mmmini": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "MS MARCO MiniLM (fast, 22M params)",
        "params": "22M",
    },
    "mmmini_12": {
        "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "description": "MS MARCO MiniLM-12 (balanced, 33M params)",
        "params": "33M",
    },
    "mmmini_multilingual": {
        "name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "description": "Multilingual MARCO mMiniLM (100+ langs, 33M params)",
        "params": "33M",
    },
    "multilingual_msmarco": {
        "name": "cross-encoder/msmarco-MiniLM-L6-en-de-v1",
        "description": "Multilingual MS MARCO (EN/DE)",
        "params": "22M",
    },
    "stsb": {
        "name": "cross-encoder/stsb-distilroberta-base",
        "description": "Semantic Textual Similarity (better for semantic matching)",
        "params": "82M",
    },
}


class Reranker:
    """
    Cross-Encoder Re-ranker using sentence_transformers.CrossEncoder.
    
    Uses the well-tested sentence_transformers implementation which
    handles tokenization, batching, and scoring correctly.
    """
    
    def __init__(
        self,
        model_type: str = "mmmini",
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        Initialize the reranker.
        
        Args:
            model_type: Type of reranker model (key in RERANKER_MODELS)
            model_name: Specific model name (overrides model_type)
            device: Device for inference
            max_length: Maximum sequence length
            batch_size: Batch size for scoring
        """
        from sentence_transformers import CrossEncoder
        
        # Resolve model name
        if model_name is None:
            model_info = RERANKER_MODELS.get(model_type, RERANKER_MODELS["mmmini"])
            model_name = model_info["name"]
        
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        logger.info(f"Loading CrossEncoder: {model_name}")
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        
        logger.info(f"CrossEncoder loaded on {device}")
    
    def load_model(self):
        """No-op: model is loaded in __init__ via CrossEncoder."""
        pass
    
    def score(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Score query-document pairs using the cross-encoder.
        
        Args:
            query: Query string
            documents: List of document strings
        
        Returns:
            Array of relevance scores
        """
        # Add prefixes ONLY if it's an E5-based model (like our custom one)
        # MS MARCO models (MiniLM) were NOT trained with these prefixes.
        is_e5 = ("e5" in self.model_name.lower()) or (self.model_type == "custom")
        
        if is_e5:
            q_prefix = "query: "
            d_prefix = "passage: "
            pairs = [[q_prefix + query, d_prefix + doc] for doc in documents]
            # logger.debug(f"Using E5 prefixes for reranking with {self.model_name}")
        else:
            pairs = [[query, doc] for doc in documents]
            # logger.debug(f"No prefixes used for reranking with {self.model_name}")
        
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.array(scores)
    
    def rerank(
        self,
        first_stage_results: Dict[str, Dict[str, Any]],
        top_k: int = 10,
        doc_max_chars: int = 512,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Re-rank first-stage retrieval results.
        
        Args:
            first_stage_results: Dict mapping query_id -> {
                "query": str,
                "retrieved": List[Dict] with 'id', 'score', 'text'
            }
            top_k: Number of top documents to return after reranking
            doc_max_chars: Maximum characters per document for scoring
            show_progress: Show progress bar
        
        Returns:
            Dict mapping query_id -> {
                "query": str,
                "first_stage": List[Dict] (top_k from first stage),
                "reranked": List[Dict] with added 'cross_encoder_score'
            }
        """
        results = {}
        
        for qid, data in tqdm(first_stage_results.items(), desc="Re-ranking", disable=not show_progress):
            query = data["query"]
            first_stage_docs = data["retrieved"]
            
            # Extract document texts (truncated)
            doc_texts = [doc.get('text', '')[:doc_max_chars] for doc in first_stage_docs]
            
            # Skip if no documents
            if not doc_texts:
                results[qid] = {
                    "query": query,
                    "first_stage": [],
                    "reranked": [],
                }
                continue
            
            # Score with cross-encoder
            scores = self.score(query, doc_texts)
            
            # Add scores and rerank
            reranked_docs = []
            for doc, ce_score in zip(first_stage_docs, scores):
                reranked_docs.append({
                    **doc,
                    'cross_encoder_score': float(ce_score),
                })
            
            reranked_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            results[qid] = {
                "query": query,
                "first_stage": first_stage_docs[:top_k],
                "reranked": reranked_docs[:top_k],
            }
            
            # Carry over extra fields
            for key in ("original_query", "expanded_query"):
                if key in data:
                    results[qid][key] = data[key]
        
        return results
    
    def unload(self):
        """Unload model from memory."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("CrossEncoder unloaded from memory")
