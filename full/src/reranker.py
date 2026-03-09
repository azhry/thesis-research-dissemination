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
        doc_max_chars: int = 1024,
        show_progress: bool = True,
        rrf_k: int = 60,  # Hyperparameter for RRF
    ) -> Dict[str, Dict[str, Any]]:
        """
        Re-rank results using Reciprocal Rank Fusion (RRF).
        
        RRF is much more robust than score normalization because it only cares 
        about the order. It prevents a 'bad' reranker from destroying performance.
        """
        results = {}
        
        for qid, data in tqdm(first_stage_results.items(), desc="Re-ranking", disable=not show_progress):
            query = data.get("original_query", data["query"])
            first_stage_docs = data["retrieved"]
            
            if not first_stage_docs:
                results[qid] = {"query": data["query"], "first_stage": [], "reranked": []}
                continue
            
            # 1. Get Reranker Scores
            doc_texts = [doc.get('text', '')[:doc_max_chars] for doc in first_stage_docs]
            ce_scores = self.score(query, doc_texts)
            
            # 2. Get Ranks for RRF
            # First-stage rank (already 1, 2, 3...)
            # Reranker rank (sort ce_scores)
            ce_order = np.argsort(ce_scores)[::-1]
            ce_ranks = {first_stage_docs[idx]['id']: rank + 1 for rank, idx in enumerate(ce_order)}
            
            # 3. Apply RRF Formula: Score = 1/(k + rank_retrieval) + 1/(k + rank_reranker)
            reranked_docs = []
            for i, doc in enumerate(first_stage_docs):
                doc_id = doc['id']
                rank_ret = i + 1
                rank_ce = ce_ranks[doc_id]
                
                rrf_score = (1.0 / (rrf_k + rank_ret)) + (1.0 / (rrf_k + rank_ce))
                
                reranked_docs.append({
                    **doc,
                    'cross_encoder_score': float(ce_scores[i]),
                    'rrf_score': float(rrf_score),
                    'ce_rank': rank_ce
                })
            
            # Sort by RRF score
            reranked_docs.sort(key=lambda x: x['rrf_score'], reverse=True)
            
            results[qid] = {
                "query": data["query"],
                "first_stage": first_stage_docs[:top_k],
                "reranked": reranked_docs[:top_k],
            }
            
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
