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
import torch
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
        
        # Patch for Jina Reranker v2: it uses a function removed in transformers >= 5.0
        if "jina" in model_name.lower():
            try:
                from transformers.models.xlm_roberta import modeling_xlm_roberta
                if not hasattr(modeling_xlm_roberta, 'create_position_ids_from_input_ids'):
                    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
                        mask = input_ids.ne(padding_idx).int()
                        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
                        return incremental_indices.long() + padding_idx
                    modeling_xlm_roberta.create_position_ids_from_input_ids = create_position_ids_from_input_ids
                    logger.info("Patched create_position_ids_from_input_ids for Jina compatibility")
            except Exception as e:
                logger.warning(f"Could not apply Jina compatibility patch: {e}")
        
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
            trust_remote_code=True,  # Required for Jina Reranker v2
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
        # Only add E5 prefixes for actual E5-based models
        # Custom models (CoSQA-trained) use raw text — NO prefixes
        is_e5 = ("e5" in self.model_name.lower()) and (self.model_type != "custom")
        
        if is_e5:
            clean_query = query.replace("query: ", "").strip()
            pairs = [["query: " + clean_query, "passage: " + doc] for doc in documents]
        else:
            # Raw text pairs for MS MARCO and CoSQA-trained models
            pairs = [[query, doc] for doc in documents]
        
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        # Convert raw logits to probabilities (0.0 to 1.0) using sigmoid.
        # This is CRITICAL because the pipeline expects probabilities for RRF fusion,
        # but newer sentence-transformers versions return raw logits by default.
        scores_arr = np.array(scores)
        probabilities = 1 / (1 + np.exp(-scores_arr))
        
        return probabilities
    
    def rerank(
        self,
        first_stage_results: Dict[str, Dict[str, Any]],
        top_k: int = 10,
        doc_max_chars: int = 1024,
        show_progress: bool = True,
        use_rrf: bool = False,
        rrf_k: int = 60,
        queries: Optional[Dict[str, str]] = None,
        confidence_threshold: float = 0.0  # Set to 0.0 to always rerank, >0 for gating
    ) -> Dict[str, Dict[str, Any]]:
        """
        Re-rank first-stage results using cross-encoder scores.
        
        Args:
            first_stage_results: Dict from retriever.retrieve or fuse_results
            top_k: Number of docs to keep after reranking
            doc_max_chars: Max characters to send to cross-encoder
            show_progress: Show progress bar
            use_rrf: Whether to use Reciprocal Rank Fusion
            rrf_k: Smoothing constant for RRF
            queries: Optional mapping of qid -> rerank_query text (e.g. English version)
        
        Standard IR reranking: the cross-encoder scores replace the initial
        ranking. Documents are sorted purely by cross-encoder relevance score.
        
        Optional: use_rrf=True will use Reciprocal Rank Fusion to combine signals.
        """
        results = {}
        
        for qid, data in tqdm(first_stage_results.items(), desc="Re-ranking", disable=not show_progress):
            # Resolve query:
            # We prioritize the original/manual query for reranking. 
            # HyDE is great for retrieval, but its 'hallucinations' can distract a Cross-Encoder.
            if queries and qid in queries:
                query = queries[qid]
            else:
                # Use original_query if available, fallback to the (possibly expanded) data["query"]
                query = data.get("original_query", data["query"])
            
            first_stage_docs = data["retrieved"]
            
            if not first_stage_docs:
                results[qid] = {"query": data["query"], "first_stage": [], "reranked": []}
                continue
            
            # 1. Get Cross-Encoder Scores
            doc_texts = [doc.get('text', '')[:doc_max_chars] for doc in first_stage_docs]
            ce_scores = self.score(query, doc_texts)
            
            # 2. Attach scores to documents
            docs_with_ce = []
            for i, doc in enumerate(first_stage_docs):
                docs_with_ce.append({
                    **doc,
                    'cross_encoder_score': float(ce_scores[i]),
                    'initial_rank': i + 1
                })
            
            # 3. Determine Final Ranking
            if use_rrf:
                # Standard RRF: equal-weight fusion of Bi-Encoder and Cross-Encoder
                # Rank by CE score first to get CE rank
                docs_with_ce.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
                for i, doc in enumerate(docs_with_ce):
                    doc['ce_rank'] = i + 1
                    # Formula: 1/(r1 + k) + 1/(r2 + k)
                    doc['rrf_score'] = (1.0 / (doc['initial_rank'] + rrf_k)) + \
                                      (1.0 / (doc['ce_rank'] + rrf_k))
                
                # Confidence Gating: Compare CE certainty vs Bi-Encoder
                # If CE scores have very low variance, it's just guessing. 
                # In that case, we revert to the Bi-Encoder's original ranking.
                if confidence_threshold > 0:
                    score_range = np.max(ce_scores) - np.min(ce_scores)
                    if score_range < confidence_threshold:
                        # Reranker is unsure → Reset to first-stage order
                        docs_with_ce.sort(key=lambda x: x['initial_rank'])
                
                # Final sort by RRF score
                docs_with_ce.sort(key=lambda x: x['rrf_score'], reverse=True)
            else:
                # Standard reranking: sort purely by CE score
                docs_with_ce.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            results[qid] = {
                "query": data["query"],
                "first_stage": first_stage_docs,
                "reranked": docs_with_ce,
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
