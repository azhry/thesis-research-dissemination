"""
Retriever module for the full pipeline.

Wraps the DenseRetriever from the QE experiment's coir module
into a clean interface for first-stage retrieval.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _setup_coir_path():
    """Add the coir module path to sys.path."""
    qe_path = Path(__file__).parent.parent.parent / "qe"
    if str(qe_path) not in sys.path:
        sys.path.insert(0, str(qe_path))


class Retriever:
    """
    First-stage dense retriever using Multilingual E5.
    
    Handles:
    - Encoding queries with 'query: ' prefix
    - Encoding corpus with 'passage: ' prefix
    - Cosine similarity search
    - Corpus embedding caching
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 512,
    ):
        """
        Initialize the retriever.
        
        Args:
            model_name: mE5 model to use
            device: Device for inference
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length
        """
        _setup_coir_path()
        from coir.dense_retriever import DenseRetriever
        
        logger.info(f"Initializing retriever with model: {model_name}")
        self._retriever = DenseRetriever(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )
        
        self.model_name = model_name
        self._corpus_encoded = False
    
    def retrieve(
        self,
        queries: Dict[str, str],
        corpus: Any,
        top_k: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve top-k documents for each query.
        
        Args:
            queries: Dict mapping query_id -> query_text
            corpus: Corpus in dict or list format
            top_k: Number of top documents to retrieve
            show_progress: Show progress bar
        
        Returns:
            Dict mapping query_id -> {
                "query": str,
                "retrieved": List[Dict] with 'id', 'score', 'rank', 'text'
            }
        """
        # Build corpus lookup for adding text to results
        if isinstance(corpus, dict):
            corpus_lookup = {doc_id: doc_data.get("text", "") for doc_id, doc_data in corpus.items()}
        elif isinstance(corpus, list):
            corpus_lookup = {doc.get("id", str(i)): doc.get("text", "") for i, doc in enumerate(corpus)}
        else:
            corpus_lookup = {}
        
        results = {}
        
        for qid, query in tqdm(queries.items(), desc="First-stage retrieval", disable=not show_progress):
            retrieved = self._retriever.retrieve(
                queries=[query],
                corpus=corpus,
                top_k=top_k,
            )
            
            # Add text to retrieved docs
            results_with_text = []
            for doc in retrieved[0]:
                doc_id = doc['id']
                results_with_text.append({
                    **doc,
                    'text': corpus_lookup.get(doc_id, ''),
                })
            
            results[qid] = {
                "query": query,
                "retrieved": results_with_text,
            }
        
        return results
    
    def fuse_results(
        self,
        results_original: Dict[str, Dict[str, Any]],
        results_expanded: Dict[str, Dict[str, Any]],
        top_k: int = 100,
        expansion_weight: float = 0.3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge scores from original and expanded retrieval results.
        """
        final_results = {}
        for qid in results_original:
            orig_data = results_original.get(qid, {"retrieved": []})
            exp_data = results_expanded.get(qid, {"retrieved": []})
            
            # Map doc_id -> score
            score_map = {}
            doc_info = {}
            
            # Add original scores
            for doc in orig_data["retrieved"]:
                doc_id = doc["id"]
                score_map[doc_id] = doc["score"]
                doc_info[doc_id] = doc
                
            # Add expanded scores (weighted)
            for doc in exp_data["retrieved"]:
                doc_id = doc["id"]
                score_map[doc_id] = score_map.get(doc_id, 0.0) + (expansion_weight * doc["score"])
                if doc_id not in doc_info:
                    doc_info[doc_id] = doc
            
            # Sort by fused score
            fused_retrieved = []
            orig_doc_ids = {doc["id"]: doc["score"] for doc in orig_data["retrieved"]}
            for doc_id, score in score_map.items():
                orig_score = orig_doc_ids.get(doc_id, 0.0)
                fused_retrieved.append({
                    **doc_info[doc_id],
                    "score": score,
                    "retrieval_score_orig": orig_score,
                    "retrieval_score_exp": score - orig_score
                })
            
            fused_retrieved.sort(key=lambda x: x["score"], reverse=True)
            
            final_results[qid] = {
                "query": results_original[qid]["query"],
                "original_query": results_original[qid].get("original_query", results_original[qid]["query"]),
                "expanded_query": results_expanded[qid].get("expanded_query", results_expanded[qid]["query"]),
                "retrieved": fused_retrieved[:top_k]
            }
        return final_results
