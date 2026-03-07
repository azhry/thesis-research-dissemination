"""
Evaluation module for the full pipeline.

Implements nDCG@K, MAP@K, Recall@K, and MRR metrics
as specified in the implementation plan.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_ndcg(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int,
) -> float:
    """
    Compute nDCG@K.
    
    Args:
        retrieved_ids: List of doc IDs in retrieval order
        relevant_ids: Set of relevant doc IDs
        k: Cutoff
    
    Returns:
        nDCG@K score
    """
    retrieved_at_k = retrieved_ids[:k]
    
    # DCG
    dcg = sum(
        1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
        for i, doc_id in enumerate(retrieved_at_k)
        if doc_id in relevant_ids
    )
    
    # IDCG
    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_map(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int,
) -> float:
    """
    Compute MAP@K (Mean Average Precision).
    
    Args:
        retrieved_ids: List of doc IDs in retrieval order
        relevant_ids: Set of relevant doc IDs
        k: Cutoff
    
    Returns:
        AP@K score
    """
    retrieved_at_k = retrieved_ids[:k]
    
    num_relevant_at_i = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved_at_k):
        if doc_id in relevant_ids:
            num_relevant_at_i += 1
            precision_at_i = num_relevant_at_i / (i + 1)
            precision_sum += precision_at_i
    
    total_relevant = min(len(relevant_ids), k)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def compute_recall(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int,
) -> float:
    """
    Compute Recall@K.
    
    Args:
        retrieved_ids: List of doc IDs in retrieval order
        relevant_ids: Set of relevant doc IDs
        k: Cutoff
    
    Returns:
        Recall@K score
    """
    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int,
) -> float:
    """
    Compute MRR@K (Mean Reciprocal Rank).
    
    Args:
        retrieved_ids: List of doc IDs in retrieval order
        relevant_ids: Set of relevant doc IDs
        k: Cutoff
    
    Returns:
        RR@K score
    """
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    results: Dict[str, Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = None,
    result_key: str = "retrieved",
) -> Dict[str, float]:
    """
    Evaluate retrieval results across multiple K values.
    
    Args:
        results: Dict mapping query_id -> {"retrieved": List[Dict]} or
                 query_id -> {"reranked": List[Dict]}
        qrels: Relevance judgments
        k_values: List of K values to evaluate at
        result_key: Key in results dict to use ('retrieved' or 'reranked')
    
    Returns:
        Dict of metric_name -> score, e.g. {"nDCG@10": 0.55, "MAP@10": 0.45, ...}
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]
    
    # Initialize accumulators
    ndcg_scores = {k: [] for k in k_values}
    map_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    mrr_scores = {k: [] for k in k_values}
    
    num_evaluated = 0
    
    for qid, data in results.items():
        if qid not in qrels:
            continue
        
        relevant_ids = set(qrels[qid].keys())
        if not relevant_ids:
            continue
        
        # Get retrieved document IDs
        docs = data.get(result_key, [])
        retrieved_ids = [doc['id'] for doc in docs]
        
        for k in k_values:
            ndcg_scores[k].append(compute_ndcg(retrieved_ids, relevant_ids, k))
            map_scores[k].append(compute_map(retrieved_ids, relevant_ids, k))
            recall_scores[k].append(compute_recall(retrieved_ids, relevant_ids, k))
            mrr_scores[k].append(compute_mrr(retrieved_ids, relevant_ids, k))
        
        num_evaluated += 1
    
    # Compute means
    metrics = {"num_queries_evaluated": num_evaluated}
    
    for k in k_values:
        metrics[f"nDCG@{k}"] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
        metrics[f"MAP@{k}"] = float(np.mean(map_scores[k])) if map_scores[k] else 0.0
        metrics[f"Recall@{k}"] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
        metrics[f"MRR@{k}"] = float(np.mean(mrr_scores[k])) if mrr_scores[k] else 0.0
    
    return metrics


def evaluate_reranking(
    results: Dict[str, Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate re-ranking results, computing metrics for both
    first-stage and re-ranked results.
    
    Args:
        results: Dict mapping query_id -> {
            "first_stage": List[Dict],
            "reranked": List[Dict]
        }
        qrels: Relevance judgments
        k_values: K values to evaluate at
    
    Returns:
        Dict with "before_rerank" and "after_rerank" metrics
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    metrics_before = evaluate_retrieval(results, qrels, k_values, result_key="first_stage")
    metrics_after = evaluate_retrieval(results, qrels, k_values, result_key="reranked")
    
    return {
        "before_rerank": metrics_before,
        "after_rerank": metrics_after,
    }
