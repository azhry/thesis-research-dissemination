"""
Standalone Cross-Encoder Re-ranking Experiment Runner.

This script runs cross-encoder re-ranking experiments using
benchmark datasets from CoIR (Code Information Retrieval) benchmark.
Benchmarks both English and Indonesian queries and saves results to CSV.
"""

import argparse
import logging
import json
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add qe directory to path for coir imports
qe_path = Path(__file__).parent.parent / "qe"
sys.path.insert(0, str(qe_path))

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our modules - handle both package and direct execution
try:
    from reranker.cross_encoder_base import CrossEncoderReranker
except ImportError:
    try:
        from .cross_encoder_base import CrossEncoderReranker
    except ImportError:
        from cross_encoder_base import CrossEncoderReranker

try:
    from reranker.mmmini_reranker import MMMiniReranker, MultilingualMMMiniReranker
except ImportError:
    try:
        from .mmmini_reranker import MMMiniReranker, MultilingualMMMiniReranker
    except ImportError:
        from mmmini_reranker import MMMiniReranker, MultilingualMMMiniReranker

try:
    from reranker.xlm_reranker import XLMReranker, MBERTReranker
except ImportError:
    try:
        from .xlm_reranker import XLMReranker, MBERTReranker
    except ImportError:
        from xlm_reranker import XLMReranker, MBERTReranker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cosqa_with_translations():
    """Load COSQA with Indonesian translations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "qe"))
    
    # Load Indonesian translations
    translations_file = Path(__file__).parent.parent / "qe" / "cosqa_queries_indonesian.csv"
    if translations_file.exists():
        trans_df = pd.read_csv(translations_file, sep="|")
        translations = dict(zip(trans_df['qid'], trans_df['query_id']))
    else:
        translations = {}
    
    # Load COSQA from HuggingFace
    from coir.data_loader import load_data_from_hf
    corpus, queries, qrels = load_data_from_hf("cosqa")
    
    # Convert to list format
    corpus_list = [
        {"id": doc_id, "text": doc_data.get("text", "")}
        for doc_id, doc_data in corpus.items()
    ]
    
    # Add Indonesian queries
    queries_english = {}
    queries_indonesian = {}
    for qid, qtext in queries.items():
        queries_english[qid] = qtext
        if qid in translations:
            queries_indonesian[qid] = translations[qid]
        else:
            queries_indonesian[qid] = qtext
    
    return corpus_list, queries_english, queries_indonesian, qrels


def run_first_stage_retrieval(queries: Dict[str, str], corpus: List[Dict], top_k: int = 100):
    """Run first-stage retrieval using mE5 embeddings."""
    from coir.dense_retriever import DenseRetriever
    
    logger.info("Running first-stage retrieval with mE5...")
    
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-base",
        device="cpu"
    )
    
    # Build corpus lookup by id
    corpus_lookup = {doc['id']: doc for doc in corpus}
    
    results = {}
    for qid, query in tqdm(queries.items(), desc="First-stage retrieval"):
        retrieved = retriever.retrieve(
            queries=[query],
            corpus=corpus,
            top_k=top_k,
        )
        
        # Add text to retrieved docs
        results_with_text = []
        for doc in retrieved[0]:
            doc_id = doc['id']
            if doc_id in corpus_lookup:
                results_with_text.append({
                    **doc,
                    'text': corpus_lookup[doc_id].get('text', '')
                })
            else:
                results_with_text.append(doc)
        
        results[qid] = {
            "query": query,
            "retrieved": results_with_text
        }
    
    return results


def run_reranking(
    first_stage_results: Dict,
    reranker,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Run cross-encoder reranking on first-stage results."""
    results = []
    
    for qid, result in tqdm(first_stage_results.items(), desc="Reranking"):
        query = result["query"]
        first_stage_docs = result["retrieved"]
        
        # Get document texts
        doc_texts = [doc['text'][:512] for doc in first_stage_docs]
        
        # Score with cross-encoder
        scores = reranker.score(query, doc_texts)
        
        # Add scores and rerank
        reranked_docs = []
        for doc, score in zip(first_stage_docs, scores):
            reranked_docs.append({
                **doc,
                'cross_encoder_score': float(score)
            })
        
        reranked_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        
        results.append({
            "qid": qid,
            "query": query,
            "first_stage": first_stage_docs[:top_k],
            "reranked": reranked_docs[:top_k]
        })
    
    return results


def evaluate_results(
    results: List[Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
    top_k: int = 10
) -> Dict[str, float]:
    """Evaluate reranking results."""
    ndcg_scores_before = []
    ndcg_scores_after = []
    map_scores_before = []
    map_scores_after = []
    
    for result in results:
        qid = result["qid"]
        relevant_docs = qrels.get(qid, {})
        
        if not relevant_docs:
            continue
        
        relevant_ids = set(relevant_docs.keys())
        
        # Before reranking
        before_docs = result.get("first_stage", [])[:top_k]
        before_ids = [doc['id'] for doc in before_docs]
        
        # NDCG before
        dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(before_ids) if doc_id in relevant_ids)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), top_k)))
        ndcg_before = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores_before.append(ndcg_before)
        
        # MAP before
        prec_sum = sum(
            sum(1 for doc_id in before_ids[:i+1] if doc_id in relevant_ids) / (i+1)
            for i in range(min(10, len(before_ids)))
            if any(doc_id in relevant_ids for doc_id in before_ids[:i+1])
        )
        map_before = prec_sum / len(relevant_ids) if relevant_ids else 0
        map_scores_before.append(map_before)
        
        # After reranking
        after_docs = result.get("reranked", [])[:top_k]
        after_ids = [doc['id'] for doc in after_docs]
        
        # NDCG after
        dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(after_ids) if doc_id in relevant_ids)
        ndcg_after = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores_after.append(ndcg_after)
        
        # MAP after
        prec_sum = sum(
            sum(1 for doc_id in after_ids[:i+1] if doc_id in relevant_ids) / (i+1)
            for i in range(min(10, len(after_ids)))
            if any(doc_id in relevant_ids for doc_id in after_ids[:i+1])
        )
        map_after = prec_sum / len(relevant_ids) if relevant_ids else 0
        map_scores_after.append(map_after)
    
    return {
        "ndcg_before": float(np.mean(ndcg_scores_before)) if ndcg_scores_before else 0.0,
        "ndcg_after": float(np.mean(ndcg_scores_after)) if ndcg_scores_after else 0.0,
        "map_before": float(np.mean(map_scores_before)) if map_scores_before else 0.0,
        "map_after": float(np.mean(map_scores_after)) if map_scores_after else 0.0,
        "num_queries": len(results)
    }


def save_detailed_results(results: List[Dict], output_path: Path, queries_english: Dict[str, str]):
    """Save detailed before-after results to CSV."""
    rows = []
    
    for result in results:
        qid = result["qid"]
        query = result["query"]
        query_en = queries_english.get(qid, "")
        
        # First stage (before reranking)
        for rank, doc in enumerate(result.get("first_stage", [])[:10], 1):
            rows.append({
                "qid": qid,
                "query_en": query_en,
                "query_id": query,
                "stage": "before_rerank",
                "rank": rank,
                "doc_id": doc.get("id", ""),
                "score_before": doc.get("score", 0),
                "score_after": doc.get("cross_encoder_score", 0) if "cross_encoder_score" in doc else 0
            })
        
        # After reranking
        for rank, doc in enumerate(result.get("reranked", [])[:10], 1):
            rows.append({
                "qid": qid,
                "query_en": query_en,
                "query_id": query,
                "stage": "after_rerank",
                "rank": rank,
                "doc_id": doc.get("id", ""),
                "score_before": doc.get("score", 0),
                "score_after": doc.get("cross_encoder_score", 0)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CE reranking experiments")
    parser.add_argument("--method", type=str, default="mmmini")
    parser.add_argument("--model-type", type=str, default="mmmini")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--first-stage-k", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="./reranker/reranker_benchmark_results.json")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit number of queries (for testing)")
    
    args = parser.parse_args()
    
    logger.info("Loading COSQA data with Indonesian translations...")
    corpus, queries_en, queries_id, qrels = load_cosqa_with_translations()
    
    # Optionally limit queries
    if args.sample_size:
        queries_en = dict(list(queries_en.items())[:args.sample_size])
        queries_id = dict(list(queries_id.items())[:args.sample_size])
        qrels = {k: v for k, v in qrels.items() if k in queries_en}
    
    logger.info(f"Loaded {len(queries_en)} queries, {len(corpus)} docs")
    
    # Initialize reranker
    if args.method == "mmmini":
        reranker = MMMiniReranker(device=args.device)
    elif args.method == "mmmini_multilingual":
        reranker = MultilingualMMMiniReranker(device=args.device)
    elif args.method == "xlm":
        reranker = XLMReranker(device=args.device)
    else:
        reranker = MMMiniReranker(device=args.device)
    
    reranker.load_model()
    
    all_results = {}
    all_metrics = {}
    
    # Run for English queries
    logger.info("=" * 60)
    logger.info("Running English queries benchmark...")
    logger.info("=" * 60)
    
    first_stage_en = run_first_stage_retrieval(queries_en, corpus, args.first_stage_k)
    results_en = run_reranking(first_stage_en, reranker, args.top_k)
    metrics_en = evaluate_results(results_en, qrels, args.top_k)
    
    logger.info(f"English - NDCG@10 Before: {metrics_en['ndcg_before']:.4f}, After: {metrics_en['ndcg_after']:.4f}")
    
    all_results["english"] = results_en
    all_metrics["english"] = metrics_en
    
    # Run for Indonesian queries
    logger.info("=" * 60)
    logger.info("Running Indonesian queries benchmark...")
    logger.info("=" * 60)
    
    first_stage_id = run_first_stage_retrieval(queries_id, corpus, args.first_stage_k)
    results_id = run_reranking(first_stage_id, reranker, args.top_k)
    metrics_id = evaluate_results(results_id, qrels, args.top_k)
    
    logger.info(f"Indonesian - NDCG@10 Before: {metrics_id['ndcg_before']:.4f}, After: {metrics_id['ndcg_after']:.4f}")
    
    all_results["indonesian"] = results_id
    all_metrics["indonesian"] = metrics_id
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "method": args.method,
        "top_k": args.top_k,
        "first_stage_k": args.first_stage_k,
        "metrics": {
            "english": all_metrics["english"],
            "indonesian": all_metrics["indonesian"]
        },
        "results": all_results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Save CSV detailed results
    csv_path = output_path.with_name(f"reranker_{args.method}_benchmark.csv")
    save_detailed_results(results_en + results_id, csv_path, queries_en)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RERANKER BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nMethod: {args.method}")
    print(f"First-stage: mE5 (top-{args.first_stage_k})")
    print(f"Reranker: {args.method}")
    print(f"Top-K: {args.top_k}")
    
    print("\n--- English Queries ---")
    print(f"NDCG@{args.top_k} Before: {all_metrics['english']['ndcg_before']:.4f}")
    print(f"NDCG@{args.top_k} After:  {all_metrics['english']['ndcg_after']:.4f}")
    print(f"MAP@{args.top_k} Before: {all_metrics['english']['map_before']:.4f}")
    print(f"MAP@{args.top_k} After:  {all_metrics['english']['map_after']:.4f}")
    
    print("\n--- Indonesian Queries ---")
    print(f"NDCG@{args.top_k} Before: {all_metrics['indonesian']['ndcg_before']:.4f}")
    print(f"NDCG@{args.top_k} After:  {all_metrics['indonesian']['ndcg_after']:.4f}")
    print(f"MAP@{args.top_k} Before: {all_metrics['indonesian']['map_before']:.4f}")
    print(f"MAP@{args.top_k} After:  {all_metrics['indonesian']['map_after']:.4f}")
    
    # Calculate improvement
    ndcg_improvement_en = all_metrics['english']['ndcg_after'] - all_metrics['english']['ndcg_before']
    ndcg_improvement_id = all_metrics['indonesian']['ndcg_after'] - all_metrics['indonesian']['ndcg_before']
    
    print("\n--- Improvement (After - Before) ---")
    print(f"English NDCG: {ndcg_improvement_en:+.4f}")
    print(f"Indonesian NDCG: {ndcg_improvement_id:+.4f}")
    print("=" * 60)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
