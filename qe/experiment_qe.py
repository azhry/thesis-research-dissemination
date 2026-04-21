"""
Standalone Query Expansion Experiment Runner using COSQA with Indonesian translations.
This script runs without depending on the original coir evaluation module.
"""

import argparse
import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import sys
import pandas as pd

# Add coir to path
coir_path = Path(__file__).parent / "coir"
sys.path.insert(0, str(coir_path))

import numpy as np
from tqdm import tqdm

# Import our modules directly (avoid __init__.py which has faiss dependency)
from coir.embedding_expander import CrossLingualEmbeddingExpander
from coir.dense_retriever import DenseRetriever, BM25Retriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cosqa_data():
    """Load COSQA dataset with Indonesian translations."""
    # Load Indonesian translations
    translations_file = Path(__file__).parent / "cosqa_queries_indonesian.csv"
    if translations_file.exists():
        trans_df = pd.read_csv(translations_file, sep="|")
        translations = dict(zip(trans_df['qid'], trans_df['query_id']))
    else:
        translations = {}
    
    # Load COSQA from HuggingFace
    from coir.data_loader import load_data_from_hf
    corpus, queries, qrels = load_data_from_hf("cosqa")
    
    # Add Indonesian queries
    queries_indonesian = {}
    for qid, qtext in queries.items():
        if qid in translations:
            queries_indonesian[qid] = translations[qid]
        else:
            # Fallback: use English if no translation
            queries_indonesian[qid] = qtext
    
    return corpus, queries, queries_indonesian, qrels


def run_embedding_qe(queries: Dict[str, str], corpus: List[Dict[str, str]], top_k: int = 10):
    """Run embedding-based query expansion."""
    logger.info("Running embedding-based QE...")
    
    # Initialize expander
    expander = CrossLingualEmbeddingExpander(
        model_name="intfloat/multilingual-e5-base"
    )
    
    # Initialize retriever
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-base",
        device="cpu"
    )
    
    results = []
    for qid, query in tqdm(queries.items(), desc="Processing queries"):
        # Expand query
        expansion = expander.expand(query, num_terms=5)
        logger.info(f"Original: {query}")
        logger.info(f"Expanded: {expansion.expanded_query}")
        
        # Retrieve
        retrieved = retriever.retrieve(
            queries=[expansion.expanded_query],
            corpus=corpus,
            top_k=top_k,
        )
        
        results.append({
            "qid": qid,
            "query": query,
            "expanded_query": expansion.expanded_query,
            "expansion_terms": expansion.expansion_terms,
            "retrieved": retrieved[0]
        })
    
    return results


def run_baseline(queries: Dict[str, str], corpus: List[Dict[str, str]], top_k: int = 10):
    """Run baseline (no QE)."""
    logger.info("Running baseline retrieval...")
    
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-base",
        device="cpu"
    )
    
    results = []
    for qid, query in tqdm(queries.items(), desc="Processing queries"):
        retrieved = retriever.retrieve(
            queries=[query],
            corpus=corpus,
            top_k=top_k,
        )
        
        results.append({
            "qid": qid,
            "query": query,
            "expanded_query": query,
            "retrieved": retrieved[0]
        })
    
    return results


def run_bm25(queries: Dict[str, str], corpus: List[Dict[str, str]], top_k: int = 10):
    """Run BM25 baseline."""
    logger.info("Running BM25...")
    
    retriever = BM25Retriever()
    retriever.fit(corpus)
    
    results = []
    for qid, query in tqdm(queries.items(), desc="Processing queries"):
        retrieved = retriever.retrieve(
            queries=[query],
            top_k=top_k,
        )
        
        results.append({
            "qid": qid,
            "query": query,
            "retrieved": retrieved[0]
        })
    
    return results


def evaluate_results(results: List[Dict], qrels: Dict[str, Dict[str, int]], top_k: int = 10):
    """Evaluate results using NDCG and Recall."""
    from collections import defaultdict
    
    # Build relevance judgments for top_k
    hits = 0
    total_relevant = 0
    ndcg_sum = 0
    evaluated = 0
    
    for result in results:
        qid = result["qid"]
        if qid not in qrels:
            continue
            
        relevant_docs = set(qrels[qid].keys())
        total_relevant += len(relevant_docs)
        
        # Get retrieved docs
        retrieved = result["retrieved"][:top_k]
        retrieved_ids = [doc["id"] for doc in retrieved]
        
        # Calculate hits
        hits += len(set(retrieved_ids) & relevant_docs)
        
        # Calculate NDCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                dcg += 1 / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate IDCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_docs), top_k)))
        
        if idcg > 0:
            ndcg_sum += dcg / idcg
            evaluated += 1
    
    recall = hits / total_relevant if total_relevant > 0 else 0
    ndcg = ndcg_sum / evaluated if evaluated > 0 else 0
    
    return {
        "NDCG@10": ndcg,
        "Recall@10": recall,
        "Hits": hits,
        "Total Relevant": total_relevant,
        "Evaluated": evaluated
    }


def save_detailed_results(results: List[Dict], output_path: Path, queries_english: Dict[str, str] = None):
    """Save detailed results to CSV."""
    rows = []
    for result in results:
        qid = result["qid"]
        query = result["query"]
        expanded = result.get("expanded_query", query)
        
        # Get top 10 retrieved docs
        retrieved = result["retrieved"][:10]
        for rank, doc in enumerate(retrieved, 1):
            rows.append({
                "qid": qid,
                "query_en": queries_english.get(qid, ""),
                "query_id": query,
                "expanded_query": expanded,
                "rank": rank,
                "doc_id": doc["id"],
                "score": doc["score"]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")


def run_experiment(queries: Dict[str, str], corpus: List[Dict[str, str]], qrels: Dict, method: str, language: str, top_k: int = 10):
    """Run a single experiment."""
    logger.info(f"Running {method} on {language} queries...")
    
    if method == "embedding":
        results = run_embedding_qe(queries, corpus, top_k)
    elif method == "bm25":
        results = run_bm25(queries, corpus, top_k)
    else:
        results = run_baseline(queries, corpus, top_k)
    
    metrics = evaluate_results(results, qrels, top_k)
    logger.info(f"{language} {method} - NDCG@10: {metrics['NDCG@10']:.4f}, Recall@10: {metrics['Recall@10']:.4f}")
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Run QE experiments on COSQA")
    parser.add_argument(
        "--method",
        type=str,
        default="embedding",
        choices=["baseline", "bm25", "embedding"],
        help="Method to run"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cosqa_benchmark_results.json",
        help="Output file"
    )
    
    args = parser.parse_args()
    
    # Load data
    corpus, queries_en, queries_id, qrels = load_cosqa_data()
    
    logger.info(f"Loaded {len(queries_en)} queries and {len(corpus)} documents")
    
    # Run experiments for both languages
    all_results = {}
    all_metrics = {}
    
    # English baseline
    logger.info("=" * 50)
    logger.info("Running English queries...")
    results_en, metrics_en = run_experiment(
        queries_en, corpus, qrels, 
        method=args.method, 
        language="english", 
        top_k=args.top_k
    )
    all_results["english"] = results_en
    all_metrics["english"] = metrics_en
    
    # Indonesian (translated) queries
    logger.info("=" * 50)
    logger.info("Running Indonesian (translated) queries...")
    results_id, metrics_id = run_experiment(
        queries_id, corpus, qrels, 
        method=args.method, 
        language="indonesian", 
        top_k=args.top_k
    )
    all_results["indonesian"] = results_id
    all_metrics["indonesian"] = metrics_id
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    output_data = {
        "method": args.method,
        "top_k": args.top_k,
        "metrics": {
            "english": all_metrics["english"],
            "indonesian": all_metrics["indonesian"]
        },
        "results": {
            "english": all_results["english"],
            "indonesian": all_results["indonesian"]
        }
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Save CSV detailed for both languages
    csv_path_en = output_path.with_name(f"cosqa_{args.method}_english.csv")
    csv_path_id = output_path.with_name(f"cosqa_{args.method}_indonesian.csv")
    
    save_detailed_results(results_en, csv_path_en, queries_en)
    save_detailed_results(results_id, csv_path_id, queries_en)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary comparison
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 60)
    print(f"\nMethod: {args.method}")
    print(f"Top-K: {args.top_k}")
    print("\n--- English Queries ---")
    print(f"NDCG@{args.top_k}: {all_metrics['english']['NDCG@10']:.4f}")
    print(f"Recall@{args.top_k}: {all_metrics['english']['Recall@10']:.4f}")
    print(f"Hits: {all_metrics['english']['Hits']}/{all_metrics['english']['Total Relevant']}")
    
    print("\n--- Indonesian (Translated) Queries ---")
    print(f"NDCG@{args.top_k}: {all_metrics['indonesian']['NDCG@10']:.4f}")
    print(f"Recall@{args.top_k}: {all_metrics['indonesian']['Recall@10']:.4f}")
    print(f"Hits: {all_metrics['indonesian']['Hits']}/{all_metrics['indonesian']['Total Relevant']}")
    
    # Calculate difference
    ndcg_diff = all_metrics['indonesian']['NDCG@10'] - all_metrics['english']['NDCG@10']
    recall_diff = all_metrics['indonesian']['Recall@10'] - all_metrics['english']['Recall@10']
    print("\n--- Difference (Indonesian - English) ---")
    print(f"NDCG@{args.top_k}: {ndcg_diff:+.4f}")
    print(f"Recall@{args.top_k}: {recall_diff:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
