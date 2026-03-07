"""
Main Experiment Runner: Full Pipeline for Indonesian Code Search.

Runs all 4 experiment configurations:
1. Baseline: mE5 only
2. TQE-Only: mE5 + Query Expansion
3. Rerank-Only: mE5 + Cross-Encoder Re-ranking
4. Full: mE5 + TQE + Cross-Encoder Re-ranking

Outputs:
- JSON results with metrics for each configuration
- CSV detailed results for analysis
- Summary comparison table
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "qe"))
sys.path.insert(0, str(Path(__file__).parent.parent / "reranker"))

from src.config import PipelineConfig, ExperimentType, QEMethod, RerankerModel
from src.dataset_loader import load_cosqa_dataset
from src.pipeline import Pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_experiment(
    experiment_type: ExperimentType,
    corpus: Dict,
    queries_en: Dict[str, str],
    queries_id: Dict[str, str],
    qrels: Dict,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    config = PipelineConfig(
        experiment_type=experiment_type,
        retriever_model=args.retriever_model,
        first_stage_top_k=args.first_stage_k,
        qe_method=QEMethod(args.qe_method),
        qe_num_terms=args.qe_num_terms,
        reranker_model_type=RerankerModel(args.reranker_model),
        reranker_top_k=args.top_k,
        device=args.device,
        sample_size=args.sample_size,
        eval_k_values=[1, 5, 10, 20, 50, 100],
    )
    
    pipeline = Pipeline(config)
    results = pipeline.run_bilingual(queries_en, queries_id, corpus, qrels)
    
    return results


def save_detailed_csv(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    queries_en: Dict[str, str],
):
    """Save detailed per-query CSV results."""
    rows = []
    
    for exp_name, exp_results in all_results.items():
        for lang in ["english", "indonesian"]:
            if lang not in exp_results:
                continue
            
            lang_results = exp_results[lang]
            raw = lang_results.get("raw_results", {})
            
            for qid, data in raw.items():
                query = data.get("query", "")
                original_query = data.get("original_query", query)
                expanded_query = data.get("expanded_query", query)
                
                # Get the final ranked list
                if "reranked" in data:
                    docs = data["reranked"][:10]
                    stage = "reranked"
                elif "retrieved" in data:
                    docs = data["retrieved"][:10]
                    stage = "retrieval_only"
                else:
                    continue
                
                for rank, doc in enumerate(docs, 1):
                    rows.append({
                        "experiment": exp_name,
                        "language": lang,
                        "qid": qid,
                        "query_en": queries_en.get(qid, ""),
                        "query_used": query,
                        "expanded_query": expanded_query if expanded_query != query else "",
                        "stage": stage,
                        "rank": rank,
                        "doc_id": doc.get("id", ""),
                        "retrieval_score": doc.get("score", 0),
                        "cross_encoder_score": doc.get("cross_encoder_score", ""),
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to {csv_path}")


def print_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print a comparison table of all experiments."""
    
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS COMPARISON")
    print("Indonesian Code Search: TQE + Cross-Encoder Re-ranking")
    print("=" * 90)
    
    # Key metrics to display
    key_metrics = ["nDCG@10", "MAP@10", "Recall@10", "MRR@10"]
    
    for lang in ["english", "indonesian"]:
        print(f"\n{'─' * 90}")
        print(f"  {lang.upper()} QUERIES")
        print(f"{'─' * 90}")
        
        # Header
        header = f"{'Experiment':<25}"
        for m in key_metrics:
            header += f" {m:>10}"
        header += f" {'Time(s)':>10}"
        print(header)
        print("-" * 90)
        
        for exp_name, exp_results in all_results.items():
            if lang not in exp_results:
                continue
            
            lang_results = exp_results[lang]
            metrics = lang_results.get("metrics", {})
            timing = lang_results.get("timing", {})
            
            # Determine which metrics dict to use
            if "after_rerank" in metrics:
                # Use after-reranking metrics
                m_dict = metrics["after_rerank"]
            elif "retrieval" in metrics:
                m_dict = metrics["retrieval"]
            else:
                m_dict = metrics
            
            row = f"{exp_name:<25}"
            for m in key_metrics:
                val = m_dict.get(m, 0.0)
                row += f" {val:>10.4f}"
            
            total_time = timing.get("total", 0.0)
            row += f" {total_time:>10.1f}"
            print(row)
        
        print()
    
    # Print before/after comparison for experiments with reranking
    print(f"\n{'─' * 90}")
    print("  RE-RANKING IMPACT (Before → After)")
    print(f"{'─' * 90}")
    
    for exp_name, exp_results in all_results.items():
        for lang in ["english", "indonesian"]:
            if lang not in exp_results:
                continue
            
            metrics = exp_results[lang].get("metrics", {})
            
            if "before_rerank" in metrics and "after_rerank" in metrics:
                before = metrics["before_rerank"]
                after = metrics["after_rerank"]
                
                ndcg_before = before.get("nDCG@10", 0)
                ndcg_after = after.get("nDCG@10", 0)
                ndcg_diff = ndcg_after - ndcg_before
                
                map_before = before.get("MAP@10", 0)
                map_after = after.get("MAP@10", 0)
                map_diff = map_after - map_before
                
                print(f"  {exp_name} ({lang}):")
                print(f"    nDCG@10: {ndcg_before:.4f} → {ndcg_after:.4f} ({ndcg_diff:+.4f})")
                print(f"    MAP@10:  {map_before:.4f} → {map_after:.4f} ({map_diff:+.4f})")
    
    # Cross-lingual gap
    print(f"\n{'─' * 90}")
    print("  CROSS-LINGUAL GAP (Indonesian − English)")
    print(f"{'─' * 90}")
    
    for exp_name, exp_results in all_results.items():
        if "english" not in exp_results or "indonesian" not in exp_results:
            continue
        
        en_metrics = exp_results["english"].get("metrics", {})
        id_metrics = exp_results["indonesian"].get("metrics", {})
        
        # Get the final metrics
        if "after_rerank" in en_metrics:
            en_m = en_metrics["after_rerank"]
            id_m = id_metrics.get("after_rerank", {})
        elif "retrieval" in en_metrics:
            en_m = en_metrics["retrieval"]
            id_m = id_metrics.get("retrieval", {})
        else:
            en_m = en_metrics
            id_m = id_metrics
        
        ndcg_gap = id_m.get("nDCG@10", 0) - en_m.get("nDCG@10", 0)
        map_gap = id_m.get("MAP@10", 0) - en_m.get("MAP@10", 0)
        recall_gap = id_m.get("Recall@10", 0) - en_m.get("Recall@10", 0)
        
        print(f"  {exp_name}:")
        print(f"    nDCG@10:   {ndcg_gap:+.4f}")
        print(f"    MAP@10:    {map_gap:+.4f}")
        print(f"    Recall@10: {recall_gap:+.4f}")
    
    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Run full Indonesian Code Search experiments"
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["baseline", "tqe_only", "rerank_only", "full"],
        choices=["baseline", "tqe_only", "rerank_only", "full"],
        help="Which experiments to run"
    )
    
    # Retriever settings
    parser.add_argument("--retriever-model", type=str, default="intfloat/multilingual-e5-small")
    parser.add_argument("--first-stage-k", type=int, default=100)
    
    # QE settings
    parser.add_argument("--qe-method", type=str, default="translation",
                        choices=["translation", "embedding", "hyde", "technical", "cot"])
    parser.add_argument("--qe-num-terms", type=int, default=5)
    
    # Reranker settings
    parser.add_argument("--reranker-model", type=str, default="mmmini",
                        choices=["mmmini", "mmmini_multilingual", "xlm", "mbert"])
    parser.add_argument("--top-k", type=int, default=10)
    
    # General settings
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Limit number of queries (for testing)")
    parser.add_argument("--output", type=str, default="./full/results/full_experiment_results.json")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading CoSQA dataset...")
    corpus, queries_en, queries_id, qrels = load_cosqa_dataset(sample_size=args.sample_size)
    
    logger.info(f"Dataset loaded: {len(queries_en)} queries, {len(corpus)} documents")
    
    # Map experiment names to types
    exp_type_map = {
        "baseline": ExperimentType.BASELINE,
        "tqe_only": ExperimentType.TQE_ONLY,
        "rerank_only": ExperimentType.RERANK_ONLY,
        "full": ExperimentType.FULL,
    }
    
    # Run experiments
    all_results = {}
    
    for exp_name in args.experiments:
        exp_type = exp_type_map[exp_name]
        
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# EXPERIMENT: {exp_name.upper()}")
        logger.info(f"{'#' * 60}")
        
        results = run_single_experiment(
            experiment_type=exp_type,
            corpus=corpus,
            queries_en=queries_en,
            queries_id=queries_id,
            qrels=qrels,
            args=args,
        )
        
        all_results[exp_name] = results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a serializable version (without raw_results which can be huge)
    serializable = {}
    for exp_name, exp_results in all_results.items():
        serializable[exp_name] = {}
        for lang, lang_results in exp_results.items():
            serializable[exp_name][lang] = {
                k: v for k, v in lang_results.items() if k != "raw_results"
            }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Results saved to {output_path}")
    
    # Save detailed CSV
    save_detailed_csv(all_results, output_path.parent, queries_en)
    
    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
