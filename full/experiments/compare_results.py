"""
Compare Results: Load and compare experiment results from JSON files.

Can be used after running individual experiments to create comparison tables.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_metrics(results: Dict, lang: str) -> Dict[str, float]:
    """Extract key metrics from results."""
    if lang not in results:
        return {}
    
    lang_results = results[lang]
    metrics = lang_results.get("metrics", {})
    
    # Handle different result formats
    if "after_rerank" in metrics:
        return metrics["after_rerank"]
    elif "retrieval" in metrics:
        return metrics["retrieval"]
    else:
        return metrics


def print_comparison_table(all_results: Dict[str, Dict], metric_keys: list = None):
    """Print a formatted comparison table."""
    if metric_keys is None:
        metric_keys = ["nDCG@10", "MAP@10", "Recall@10", "MRR@10"]
    
    for lang in ["english", "indonesian"]:
        print(f"\n{'='*80}")
        print(f"  {lang.upper()} QUERIES - COMPARISON")
        print(f"{'='*80}")
        
        # Header
        header = f"  {'Experiment':<30}"
        for m in metric_keys:
            header += f" {m:>10}"
        print(header)
        print(f"  {'-'*70}")
        
        for exp_name, results in all_results.items():
            metrics = extract_metrics(results, lang)
            
            row = f"  {exp_name:<30}"
            for m in metric_keys:
                val = metrics.get(m, 0.0)
                row += f" {val:>10.4f}"
            print(row)
        
        print()
    
    # Improvement over baseline
    if "baseline" in all_results:
        print(f"\n{'='*80}")
        print(f"  IMPROVEMENT OVER BASELINE")
        print(f"{'='*80}")
        
        baseline = all_results["baseline"]
        
        for lang in ["english", "indonesian"]:
            print(f"\n  {lang.upper()}:")
            baseline_metrics = extract_metrics(baseline, lang)
            baseline_ndcg = baseline_metrics.get("nDCG@10", 0)
            
            for exp_name, results in all_results.items():
                if exp_name == "baseline":
                    continue
                
                metrics = extract_metrics(results, lang)
                ndcg = metrics.get("nDCG@10", 0)
                diff = ndcg - baseline_ndcg
                pct = (diff / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
                
                print(f"    {exp_name:<30} nDCG@10: {diff:+.4f} ({pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Result JSON files to compare (format: name:path)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./full/results",
        help="Directory containing result JSON files"
    )
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.results:
        # Load specified files
        for spec in args.results:
            if ":" in spec:
                name, path = spec.split(":", 1)
            else:
                name = Path(spec).stem
                path = spec
            
            try:
                all_results[name] = load_results(path)
                logger.info(f"Loaded: {name} from {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
    else:
        # Auto-discover from results directory
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            for json_file in sorted(results_dir.glob("*.json")):
                name = json_file.stem
                try:
                    all_results[name] = load_results(str(json_file))
                    logger.info(f"Loaded: {name}")
                except Exception as e:
                    logger.error(f"Failed to load {json_file}: {e}")
        
        # Check for the full results file
        full_results_path = results_dir / "full_experiment_results.json"
        if full_results_path.exists():
            full_data = load_results(str(full_results_path))
            # This file has experiment names as top-level keys
            if any(k in full_data for k in ["baseline", "tqe_only", "rerank_only", "full"]):
                all_results = full_data
    
    if not all_results:
        print("No results found. Run experiments first or specify --results paths.")
        return
    
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
