"""
Run Query Expansion experiments.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coir.config import QEConfig
from coir.qe_pipeline import QEPipeline, IndonesianQEPipeline
from coir.dense_retriever import DenseRetriever, BM25Retriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data() -> tuple:
    """
    Load sample Indonesian queries and code corpus for testing.
    
    Returns:
        Tuple of (queries, corpus)
    """
    # Sample Indonesian queries
    indonesian_queries = [
        "cara membuat fungsi di python",
        "cara menyimpan data ke database",
        "cara mengurutkan list di python",
        "cara membuat API dengan flask",
        "cara menggunakan pandas dataframe",
        "cara looping through dictionary",
        "cara membuat class di python",
        "cara membaca file JSON",
        "cara membuat REST API",
        "cara menggunakan numpy array",
    ]
    
    # Sample code corpus (simplified for demo)
    code_corpus = [
        {"id": "0", "text": "def function_name(params):\n    '''Function documentation'''\n    return params"},
        {"id": "1", "text": "import pandas as pd\ndf = pd.DataFrame(data)\ndf.to_sql('table_name', engine)"},
        {"id": "2", "text": "sorted_list = sorted(items, key=lambda x: x['value'])"}, 
        {"id": "3", "text": "from flask import Flask, jsonify\napp = Flask(__name__)\n@app.route('/api')"},
        {"id": "4", "text": "import pandas as pd\nimport numpy as np\narr = np.array([1, 2, 3])"},
        {"id": "5", "text": "for key, value in dictionary.items():\n    print(key, value)"},
        {"id": "6", "text": "class MyClass:\n    def __init__(self):\n        self.data = None"},
        {"id": "7", "text": "import json\nwith open('file.json', 'r') as f:\n    data = json.load(f)"},
        {"id": "8", "text": "from flask import Flask, request, jsonify\n@app.route('/api', methods=['GET', 'POST'])"},
        {"id": "9", "text": "import numpy as np\narr = np.array([1, 2, 3, 4, 5])\nprint(arr.mean())"},
    ]
    
    return indonesian_queries, code_corpus


def calculate_metrics(
    results: List[Dict[str, Any]],
    qrels: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Calculate IR metrics.
    
    Args:
        results: Retrieved results
        qrels: Query relevance judgments
        
    Returns:
        Dictionary of metrics
    """
    # Simplified metrics calculation
    ndcg_scores = []
    map_scores = []
    recall_scores = []
    
    for query_id, retrieved in enumerate(results):
        retrieved_ids = [doc["id"] for doc in retrieved]
        relevant = set(qrels.get(str(query_id), []))
        
        if not relevant:
            continue
        
        # Calculate NDCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids[:10]):
            if doc_id in relevant:
                dcg += 1 / np.log2(i + 2)
        
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Calculate MAP
        ap = 0
        num_relevant = 0
        for i, doc_id in enumerate(retrieved_ids[:10]):
            if doc_id in relevant:
                num_relevant += 1
                ap += num_relevant / (i + 1)
        ap = ap / len(relevant) if relevant else 0
        map_scores.append(ap)
        
        # Calculate Recall
        retrieved_relevant = len([d for d in retrieved_ids if d in relevant])
        recall = retrieved_relevant / len(relevant) if relevant else 0
        recall_scores.append(recall)
    
    return {
        "ndcg@10": np.mean(ndcg_scores) if ndcg_scores else 0,
        "map@10": np.mean(map_scores) if map_scores else 0,
        "recall@10": np.mean(recall_scores) if recall_scores else 0,
    }


def run_baseline_experiment(args):
    """Run baseline (no QE) experiment."""
    logger.info("Running baseline experiment...")
    
    # Load data
    queries, corpus = load_sample_data()
    
    # Create retriever
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
    )
    
    # Retrieve
    results = retriever.retrieve(
        queries=queries,
        corpus=corpus,
        top_k=args.top_k,
    )
    
    # Calculate metrics (using dummy qrels for demo)
    qrels = {str(i): [str(i)] for i in range(len(queries))}
    metrics = calculate_metrics(results, qrels)
    
    logger.info(f"Baseline metrics: {metrics}")
    return results, metrics


def run_qe_experiment(args):
    """Run Query Expansion experiment."""
    logger.info(f"Running QE experiment with method: {args.method}")
    
    # Load data
    queries, corpus = load_sample_data()
    
    # Create pipeline
    config = QEConfig()
    config.EXPANSION_METHOD = args.method
    
    pipeline = QEPipeline(config)
    
    # Run
    if args.method == "bm25":
        results = pipeline.run_bm25_baseline(queries, corpus, args.top_k)
    elif args.method == "baseline":
        results = pipeline.run_baseline(queries, corpus, args.top_k)
    else:
        results = pipeline.run_qe(queries, corpus, args.top_k)
    
    # Extract retrieved docs
    retrieved = [r.retrieved_docs for r in results]
    
    # Calculate metrics
    qrels = {str(i): [str(i)] for i in range(len(queries))}
    metrics = calculate_metrics(retrieved, qrels)
    
    logger.info(f"{args.method} metrics: {metrics}")
    return results, metrics


def run_comparison_experiment(args):
    """Run comparison of multiple methods."""
    logger.info("Running comparison experiment...")
    
    # Load data
    queries, corpus = load_sample_data()
    
    # Methods to compare
    methods = ["baseline", "embedding"]
    
    # Create pipeline
    config = QEConfig()
    pipeline = QEPipeline(config)
    
    # Run comparison
    comparison = pipeline.run_comparison(
        queries=queries,
        corpus=corpus,
        methods=methods,
        top_k=args.top_k,
    )
    
    # Calculate metrics for each
    qrels = {str(i): [str(i)] for i in range(len(queries))}
    
    results = {}
    for method, qe_results in comparison.items():
        retrieved = [r.retrieved_docs for r in qe_results]
        metrics = calculate_metrics(retrieved, qrels)
        results[method] = metrics
        logger.info(f"{method}: {metrics}")
    
    return comparison, results


def main():
    parser = argparse.ArgumentParser(description="Run QE experiments")
    parser.add_argument(
        "--method",
        type=str,
        default="embedding",
        choices=["baseline", "bm25", "hyde", "embedding", "prf", "combined", "sequential"],
        help="Query expansion method"
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
        default="results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison of multiple methods"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    if args.compare:
        results, metrics = run_comparison_experiment(args)
    else:
        results, metrics = run_qe_experiment(args)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "method": args.method,
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return results, metrics


if __name__ == "__main__":
    main()
