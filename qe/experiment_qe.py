"""
Standalone Query Expansion Experiment Runner.
This script runs without depending on the original coir evaluation module.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

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


def load_sample_data() -> tuple:
    """Load sample Indonesian queries and code corpus."""
    indonesian_queries = [
        "cara membuat fungsi di python",
        "cara menyimpan data ke database", 
        "cara mengurutkan list di python",
        "cara membuat API dengan flask",
        "cara menggunakan pandas dataframe",
    ]
    
    code_corpus = [
        {"id": "0", "text": "def function_name(params):\n    return params"},
        {"id": "1", "text": "import pandas as pd\ndf = pd.DataFrame(data)\ndf.to_sql('table', engine)"},
        {"id": "2", "text": "sorted_list = sorted(items, key=lambda x: x['value'])"},
        {"id": "3", "text": "from flask import Flask, jsonify\napp = Flask(__name__)"},
        {"id": "4", "text": "import pandas as pd\nimport numpy as np\narr = np.array([1, 2, 3])"},
    ]
    
    return indonesian_queries, code_corpus


def run_embedding_qe(queries: List[str], corpus: List[Dict[str, str]], top_k: int = 10):
    """Run embedding-based query expansion."""
    logger.info("Running embedding-based QE...")
    
    # Initialize expander
    expander = CrossLingualEmbeddingExpander(
        model_name="intfloat/multilingual-e5-small"
    )
    
    # Initialize retriever
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-small",
        device="cpu"
    )
    
    results = []
    for query in tqdm(queries, desc="Processing queries"):
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
            "query": query,
            "expanded_query": expansion.expanded_query,
            "expansion_terms": expansion.expansion_terms,
            "retrieved": retrieved[0]
        })
    
    return results


def run_baseline(queries: List[str], corpus: List[Dict[str, str]], top_k: int = 10):
    """Run baseline (no QE)."""
    logger.info("Running baseline retrieval...")
    
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-small",
        device="cpu"
    )
    
    results = []
    for query in tqdm(queries, desc="Processing queries"):
        retrieved = retriever.retrieve(
            queries=[query],
            corpus=corpus,
            top_k=top_k,
        )
        
        results.append({
            "query": query,
            "expanded_query": query,
            "retrieved": retrieved[0]
        })
    
    return results


def run_bm25(queries: List[str], corpus: List[Dict[str, str]], top_k: int = 10):
    """Run BM25 baseline."""
    logger.info("Running BM25...")
    
    retriever = BM25Retriever()
    retriever.fit(corpus)
    
    results = []
    for query in tqdm(queries, desc="Processing queries"):
        retrieved = retriever.retrieve(
            queries=[query],
            top_k=top_k,
        )
        
        results.append({
            "query": query,
            "retrieved": retrieved[0]
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run QE experiments")
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
        default="results.json",
        help="Output file"
    )
    
    args = parser.parse_args()
    
    # Load data
    queries, corpus = load_sample_data()
    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} documents")
    
    # Run experiment
    if args.method == "embedding":
        results = run_embedding_qe(queries, corpus, args.top_k)
    elif args.method == "bm25":
        results = run_bm25(queries, corpus, args.top_k)
    else:
        results = run_baseline(queries, corpus, args.top_k)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n=== Results Summary ===")
    for r in results:
        print(f"\nQuery: {r['query']}")
        if 'expanded_query' in r and r['expanded_query'] != r['query']:
            print(f"Expanded: {r['expanded_query']}")
            print(f"Terms: {r.get('expansion_terms', [])}")
        print(f"Top 3: {[d['id'] for d in r['retrieved'][:3]]}")


if __name__ == "__main__":
    main()
