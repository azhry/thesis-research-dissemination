"""
Standalone Cross-Encoder Re-ranking Experiment Runner.

This script runs cross-encoder re-ranking experiments using
benchmark datasets from CoIR (Code Information Retrieval) benchmark.
"""

import argparse
import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

# Import our modules
from .cross_encoder_base import CrossEncoderReranker
from .mmmini_reranker import MMMiniReranker, MultilingualMMMiniReranker
from .xlm_reranker import XLMReranker, MBERTReranker
from .fine_tuner import CrossEncoderFineTuner, FineTuningConfig, create_hard_negatives

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Benchmark dataset configurations
BENCHMARK_CONFIGS = {
    "cosqa": {
        "name": "CoSQA",
        "description": "Code Search Q&A - Natural language to code",
        "language": "English",
        "task_type": "code-retrieval"
    },
    "codetrans_dl": {
        "name": "CodeTrans-DL",
        "description": "Code Translation Deep Learning",
        "language": "Multiple",
        "task_type": "code-translation"
    },
    "stackoverflow_qa": {
        "name": "StackOverflow QA",
        "description": "StackOverflow Question Answering",
        "language": "English",
        "task_type": "code-retrieval"
    },
}


def load_benchmark_data(
    dataset_name: str = "cosqa",
    split: str = "test",
    sample_size: Optional[int] = None
) -> tuple:
    """
    Load benchmark dataset from CoIR.
    
    CoIR format: separate subsets for queries, corpus, and qrels
    """
    logger.info(f"Loading benchmark dataset: {dataset_name}")
    
    try:
        from datasets import load_dataset
        from coir.data_loader import load_data_from_hf
        
        # Try to load properly from CoIR
        try:
            corpus, queries, qrels = load_data_from_hf(dataset_name)
            
            # Convert corpus to list format
            corpus_list = [
                {"id": doc_id, "text": doc_data.get("text", "")}
                for doc_id, doc_data in corpus.items()
            ]
            
            # Convert queries dict to list with id and text
            queries_list = [
                {"id": qid, "query": qtext, "text": qtext}
                for qid, qtext in queries.items()
            ]
            
            # Convert qrels to ground truth format
            relevance_labels = {}
            for qid, doc_scores in qrels.items():
                # Get all relevant doc IDs (score > 0)
                relevant_docs = [doc_id for doc_id, score in doc_scores.items() if score > 0]
                if relevant_docs:
                    relevance_labels[qid] = relevant_docs
            
            # Optionally limit queries
            if sample_size and sample_size < len(queries_list):
                queries_list = queries_list[:sample_size]
                # Also limit qrels
                query_ids = [q["id"] for q in queries_list]
                relevance_labels = {k: v for k, v in relevance_labels.items() if k in query_ids}
            
            logger.info(f"Loaded {len(queries_list)} queries, {len(corpus_list)} docs, {len(relevance_labels)} qrels")
            return queries_list, corpus_list, relevance_labels
            
        except Exception as e:
            logger.warning(f"Failed to load with load_data_from_hf: {e}")
        
        # Fallback to direct dataset loading
        queries = []
        corpus_dict = {}
        relevance_labels = {}
        
        # Load corpus
        logger.info(f"Loading corpus for {dataset_name}...")
        try:
            corpus_ds = load_dataset(f"CoIR-Retrieval/{dataset_name}-queries-corpus", "corpus", split="corpus")
            logger.info(f"Corpus: {len(corpus_ds)} samples")
            for idx, item in enumerate(tqdm(corpus_ds, desc="Processing corpus")):
                doc_id = str(item.get('doc-id', item.get('doc_id', item.get('_id', idx))))
                doc_text = item.get('text', item.get('code', str(item)))
                if not doc_id or doc_id == '':
                    doc_id = str(idx)
                corpus_dict[doc_id] = {'id': doc_id, 'text': doc_text}
        except Exception as e:
            logger.warning(f"Could not load corpus: {e}")
        
        # Load queries
        logger.info(f"Loading queries for {dataset_name}...")
        try:
            queries_ds = None
            try:
                queries_ds = load_dataset(f"CoIR-Retrieval/{dataset_name}-queries-corpus", "queries", split="queries")
            except:
                try:
                    queries_ds = load_dataset(f"CoIR-Retrieval/{dataset_name}", "default", split="test")
                except:
                    try:
                        queries_ds = load_dataset(f"CoIR-Retrieval/{dataset_name}", split="test")
                    except:
                        pass
            
            if queries_ds:
                logger.info(f"Queries: {len(queries_ds)} samples")
                query_items = list(queries_ds)
                if sample_size:
                    query_items = query_items[:sample_size]
                
                for idx, item in enumerate(query_items):
                    query_id = str(item.get('query-id', item.get('query_id', item.get('_id', idx))))
                    query_text = item.get('text', item.get('query', str(item)))
                    
                    queries.append({
                        'id': query_id,
                        'query': query_text,
                        'text': query_text
                    })
        except Exception as e:
            logger.warning(f"Could not load queries: {e}")
        
        # Load qrels
        logger.info(f"Loading qrels for {dataset_name}...")
        try:
            qrels_ds = load_dataset(f"CoIR-Retrieval/{dataset_name}-qrels", "test", split="test")
            logger.info(f"Qrels: {len(qrels_ds)} samples")
            for item in qrels_ds:
                query_id = str(item.get('query-id', item.get('query_id', '')))
                doc_id = str(item.get('corpus-id', item.get('corpus_id', '')))
                score = int(item.get('score', 1))
                
                if score > 0:
                    if query_id not in relevance_labels:
                        relevance_labels[query_id] = []
                    relevance_labels[query_id].append(doc_id)
        except Exception as e:
            logger.warning(f"Could not load qrels: {e}")
        
        # Fallback: create pseudo relevance if no qrels
        if not relevance_labels and corpus_dict:
            logger.info("No qrels found, using first doc as relevant (pseudo)")
            for q in queries:
                first_doc_id = list(corpus_dict.keys())[0]
                relevance_labels[q['id']] = [first_doc_id]
        
        corpus = list(corpus_dict.values())
        
        logger.info(f"Final: {len(queries)} queries, {len(corpus)} documents, {len(relevance_labels)} qrels")
        
        return queries, corpus, relevance_labels
        
    except ImportError:
        logger.warning("datasets library not available, using sample data")
        return load_sample_data()
    except Exception as e:
        logger.warning(f"Failed to load benchmark dataset: {e}")
        import traceback
        traceback.print_exc()
        return load_sample_data()


def load_sample_data() -> tuple:
    """Load sample Indonesian queries and code corpus."""
    indonesian_queries = [
        {"id": "0", "query": "cara membuat fungsi di python", "text": "cara membuat fungsi di python"},
        {"id": "1", "query": "cara menyimpan data ke database", "text": "cara menyimpan data ke database"}, 
        {"id": "2", "query": "cara mengurutkan list di python", "text": "cara mengurutkan list di python"},
        {"id": "3", "query": "cara membuat API dengan flask", "text": "cara membuat API dengan flask"},
        {"id": "4", "query": "cara menggunakan pandas dataframe", "text": "cara menggunakan pandas dataframe"},
    ]
    
    code_corpus = [
        {"id": "0", "text": "def function_name(params):\n    return params"},
        {"id": "1", "text": "import pandas as pd\ndf = pd.DataFrame(data)\ndf.to_sql('table', engine)"},
        {"id": "2", "text": "sorted_list = sorted(items, key=lambda x: x['value'])"},
        {"id": "3", "text": "from flask import Flask, jsonify\napp = Flask(__name__)"},
        {"id": "4", "text": "import pandas as pd\nimport numpy as np\narr = np.array([1, 2, 3])"},
    ]
    
    relevance_labels = {
        "0": ["0"],
        "1": ["1"],
        "2": ["2"],
        "3": ["3"],
        "4": ["1", "4"],
    }
    
    return indonesian_queries, code_corpus, relevance_labels


def run_cross_encoder_reranking(
    queries: List[Dict],
    corpus: List[Dict[str, str]],
    model_type: str = "mmmini",
    top_k: int = 10,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """Run cross-encoder re-ranking."""
    logger.info(f"Running {model_type} cross-encoder reranking...")
    
    if not corpus or not queries:
        logger.error("No corpus or queries available!")
        return []
    
    if model_type == "mmmini":
        reranker = MMMiniReranker(device=device)
    elif model_type == "mmmini_multilingual":
        reranker = MultilingualMMMiniReranker(device=device)
    elif model_type == "xlm":
        reranker = XLMReranker(device=device)
    else:
        reranker = MMMiniReranker(device=device)
    
    reranker.load_model()
    
    results = []
    corpus_texts = [doc['text'][:512] for doc in corpus]  # Truncate for efficiency
    
    for query_item in tqdm(queries, desc="Processing queries"):
        query_id = query_item.get('id', 'unknown')
        query_text = query_item.get('text', query_item.get('query', ''))[:512]
        
        if not query_text:
            continue
        
        try:
            scores = reranker.score(query_text, corpus_texts)
            
            scored_docs = [
                {**doc, 'cross_encoder_score': float(score)}
                for doc, score in zip(corpus, scores)
            ]
            
            scored_docs.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            
            results.append({
                "query_id": query_id,
                "query": query_text,
                "reranked": scored_docs[:top_k]
            })
        except Exception as e:
            logger.warning(f"Error processing query {query_id}: {e}")
            continue
    
    return results


def run_baseline(
    queries: List[Dict],
    corpus: List[Dict[str, str]],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Run baseline retrieval."""
    results = []
    for query_item in queries:
        results.append({
            "query_id": query_item.get('id', 'unknown'),
            "query": query_item.get('text', query_item.get('query', '')),
            "retrieved": corpus[:top_k]
        })
    return results


def evaluate_results(
    results: List[Dict[str, Any]],
    ground_truth: Dict[str, List[str]],
    output_path: str = None
) -> Dict[str, float]:
    """Evaluate reranking results and save detailed output."""
    ndcg_scores = []
    map_scores = []
    mrr_scores = []
    
    detailed_results = []
    
    for result in results:
        query_id = result.get('query_id', '')
        reranked = result.get('reranked', result.get('retrieved', []))
        
        relevant_ids = set(ground_truth.get(query_id, []))
        retrieved_ids = [doc['id'] for doc in reranked]
        
        # Save detailed per-query results
        for rank, doc in enumerate(reranked, 1):
            doc_id = doc.get('id', '')
            score = doc.get('cross_encoder_score', 0)
            is_relevant = doc_id in relevant_ids if relevant_ids else False
            
            detailed_results.append({
                'query_id': query_id,
                'query': result.get('query', ''),
                'predicted_doc_id': doc_id,
                'predicted_rank': rank,
                'cross_encoder_score': score,
                'ground_truth_doc_ids': ';'.join(relevant_ids) if relevant_ids else '',
                'is_relevant': is_relevant
            })
        
        if not relevant_ids:
            continue
        
        # NDCG
        dcg = sum(1.0 / np.log2(i + 2) for i, doc_id in enumerate(retrieved_ids[:10]) if doc_id in relevant_ids)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), 10)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
        
        # MAP
        prec_sum = sum(sum(1 for doc_id in retrieved_ids[:i+1] if doc_id in relevant_ids) / (i+1) 
                      for i in range(min(10, len(retrieved_ids))) 
                      if any(doc_id in relevant_ids for doc_id in retrieved_ids[:i+1]))
        map_scores.append(prec_sum / len(relevant_ids) if relevant_ids else 0)
        
        # MRR
        for i, doc_id in enumerate(retrieved_ids[:10]):
            if doc_id in relevant_ids:
                mrr_scores.append(1.0 / (i + 1))
                break
        else:
            mrr_scores.append(0)
    
    # Save detailed results to CSV
    if output_path and detailed_results:
        csv_path = str(output_path).replace('.json', '_detailed.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'query_id', 'query', 'predicted_doc_id', 'predicted_rank',
                'cross_encoder_score', 'ground_truth_doc_ids', 'is_relevant'
            ])
            writer.writeheader()
            writer.writerows(detailed_results)
        logger.info(f"Detailed results saved to {csv_path}")
    
    return {
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "map@10": float(np.mean(map_scores)) if map_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "num_queries": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Run CE reranking experiments")
    parser.add_argument("--method", type=str, default="mmmini")
    parser.add_argument("--model-type", type=str, default="mmmini")
    parser.add_argument("--dataset", type=str, default="cosqa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="./reranker/reranker_results.json")
    
    args = parser.parse_args()
    
    logger.info(f"=== Benchmark: {args.dataset} ===")
    
    queries, corpus, relevance_labels = load_benchmark_data(
        dataset_name=args.dataset,
        sample_size=args.sample_size
    )
    logger.info(f"Loaded {len(queries)} queries, {len(corpus)} docs")
    
    if not queries or not corpus:
        logger.error("Failed to load data!")
        return
    
    if args.method == "baseline":
        results = run_baseline(queries, corpus, args.top_k)
    else:
        results = run_cross_encoder_reranking(queries, corpus, args.model_type, args.top_k, args.device)
    
    metrics = evaluate_results(results, relevance_labels, args.output) if relevance_labels else {}
    
    logger.info(f"=== Metrics: {metrics}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "method": args.method,
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {len(queries)}, Docs: {len(corpus)}")
    if metrics:
        print(f"nDCG@10: {metrics.get('ndcg@10', 0):.4f}")
        print(f"MAP@10: {metrics.get('map@10', 0):.4f}")
        print(f"MRR: {metrics.get('mrr', 0):.4f}")


if __name__ == "__main__":
    main()
