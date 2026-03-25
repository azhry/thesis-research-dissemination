
import os
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add qe to path
QE_PATH = Path(__file__).parent.parent.parent / "qe"
if str(QE_PATH) not in sys.path:
    sys.path.insert(0, str(QE_PATH))

from coir.dense_retriever import DenseRetriever
from datasets import load_dataset

def mine_negatives(num_queries=5000, top_k=50, num_hard_negatives=5):
    """
    Mine hard negatives for CoSQA training.
    """
    logger.info("Loading CoSQA data...")
    dataset = load_dataset("CoIR-Retrieval/cosqa-queries-corpus")
    qrels_dataset = load_dataset("CoIR-Retrieval/cosqa-qrels")
    
    # Process queries (train set only)
    logger.info("Processing training queries...")
    train_qrel_dict = {}
    for row in qrels_dataset['train']:
        qid = row['query_id']
        cid = row['corpus_id']
        if qid not in train_qrel_dict:
            train_qrel_dict[qid] = []
        train_qrel_dict[qid].append(cid)
    
    # Filter only queries that have text and qrels
    all_queries = {q['_id']: q['text'] for q in dataset['queries']}
    active_qids = [qid for qid in train_qrel_dict if qid in all_queries]
    
    if num_queries:
        active_qids = active_qids[:num_queries]
    
    query_list = [all_queries[qid] for qid in active_qids]
    
    logger.info(f"Mining negatives for {len(active_qids)} queries.")
    
    # Initialize Bi-Encoder
    retriever = DenseRetriever(
        model_name="intfloat/multilingual-e5-base",
        device="cpu", # Change to cuda if available
        batch_size=32
    )
    
    # Encode corpus
    corpus_data = dataset['corpus']
    logger.info("Encoding corpus (this might take a while)...")
    corpus_embeddings, corpus_ids = retriever.encode_corpus(corpus_data)
    
    # Encode queries in batches to avoid OOM
    logger.info("Encoding queries...")
    query_embeddings = retriever.encode_queries(query_list)
    
    # Find hard negatives
    logger.info("Calculating similarity and picking hard negatives...")
    hard_negatives = {}
    
    # Dot product for speed
    similarities = np.matmul(query_embeddings, corpus_embeddings.T)
    
    for i, qid in enumerate(tqdm(active_qids, desc="Mining")):
        scores = similarities[i]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        ground_truth = set(train_qrel_dict[qid])
        negatives = []
        
        for idx in top_indices:
            cid = corpus_ids[idx]
            if cid not in ground_truth:
                negatives.append(cid)
                if len(negatives) >= num_hard_negatives:
                    break
        
        hard_negatives[qid] = negatives
        
    # Save results
    output_path = Path("./full/data/hard_negatives.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hard_negatives, f)
    
    logger.info(f"Saved hard negatives to {output_path}")

if __name__ == "__main__":
    mine_negatives(num_queries=10000) # Only mine top 10k for speed
