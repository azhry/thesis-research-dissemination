"""
Fine-tune a Cross-Encoder on CoSQA query-code pairs.

KEY INSIGHT: MS MARCO cross-encoders are trained on (query, passage) pairs 
where passages are natural language. For code search, we need to fine-tune 
on (query, code) pairs so the model learns what "relevant code" looks like.

This script:
1. Loads CoSQA query-code pairs from HuggingFace
2. Mines hard negatives using the Bi-Encoder (mE5) to find plausible-but-wrong code
3. Fine-tunes a cross-encoder to distinguish relevant code from hard negatives
4. Saves per-epoch checkpoints for evaluation
"""

import logging
import random
import sys
import json
import os
import argparse
from pathlib import Path

from datasets import load_dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load CoSQA queries, code corpus, and relevance judgments."""
    logger.info("Loading CoSQA dataset from HuggingFace...")
    queries_corpus = load_dataset("CoIR-Retrieval/cosqa-queries-corpus")
    qrels = load_dataset("CoIR-Retrieval/cosqa-qrels")
    
    # Process queries
    logger.info("Processing queries...")
    queries = {}
    for q in queries_corpus['queries']:
        queries[q['_id']] = q['text']
        
    # Process corpus (these are CODE SNIPPETS, not passages)
    logger.info("Processing code corpus...")
    corpus = {}
    corpus_ids = []
    for doc in queries_corpus['corpus']:
        corpus[doc['_id']] = doc.get('text', '')
        corpus_ids.append(doc['_id'])
        
    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} code snippets.")
    return queries, corpus, corpus_ids, qrels


def mine_hard_negatives_with_retriever(queries, corpus, corpus_ids, qrels_dict, top_k=20):
    """
    Mine hard negatives using the mE5 Bi-Encoder.
    
    Hard negatives = code snippets that the retriever thinks are relevant 
    but are actually NOT the correct answer. These are the most informative 
    training examples for the cross-encoder.
    """
    logger.info("Mining hard negatives with mE5 Bi-Encoder...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        
        # Encode queries with "query: " prefix (mE5 convention)
        query_ids = list(queries.keys())
        query_texts = ["query: " + queries[qid] for qid in query_ids]
        
        # Encode corpus with "passage: " prefix
        doc_ids = list(corpus.keys())
        doc_texts = ["passage: " + corpus[did] for did in doc_ids]
        
        logger.info(f"Encoding {len(query_texts)} queries...")
        query_embs = model.encode(query_texts, batch_size=64, show_progress_bar=True)
        
        logger.info(f"Encoding {len(doc_texts)} code snippets...")
        doc_embs = model.encode(doc_texts, batch_size=64, show_progress_bar=True)
        
        # For each query, find the top-K most similar code (by Bi-Encoder)
        # but exclude the actual positives → these are hard negatives
        hard_negatives = {}
        
        for i, qid in enumerate(query_ids):
            if qid not in qrels_dict:
                continue
                
            # Cosine similarity
            scores = np.dot(doc_embs, query_embs[i]) / (
                np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_embs[i])
            )
            
            # Get top-K indices
            top_indices = np.argsort(scores)[::-1][:top_k + 10]  # Extra buffer
            
            # Filter out positives
            positives = qrels_dict[qid]
            neg_ids = []
            for idx in top_indices:
                did = doc_ids[idx]
                if did not in positives:
                    neg_ids.append(did)
                if len(neg_ids) >= top_k:
                    break
            
            hard_negatives[qid] = neg_ids
        
        logger.info(f"Mined hard negatives for {len(hard_negatives)} queries.")
        
        # Clean up GPU memory
        del model, query_embs, doc_embs
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return hard_negatives
        
    except Exception as e:
        logger.warning(f"Hard negative mining failed: {e}. Falling back to random negatives.")
        return None


def build_examples(split_qrels, queries, corpus, corpus_ids, num_negatives=4, hard_negatives=None):
    """
    Build training examples for cross-encoder.
    
    IMPORTANT: No "query: " or "passage: " prefixes!
    MS MARCO cross-encoders are trained on raw text pairs.
    The prefixes were an E5-specific convention that was confusing the model.
    """
    examples = []
    qrel_dict = {}
    
    for row in split_qrels:
        qid = row['query_id']
        cid = row['corpus_id']
        if qid not in qrel_dict:
            qrel_dict[qid] = set()
        qrel_dict[qid].add(cid)
        
        # Positive example: raw query + raw code (NO prefixes)
        score = float(row['score'])
        examples.append(InputExample(
            texts=[queries[qid], corpus[cid]], 
            label=score
        ))
    
    logger.info(f"Added {len(examples)} positive pairs.")
    
    # Add negative examples
    negatives_added = 0
    for qid, cids in qrel_dict.items():
        query_text = queries[qid]
        
        # Try hard negatives first (much more informative)
        query_hard_negs = hard_negatives.get(qid, []) if hard_negatives else []
        
        for i in range(num_negatives):
            if i < len(query_hard_negs):
                neg_cid = query_hard_negs[i]
            else:
                # Random fallback
                neg_cid = random.choice(corpus_ids)
                while neg_cid in cids:
                    neg_cid = random.choice(corpus_ids)
            
            # Raw query + raw code (NO prefixes)        
            examples.append(InputExample(
                texts=[query_text, corpus[neg_cid]], 
                label=0.0
            ))
            negatives_added += 1
            
    logger.info(f"Added {negatives_added} negative pairs ({num_negatives} per query).")
    random.shuffle(examples)
    return examples, qrel_dict


def run_finetuning():
    parser = argparse.ArgumentParser(description="Fine-tune Cross-Encoder on CoSQA query-code pairs")
    parser.add_argument("--model-name", type=str, 
                        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                        help="Base cross-encoder model (must be a real cross-encoder, NOT a bi-encoder)")
    parser.add_argument("--output-dir", type=str, 
                        default="./full/models/cross-encoder-cosqa",
                        help="Path to save the fine-tuned model")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-negatives", type=int, default=4, 
                        help="Negative samples per positive pair")
    parser.add_argument("--mine-hard-negatives", action="store_true",
                        help="Mine hard negatives using mE5 Bi-Encoder (slower but better)")
    parser.add_argument("--hard-negatives-file", type=str, 
                        default="./full/data/hard_negatives_cosqa.json",
                        help="Path to save/load pre-mined hard negatives")
    parser.add_argument("--train-samples", type=int, default=None, 
                        help="Limit training samples (for quick testing)")
    parser.add_argument("--val-samples", type=int, default=None, 
                        help="Limit validation samples")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ── Step 1: Load CoSQA Data ──
    queries, corpus, corpus_ids, qrels = load_data()
    
    # ── Step 2: Get Hard Negatives ──
    hard_negs = None
    hn_file = Path(args.hard_negatives_file)
    
    if hn_file.exists() and not args.mine_hard_negatives:
        logger.info(f"Loading pre-mined hard negatives from {hn_file}")
        with open(hn_file, 'r') as f:
            hard_negs = json.load(f)
        logger.info(f"Loaded hard negatives for {len(hard_negs)} queries.")
    elif args.mine_hard_negatives:
        # Build qrels dict for mining
        qrels_dict = {}
        for row in qrels['train']:
            qid = row['query_id']
            if qid not in qrels_dict:
                qrels_dict[qid] = set()
            qrels_dict[qid].add(row['corpus_id'])
        
        hard_negs = mine_hard_negatives_with_retriever(
            queries, corpus, corpus_ids, qrels_dict, top_k=20
        )
        
        if hard_negs:
            # Save for reuse
            hn_file.parent.mkdir(parents=True, exist_ok=True)
            # Convert sets in qrels_dict aren't in hard_negs (it's already list-based)
            with open(hn_file, 'w') as f:
                json.dump(hard_negs, f)
            logger.info(f"Saved hard negatives to {hn_file}")
    else:
        logger.info("Using random negatives (use --mine-hard-negatives for better results).")
    
    # ── Step 3: Build Training Examples ──
    logger.info("Building training examples...")
    train_qrels = qrels['train']
    if args.train_samples:
        train_qrels = [row for i, row in enumerate(train_qrels) if i < args.train_samples][:args.train_samples]
    
    train_examples, _ = build_examples(
        train_qrels, queries, corpus, corpus_ids, args.num_negatives, hard_negs
    )
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # ── Step 4: Build Validation Examples ──
    logger.info("Building validation examples...")
    if 'valid' in qrels:
        val_qrels = qrels['valid']
    elif 'dev' in qrels:
        val_qrels = qrels['dev']
    else:
        val_qrels = qrels['test']
        
    if args.val_samples:
        val_qrels = [row for i, row in enumerate(val_qrels) if i < args.val_samples]
    else:
        val_qrels = [row for i, row in enumerate(val_qrels) if i < 1000]
        
    dev_examples, _ = build_examples(
        val_qrels, queries, corpus, corpus_ids, num_negatives=1, hard_negatives=hard_negs
    )
    
    dev_sentence_pairs = [[ex.texts[0], ex.texts[1]] for ex in dev_examples]
    dev_labels = [ex.label for ex in dev_examples]
    evaluator = CEBinaryClassificationEvaluator(dev_sentence_pairs, dev_labels)
    
    # ── Step 5: Initialize Cross-Encoder ──
    logger.info(f"Initializing Cross-Encoder: {args.model_name}")
    logger.info(f"  This model will learn to score (query, code) pairs")
    logger.info(f"  Training data: {len(train_examples)} examples")
    logger.info(f"  Validation data: {len(dev_examples)} examples")
    
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)
    
    warmup_steps = int(len(train_dataloader) * args.num_epochs * 0.1)
    
    # ── Step 6: Train! ──
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=len(train_dataloader),  # Evaluate every epoch
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        output_path=str(output_path),
        use_amp=True,
        show_progress_bar=True,
    )
    
    # Save final model
    logger.info("Saving final model...")
    model.save(str(output_path))
    
    logger.info(f"Training complete! Model saved to {output_path}")
    logger.info(f"")
    logger.info(f"To use this model in experiments, run:")
    logger.info(f"  python run_experiments.py --experiments full --reranker-model custom \\")
    logger.info(f"    --qe-method hyde --llm-provider local --llm-model llama3")


if __name__ == "__main__":
    run_finetuning()
