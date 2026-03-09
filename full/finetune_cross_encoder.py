import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
    CECorrelationEvaluator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    logger.info("Loading datasets from HuggingFace...")
    queries_corpus = load_dataset("CoIR-Retrieval/cosqa-queries-corpus")
    qrels = load_dataset("CoIR-Retrieval/cosqa-qrels")
    
    # Process queries
    logger.info("Processing queries...")
    queries = {}
    for q in queries_corpus['queries']:
        queries[q['_id']] = q['text']
        
    # Process corpus
    logger.info("Processing corpus...")
    corpus = {}
    corpus_ids = []
    for doc in queries_corpus['corpus']:
        corpus[doc['_id']] = doc.get('text', '')
        corpus_ids.append(doc['_id'])
        
    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} documents.")
    return queries, corpus, corpus_ids, qrels

def build_examples(split_qrels, queries, corpus, corpus_ids, num_negatives=4, hard_negatives=None):
    """
    Builds InputExamples for the cross-encoder.
    Uses hard negatives if available, falling back to random samples.
    """
    examples = []
    qrel_dict = {}
    
    for row in split_qrels:
        qid = row['query_id']
        cid = row['corpus_id']
        if qid not in qrel_dict:
            qrel_dict[qid] = set()
        qrel_dict[qid].add(cid)
        
        # Add positive example
        score = float(row['score'])
        # Add 'query: ' and 'passage: ' prefixes for mE5 compatibility
        examples.append(InputExample(texts=["query: " + queries[qid], "passage: " + corpus[cid]], label=score))
    
    logger.info(f"Added {len(examples)} positive pairs.")
    
    # Add negative examples
    negatives_added = 0
    for qid, cids in qrel_dict.items():
        query_text = queries[qid]
        
        # Try to use hard negatives first
        query_hard_negs = hard_negatives.get(qid, []) if hard_negatives else []
        
        for i in range(num_negatives):
            if i < len(query_hard_negs):
                neg_cid = query_hard_negs[i]
            else:
                # Random fallback
                neg_cid = random.choice(corpus_ids)
                while neg_cid in cids:
                    neg_cid = random.choice(corpus_ids)
                    
            examples.append(InputExample(texts=["query: " + query_text, "passage: " + corpus[neg_cid]], label=0.0))
            negatives_added += 1
            
    logger.info(f"Added {negatives_added} negative pairs.")
    random.shuffle(examples)
    return examples

def run_finetuning():
    import argparse
    import json
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-small",
                       help="Base model to fine-tune as cross-encoder")
    parser.add_argument("--output-dir", type=str, default="./full/models/cross-encoder-me5-small-full-hn",
                       help="Path to save the fine-tuned model")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-negatives", type=int, default=4, help="Negative samples per positive")
    parser.add_argument("--train-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--val-samples", type=int, default=None, help="Limit validation samples")
    parser.add_argument("--hard-negatives", type=str, default="./full/data/hard_negatives.json", 
                        help="Path to pre-mined hard negatives")
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    queries, corpus, corpus_ids, qrels = load_data()
    
    # Load hard negatives
    hard_negs = None
    if os.path.exists(args.hard_negatives):
        logger.info(f"Loading hard negatives from {args.hard_negatives}")
        with open(args.hard_negatives, 'r') as f:
            hard_negs = json.load(f)
    
    logger.info("Building training examples...")
    train_qrels = qrels['train']
    if args.train_samples:
        train_qrels = [row for i, row in enumerate(train_qrels) if i < args.train_samples][:args.train_samples]
    
    train_examples = build_examples(train_qrels, queries, corpus, corpus_ids, args.num_negatives, hard_negs)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
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
        
    dev_examples = build_examples(val_qrels, queries, corpus, corpus_ids, num_negatives=1, hard_negatives=hard_negs)
    
    # For evaluation, BCE needs arrays of texts and labels
    dev_sentence_pairs = [[ex.texts[0], ex.texts[1]] for ex in dev_examples]
    dev_labels = [ex.label for ex in dev_examples]
    evaluator = CEBinaryClassificationEvaluator(dev_sentence_pairs, dev_labels)
    
    logger.info(f"Initializing Cross-Encoder with base model: {args.model_name}")
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)
    
    warmup_steps = int(len(train_dataloader) * args.num_epochs * 0.1)
    
    logger.info(f"Starting training for {args.num_epochs} epochs. Output dir: {output_path}")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=9999999, # Evaluate only at end of epoch to speed up
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        output_path=str(output_path),
        use_amp=True,
        show_progress_bar=True,
    )
    
    logger.info("Finetuning finished. Explicitly saving model...")
    model.save(str(output_path))
    
    logger.info("Training complete!")
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    run_finetuning()
