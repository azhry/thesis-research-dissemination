import sys
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add qe to path
QE_PATH = Path(__file__).parent.parent.parent / "qe"
if str(QE_PATH) not in sys.path:
    sys.path.insert(0, str(QE_PATH))

FULL_PATH = Path(__file__).parent.parent
if str(FULL_PATH) not in sys.path:
    sys.path.insert(0, str(FULL_PATH))

from datasets import load_dataset
from src.dataset_loader import load_indonesian_translations
from src.retriever import Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_translation_overlap(queries_en, queries_id):
    logger.info("Evaluating Translation Overlap...")
    # Check if code-like terminology was preserved vs translated.
    # Since programming is in English mostly, if "for loop" was translated as "perulangan untuk",
    # the exact code overlaps will differ.
    
    # Simple check: calculate average length, specific code keywords in both
    common_code_words = ["for", "while", "if", "else", "def", "function", "class", "print", "return"]
    
    en_code_kw_count = 0
    id_code_kw_count = 0
    
    for qid in queries_en:
        en_text = queries_en[qid].lower().split()
        id_text = queries_id.get(qid, "").lower().split()
        
        en_code_kw_count += sum([1 for w in en_text if w in common_code_words])
        id_code_kw_count += sum([1 for w in id_text if w in common_code_words])
    
    logger.info(f"Total English code keywords preserved: {en_code_kw_count}")
    logger.info(f"Total Indonesian code keywords preserved: {id_code_kw_count}")
    if en_code_kw_count > 0:
        logger.info(f"Preservation Ratio: {id_code_kw_count / en_code_kw_count:.2%}")
        
    return id_code_kw_count / max(en_code_kw_count, 1)

def evaluate_embedding_alignment(queries_en, queries_id, retriever):
    logger.info("Evaluating Embedding Alignment (ID vs EN)...")
    
    qids = list(queries_id.keys())[:2000] # Use a subset for speed
    
    en_texts = [queries_en[qid] for qid in qids]
    id_texts = [queries_id[qid] for qid in qids]
    
    # We directly use the DenseRetriever inside the retriever
    en_embeddings = retriever._retriever.encode_queries(en_texts)
    id_embeddings = retriever._retriever.encode_queries(id_texts)
    
    # Calculate cosine similarity between EN and ID pairs
    # Since encode_queries L2-normalizes by default, dot product = cosine sim
    similarities = np.sum(en_embeddings * id_embeddings, axis=1)
    
    mean_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    logger.info(f"Mean Cosine Sim (EN vs ID embeddings): {mean_sim:.4f}")
    logger.info(f"Min Cosine Sim: {min_sim:.4f}")
    logger.info(f"Max Cosine Sim: {max_sim:.4f}")
    
    # Look at bottom 5 poorly aligned queries
    bottom_indices = np.argsort(similarities)[:5]
    logger.info("\nBottom 5 Poorly Aligned Queries:")
    for idx in bottom_indices:
        qid = qids[idx]
        logger.info(f"[{similarities[idx]:.4f}] EN: {en_texts[idx]}")
        logger.info(f"         ID: {id_texts[idx]}")

def check_reranker_sensitivity(queries_en, queries_id, qrels, dataset, reranker_model_path):
    logger.info("Evaluating Reranker Sensitivity (ID vs EN)...")
    from src.reranker import Reranker
    
    reranker = Reranker(model_type="custom", model_name=reranker_model_path, device="cpu")
    
    corpus_dict = {}
    for row in dataset['corpus']:
        corpus_dict[row['_id']] = row['text']
        
    test_qrels = qrels['test']
    
    qrel_dict = {}
    for row in test_qrels:
        qid = row['query_id']
        cid = row['corpus_id']
        if qid not in qrel_dict:
            qrel_dict[qid] = []
        qrel_dict[qid].append(cid)
        
    qids = [qid for qid in qrel_dict if qid in queries_en and qid in queries_id][:200]
    
    en_scores = []
    id_scores = []
    
    for qid in tqdm(qids, desc="Reranking positive pairs"):
        cid = qrel_dict[qid][0] # take first positive
        doc_text = corpus_dict[cid][:1024]
        
        en_q = queries_en[qid]
        id_q = queries_id[qid]
        
        # score returns a 1D array
        en_score = reranker.score(en_q, [doc_text])[0]
        id_score = reranker.score(id_q, [doc_text])[0]
        
        en_scores.append(en_score)
        id_scores.append(id_score)
        
    mean_en = np.mean(en_scores)
    mean_id = np.mean(id_scores)
    
    logger.info(f"Mean Reranker Score for Positive Docs (EN): {mean_en:.4f}")
    logger.info(f"Mean Reranker Score for Positive Docs (ID): {mean_id:.4f}")
    logger.info(f"Gap: {mean_en - mean_id:.4f}")

def main():
    logger.info("Loading Data...")
    queries_corpus = load_dataset("CoIR-Retrieval/cosqa-queries-corpus")
    qrels = load_dataset("CoIR-Retrieval/cosqa-qrels")
    
    queries_en = {q['_id']: q['text'] for q in queries_corpus['queries']}
    queries_id = load_indonesian_translations()
    
    if len(queries_id) == 0:
        logger.error("No Indonesian translations loaded. Generating a dummy sample to avoid crash.")
        # fallback
        queries_id = {k: "dummy translation for " + v for k, v in list(queries_en.items())[:5000]}
    
    evaluate_translation_overlap(queries_en, queries_id)
    
    retriever = Retriever(model_name="intfloat/multilingual-e5-small")
    evaluate_embedding_alignment(queries_en, queries_id, retriever)
    
    model_path = str(FULL_PATH / "models" / "cross-encoder-me5-small-full-hn-v1")
    if Path(model_path).exists():
        check_reranker_sensitivity(queries_en, queries_id, qrels, queries_corpus, model_path)
    else:
        logger.warning(f"Reranker model not found at {model_path}. Skipping.")

if __name__ == "__main__":
    main()
