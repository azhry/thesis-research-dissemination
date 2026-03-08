"""
Dataset Loader for the full pipeline.

Loads the CoSQA dataset with Indonesian translations.
Reuses the coir data_loader from the QE experiment.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _setup_coir_path():
    """Add the coir module path to sys.path."""
    coir_path = Path(__file__).parent.parent.parent / "qe" / "coir"
    if str(coir_path) not in sys.path:
        sys.path.insert(0, str(coir_path.parent))  # Add qe/ so 'coir.X' works
    return coir_path


def load_indonesian_translations() -> Dict[str, str]:
    """
    Load Indonesian translations from the CSV file.
    
    Returns:
        Dict mapping query ID to Indonesian query text
    """
    translations_file = Path(__file__).parent.parent.parent / "qe" / "cosqa_queries_indonesian.csv"
    if translations_file.exists():
        trans_df = pd.read_csv(translations_file, sep="|")
        translations = dict(zip(trans_df['qid'], trans_df['query_id']))
        logger.info(f"Loaded {len(translations)} Indonesian translations")
        return translations
    else:
        logger.warning(f"Indonesian translations file not found: {translations_file}")
        return {}


def load_cosqa_dataset(
    sample_size: Optional[int] = None,
) -> Tuple[Dict, Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load the CoSQA dataset with English and Indonesian queries.
    
    Args:
        sample_size: Optional limit on number of queries (for testing)
    
    Returns:
        Tuple of (corpus, queries_english, queries_indonesian, qrels)
        - corpus: Dict[str, Dict] mapping doc_id -> {text: str, title: str}
        - queries_english: Dict[str, str] mapping query_id -> English query text
        - queries_indonesian: Dict[str, str] mapping query_id -> Indonesian query text
        - qrels: Dict[str, Dict[str, int]] mapping query_id -> {doc_id: relevance}
    """
    _setup_coir_path()
    
    from coir.data_loader import load_data_from_hf
    
    logger.info("Loading CoSQA dataset from HuggingFace...")
    corpus, queries, qrels = load_data_from_hf("cosqa")
    
    logger.info(f"Loaded {len(queries)} queries and {len(corpus)} documents")
    
    # Load Indonesian translations
    translations = load_indonesian_translations()
    
    # Build English and Indonesian query dicts
    queries_english = {}
    queries_indonesian = {}
    
    for qid, qtext in queries.items():
        queries_english[qid] = qtext
        if qid in translations:
            queries_indonesian[qid] = translations[qid]
        else:
            # Fallback: use English if no translation available
            queries_indonesian[qid] = qtext
    
    # Apply sample size limit if specified
    if sample_size is not None and sample_size > 0:
        query_ids = list(queries_english.keys())[:sample_size]
        queries_english = {qid: queries_english[qid] for qid in query_ids}
        queries_indonesian = {qid: queries_indonesian[qid] for qid in query_ids}
        qrels = {qid: v for qid, v in qrels.items() if qid in queries_english}
        logger.info(f"Sampled {len(queries_english)} queries for testing")
    
    return corpus, queries_english, queries_indonesian, qrels


def corpus_to_list(corpus: Dict) -> List[Dict[str, str]]:
    """
    Convert corpus dict to list format for retrieval.
    
    Args:
        corpus: Dict[str, Dict] from load_cosqa_dataset
    
    Returns:
        List of dicts with 'id' and 'text' keys
    """
    corpus_list = []
    for doc_id, doc_data in corpus.items():
        corpus_list.append({
            "id": doc_id,
            "text": doc_data.get("text", ""),
            "title": doc_data.get("title", ""),
        })
    return corpus_list
