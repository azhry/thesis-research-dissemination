"""
Dense Retriever using Multilingual E5 (mE5) embeddings.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense Retriever using Multilingual E5 embeddings.
    
    Supports:
    - mE5-small, mE5-base, mE5-large-instruct
    - Cross-lingual retrieval (Indonesian query -> English code)
    - Batch processing for efficiency
    """
    
    # Instruction prefixes for mE5
    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: Name of the mE5 model
            device: Device to use ("cuda" or "cpu")
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Cache for encoded corpus
        self.corpus_embeddings = None
        self.corpus_ids = None
    
    def encode_queries(
        self,
        queries: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode queries into embeddings.
        
        Args:
            queries: List of query strings
            show_progress: Show progress bar
            
        Returns:
            Query embeddings (num_queries x embedding_dim)
        """
        # Add query prefix
        prefixed_queries = [self.QUERY_PREFIX + q for q in queries]
        
        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )
        
        return embeddings
    
    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode corpus documents into embeddings.
        
        Args:
            corpus: List of documents with "id" and "text" fields
            show_progress: Show progress bar
            
        Returns:
            Tuple of (embeddings, ids)
        """
        # Extract text and IDs
        texts = [doc.get("text", "") or doc.get("code", "") for doc in corpus]
        doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(corpus)]
        
        # Add passage prefix
        prefixed_texts = [self.PASSAGE_PREFIX + t for t in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )
        
        # Cache embeddings
        self.corpus_embeddings = embeddings
        self.corpus_ids = doc_ids
        
        return embeddings, doc_ids
    
    def retrieve(
        self,
        queries: List[str],
        corpus: List[Dict[str, str]],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> List[List[Dict[str, any]]]:
        """
        Retrieve top-k documents for each query.
        
        Args:
            queries: List of queries
            corpus: List of documents
            top_k: Number of top documents to retrieve
            show_progress: Show progress bar
            
        Returns:
            List of retrieved results for each query
        """
        # Encode queries
        logger.info(f"Encoding {len(queries)} queries...")
        query_embeddings = self.encode_queries(queries, show_progress=show_progress)
        
        # Encode corpus (use cache if available)
        if self.corpus_embeddings is None:
            logger.info(f"Encoding {len(corpus)} corpus documents...")
            corpus_embeddings, doc_ids = self.encode_corpus(corpus, show_progress=show_progress)
        else:
            logger.info("Using cached corpus embeddings")
            corpus_embeddings = self.corpus_embeddings
            doc_ids = self.corpus_ids
        
        # Calculate similarities
        logger.info("Calculating similarities...")
        similarities = np.matmul(query_embeddings, corpus_embeddings.T)
        
        # Get top-k for each query
        results = []
        for i, query in enumerate(tqdm(queries, desc="Retrieving", disable=not show_progress)):
            scores = similarities[i]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Build result list
            result = []
            for idx in top_indices:
                result.append({
                    "id": doc_ids[idx],
                    "score": float(scores[idx]),
                    "rank": len(result) + 1,
                })
            
            results.append(result)
        
        return results
    
    def retrieve_with_expanded_query(
        self,
        expanded_queries: List[str],
        original_queries: List[str],
        corpus: List[Dict[str, str]],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> List[List[Dict[str, any]]]:
        """
        Retrieve using expanded queries.
        
        Args:
            expanded_queries: List of expanded queries
            original_queries: List of original queries
            corpus: List of documents
            top_k: Number of top documents
            show_progress: Show progress bar
            
        Returns:
            List of retrieved results
        """
        # Encode expanded queries
        logger.info(f"Encoding {len(expanded_queries)} expanded queries...")
        query_embeddings = self.encode_queries(expanded_queries, show_progress=show_progress)
        
        # Use cached corpus embeddings
        if self.corpus_embeddings is None:
            logger.info(f"Encoding {len(corpus)} corpus documents...")
            corpus_embeddings, doc_ids = self.encode_corpus(corpus, show_progress=show_progress)
        else:
            corpus_embeddings = self.corpus_embeddings
            doc_ids = self.corpus_ids
        
        # Calculate similarities
        similarities = np.matmul(query_embeddings, corpus_embeddings.T)
        
        # Get top-k
        results = []
        for i in tqdm(range(len(expanded_queries)), desc="Retrieving", disable=not show_progress):
            scores = similarities[i]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            result = []
            for idx in top_indices:
                result.append({
                    "id": doc_ids[idx],
                    "score": float(scores[idx]),
                    "rank": len(result) + 1,
                    "original_query": original_queries[i],
                    "expanded_query": expanded_queries[i],
                })
            
            results.append(result)
        
        return results
    
    def get_documents(
        self,
        doc_ids: List[str],
        corpus: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Get document contents by IDs.
        
        Args:
            doc_ids: List of document IDs
            corpus: Full corpus
            
        Returns:
            List of documents
        """
        id_to_doc = {doc.get("id", str(i)): doc for i, doc in enumerate(corpus)}
        
        return [id_to_doc.get(doc_id, {}) for doc_id in doc_ids]
    
    def encode_corpus_cached(
        self,
        corpus: List[Dict[str, str]],
        cache_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode and cache corpus embeddings.
        
        Args:
            corpus: List of documents
            cache_path: Optional path to save cached embeddings
            
        Returns:
            Tuple of (embeddings, ids)
        """
        if cache_path and self.corpus_embeddings is not None:
            logger.info(f"Loading cached embeddings from {cache_path}")
            self.corpus_embeddings = np.load(cache_path)
            return self.corpus_embeddings, self.corpus_ids
        
        embeddings, ids = self.encode_corpus(corpus)
        
        if cache_path:
            logger.info(f"Saving embeddings to {cache_path}")
            np.save(cache_path, embeddings)
        
        return embeddings, ids


class BM25Retriever:
    """
    BM25 Retriever for baseline comparison.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: BM25 parameter k1
            b: BM25 parameter b
        """
        self.k1 = k1
        self.b = b
        self.corpus = None
        self.doc_len = None
        self.avgdl = None
        self.doc_freqs = None
        self.idf = None
        self.vocab = None
    
    def fit(self, corpus: List[Dict[str, str]]):
        """
        Fit BM25 on corpus.
        
        Args:
            corpus: List of documents
        """
        self.corpus = corpus
        self.doc_len = []
        self.doc_freqs = []
        self.vocab = {}
        
        # Count document frequencies
        for doc in corpus:
            text = doc.get("text", "") or doc.get("code", "")
            tokens = text.lower().split()
            self.doc_len.append(len(tokens))
            
            freqs = {}
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                freqs[token] = freqs.get(token, 0) + 1
            
            self.doc_freqs.append(freqs)
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # Calculate IDF
        N = len(corpus)
        self.idf = {}
        df = Counter()
        for freqs in self.doc_freqs:
            for token in freqs:
                df[token] += 1
        
        for token, freq in df.items():
            self.idf[token] = np.log((N - freq + 0.5) / (freq + 0.5) + 1)
    
    def retrieve(
        self,
        queries: List[str],
        top_k: int = 100,
    ) -> List[List[Dict[str, any]]]:
        """
        Retrieve documents using BM25.
        
        Args:
            queries: List of queries
            top_k: Number of top documents
            
        Returns:
            List of retrieved results
        """
        results = []
        
        for query in queries:
            query_tokens = query.lower().split()
            scores = []
            
            for i, doc in enumerate(self.corpus):
                score = self._score(query_tokens, i)
                scores.append((i, score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k
            result = []
            for rank, (idx, score) in enumerate(scores[:top_k]):
                result.append({
                    "id": self.corpus[idx].get("id", str(idx)),
                    "score": float(score),
                    "rank": rank + 1,
                })
            
            results.append(result)
        
        return results
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a query and document."""
        doc_len = self.doc_len[doc_idx]
        freqs = self.doc_freqs[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token not in freqs:
                continue
            
            freq = freqs[token]
            idf = self.idf.get(token, 0)
            
            # BM25 formula
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        
        return score
