"""
Embedding-based Query Expansion module.
Uses multilingual embeddings to find related terms for query expansion.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .llm_expander import ExpansionResult

logger = logging.getLogger(__name__)


class EmbeddingExpander:
    """
    Embedding-based Query Expander.
    
    Uses multilingual embeddings to find semantically similar terms
    for expanding Indonesian queries to English terms.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding expander.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Load vocabulary for term expansion
        self._term_vocabulary = self._load_term_vocabulary()
    
    def _load_term_vocabulary(self) -> List[str]:
        """
        Load vocabulary of technical terms for expansion.
        
        Returns:
            List of technical terms
        """
        # Common programming/technical terms in English
        # In production, this could be loaded from a file or database
        technical_terms = [
            # General programming
            "function", "class", "method", "variable", "parameter", "return",
            "loop", "condition", "array", "list", "dictionary", "string", "integer",
            "boolean", "float", "object", "exception", "error", "debug",
            
            # Web development
            "http", "request", "response", "api", "endpoint", "route", "controller",
            "database", "sql", "query", "post", "get", "put", "delete",
            "json", "xml", "html", "css", "javascript", "frontend", "backend",
            
            # Data science
            "pandas", "numpy", "sklearn", "tensorflow", "pytorch", "machine learning",
            "neural network", "deep learning", "model", "training", "prediction",
            "feature", "label", "dataset", "accuracy", "loss", "optimizer",
            
            # Python specific
            "import", "def", "return", "print", "len", "range", "enumerate",
            "list comprehension", "dictionary comprehension", "lambda", "decorator",
            "self", "init", "py", "pip", "venv", "virtualenv",
            
            # File operations
            "read", "write", "open", "close", "file", "path", "directory",
            "csv", "txt", "json", "pickle", "yaml", "config",
            
            # Testing
            "test", "unittest", "pytest", "assert", "mock", "fixture",
            "coverage", "assertion", "failure", "success",
            
            # Version control
            "git", "commit", "push", "pull", "merge", "branch", "repository",
            
            # Indonesian technical terms (for mapping)
            "fungsi", "kelas", "metode", "variabel", "parameter", "mengembalikan",
            "perulangan", "kondisi", "array", "daftar", "kamus", "teks", "bilangan",
            "boolean", "objek", "kesalahan", "debbuging",
        ]
        
        return technical_terms
    
    def expand(
        self,
        query: str,
        num_terms: int = 5,
    ) -> ExpansionResult:
        """
        Expand a query using embeddings.
        
        Args:
            query: Input query
            num_terms: Number of terms to expand with
            
        Returns:
            ExpansionResult with expanded query
        """
        # Get embedding for the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False,
        )
        
        # Get embeddings for all terms in vocabulary
        term_embeddings = self.model.encode(
            self._term_vocabulary,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False,
        )
        
        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_embedding, term_embeddings)[0]
        
        # Get top-k most similar terms
        top_k_indices = np.argsort(similarities)[-num_terms:][::-1]
        
        # Get the expansion terms
        expansion_terms = [
            self._term_vocabulary[i]
            for i in top_k_indices
            if similarities[i] > 0  # Only positive similarities
        ][:num_terms]
        
        # Create expanded query
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            method="embedding",
            metadata={
                "similarities": [float(similarities[i]) for i in top_k_indices],
                "model": self.model_name,
            }
        )
    
    def expand_batch(
        self,
        queries: List[str],
        num_terms: int = 5,
    ) -> List[ExpansionResult]:
        """
        Expand multiple queries.
        
        Args:
            queries: List of input queries
            num_terms: Number of terms per query
            
        Returns:
            List of ExpansionResults
        """
        results = []
        for query in queries:
            result = self.expand(query, num_terms)
            results.append(result)
        
        return results
    
    def _cosine_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (shape: N x D)
            embeddings2: Second set of embeddings (shape: M x D)
            
        Returns:
            Similarity matrix (shape: N x M)
        """
        # Normalize embeddings
        norm1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Calculate similarity
        return np.dot(norm1, norm2.T)


class CrossLingualEmbeddingExpander(EmbeddingExpander):
    """
    Cross-lingual embedding expander that uses parallel vocabularies.
    
    Specifically designed for Indonesian-to-English expansion.
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: Optional[str] = None,
    ):
        """
        Initialize cross-lingual embedding expander.
        
        Args:
            model_name: Name of the multilingual model
            device: Device to use
        """
        super().__init__(model_name, device)
        self._term_vocabulary = self._load_bilingual_vocabulary()
    
    def _load_bilingual_vocabulary(self) -> List[str]:
        """
        Load bilingual (Indonesian-English) vocabulary.
        
        Returns:
            List of English technical terms
        """
        # Indonesian to English technical term mappings
        # These are common terms Indonesian developers use
        indonesian_to_english = {
            # Basic programming
            "fungsi": "function",
            "kelas": "class",
            "metode": "method",
            "variabel": "variable",
            "parameter": "parameter",
            "mengembalikan": "return",
            "perulangan": "loop",
            "kondisi": "condition",
            
            # Data structures
            "array": "array",
            "daftar": "list",
            "kamus": "dictionary",
            "teks": "string",
            "bilangan": "integer",
            "boolean": "boolean",
            
            # File operations
            "baca": "read",
            "tulis": "write",
            "buka": "open",
            "tutup": "close",
            "file": "file",
            
            # Web/Database
            "data": "data",
            "database": "database",
            "tabel": "table",
            "baris": "row",
            "kolom": "column",
            "query": "query",
            
            # Common programming actions
            "simpan": "save",
            "ambil": "get",
            "hapus": "delete",
            "ubah": "update",
            "tambah": "add",
            "cari": "search",
            "urut": "sort",
            "filter": "filter",
            
            # Error handling
            "error": "error",
            "kesalahan": "error",
            "peringatan": "warning",
        }
        
        # Get unique English terms
        return list(set(indonesian_to_english.values()))
    
    def expand(
        self,
        query: str,
        num_terms: int = 5,
    ) -> ExpansionResult:
        """
        Expand an Indonesian query using cross-lingual embeddings.
        
        Args:
            query: Input query (Indonesian)
            num_terms: Number of terms to expand with
            
        Returns:
            ExpansionResult with expanded query
        """
        # First, try to identify Indonesian terms in the query
        indonesian_terms = self._identify_indonesian_terms(query)
        
        # Get embedding for the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False,
        )
        
        # Get embeddings for vocabulary terms
        term_embeddings = self.model.encode(
            self._term_vocabulary,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False,
        )
        
        # Calculate similarity
        similarities = self._cosine_similarity(query_embedding, term_embeddings)[0]
        
        # Get top-k most similar terms
        top_k_indices = np.argsort(similarities)[-num_terms:][::-1]
        
        expansion_terms = [
            self._term_vocabulary[i]
            for i in top_k_indices
            if similarities[i] > 0.1  # Threshold for relevance
        ][:num_terms]
        
        # Add any direct translations found
        for indo_term in indonesian_terms:
            eng_term = self._translate_term(indo_term)
            if eng_term and eng_term not in expansion_terms:
                expansion_terms.insert(0, eng_term)
        
        # Create expanded query
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            method="cross_lingual_embedding",
            metadata={
                "indonesian_terms_found": indonesian_terms,
                "model": self.model_name,
            }
        )
    
    def _identify_indonesian_terms(self, query: str) -> List[str]:
        """Identify Indonesian technical terms in the query."""
        query_lower = query.lower()
        found_terms = []
        
        # Common Indonesian technical terms
        indo_terms = [
            "fungsi", "kelas", "metode", "variabel", "parameter", "loop",
            "array", "daftar", "kamus", "string", "integer", "boolean",
            "baca", "tulis", "buka", "tutup", "file", "data", "database",
            "tabel", "baris", "kolom", "query", "simpan", "ambil", "hapus",
            "ubah", "tambah", "cari", "urut", "filter", "error", "kesalahan",
        ]
        
        for term in indo_terms:
            if term in query_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _translate_term(self, indonesian_term: str) -> Optional[str]:
        """Translate Indonesian term to English."""
        mapping = {
            "fungsi": "function",
            "kelas": "class",
            "metode": "method",
            "variabel": "variable",
            "parameter": "parameter",
            "loop": "loop",
            "array": "array",
            "daftar": "list",
            "kamus": "dictionary",
            "string": "string",
            "integer": "integer",
            "boolean": "boolean",
            "baca": "read",
            "tulis": "write",
            "buka": "open",
            "tutup": "close",
            "file": "file",
            "data": "data",
            "database": "database",
            "tabel": "table",
            "baris": "row",
            "kolom": "column",
            "query": "query",
            "simpan": "save",
            "ambil": "get",
            "hapus": "delete",
            "ubah": "update",
            "tambah": "add",
            "cari": "search",
            "urut": "sort",
            "filter": "filter",
            "error": "error",
            "kesalahan": "error",
        }
        
        return mapping.get(indonesian_term.lower())
