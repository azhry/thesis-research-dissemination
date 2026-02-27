"""
Pseudo-Relevance Feedback (PRF) based Query Expansion.
Uses top-retrieved documents to extract expansion terms.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from collections import Counter
import re

from .llm_expander import ExpansionResult

logger = logging.getLogger(__name__)


class PRFExpander:
    """
    Pseudo-Relevance Feedback Query Expander.
    
    Uses the top-retrieved documents to extract expansion terms.
    This is adapted for cross-lingual settings where initial retrieval
    is done with translated or expanded queries.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        num_terms: int = 10,
        use_frequency: bool = True,
        use_bert_score: bool = False,
    ):
        """
        Initialize the PRF expander.
        
        Args:
            top_k: Number of top documents to use for feedback
            num_terms: Number of terms to extract
            use_frequency: Use term frequency for scoring
            use_bert_score: Use BERT-based relevance scoring
        """
        self.top_k = top_k
        self.num_terms = num_terms
        self.use_frequency = use_frequency
        self.use_bert_score = use_bert_score
        
        # Common stopwords to filter
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load stopwords for filtering."""
        return {
            # English stopwords
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "this", "that", "these", "those", "what", "which",
            "who", "whom", "its", "it", "i", "you", "he", "she", "we", "they",
            # Programming keywords
            "def", "class", "function", "return", "import", "from", "if", "else",
            "elif", "for", "while", "try", "except", "finally", "with", "as",
            "pass", "break", "continue", "and", "or", "not", "in", "is", "True", "False",
            # Generic
            "null", "none", "void", "int", "float", "string", "bool", "list", "dict",
        }
    
    def expand(
        self,
        query: str,
        retrieved_docs: List[Dict[str, str]],
    ) -> ExpansionResult:
        """
        Expand query using PRF.
        
        Args:
            query: Original query
            retrieved_docs: List of retrieved documents with "text" field
            
        Returns:
            ExpansionResult with expanded query
        """
        if not retrieved_docs:
            logger.warning("No documents provided for PRF")
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                method="prf",
                metadata={"error": "No documents provided"}
            )
        
        # Get top-k documents
        top_docs = retrieved_docs[:self.top_k]
        
        # Extract terms from top documents
        term_scores = self._extract_terms(top_docs)
        
        # Get top expansion terms
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, score in top_terms[:self.num_terms]]
        
        # Create expanded query
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            method="prf",
            metadata={
                "term_scores": dict(top_terms[:self.num_terms]),
                "num_docs_used": len(top_docs),
            }
        )
    
    def _extract_terms(
        self,
        documents: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Extract terms from documents with scoring.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary of term -> score
        """
        term_counts = Counter()
        doc_count = len(documents)
        
        for doc in documents:
            text = doc.get("text", "") or doc.get("code", "") or ""
            
            # Tokenize
            tokens = self._tokenize(text)
            
            # Count terms (unique per document to avoid bias)
            unique_tokens = set(tokens)
            term_counts.update(unique_tokens)
        
        # Calculate scores
        if self.use_frequency:
            # TF-IDF-like scoring
            term_scores = {}
            for term, count in term_counts.items():
                # Term frequency * inverse document frequency
                tf = count / len(documents)
                idf = np.log(doc_count / (count + 1)) + 1
                term_scores[term] = tf * idf
        else:
            # Simple frequency
            term_scores = {term: count for term, count in term_counts.items()}
        
        # Filter stopwords and short terms
        term_scores = {
            term: score
            for term, score in term_scores.items()
            if term.lower() not in self.stopwords and len(term) > 2
        }
        
        return term_scores
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep underscores and alphanumeric
        text = re.sub(r'[^a-z0-9\s_]', ' ', text)
        
        # Split by whitespace
        tokens = text.split()
        
        return tokens


class CrossLingualPRFExpander(PRFExpander):
    """
    Cross-Lingual Pseudo-Relevance Feedback.
    
    Adapts PRF for cross-lingual settings where documents are in English
    and queries are in Indonesian.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        num_terms: int = 10,
        translation_model: Optional[Any] = None,
    ):
        """
        Initialize cross-lingual PRF expander.
        
        Args:
            top_k: Number of top documents
            num_terms: Number of terms to extract
            translation_model: Optional translation model
        """
        super().__init__(top_k, num_terms)
        self.translation_model = translation_model
    
    def expand(
        self,
        query: str,
        retrieved_docs: List[Dict[str, str]],
        translated_query: Optional[str] = None,
    ) -> ExpansionResult:
        """
        Expand query using cross-lingual PRF.
        
        Args:
            query: Original Indonesian query
            retrieved_docs: Retrieved English documents
            translated_query: Optional translated query for reference
            
        Returns:
            ExpansionResult
        """
        if not retrieved_docs:
            logger.warning("No documents provided for cross-lingual PRF")
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                method="cross_lingual_prf",
                metadata={"error": "No documents provided"}
            )
        
        # Extract English terms
        term_scores = self._extract_terms(retrieved_docs)
        
        # Get top terms
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, score in top_terms[:self.num_terms]]
        
        # Create expanded query - keep original Indonesian + add English terms
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            method="cross_lingual_prf",
            metadata={
                "term_scores": dict(top_terms[:self.num_terms]),
                "translated_query": translated_query,
                "num_docs_used": len(retrieved_docs),
            }
        )


class RM3Expander(PRFExpander):
    """
    RM3 (Relevance Model 3) inspired Query Expander.
    
    A more sophisticated PRF approach that uses relevance models.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        num_terms: int = 10,
        lambda_mix: float = 0.5,
    ):
        """
        Initialize RM3-style expander.
        
        Args:
            top_k: Number of top documents
            num_terms: Number of terms
            lambda_mix: Mixing parameter for combining original and expanded
        """
        super().__init__(top_k, num_terms)
        self.lambda_mix = lambda_mix
    
    def expand(
        self,
        query: str,
        retrieved_docs: List[Dict[str, str]],
        original_scores: Optional[List[float]] = None,
    ) -> ExpansionResult:
        """
        Expand using RM3-style scoring.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            original_scores: Original retrieval scores
            
        Returns:
            ExpansionResult
        """
        if not retrieved_docs:
            return ExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                method="rm3",
                metadata={"error": "No documents"}
            )
        
        # Get document probabilities
        doc_probs = self._get_document_probabilities(original_scores)
        
        # Extract term probabilities from relevant documents
        term_probs = self._get_term_probabilities(retrieved_docs, doc_probs)
        
        # Get expansion terms
        top_terms = sorted(term_probs.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, prob in top_terms[:self.num_terms]]
        
        # Mix original query with expansion terms
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            method="rm3",
            metadata={
                "term_probs": dict(top_terms[:self.num_terms]),
                "lambda_mix": self.lambda_mix,
            }
        )
    
    def _get_document_probabilities(
        self,
        scores: Optional[List[float]],
    ) -> np.ndarray:
        """Get document probabilities from scores."""
        if scores is None:
            n = len(self.top_k)
            return np.ones(n) / n
        
        scores = np.array(scores)
        # Softmax-like normalization
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        return probs
    
    def _get_term_probabilities(
        self,
        documents: List[Dict[str, str]],
        doc_probs: np.ndarray,
    ) -> Dict[str, float]:
        """Get term probabilities from documents."""
        term_doc_count = Counter()
        total_terms = 0
        
        for i, doc in enumerate(documents):
            text = doc.get("text", "") or doc.get("code", "") or ""
            tokens = self._tokenize(text)
            
            # Count unique terms per document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token.lower() not in self.stopwords and len(token) > 2:
                    term_doc_count[token] += 1
            
            total_terms += len(tokens)
        
        # Calculate probability
        term_probs = {}
        for term, count in term_doc_count.items():
            # P(term|relevant) * P(relevant)
            p_term_given_relevant = count / len(documents)
            p_term = (count + 0.01) / (total_terms + 1)
            
            # Smoothed mixture
            term_probs[term] = (
                self.lambda_mix * p_term_given_relevant +
                (1 - self.lambda_mix) * p_term
            )
        
        return term_probs
