"""
Query Expansion Combiner.
Combines multiple query expansion methods for better results.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .llm_expander import ExpansionResult, LLMExpander
from .embedding_expander import EmbeddingExpander
from .prf_expander import PRFExpander

logger = logging.getLogger(__name__)


@dataclass
class CombinedExpansionResult:
    """Result of combined query expansion."""
    original_query: str
    expanded_query: str
    expansion_terms: List[str]
    method: str
    metadata: Dict
    individual_results: List[ExpansionResult]


class QECombiner:
    """
    Combines multiple query expansion methods.
    
    Supports different combination strategies:
    - Union: Combine all terms from all methods
    - Intersection: Keep only common terms
    - Weighted: Weight terms by method confidence
    - Sequential: Apply methods one after another
    """
    
    def __init__(
        self,
        methods: List[str] = ["llm", "embedding"],
        combination_strategy: str = "union",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the combiner.
        
        Args:
            methods: List of methods to combine
            combination_strategy: Strategy for combining ("union", "intersection", "weighted", "sequential")
            weights: Weights for each method (for weighted strategy)
        """
        self.methods = methods
        self.combination_strategy = combination_strategy
        self.weights = weights or {m: 1.0 for m in methods}
        
        # Initialize expanders
        self.expanders = {}
        self._initialize_expanders()
    
    def _initialize_expanders(self):
        """Initialize the expanders for each method."""
        if "llm" in self.methods:
            try:
                self.expanders["llm"] = LLMExpander()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM expander: {e}")
        
        if "embedding" in self.methods:
            try:
                self.expanders["embedding"] = EmbeddingExpander()
            except Exception as e:
                logger.warning(f"Failed to initialize embedding expander: {e}")
        
        if "prf" in self.methods:
            self.expanders["prf"] = PRFExpander()
    
    def expand(
        self,
        query: str,
        method_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> CombinedExpansionResult:
        """
        Expand query using combined methods.
        
        Args:
            query: Input query
            method_kwargs: Keyword arguments for each method
            
        Returns:
            CombinedExpansionResult
        """
        method_kwargs = method_kwargs or {}
        results = []
        
        # Get individual expansions
        for method in self.methods:
            if method not in self.expanders:
                continue
            
            expander = self.expanders[method]
            kwargs = method_kwargs.get(method, {})
            
            try:
                if method == "prf":
                    # PRF needs documents, skip for now
                    continue
                
                result = expander.expand(query, **kwargs)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in {method} expansion: {e}")
        
        # Combine results
        combined = self._combine_results(query, results)
        
        return combined
    
    def expand_with_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, str]],
        method_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> CombinedExpansionResult:
        """
        Expand query with retrieval context (includes PRF).
        
        Args:
            query: Input query
            retrieved_docs: Retrieved documents for PRF
            method_kwargs: Keyword arguments for each method
            
        Returns:
            CombinedExpansionResult
        """
        method_kwargs = method_kwargs or {}
        results = []
        
        # Get individual expansions
        for method in self.methods:
            if method not in self.expanders:
                continue
            
            expander = self.expanders[method]
            kwargs = method_kwargs.get(method, {})
            
            try:
                if method == "prf":
                    # PRF needs documents
                    result = expander.expand(query, retrieved_docs, **kwargs)
                else:
                    result = expander.expand(query, **kwargs)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in {method} expansion: {e}")
        
        # Combine results
        combined = self._combine_results(query, results)
        
        return combined
    
    def _combine_results(
        self,
        query: str,
        results: List[ExpansionResult],
    ) -> CombinedExpansionResult:
        """Combine multiple expansion results."""
        
        if not results:
            return CombinedExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                method="combined",
                metadata={"error": "No results to combine"},
                individual_results=[],
            )
        
        if self.combination_strategy == "union":
            return self._union_strategy(query, results)
        elif self.combination_strategy == "intersection":
            return self._intersection_strategy(query, results)
        elif self.combination_strategy == "weighted":
            return self._weighted_strategy(query, results)
        elif self.combination_strategy == "sequential":
            return self._sequential_strategy(query, results)
        else:
            # Default to union
            return self._union_strategy(query, results)
    
    def _union_strategy(
        self,
        query: str,
        results: List[ExpansionResult],
    ) -> CombinedExpansionResult:
        """Union: Combine all terms from all methods."""
        all_terms = []
        
        for result in results:
            all_terms.extend(result.expansion_terms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        expanded_query = f"{query} {' '.join(unique_terms)}"
        
        return CombinedExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=unique_terms,
            method=f"combined_union_{len(results)}_methods",
            metadata={"strategies": [r.method for r in results]},
            individual_results=results,
        )
    
    def _intersection_strategy(
        self,
        query: str,
        results: List[ExpansionResult],
    ) -> CombinedExpansionResult:
        """Intersection: Keep only terms common to all methods."""
        if not results:
            return CombinedExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                method="combined_intersection",
                metadata={},
                individual_results=results,
            )
        
        # Get term sets
        term_sets = [set(t.lower() for t in r.expansion_terms) for r in results]
        
        # Intersect all sets
        common_terms = term_sets[0]
        for term_set in term_sets[1:]:
            common_terms = common_terms.intersection(term_set)
        
        # Convert back to list (preserve original case from first result)
        term_list = list(common_terms)
        
        # Preserve original case
        final_terms = []
        for result in results:
            for term in result.expansion_terms:
                if term.lower() in common_terms and term not in final_terms:
                    final_terms.append(term)
        
        expanded_query = f"{query} {' '.join(final_terms)}"
        
        return CombinedExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=final_terms,
            method=f"combined_intersection_{len(results)}_methods",
            metadata={"strategies": [r.method for r in results]},
            individual_results=results,
        )
    
    def _weighted_strategy(
        self,
        query: str,
        results: List[ExpansionResult],
    ) -> CombinedExpansionResult:
        """Weighted: Weight terms by method confidence."""
        term_scores = {}
        
        for result in results:
            method = result.method
            weight = self.weights.get(method, 1.0)
            
            # Use metadata for scoring if available
            if "similarities" in result.metadata:
                for term, score in zip(
                    result.expansion_terms,
                    result.metadata.get("similarities", [])
                ):
                    if term not in term_scores:
                        term_scores[term] = 0
                    term_scores[term] += score * weight
            else:
                # Equal weight for each term
                for term in result.expansion_terms:
                    if term not in term_scores:
                        term_scores[term] = 0
                    term_scores[term] += weight
        
        # Sort by score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        final_terms = [term for term, score in sorted_terms]
        
        expanded_query = f"{query} {' '.join(final_terms)}"
        
        return CombinedExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=final_terms,
            method=f"combined_weighted_{len(results)}_methods",
            metadata={
                "strategies": [r.method for r in results],
                "weights": self.weights,
                "term_scores": dict(sorted_terms[:20]),
            },
            individual_results=results,
        )
    
    def _sequential_strategy(
        self,
        query: str,
        results: List[ExpansionResult],
    ) -> CombinedExpansionResult:
        """Sequential: Apply methods one after another."""
        if not results:
            return CombinedExpansionResult(
                original_query=query,
                expanded_query=query,
                expansion_terms=[],
                method="combined_sequential",
                metadata={},
                individual_results=results,
            )
        
        # Use the last result (most refined)
        last_result = results[-1]
        
        # But also include terms from earlier results
        all_terms = []
        for result in results:
            for term in result.expansion_terms:
                if term not in all_terms:
                    all_terms.append(term)
        
        expanded_query = last_result.expanded_query
        
        return CombinedExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=all_terms,
            method=f"combined_sequential_{len(results)}_methods",
            metadata={
                "strategies": [r.method for r in results],
                "final_method": last_result.method,
            },
            individual_results=results,
        )


class SequentialQE:
    """
    Sequential Query Expansion pipeline.
    
    Applies query expansion methods in sequence:
    1. LLM expansion (if available)
    2. PRF expansion (if documents available)
    3. Return final expanded query
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        use_embedding: bool = True,
        use_prf: bool = True,
    ):
        """
        Initialize sequential QE.
        
        Args:
            use_llm: Use LLM-based expansion
            use_embedding: Use embedding-based expansion
            use_prf: Use PRF expansion
        """
        self.use_llm = use_llm
        self.use_embedding = use_embedding
        self.use_prf = use_prf
        
        # Initialize expanders
        self._init_expanders()
    
    def _init_expanders(self):
        """Initialize expanders."""
        self.expanders = {}
        
        if self.use_llm:
            try:
                self.expanders["llm"] = LLMExpander()
            except Exception as e:
                logger.warning(f"LLM expander unavailable: {e}")
        
        if self.use_embedding:
            try:
                self.expanders["embedding"] = EmbeddingExpander()
            except Exception as e:
                logger.warning(f"Embedding expander unavailable: {e}")
        
        if self.use_prf:
            self.expanders["prf"] = PRFExpander()
    
    def expand(
        self,
        query: str,
        retrieved_docs: Optional[List[Dict[str, str]]] = None,
    ) -> CombinedExpansionResult:
        """
        Expand query sequentially.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents (for PRF)
            
        Returns:
            CombinedExpansionResult
        """
        current_query = query
        all_results = []
        
        # Step 1: LLM expansion
        if "llm" in self.expanders:
            try:
                result = self.expanders["llm"].expand(query)
                current_query = result.expanded_query
                all_results.append(result)
            except Exception as e:
                logger.error(f"LLM expansion failed: {e}")
        
        # Step 2: Embedding expansion
        if "embedding" in self.expanders:
            try:
                result = self.expanders["embedding"].expand(current_query)
                current_query = result.expanded_query
                all_results.append(result)
            except Exception as e:
                logger.error(f"Embedding expansion failed: {e}")
        
        # Step 3: PRF expansion
        if "prf" in self.expanders and retrieved_docs:
            try:
                result = self.expanders["prf"].expand(current_query, retrieved_docs)
                current_query = result.expanded_query
                all_results.append(result)
            except Exception as e:
                logger.error(f"PRF expansion failed: {e}")
        
        # Combine all terms
        all_terms = []
        for result in all_results:
            for term in result.expansion_terms:
                if term.lower() not in [t.lower() for t in all_terms]:
                    all_terms.append(term)
        
        return CombinedExpansionResult(
            original_query=query,
            expanded_query=current_query,
            expansion_terms=all_terms,
            method="sequential_qe",
            metadata={
                "steps": [r.method for r in all_results],
            },
            individual_results=all_results,
        )
