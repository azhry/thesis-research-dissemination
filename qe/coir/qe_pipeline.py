"""
Query Expansion Pipeline.
Complete pipeline for Indonesian code search with Query Expansion.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .config import QEConfig
from .llm_expander import LLMExpander
from .embedding_expander import EmbeddingExpander, CrossLingualEmbeddingExpander
from .prf_expander import PRFExpander, CrossLingualPRFExpander
from .combiner import QECombiner, SequentialQE
from .dense_retriever import DenseRetriever, BM25Retriever

logger = logging.getLogger(__name__)


@dataclass
class QEResult:
    """Result of QE pipeline."""
    query: str
    expanded_query: str
    expansion_method: str
    retrieved_docs: List[Dict[str, Any]]
    metrics: Dict[str, float]


class QEPipeline:
    """
    Complete Query Expansion Pipeline for Indonesian Code Search.
    
    Supports multiple configurations:
    - Baseline: No QE
    - LLM QE: LLM-based expansion
    - Embedding QE: Embedding-based expansion
    - PRF QE: Pseudo-relevance feedback
    - Combined: Multiple methods combined
    - Sequential: Sequential application
    """
    
    def __init__(self, config: Optional[QEConfig] = None):
        """
        Initialize the QE pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config or QEConfig()
        
        # Initialize components
        self.retriever = None
        self.expander = None
        self.bm25_retriever = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize retrieval and expansion components."""
        # Initialize retriever
        self.retriever = DenseRetriever(
            model_name=self.config.DEFAULT_EMBEDDING_MODEL,
            device=self.config.DEVICE,
            batch_size=self.config.RETRIEVAL_BATCH_SIZE,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
        )
        
        # Initialize BM25 baseline
        self.bm25_retriever = BM25Retriever()
        
        # Initialize expander based on method
        if self.config.EXPANSION_METHOD == "hyde":
            self.expander = LLMExpander(
                provider=self.config.LLM_PROVIDER,
                model=self.config.OPENAI_MODEL if self.config.LLM_PROVIDER == "openai" else self.config.GOOGLE_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
            )
        elif self.config.EXPANSION_METHOD == "embedding":
            self.expander = CrossLingualEmbeddingExpander(
                model_name=self.config.DEFAULT_EMBEDDING_MODEL,
                device=self.config.DEVICE,
            )
        elif self.config.EXPANSION_METHOD == "prf":
            self.expander = PRFExpander(
                top_k=self.config.TOP_K,
                num_terms=self.config.NUM_EXPANSION_TERMS,
            )
        elif self.config.EXPANSION_METHOD == "combined":
            self.expander = QECombiner(
                methods=["llm", "embedding"],
                combination_strategy="union",
            )
        elif self.config.EXPANSION_METHOD == "sequential":
            self.expander = SequentialQE(
                use_llm=True,
                use_embedding=True,
                use_prf=True,
            )
    
    def run_baseline(
        self,
        queries: List[str],
        corpus: List[Dict[str, str]],
        top_k: int = 100,
    ) -> List[QEResult]:
        """
        Run baseline retrieval (no QE).
        
        Args:
            queries: List of queries
            corpus: Document corpus
            top_k: Number of documents to retrieve
            
        Returns:
            List of QEResults
        """
        logger.info("Running baseline retrieval...")
        
        # Retrieve documents
        results = self.retriever.retrieve(
            queries=queries,
            corpus=corpus,
            top_k=top_k,
        )
        
        # Build return objects
        qe_results = []
        for query, retrieved in zip(queries, results):
            qe_results.append(QEResult(
                query=query,
                expanded_query=query,  # No expansion
                expansion_method="baseline",
                retrieved_docs=retrieved,
                metrics={},
            ))
        
        return qe_results
    
    def run_bm25_baseline(
        self,
        queries: List[str],
        corpus: List[Dict[str, str]],
        top_k: int = 100,
    ) -> List[QEResult]:
        """
        Run BM25 baseline.
        
        Args:
            queries: List of queries
            corpus: Document corpus
            top_k: Number of documents to retrieve
            
        Returns:
            List of QEResults
        """
        logger.info("Running BM25 baseline...")
        
        # Fit BM25
        self.bm25_retriever.fit(corpus)
        
        # Retrieve documents
        results = self.bm25_retriever.retrieve(
            queries=queries,
            top_k=top_k,
        )
        
        # Build return objects
        qe_results = []
        for query, retrieved in zip(queries, results):
            qe_results.append(QEResult(
                query=query,
                expanded_query=query,
                expansion_method="bm25",
                retrieved_docs=retrieved,
                metrics={},
            ))
        
        return qe_results
    
    def run_qe(
        self,
        queries: List[str],
        corpus: List[Dict[str, str]],
        top_k: int = 100,
        use_prf: bool = False,
    ) -> List[QEResult]:
        """
        Run retrieval with Query Expansion.
        
        Args:
            queries: List of queries
            corpus: Document corpus
            top_k: Number of documents to retrieve
            use_prf: Whether to use PRF (requires initial retrieval)
            
        Returns:
            List of QEResults
        """
        if self.expander is None:
            logger.warning("No expander configured, running baseline")
            return self.run_baseline(queries, corpus, top_k)
        
        logger.info(f"Running QE with method: {self.config.EXPANSION_METHOD}")
        
        results = []
        
        # First retrieval for PRF (if needed)
        if use_prf and isinstance(self.expander, (PRFExpander, CrossLingualPRFExpander)):
            # Initial retrieval
            initial_results = self.retriever.retrieve(
                queries=queries,
                corpus=corpus,
                top_k=top_k,
            )
            
            # Get retrieved docs for each query
            for i, query in enumerate(queries):
                retrieved = initial_results[i]
                retrieved_docs = [
                    {"id": r["id"], "text": corpus[int(r["id"])]["text"]}
                    for r in retrieved
                ]
                
                # Expand query with PRF
                expansion_result = self.expander.expand(query, retrieved_docs)
                
                # Re-retrieve with expanded query
                final_results = self.retriever.retrieve(
                    queries=[expansion_result.expanded_query],
                    corpus=corpus,
                    top_k=top_k,
                )
                
                results.append(QEResult(
                    query=query,
                    expanded_query=expansion_result.expanded_query,
                    expansion_method=self.config.EXPANSION_METHOD,
                    retrieved_docs=final_results[0],
                    metrics={},
                ))
        else:
            # Regular QE without PRF
            for query in queries:
                # Expand query
                expansion_result = self.expander.expand(query)
                
                # Retrieve with expanded query
                retrieved = self.retriever.retrieve(
                    queries=[expansion_result.expanded_query],
                    corpus=corpus,
                    top_k=top_k,
                )
                
                results.append(QEResult(
                    query=query,
                    expanded_query=expansion_result.expanded_query,
                    expansion_method=self.config.EXPANSION_METHOD,
                    retrieved_docs=retrieved[0],
                    metrics={},
                ))
        
        return results
    
    def run_comparison(
        self,
        queries: List[str],
        corpus: List[Dict[str, str]],
        methods: List[str],
        top_k: int = 100,
    ) -> Dict[str, List[QEResult]]:
        """
        Run comparison of multiple methods.
        
        Args:
            queries: List of queries
            corpus: Document corpus
            methods: List of methods to compare
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary of method -> results
        """
        comparison_results = {}
        
        for method in methods:
            logger.info(f"Running method: {method}")
            
            # Create config for this method
            config = QEConfig()
            config.EXPANSION_METHOD = method
            
            # Create pipeline
            pipeline = QEPipeline(config)
            
            # Run
            if method == "bm25":
                results = pipeline.run_bm25_baseline(queries, corpus, top_k)
            elif method == "baseline":
                results = pipeline.run_baseline(queries, corpus, top_k)
            else:
                results = pipeline.run_qe(queries, corpus, top_k)
            
            comparison_results[method] = results
        
        return comparison_results


class IndonesianQEPipeline(QEPipeline):
    """
    Specialized pipeline for Indonesian Code Search.
    
    Optimized for Indonesian queries with specific handling for:
    - Indonesian-English translation
    - Technical term mapping
    - Code-specific expansion
    """
    
    def __init__(self, config: Optional[QEConfig] = None):
        """Initialize Indonesian QE pipeline."""
        super().__init__(config)
        
        # Use specialized expanders for Indonesian
        self.expander = CrossLingualEmbeddingExpander(
            model_name=self.config.DEFAULT_EMBEDDING_MODEL,
            device=self.config.DEVICE,
        )
    
    def run_qe(
        self,
        queries: List[str],
        corpus: List[Dict[str, str]],
        top_k: int = 100,
    ) -> List[QEResult]:
        """
        Run Indonesian QE.
        
        Args:
            queries: Indonesian queries
            corpus: English code corpus
            top_k: Number of documents to retrieve
            
        Returns:
            List of QEResults
        """
        logger.info("Running Indonesian QE pipeline...")
        
        results = []
        
        for query in queries:
            # Expand with cross-lingual embeddings
            expansion_result = self.expander.expand(
                query,
                num_terms=self.config.NUM_EXPANSION_TERMS,
            )
            
            # Retrieve
            retrieved = self.retriever.retrieve(
                queries=[expansion_result.expanded_query],
                corpus=corpus,
                top_k=top_k,
            )
            
            results.append(QEResult(
                query=query,
                expanded_query=expansion_result.expanded_query,
                expansion_method="indonesian_qe",
                retrieved_docs=retrieved[0],
                metadata=expansion_result.metadata,
            ))
        
        return results
