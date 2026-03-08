"""
Complete Pipeline: Indonesian Code Search with TQE and Cross-Encoder Re-ranking.

Orchestrates all components:
1. Dataset loading
2. Query Expansion (optional)
3. First-stage dense retrieval (mE5)
4. Cross-Encoder re-ranking (optional)
5. Evaluation

Supports 4 experiment configurations:
- Baseline: mE5 only
- TQE-Only: mE5 + Query Expansion
- Rerank-Only: mE5 + Cross-Encoder
- Full: mE5 + TQE + Cross-Encoder
"""

import logging
import time
from typing import Dict, Any, Optional

from .config import PipelineConfig, ExperimentType
from .dataset_loader import load_cosqa_dataset
from .query_expansion import QueryExpander
from .retriever import Retriever
from .reranker import Reranker
from .evaluator import evaluate_retrieval, evaluate_reranking

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Full Indonesian Code Search Pipeline.
    
    Orchestrates mE5 retrieval, TQE, and cross-encoder re-ranking.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Components (lazy-loaded)
        self._retriever = None
        self._expander = None
        self._reranker = None
        
        logger.info(f"Pipeline initialized: {config.experiment_type.value}")
        logger.info(f"  TQE: {'enabled (' + config.qe_method.value + ')' if config.enable_tqe else 'disabled'}")
        logger.info(f"  Reranker: {'enabled (' + config.reranker_model_type.value + ')' if config.enable_reranker else 'disabled'}")
    
    def _get_retriever(self) -> Retriever:
        """Get or create the retriever."""
        if self._retriever is None:
            self._retriever = Retriever(
                model_name=self.config.retriever_model,
                device=self.config.device,
                batch_size=self.config.retriever_batch_size,
                max_seq_length=self.config.retriever_max_seq_length,
            )
        return self._retriever
    
    def _get_expander(self) -> QueryExpander:
        """Get or create the query expander."""
        if self._expander is None:
            self._expander = QueryExpander(
                method=self.config.qe_method.value,
                num_terms=self.config.qe_num_terms,
                embedding_model=self.config.retriever_model,
                llm_provider=self.config.llm_provider,
                llm_model=self.config.llm_model,
                llm_temperature=self.config.llm_temperature,
                llm_max_tokens=self.config.llm_max_tokens,
                device=self.config.device,
                cache_path=str(self.config.project_root / "results" / "llm_expansion_cache.json")
            )
        return self._expander
    
    def _get_reranker(self) -> Reranker:
        """Get or create the reranker."""
        if self._reranker is None:
            self._reranker = Reranker(
                model_type=self.config.reranker_model_type.value,
                model_name=self.config.reranker_model_name,
                device=self.config.device,
                max_length=self.config.reranker_max_length,
                batch_size=self.config.reranker_batch_size,
            )
            self._reranker.load_model()
        return self._reranker
    
    def run(
        self,
        queries: Dict[str, str],
        corpus: Any,
        qrels: Dict[str, Dict[str, int]],
        language: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Run the pipeline on a set of queries.
        
        Args:
            queries: Dict mapping query_id -> query_text
            corpus: Corpus data
            qrels: Relevance judgments
            language: Language label for logging
        
        Returns:
            Dict with:
                - "results": per-query results
                - "metrics": evaluation metrics
                - "timing": execution time per stage
                - "config": pipeline configuration used
        """
        logger.info(f"{'='*60}")
        logger.info(f"Running pipeline on {language} queries ({len(queries)} queries)")
        logger.info(f"Experiment: {self.config.experiment_type.value}")
        logger.info(f"{'='*60}")
        
        timing = {}
        
        # ── Step 1: Query Expansion (if enabled) ──
        expanded_queries = None
        qe_results = None
        
        if self.config.enable_tqe:
            logger.info("Step 1: Query Expansion...")
            t0 = time.time()
            
            expander = self._get_expander()
            qe_results = expander.expand_batch(queries)
            expanded_queries = {qid: r.expanded_query for qid, r in qe_results.items()}
            
            timing["query_expansion"] = time.time() - t0
            logger.info(f"  Query expansion completed in {timing['query_expansion']:.2f}s")
        else:
            logger.info("Step 1: Query Expansion SKIPPED")
            expanded_queries = queries  # Use original queries
        
        # ── Step 2: First-Stage Retrieval ──
        logger.info("Step 2: First-Stage Retrieval (mE5)...")
        t0 = time.time()
        
        retriever = self._get_retriever()
        
        if self.config.enable_tqe:
            first_stage_results = retriever.retrieve_with_expanded_queries(
                original_queries=queries,
                expanded_queries=expanded_queries,
                corpus=corpus,
                top_k=self.config.first_stage_top_k,
            )
        else:
            first_stage_results = retriever.retrieve(
                queries=queries,
                corpus=corpus,
                top_k=self.config.first_stage_top_k,
            )
        
        timing["retrieval"] = time.time() - t0
        logger.info(f"  Retrieval completed in {timing['retrieval']:.2f}s")
        
        # ── Step 3: Cross-Encoder Re-ranking (if enabled) ──
        reranked_results = None
        
        if self.config.enable_reranker:
            logger.info("Step 3: Cross-Encoder Re-ranking...")
            t0 = time.time()
            
            reranker = self._get_reranker()
            reranked_results = reranker.rerank(
                first_stage_results=first_stage_results,
                top_k=self.config.reranker_top_k,
            )
            
            timing["reranking"] = time.time() - t0
            logger.info(f"  Re-ranking completed in {timing['reranking']:.2f}s")
        else:
            logger.info("Step 3: Cross-Encoder Re-ranking SKIPPED")
        
        # ── Step 4: Evaluation ──
        logger.info("Step 4: Evaluation...")
        t0 = time.time()
        
        k_values = self.config.eval_k_values
        
        if self.config.enable_reranker and reranked_results is not None:
            # Evaluate both first-stage and reranked
            eval_metrics = evaluate_reranking(reranked_results, qrels, k_values)
        else:
            # Evaluate first-stage only
            eval_metrics = {
                "retrieval": evaluate_retrieval(first_stage_results, qrels, k_values),
            }
        
        timing["evaluation"] = time.time() - t0
        timing["total"] = sum(timing.values())
        
        # ── Build output ──
        output = {
            "language": language,
            "experiment_type": self.config.experiment_type.value,
            "config": self.config.to_dict(),
            "metrics": eval_metrics,
            "timing": timing,
            "num_queries": len(queries),
        }
        
        # Add QE info if used
        if qe_results is not None:
            output["qe_info"] = {
                "method": self.config.qe_method.value,
                "num_terms": self.config.qe_num_terms,
                "sample_expansions": [
                    {
                        "qid": qid,
                        "original": r.original_query,
                        "expanded": r.expanded_query,
                        "terms": r.expansion_terms,
                    }
                    for qid, r in list(qe_results.items())[:5]
                ],
            }
        
        # Store raw results for detailed output
        if reranked_results is not None:
            output["raw_results"] = reranked_results
        else:
            output["raw_results"] = first_stage_results
        
        return output
    
    def run_bilingual(
        self,
        queries_english: Dict[str, str],
        queries_indonesian: Dict[str, str],
        corpus: Any,
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        """
        Run the pipeline on both English and Indonesian queries.
        
        Args:
            queries_english: English queries
            queries_indonesian: Indonesian queries
            corpus: Corpus data
            qrels: Relevance judgments
        
        Returns:
            Dict with results for both languages
        """
        results = {}
        
        # Run English queries
        results["english"] = self.run(queries_english, corpus, qrels, language="english")
        
        # Run Indonesian queries
        results["indonesian"] = self.run(queries_indonesian, corpus, qrels, language="indonesian")
        
        return results
