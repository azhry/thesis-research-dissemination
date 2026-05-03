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
        rerank_queries: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run the pipeline on a set of queries.
        
        Args:
            queries: Dict mapping query_id -> query_text
            corpus: Corpus data
            qrels: Relevance judgments
            language: Language label for logging
            rerank_queries: Optional mapping of qid -> alternative query for reranking (e.g. English)
        
        Returns:
            Dict with metrics and results
        """
        logger.info(f"{'='*60}")
        logger.info(f"Running pipeline on {language} queries ({len(queries)} queries)")
        logger.info(f"Experiment: {self.config.experiment_type.value}")
        logger.info(f"{'='*60}")
        
        timing = {}
        diagnostics = {}
        
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
            
            # Log sample expansions
            sample_count = min(2, len(expanded_queries))
            sample_qids = list(expanded_queries.keys())[:sample_count]
            for qid in sample_qids:
                logger.info(f"  [TQE DEBUG] QID {qid}: {queries[qid]} -> {expanded_queries[qid][:100]}...")
        else:
            logger.info("Step 1: Query Expansion SKIPPED")
            expanded_queries = queries
        
        # ── Step 2: First-Stage Retrieval ──
        logger.info("Step 2: First-Stage Retrieval (mE5)...")
        t0 = time.time()
        
        retriever = self._get_retriever()
        
        if self.config.enable_tqe:
            # Baseline for comparison
            results_original = retriever.retrieve(queries=queries, corpus=corpus, top_k=self.config.retrieval_depth, show_progress=False)
            results_expanded = retriever.retrieve(queries=expanded_queries, corpus=corpus, top_k=self.config.retrieval_depth, show_progress=False)
            
            # Hybrid Fusion
            first_stage_results = retriever.fuse_results(
                results_original=results_original,
                results_expanded=results_expanded,
                top_k=self.config.retrieval_depth,
                expansion_weight=0.3
            )
            
            # Log impact
            baseline_eval = evaluate_retrieval(results_original, qrels, [10])
            tqe_eval = evaluate_retrieval(first_stage_results, qrels, [10])
            diff = tqe_eval['nDCG@10'] - baseline_eval['nDCG@10']
            
            print(f"\n{'-'*60}")
            print(f"  [TQE DIAGNOSTIC] Impact on First-Stage Retrieval ({language}):")
            print(f"  Baseline nDCG@10:    {baseline_eval['nDCG@10']:.4f}")
            print(f"  TQE-Boosted nDCG@10: {tqe_eval['nDCG@10']:.4f} ({diff:+.4f})")
            print(f"{'-'*60}\n")
            
            diagnostics["tqe_impact"] = {
                "baseline_ndcg": baseline_eval['nDCG@10'],
                "boosted_ndcg": tqe_eval['nDCG@10'],
                "diff": diff
            }
        else:
            first_stage_results = retriever.retrieve(
                queries=queries,
                corpus=corpus,
                top_k=self.config.retrieval_depth,
            )
        
        # Inject rerank queries if available
        for qid, res in first_stage_results.items():
            if rerank_queries and qid in rerank_queries:
                res["rerank_query"] = rerank_queries[qid]
            # Ensure we have the original query for precision reranking
            res["original_query"] = queries.get(qid, res.get("query"))
            
        timing["retrieval"] = time.time() - t0
        
        # ── Step 3: Cross-Encoder Re-ranking ──
        reranked_results = None
        if self.config.enable_reranker:
            logger.info("Step 3: Cross-Encoder Re-ranking...")
            t0 = time.time()
            
            # Collect reranking queries for the Cross-Encoder
            # Priority: rerank_query (English translation for Indonesian) > original_query > query
            # This ensures the Cross-Encoder always sees English text for optimal matching
            rerank_query_texts = {}
            for qid, data in first_stage_results.items():
                rerank_query_texts[qid] = data.get("rerank_query", data.get("original_query", data["query"]))
            
            reranker = self._get_reranker()
            logger.info(f"Step 3: Cross-Encoder Re-ranking ({language})...")
            reranked_results = reranker.rerank(
                first_stage_results=first_stage_results,
                queries=rerank_query_texts,
                top_k=self.config.top_k,
                use_rrf=self.config.reranker_use_rrf,
                rrf_k=self.config.reranker_rrf_k
            )
            timing["reranking"] = time.time() - t0
        
        # ── Step 4: Evaluation ──
        logger.info("Step 4: Evaluation...")
        t0 = time.time()
        k_values = self.config.eval_k_values
        
        if self.config.enable_reranker and reranked_results is not None:
            eval_metrics = evaluate_reranking(reranked_results, qrels, k_values)
        else:
            eval_metrics = {"retrieval": evaluate_retrieval(first_stage_results, qrels, k_values)}
            
        timing["evaluation"] = time.time() - t0
        timing["total"] = sum(timing.values())
        
        return {
            "language": language,
            "experiment_type": self.config.experiment_type.value,
            "metrics": eval_metrics,
            "timing": timing,
            "diagnostics": diagnostics,
            "expanded_queries": expanded_queries,
            "raw_results": reranked_results if reranked_results else first_stage_results
        }

    def run_bilingual(
        self,
        queries_english: Dict[str, str],
        queries_indonesian: Dict[str, str],
        corpus: Any,
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        """Run on both languages."""
        results = {}
        # English: Direct run
        results["english"] = self.run(queries_english, corpus, qrels, language="english")
        
        # Indonesian: Use English as Translate-for-Rerank queries
        results["indonesian"] = self.run(
            queries_indonesian, 
            corpus, 
            qrels, 
            language="indonesian",
            rerank_queries=queries_english
        )
        return results
