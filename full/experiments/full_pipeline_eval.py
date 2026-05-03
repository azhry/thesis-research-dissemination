"""
Full Pipeline Experiment: mE5 + TQE + Cross-Encoder Re-ranking.

Runs the complete pipeline with all components enabled.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qe"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "reranker"))

from src.config import PipelineConfig, ExperimentType, QEMethod, RerankerModel
from src.dataset_loader import load_cosqa_dataset
from src.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: mE5 + TQE + Cross-Encoder")
    parser.add_argument("--retriever-model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--first-stage-k", type=int, default=100)
    parser.add_argument("--qe-method", type=str, default="embedding",
                        choices=["embedding", "hyde", "technical", "cot"])
    parser.add_argument("--qe-num-terms", type=int, default=5)
    parser.add_argument("--reranker-model", type=str, default="mmmini",
                        choices=["mmmini", "mmmini_multilingual", "xlm", "mbert"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output", type=str, default="./full/results/full_pipeline_results.json")
    
    args = parser.parse_args()
    
    corpus, queries_en, queries_id, qrels = load_cosqa_dataset(sample_size=args.sample_size)
    logger.info(f"Loaded {len(queries_en)} queries, {len(corpus)} documents")
    
    config = PipelineConfig(
        experiment_type=ExperimentType.FULL,
        retriever_model=args.retriever_model,
        retrieval_depth=args.first_stage_k,
        qe_method=QEMethod(args.qe_method),
        qe_num_terms=args.qe_num_terms,
        reranker_model_type=RerankerModel(args.reranker_model),
        top_k=args.top_k,
        device=args.device,
        sample_size=args.sample_size,
    )
    
    pipeline = Pipeline(config)
    results = pipeline.run_bilingual(queries_en, queries_id, corpus, qrels)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable = {}
    for lang, lang_results in results.items():
        serializable[lang] = {k: v for k, v in lang_results.items() if k != "raw_results"}
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    
    logger.info(f"Full pipeline results saved to {output_path}")
    
    # --- Qualitative Trace ---
    trace_path = output_path.parent / "full_pipeline_trace.json"
    trace_data = []
    
    # We focus on Indonesian queries for qualitative analysis
    indo_results = results.get("indonesian", {})
    raw_results = indo_results.get("raw_results", {})
    expanded_queries = indo_results.get("expanded_queries", {})
    
    # Sample up to 20 queries for the trace to keep file size reasonable
    trace_qids = list(raw_results.keys())[:20]
    
    for qid in trace_qids:
        query_data = raw_results[qid]
        expanded_q = expanded_queries.get(qid, "")
        
        # Get ground truth relevant docs
        relevant_docs = list(qrels.get(qid, {}).keys())
        

        trace_entry = {
            "qid": qid,
            "query_id": queries_id.get(qid),
            "query_en": queries_en.get(qid),
            "expanded_query": expanded_q,
            "relevant_doc_ids": relevant_docs,
            "top_results": []
        }
        
        # Get content for top 10 results
        # Keys depend on whether reranking was used ("reranked") or only retrieval ("retrieved")
        top_results = query_data.get("reranked", query_data.get("retrieved", []))
        
        for res in top_results[:10]:
            doc_id = res["id"]
            doc_data = corpus.get(doc_id, {})
            
            trace_entry["top_results"].append({
                "doc_id": doc_id,
                "score": res["score"],
                "rank": res["rank"],
                "is_relevant": doc_id in relevant_docs,
                "title": doc_data.get("title", ""),
                "text": doc_data.get("text", "")[:500] + "..." if len(doc_data.get("text", "")) > 500 else doc_data.get("text", "")
            })
            
        trace_data.append(trace_entry)
        
    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2)
        
    logger.info(f"Qualitative trace saved to {trace_path}")
    
    for lang in ["english", "indonesian"]:
        metrics = results[lang]["metrics"]
        before = metrics.get("before_rerank", {})
        after = metrics.get("after_rerank", {})
        
        print(f"\n{lang.upper()} Full Pipeline ({args.qe_method} + {args.reranker_model}):")
        print(f"  nDCG@10 Before: {before.get('nDCG@10', 0):.4f}")
        print(f"  nDCG@10 After:  {after.get('nDCG@10', 0):.4f}")
        print(f"  MAP@10 Before:  {before.get('MAP@10', 0):.4f}")
        print(f"  MAP@10 After:   {after.get('MAP@10', 0):.4f}")
        print(f"  Recall@10:      {after.get('Recall@10', 0):.4f}")


if __name__ == "__main__":
    main()
