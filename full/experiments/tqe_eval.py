"""
TQE Experiment: mE5 + Technical Query Expansion.

Runs mE5 retrieval with technical query expansion.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "qe"))

from src.config import PipelineConfig, ExperimentType, QEMethod
from src.dataset_loader import load_cosqa_dataset
from src.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TQE experiment: mE5 + Query Expansion")
    parser.add_argument("--retriever-model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--qe-method", type=str, default="embedding",
                        choices=["embedding", "hyde", "technical", "cot"])
    parser.add_argument("--qe-num-terms", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output", type=str, default="./full/results/tqe_results.json")
    
    args = parser.parse_args()
    
    corpus, queries_en, queries_id, qrels = load_cosqa_dataset(sample_size=args.sample_size)
    logger.info(f"Loaded {len(queries_en)} queries, {len(corpus)} documents")
    
    config = PipelineConfig(
        experiment_type=ExperimentType.TQE_ONLY,
        retriever_model=args.retriever_model,
        first_stage_top_k=args.top_k,
        qe_method=QEMethod(args.qe_method),
        qe_num_terms=args.qe_num_terms,
        device=args.device,
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
    
    logger.info(f"TQE results saved to {output_path}")
    
    for lang in ["english", "indonesian"]:
        metrics = results[lang]["metrics"].get("retrieval", {})
        print(f"\n{lang.upper()} TQE ({args.qe_method}):")
        print(f"  nDCG@10:   {metrics.get('nDCG@10', 0):.4f}")
        print(f"  MAP@10:    {metrics.get('MAP@10', 0):.4f}")
        print(f"  Recall@10: {metrics.get('Recall@10', 0):.4f}")


if __name__ == "__main__":
    main()
