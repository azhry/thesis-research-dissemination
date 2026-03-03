# CoIR - Code Information Retrieval Benchmark
# Query Expansion modules added

# Try to import original modules (may fail if faiss not installed)
try:
    from .data_loader import get_tasks, load_data_from_hf
    from .evaluation import COIR
    _HAS_ORIGINAL = True
except ImportError as e:
    _HAS_ORIGINAL = False
    print(f"Warning: Original CoIR modules not available: {e}")
except Exception as e:
    # Handle other errors like from evaluation.py importing faiss
    _HAS_ORIGINAL = False
    print(f"Warning: Original CoIR modules not available: {e}")

# Query Expansion modules
from .config import QEConfig, config
from .llm_expander import LLMExpander, ExpansionResult, LocalLLMExpander
from .embedding_expander import EmbeddingExpander, CrossLingualEmbeddingExpander
from .prf_expander import PRFExpander, CrossLingualPRFExpander, RM3Expander
from .combiner import QECombiner, SequentialQE, CombinedExpansionResult
from .dense_retriever import DenseRetriever, BM25Retriever
from .qe_pipeline import QEPipeline, IndonesianQEPipeline, QEResult
from .run_qe_experiment import main, run_qe_experiment, run_baseline_experiment

__all__ = [
    # Config
    "QEConfig",
    "config",
    # Expanders
    "LLMExpander",
    "ExpansionResult",
    "LocalLLMExpander",
    "EmbeddingExpander",
    "CrossLingualEmbeddingExpander",
    "PRFExpander",
    "CrossLingualPRFExpander",
    "RM3Expander",
    # Combiner
    "QECombiner",
    "SequentialQE",
    "CombinedExpansionResult",
    # Retrievers
    "DenseRetriever",
    "BM25Retriever",
    # Pipeline
    "QEPipeline",
    "IndonesianQEPipeline",
    "QEResult",
    # Experiments
    "main",
    "run_qe_experiment",
    "run_baseline_experiment",
]
