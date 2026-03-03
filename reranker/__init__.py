"""
Cross-Encoder Re-ranker Package for Indonesian Code Search.

This package provides cross-encoder re-ranking models for improving
Indonesian code search results using multilingual transformers.

Based on the research plan: Cross-Encoder Re-ranking for Indonesian 
Information Retrieval (Cross-lingual Code Search)

Modules:
- cross_encoder_base: Base class for all cross-encoder rerankers
- mmmini_reranker: Mmarco-mMiniLMv2 implementation (recommended)
- xlm_reranker: XLM-RoBERTa implementation
- fine_tuner: Fine-tuning module with hard negatives
- config: Configuration classes and constants
- experiment_ce: Experiment runner

Usage:
    from reranker import MMMiniReranker
    
    reranker = MMMiniReranker(device="cuda")
    reranker.load_model()
    
    results = reranker.rerank(
        query="cara membuat fungsi di python",
        documents=code_corpus,
        top_k=10
    )
"""

from .cross_encoder_base import CrossEncoderReranker
from .mmmini_reranker import MMMiniReranker, MultilingualMMMiniReranker
from .xlm_reranker import XLMReranker, MBERTReranker
from .fine_tuner import CrossEncoderFineTuner, FineTuningConfig, create_hard_negatives
from .config import (
    RerankerConfig,
    EvaluationConfig,
    FineTuningTrainingConfig,
    ModelType,
    MODEL_CONFIGS,
    EXPERIMENT_CONFIGS,
    METRIC_TARGETS,
    BENCHMARK_DATASETS,
    get_model_config,
    get_experiment_config,
    create_reranker_config
)

__version__ = "0.1.0"
__all__ = [
    # Base classes
    "CrossEncoderReranker",
    
    # Reranker implementations
    "MMMiniReranker",
    "MultilingualMMMiniReranker",
    "XLMReranker",
    "MBERTReranker",
    
    # Fine-tuning
    "CrossEncoderFineTuner",
    "FineTuningConfig",
    "create_hard_negatives",
    
    # Configuration
    "RerankerConfig",
    "EvaluationConfig",
    "FineTuningTrainingConfig",
    "ModelType",
    "MODEL_CONFIGS",
    "EXPERIMENT_CONFIGS",
    "METRIC_TARGETS",
    "BENCHMARK_DATASETS",
    "get_model_config",
    "get_experiment_config",
    "create_reranker_config",
]
