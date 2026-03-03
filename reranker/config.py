"""
Configuration for Cross-Encoder Re-ranker.

This module provides configuration classes and constants
for the cross-encoder reranker based on the research plan.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelType(Enum):
    """Available cross-encoder model types."""
    MMMINI = "mmmini"
    MMMINI_MULTILINGUAL = "mmmini_multilingual"
    XLM_ROBERTA = "xlm"
    MBERT = "mbert"
    CUSTOM = "custom"


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    
    # Model settings
    model_type: ModelType = ModelType.MMMINI
    model_name: str = "sentence-transformers/ms-marco-MiniLM-L-12-v2-cross-encoder"
    device: str = "cuda"  # or "cpu"
    max_length: int = 512
    batch_size: int = 8
    
    # Re-ranking settings
    top_k: int = 10
    first_stage_k: int = 100  # Number of candidates from first stage
    score_threshold: Optional[float] = None
    
    # Multilingual settings
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    
    # Performance settings
    use_fp16: bool = False
    num_workers: int = 0
    
    def __post_init__(self):
        """Set defaults based on model type."""
        if self.model_type == ModelType.XLM_ROBERTA:
            self.batch_size = 4  # Smaller batch for larger models
        elif self.model_type == ModelType.MBERT:
            self.batch_size = 4


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["ndcg@10", "map@10", "mrr", "recall@10"])
    
    # Evaluation settings
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50, 100])
    
    # Output settings
    output_dir: str = "./results"
    save_predictions: bool = True
    
    # Ground truth
    relevance_threshold: int = 1


@dataclass
class FineTuningTrainingConfig:
    """Configuration for cross-encoder fine-tuning."""
    
    # Model to fine-tune
    base_model: str = "sentence-transformers/ms-marco-MiniLM-L-12-v2-cross-encoder"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    
    # Loss settings
    margin: float = 0.5
    loss_type: str = "margin_ranking"  # or "cross_entropy"
    
    # Data settings
    max_seq_length: int = 512
    train_split: float = 0.9
    
    # Hard negative settings
    use_hard_negatives: bool = True
    hard_negative_margin: float = 1.0
    num_hard_negatives: int = 4
    
    # Output settings
    save_path: str = "./models/finetuned_ce"
    save_steps: int = 500
    logging_steps: int = 100
    
    # Device
    device: str = "cuda"
    
    # Mixed precision
    use_fp16: bool = True


# Model configurations from the research plan
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mmmini": {
        "name": "sentence-transformers/ms-marco-MiniLM-L-12-v2-cross-encoder",
        "params": "33M",
        "languages": "100+",
        "latency_ms": 65,
        "batch_size": 16,
        "description": "Recommended - Small multilingual re-ranker"
    },
    "multilingual_mpnet": {
        "name": "sentence-transformers/multi-qa-mpnet-base-coss",
        "params": "278M",
        "languages": "50+",
        "latency_ms": 200,
        "batch_size": 8,
        "description": "High quality multilingual model"
    },
    "xlm_roberta_base": {
        "name": "xlm-roberta-base",
        "params": "278M",
        "languages": "100+",
        "latency_ms": 250,
        "batch_size": 4,
        "description": "XLM-RoBERTa base for re-ranking"
    },
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "params": "178M",
        "languages": "104+",
        "latency_ms": 220,
        "batch_size": 4,
        "description": "Baseline multilingual BERT"
    }
}


# Experimental configurations from the research plan
EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "config_1": {
        "name": "mE5-only (baseline)",
        "retrieval": "mE5",
        "reranker": None,
        "finetuned": False
    },
    "config_2": {
        "name": "mE5 + mBERT (zero-shot)",
        "retrieval": "mE5",
        "reranker": "mbert",
        "finetuned": False
    },
    "config_3": {
        "name": "mE5 + mmarco-mMiniLMv2 (zero-shot)",
        "retrieval": "mE5",
        "reranker": "mmmini",
        "finetuned": False
    },
    "config_4": {
        "name": "mE5 + XLM-RoBERTa (zero-shot)",
        "retrieval": "mE5",
        "reranker": "xlm",
        "finetuned": False
    },
    "config_5": {
        "name": "mE5 + mmarco-mMiniLMv2 (fine-tuned)",
        "retrieval": "mE5",
        "reranker": "mmmini",
        "finetuned": True
    },
    "config_6": {
        "name": "mE5 + Custom CE (hard negatives)",
        "retrieval": "mE5",
        "reranker": "custom",
        "finetuned": True
    }
}


# Evaluation metrics targets from the research plan
METRIC_TARGETS = {
    "ndcg@10": 0.55,
    "map@10": 0.45,
    "mrr": 0.65,
    "recall@50": 0.75,
    "latency_ms": 500
}

# Benchmark dataset configurations (CoIR)
BENCHMARK_DATASETS = {
    "cosqa": {
        "name": "CoSQA",
        "hf_name": "cosqa",
        "description": "Code Search Q&A - Natural language to code",
        "language": "English",
        "task_type": "code-retrieval"
    },
    "codetrans_dl": {
        "name": "CodeTrans-DL", 
        "hf_name": "codetrans-dl",
        "description": "Code Translation Deep Learning",
        "language": "Multiple",
        "task_type": "code-translation"
    },
    "stackoverflow_qa": {
        "name": "StackOverflow QA",
        "hf_name": "stackoverflow-qa",
        "description": "StackOverflow Question Answering",
        "language": "English",
        "task_type": "code-retrieval"
    },
    "apps": {
        "name": "Apps",
        "hf_name": "apps",
        "description": "Apps Dataset - Python code descriptions",
        "language": "English",
        "task_type": "code-retrieval"
    },
    "codefeedback_mt": {
        "name": "CodeFeedback MT",
        "hf_name": "codefeedback-mt",
        "description": "Code Feedback Multilingual",
        "language": "Multilingual",
        "task_type": "code-retrieval"
    },
    "codefeedback_st": {
        "name": "CodeFeedback ST",
        "hf_name": "codefeedback-st",
        "description": "Code Feedback Single Turn",
        "language": "English",
        "task_type": "code-retrieval"
    },
    "codetrans_contest": {
        "name": "CodeTrans Contest",
        "hf_name": "codetrans-contest",
        "description": "Code Translation Contest",
        "language": "Multiple",
        "task_type": "code-translation"
    },
    "synthetic_text2sql": {
        "name": "Synthetic Text2SQL",
        "hf_name": "synthetic-text2sql",
        "description": "Synthetic Text to SQL dataset",
        "language": "English",
        "task_type": "text2sql"
    },
}


def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get model configuration by type."""
    return MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["mmmini"])


def get_experiment_config(exp_id: str) -> Dict[str, Any]:
    """Get experiment configuration by ID."""
    return EXPERIMENT_CONFIGS.get(exp_id, EXPERIMENT_CONFIGS["config_3"])


def create_reranker_config(
    model_type: str = "mmmini",
    device: str = "cuda",
    **kwargs
) -> RerankerConfig:
    """Create reranker configuration."""
    model_map = {
        "mmmini": ModelType.MMMINI,
        "mmmini_multilingual": ModelType.MMMINI_MULTILINGUAL,
        "xlm": ModelType.XLM_ROBERTA,
        "mbert": ModelType.MBERT
    }
    
    return RerankerConfig(
        model_type=model_map.get(model_type, ModelType.MMMINI),
        model_name=get_model_config(model_type)["name"],
        device=device,
        **kwargs
    )
