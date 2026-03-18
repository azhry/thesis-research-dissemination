"""
Configuration settings for the full Indonesian Code Search pipeline.

Consolidates configuration from both QE and reranker experiments
into a single unified config module.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class ExperimentType(Enum):
    """Experiment configurations as defined in the implementation plan."""
    BASELINE = "baseline"           # mE5 only
    TQE_ONLY = "tqe_only"         # mE5 + TQE
    RERANK_ONLY = "rerank_only"   # mE5 + Cross-Encoder
    FULL = "full"                  # mE5 + TQE + Cross-Encoder


class RerankerModel(Enum):
    """Available cross-encoder reranker models."""
    MMMINI = "mmmini"
    MMMINI_MULTILINGUAL = "mmmini_multilingual"
    XLM_ROBERTA = "xlm"
    MBERT = "mbert"
    CUSTOM = "custom"


class QEMethod(Enum):
    """Query expansion methods."""
    TRANSLATION = "translation"    # Direct ID→EN term translation (recommended)
    EMBEDDING = "embedding"        # Cross-lingual embedding expansion
    HYDE = "hyde"                   # HyDE-style LLM expansion
    TECHNICAL = "technical"        # Technical enrichment via LLM
    COT = "cot"                    # Chain-of-thought expansion


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    
    # === Project paths ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    output_dir: str = "results"  # Will be resolved relative to project_root in __post_init__
    
    # === Experiment type ===
    experiment_type: ExperimentType = ExperimentType.FULL
    
    # === Dataset ===
    dataset: str = "cosqa"
    
    # === First-stage retrieval (mE5) ===
    retriever_model: str = "intfloat/multilingual-e5-small"
    retriever_batch_size: int = 32
    retriever_max_seq_length: int = 512
    first_stage_top_k: int = 100  # Candidates for reranking
    
    # === Query Expansion (TQE) ===
    enable_tqe: bool = True
    qe_method: QEMethod = QEMethod.HYDE
    qe_num_terms: int = 5
    # LLM settings (for HyDE/technical/CoT methods)
    llm_provider: str = "google"
    llm_model: str = "models/gemini-flash-latest"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    
    # === Cross-Encoder Re-ranking ===
    enable_reranker: bool = True
    reranker_model_type: RerankerModel = RerankerModel.MMMINI
    reranker_model_name: Optional[str] = None  # Auto-set from model_type if None
    reranker_max_length: int = 512
    reranker_batch_size: int = 8
    reranker_top_k: int = 10
    reranker_use_rrf: bool = True  # Use RRF to combine Bi-Encoder + Cross-Encoder
    reranker_rrf_k: int = 100
    
    # === Evaluation ===
    eval_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    
    # === Device ===
    device: str = "cpu"
    
    # === Sampling (for testing) ===
    sample_size: Optional[int] = None
    
    def __post_init__(self):
        """Set derived values."""
        # Set experiment flags based on experiment type
        if self.experiment_type == ExperimentType.BASELINE:
            self.enable_tqe = False
            self.enable_reranker = False
        elif self.experiment_type == ExperimentType.TQE_ONLY:
            self.enable_tqe = True
            self.enable_reranker = False
        elif self.experiment_type == ExperimentType.RERANK_ONLY:
            self.enable_tqe = False
            self.enable_reranker = True
        elif self.experiment_type == ExperimentType.FULL:
            self.enable_tqe = True
            self.enable_reranker = True
            
        # Ensure output_dir is absolute relative to project_root
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str(self.project_root / self.output_dir)
        
        # Set default reranker model name
        if self.reranker_model_name is None:
            self.reranker_model_name = RERANKER_MODEL_MAP.get(
                self.reranker_model_type,
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "experiment_type": self.experiment_type.value,
            "dataset": self.dataset,
            "retriever_model": self.retriever_model,
            "first_stage_top_k": self.first_stage_top_k,
            "enable_tqe": self.enable_tqe,
            "qe_method": self.qe_method.value if self.enable_tqe else None,
            "qe_num_terms": self.qe_num_terms,
            "enable_reranker": self.enable_reranker,
            "reranker_model_type": self.reranker_model_type.value if self.enable_reranker else None,
            "reranker_model_name": self.reranker_model_name,
            "reranker_top_k": self.reranker_top_k,
            "eval_k_values": self.eval_k_values,
            "device": self.device,
            "sample_size": self.sample_size,
        }


PROJECT_DIR = Path(__file__).parent.parent.absolute()

# === Model name mappings ===
RERANKER_MODEL_MAP = {
    RerankerModel.MMMINI: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    RerankerModel.MMMINI_MULTILINGUAL: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    RerankerModel.XLM_ROBERTA: "cross-encoder/stsb-roberta-base",
    RerankerModel.MBERT: "cross-encoder/ms-marco-MiniLM-L-12-v2",
    RerankerModel.CUSTOM: str(PROJECT_DIR / "models" / "cross-encoder-me5-small-full-hn-v2"),
}

RETRIEVER_MODEL_MAP = {
    "small": "intfloat/multilingual-e5-small",
    "base": "intfloat/multilingual-e5-base",
    "large": "intfloat/multilingual-e5-large-instruct",
}

# === Pre-defined experiment configurations ===
EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "experiment_type": ExperimentType.BASELINE,
        "description": "mE5-only baseline retrieval",
    },
    "tqe_translation": {
        "experiment_type": ExperimentType.TQE_ONLY,
        "qe_method": QEMethod.TRANSLATION,
        "description": "mE5 + Translation-based TQE (recommended)",
    },
    "tqe_embedding": {
        "experiment_type": ExperimentType.TQE_ONLY,
        "qe_method": QEMethod.EMBEDDING,
        "description": "mE5 + Embedding-based TQE",
    },
    "tqe_hyde": {
        "experiment_type": ExperimentType.TQE_ONLY,
        "qe_method": QEMethod.HYDE,
        "description": "mE5 + HyDE TQE (requires LLM API)",
    },
    "rerank_mmmini": {
        "experiment_type": ExperimentType.RERANK_ONLY,
        "reranker_model_type": RerankerModel.MMMINI,
        "description": "mE5 + mMiniLMv2 re-ranking",
    },
    "rerank_xlm": {
        "experiment_type": ExperimentType.RERANK_ONLY,
        "reranker_model_type": RerankerModel.XLM_ROBERTA,
        "description": "mE5 + XLM-RoBERTa re-ranking",
    },
    "full_translation_mmmini": {
        "experiment_type": ExperimentType.FULL,
        "qe_method": QEMethod.TRANSLATION,
        "reranker_model_type": RerankerModel.MMMINI,
        "description": "Full pipeline: mE5 + Translation TQE + mMiniLMv2",
    },
    "full_embedding_mmmini": {
        "experiment_type": ExperimentType.FULL,
        "qe_method": QEMethod.EMBEDDING,
        "reranker_model_type": RerankerModel.MMMINI,
        "description": "Full pipeline: mE5 + Embedding TQE + mMiniLMv2",
    },
    "full_hyde_mmmini": {
        "experiment_type": ExperimentType.FULL,
        "qe_method": QEMethod.HYDE,
        "reranker_model_type": RerankerModel.MMMINI,
        "description": "Full pipeline: mE5 + HyDE TQE + mMiniLMv2",
    },
    "full_hyde_custom": {
        "experiment_type": ExperimentType.FULL,
        "qe_method": QEMethod.HYDE,
        "reranker_model_type": RerankerModel.CUSTOM,
        "description": "Full pipeline: mE5 + HyDE TQE + Fine-tuned mE5 Cross-Encoder",
    },
}
