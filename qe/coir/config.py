"""
Configuration settings for Query Expansion experiments.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


class QEConfig:
    """Configuration for Query Expansion experiments."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Model settings
    DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_MODELS = {
        "small": "intfloat/multilingual-e5-small",
        "base": "intfloat/multilingual-e5-base",
        "large": "intfloat/multilingual-e5-large-instruct",
    }
    
    # LLM settings for Query Expansion
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "google"
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp")
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 512
    
    # Query Expansion settings
    EXPANSION_METHOD = "hyde"  # "hyde", "embedding", "prf", "combined"
    NUM_EXPANSION_TERMS = 5
    MAX_EXPANSION_TERMS = 20
    
    # Retrieval settings
    TOP_K = 100
    RETRIEVAL_BATCH_SIZE = 32
    MAX_SEQ_LENGTH = 512
    
    # Evaluation settings
    EVAL_METRICS = ["ndcg@10", "map@10", "recall@100", "mrr"]
    
    # Dataset settings
    DATASET = "cosqa"
    LANGUAGE = "indonesian"  # "indonesian" or "english"
    
    # Device settings
    DEVICE = "cpu"  # "cuda" or "cpu"
    
    # Cache settings
    USE_CACHE = True
    CACHE_DIR = DATA_DIR / "cache"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QEConfig":
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }
    
    def __repr__(self) -> str:
        return f"QEConfig({self.to_dict()})"


# Default configuration instance
config = QEConfig()
