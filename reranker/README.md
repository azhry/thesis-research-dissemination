# Cross-Encoder Re-ranker Module

This module implements Cross-Encoder Re-ranking for improving Indonesian code search.

## Quick Start

```bash
# Activate virtual environment
cd reranker
call venv\Scripts\activate.bat

# Run experiment (quick test with 10 samples)
python -m experiment_ce --dataset cosqa --sample-size 10 --top-k 5
```

## Running with Full Dataset

For full benchmark evaluation (uses all 20K+ queries), use GPU for reasonable speed:

```bash
# With GPU (recommended)
python -m experiment_ce --dataset cosqa --device cuda

# With CPU only (very slow - not recommended)
python -m experiment_ce --dataset cosqa --device cpu
```

**Note:** Full evaluation on CPU is extremely slow (~hours). GPU is highly recommended.

## Available Models

| Model | Description | Latency |
|-------|-------------|---------|
| `mmmini` | mmarco-mMiniLMv2-L12 (recommended) | ~65ms |
| `mmmini_multilingual` | Multilingual BERT variant | ~100ms |
| `xlm` | XLM-RoBERTa base | ~250ms |

## Options

```bash
python -m experiment_ce --method mmmini --dataset cosqa --sample-size 100 --top-k 10 --device cpu
```

- `--method`: Model to use (mmmini, mmmini_multilingual, xlm)
- `--dataset`: Benchmark dataset (cosqa, codetrans_dl, stackoverflow_qa)
- `--sample-size`: Number of queries to test (omit for full dataset)
- `--top-k`: Number of documents to retrieve (default: 5)
- `--device`: Device to use (cpu, cuda)
- `--output`: Output file path

## Python Usage

```python
from reranker import MMMiniReranker

# Initialize reranker
reranker = MMMiniReranker(device="cpu")
reranker.load_model()

# Re-rank documents
results = reranker.rerank(
    query="cara membuat fungsi di python",
    documents=code_corpus,
    top_k=10
)

# Print results
for doc in results:
    print(f"Score: {doc['cross_encoder_score']:.3f} - {doc['id']}")
```

## Fine-tuning

```python
from reranker import CrossEncoderFineTuner, FineTuningConfig

# Configure fine-tuning
config = FineTuningConfig(
    model_name="sentence-transformers/ms-marco-MiniLM-L-12-v2-cross-encoder",
    learning_rate=2e-5,
    num_epochs=3,
    batch_size=16
)

# Fine-tune
tuner = CrossEncoderFineTuner(config)
tuner.train(train_dataset)
```

## Installation

```bash
cd reranker
pip install -r requirements.txt
```
