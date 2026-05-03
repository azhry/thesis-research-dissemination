# Query Expansion (QE) Module

This module implements Query Expansion techniques for improving Indonesian code search.

## Quick Start

```bash
# Activate virtual environment
cd qe
call venv\Scripts\activate.bat

# Run experiment
python experiment_qe.py --method embedding --top-k 10
```

## Available Methods

| Method | Description |
|--------|-------------|
| `baseline` | No query expansion |
| `bm25` | BM25 lexical retrieval |
| `embedding` | Embedding-based expansion |

## Options

```bash
python experiment_qe.py --method embedding --top-k 10 --output results.json
```

- `--method`: Method to run (baseline, bm25, embedding)
- `--top-k`: Number of documents to retrieve (default: 10)
- `--output`: Output file path (default: results.json)

## Python Usage

```python
from qe.coir.embedding_expander import CrossLingualEmbeddingExpander
from qe.coir.dense_retriever import DenseRetriever

# Initialize expander
expander = CrossLingualEmbeddingExpander(model_name="intfloat/multilingual-e5-base")

# Expand query
expansion = expander.expand("cara membuat fungsi di python", num_terms=5)

# Retrieve
retriever = DenseRetriever(model_name="intfloat/multilingual-e5-base", device="cpu")
results = retriever.retrieve(queries=[expansion.expanded_query], corpus=code_corpus, top_k=10)
```

## Installation

```bash
cd qe
pip install -r requirements.txt
```
