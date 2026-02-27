# Research on Indonesian Code Search with Query Expansion and Cross-Encoder Re-ranking

This project implements a research framework for **Improving Indonesian Code Search via Technical Query Expansion and Cross-Encoder Re-ranking** using Multilingual E5 (mE5) embeddings and the CoSQA dataset.

## Project Overview

This research focuses on cross-lingual code retrieval - searching English Python code using Indonesian natural language queries. The framework implements a multi-stage pipeline:

1. **First-Stage Retrieval**: Multilingual E5 embeddings for semantic search
2. **Query Expansion (QE)**: Technical Query Expansion to bridge the Indonesian-English vocabulary gap
3. **Re-Ranking**: Cross-Encoder re-ranking for precision refinement (future work)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Indonesian    │────▶│  Query Expansion │────▶│  Multilingual   │
│     Query       │     │  (HyDE/Embed/PRF)│     │      E5         │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Results       │◀────│ Cross-Encoder    │◀────│  Top-K          │
│   (Ranked)      │     │ Re-Ranking       │     │  Candidates     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Directory Structure

```
research-dissemination/
├── README.md                    # This file
├── qe/                         # Query Expansion module
│   ├── coir/                   # Main package
│   │   ├── config.py           # Configuration settings
│   │   ├── llm_expander.py    # LLM-based Query Expansion (HyDE)
│   │   ├── embedding_expander.py  # Embedding-based QE
│   │   ├── prf_expander.py    # Pseudo-Relevance Feedback QE
│   │   ├── combiner.py        # Combine multiple QE methods
│   │   ├── dense_retriever.py # mE5 dense retrieval
│   │   ├── qe_pipeline.py     # Complete QE pipeline
│   │   ├── run_qe_experiment.py  # Experiment runner
│   │   └── evaluation.py       # Evaluation metrics
│   ├── requirements.txt        # Python dependencies
│   └── setup.py                # Package setup
├── plans/                      # Research plans
│   ├── implementation_plan.md
│   ├── query_expansion_indonesian_ir_plan.md
│   └── cross_encoder_reranker_plan.md
├── latex/                      # LaTeX paper files
└── sources/                   # Research source materials
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd research-dissemination
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install QE module dependencies
cd qe
pip install -r requirements.txt

# Or install the package in development mode
pip install -e .
```

### 4. Install Additional Dependencies (Optional)

For GPU support with FAISS:
```bash
pip install faiss-gpu
```

For LLM-based Query Expansion:
```bash
# OpenAI
pip install openai

# Google Gemini
pip install google-generativeai
```

### 5. Environment Variables (Optional)

For LLM-based Query Expansion, set your API keys:

```bash
# Windows
set OPENAI_API_KEY=your_openai_key
set GOOGLE_API_KEY=your_google_key

# macOS/Linux
export OPENAI_API_KEY=your_openai_key
export GOOGLE_API_KEY=your_google_key
```

## Usage

### Running Query Expansion Experiments

The main experiment script is `qe/coir/run_qe_experiment.py`.

#### Basic Usage

```python
from qe.coir import QEPipeline, QEConfig

# Create configuration
config = QEConfig()
config.DATASET = "cosqa"
config.LANGUAGE = "indonesian"
config.EXPANSION_METHOD = "hyde"  # Options: "hyde", "embedding", "prf", "combined"
config.DEVICE = "cpu"  # Use "cuda" for GPU

# Create and run pipeline
pipeline = QEPipeline(config)
results = pipeline.run()

# Print results
print(f"nDCG@10: {results['ndcg@10']:.3f}")
print(f"MAP@10: {results['map@10']:.3f}")
print(f"Recall@10: {results['recall@10']:.3f}")
```

#### Running via Command Line

```bash
cd qe
python -m coir.run_qe_experiment
```

### Configuration Options

Edit `qe/coir/config.py` or pass parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DEFAULT_EMBEDDING_MODEL` | Multilingual E5 model | `intfloat/multilingual-e5-large-instruct` |
| `EXPANSION_METHOD` | QE method: `hyde`, `embedding`, `prf`, `combined` | `hyde` |
| `TOP_K` | Number of retrieved candidates | `100` |
| `DEVICE` | Compute device: `cpu` or `cuda` | `cpu` |
| `LLM_PROVIDER` | LLM provider: `openai` or `google` | `openai` |

### Query Expansion Methods

#### 1. HyDE (Hypothetical Document Embeddings)

Uses LLM to generate a hypothetical code snippet from the query, then uses that for retrieval.

```python
from qe.coir.llm_expander import LLMExpander

expander = LLMExpander(provider="openai", model="gpt-4o")
expanded_query = expander.expand("cara simpan data ke database")
```

#### 2. Embedding-based Query Expansion

Uses cross-lingual embeddings to find similar terms in the code corpus.

```python
from qe.coir.embedding_expander import EmbeddingExpander

expander = EmbeddingExpander(model_name="intfloat/multilingual-e5-large-instruct")
expanded_query = expander.expand("cara simpan data ke database", corpus_embeddings)
```

#### 3. Pseudo-Relevance Feedback (PRF)

Uses top-retrieved documents to expand the query.

```python
from qe.coir.prf_expander import PRFExpander

expander = PRFExpander(top_k=10)
expanded_query = expander.expand(query, initial_results)
```

#### 4. Combined Method

Combines multiple QE methods for better results.

```python
from qe.coir.combiner import QECombiner

combiner = QECombiner(methods=["hyde", "embedding", "prf"])
expanded_query = combiner.combine(original_query, retrieval_results)
```

## Datasets

### CoSQA (Code Search QA)

- **Description**: 20,604 human-annotated pairs of natural language queries and Python code
- **Source**: [CoSQA: 20,000+ Web Queries for Code Search and Question Answering](https://github.com/Junqiang-Alpha/CoSQA)
- **Usage**: The dataset is automatically downloaded when running experiments

### Converting to Indonesian

For Indonesian code search, queries can be translated using:

```python
from qe.coir.utils import translate_queries_to_indonesian

indonesian_queries = translate_queries_to_indonesian(cosqa_queries)
```

## Evaluation Metrics

The framework evaluates using standard information retrieval metrics:

- **nDCG@10**: Normalized Discounted Cumulative Gain at 10
- **MAP@10**: Mean Average Precision at 10
- **Recall@K**: Recall at K (typically K=100)
- **MRR**: Mean Reciprocal Rank

### Running Evaluation

```python
from qe.coir.evaluation import evaluate_retrieval

metrics = evaluate_retrieval(
    queries=query_list,
    results=retrieved_docs,
    ground_truth=ground_truth,
    metrics=["ndcg@10", "map@10", "recall@100", "mrr"]
)
```

## Models

### First-Stage Retrieval: Multilingual E5

| Model | Parameters | Description |
|-------|------------|-------------|
| `intfloat/multilingual-e5-small` | 118M | Lightweight multilingual embeddings |
| `intfloat/multilingual-e5-base` | 278M | Medium-size multilingual embeddings |
| `intfloat/multilingual-e5-large-instruct` | 560M | **Recommended** - Best performance with instruction tuning |

### Re-Ranking (Future Work)

| Model | Description |
|-------|-------------|
| `mmarco-mMiniLMv2-L12` | Multilingual re-ranker with low latency |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | English re-ranker |
| `Cohere Rerank 4 Pro` | Proprietary high-quality re-ranker |

## Research Papers

This framework is based on the following research:

1. **COIR: A Comprehensive Benchmark for Code Information Retrieval Models** (ACL 2025)
2. **CoSQA: 20,000+ Web Queries for Code Search and Question Answering** (ACL 2021)
3. **Multilingual E5 Text Embeddings: A Technical Report** (arXiv 2024)
4. **Query2doc: Query Expansion with Large Language Models** (EMNLP 2023)
5. **What Drives Cross-lingual Ranking?** (arXiv 2025)

## Troubleshooting

### CUDA/GPU Issues

If you encounter CUDA errors, set the device to CPU in `qe/coir/config.py`:

```python
DEVICE = "cpu"  # Change from "cuda" to "cpu"
```

### FAISS Import Errors

If FAISS fails to import, the code will automatically fall back to NumPy-based retrieval. Install FAISS for better performance:

```bash
pip install faiss-cpu  # CPU version
# or
pip install faiss-gpu  # GPU version
```

### LLM API Errors

Ensure you have set up your API keys correctly. You can also disable LLM-based QE and use other methods:

```python
config.EXPANSION_METHOD = "embedding"  # Use embedding-based QE instead
```

## License

This project is for research purposes. See `qe/LICENSE` for details.

## Contact

For questions or issues, please contact the research team.

---

**Note**: This README covers the Query Expansion module. For Cross-Encoder Re-ranking implementation, see `plans/cross_encoder_reranker_plan.md`.
