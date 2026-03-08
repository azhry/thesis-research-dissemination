# Full Pipeline: Indonesian Code Search with TQE + Cross-Encoder Re-ranking

This directory contains the complete, unified pipeline implementation for the research:
**"Improving Indonesian Code Search via Technical Query Expansion and Cross-Encoder Re-ranking"**

## Architecture

```
Indonesian Query → [Query Expansion] → [mE5 Dense Retrieval] → [Cross-Encoder Re-ranking] → Evaluated Results
```

## Directory Structure

```
full/
├── src/
│   ├── __init__.py            # Package init
│   ├── config.py              # Unified configuration
│   ├── dataset_loader.py      # CoSQA dataset with Indonesian translations
│   ├── query_expansion.py     # TQE module (embedding + LLM-based)
│   ├── retriever.py           # mE5 dense retrieval
│   ├── reranker.py            # Cross-Encoder re-ranking
│   ├── pipeline.py            # Complete pipeline orchestrator
│   └── evaluator.py           # nDCG, MAP, Recall, MRR metrics
├── experiments/
│   ├── baseline_eval.py       # mE5-only baseline
│   ├── tqe_eval.py            # mE5 + TQE
│   ├── full_pipeline_eval.py  # mE5 + TQE + CrossEncoder
│   └── compare_results.py     # Results comparison
├── results/                   # Output directory
├── run_experiments.py         # Main runner (all experiments)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Experiment Configurations

| Experiment   | mE5 | TQE | Cross-Encoder | Description              |
|-------------|-----|-----|---------------|--------------------------|
| Baseline    | ✓   | ✗   | ✗             | Direct mE5 retrieval     |
| TQE-Only    | ✓   | ✓   | ✗             | mE5 + query expansion    |
| Rerank-Only | ✓   | ✗   | ✓             | mE5 + re-ranking         |
| Full        | ✓   | ✓   | ✓             | Complete pipeline        |

## Quick Start

### Run all experiments (with sampling for quick test)
```bash
python full/run_experiments.py --sample-size 10
```

### Run specific experiments
```bash
# Baseline only
python full/run_experiments.py --experiments baseline --sample-size 10

# TQE + Full
python full/run_experiments.py --experiments tqe_only full --sample-size 10

# Full pipeline with specific models
python full/run_experiments.py \
    --experiments full \
    --retriever-model intfloat/multilingual-e5-small \
    --qe-method embedding \
    --reranker-model mmmini \
    --sample-size 10
```

### Run individual experiments
```bash
# Baseline
python full/experiments/baseline_eval.py --sample-size 10

# TQE
python full/experiments/tqe_eval.py --qe-method embedding --sample-size 10

# Full pipeline
python full/experiments/full_pipeline_eval.py --sample-size 10
```

### Compare results
```bash
python full/experiments/compare_results.py --results-dir ./full/results
```

## Fine-tuning the Cross-Encoder

To fine-tune the multilingual E5 model as a cross-encoder on the CoSQA dataset:

```bash
python full/finetune_cross_encoder.py \
    --model-name intfloat/multilingual-e5-small \
    --output-dir ./full/models/cross-encoder-me5-small \
    --num-epochs 3 \
    --batch-size 16
```

Use the fine-tuned model in experiments:

```bash
python full/run_experiments.py \
    --experiments full \
    --qe-method hyde \
    --reranker-model custom \
    --sample-size 10
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiments` | all 4 | Which experiments to run |
| `--retriever-model` | `intfloat/multilingual-e5-small` | mE5 model variant |
| `--first-stage-k` | 100 | Number of candidates for reranking |
| `--qe-method` | `hyde` | QE method: embedding, hyde, technical, cot, translation |
| `--qe-num-terms` | 5 | Number of expansion terms |
| `--reranker-model` | `mmmini` | Reranker: mmmini, mmmini_multilingual, xlm, mbert, custom |
| `--top-k` | 10 | Final top-K results |
| `--device` | `cpu` | Device: cpu or cuda |
| `--sample-size` | None | Limit queries for testing |
| `--output` | `./full/results/...` | Output file path |

## Evaluation Metrics

- **nDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Recall@K**: Recall at K
- **MRR@K**: Mean Reciprocal Rank

Computed at K = {1, 5, 10, 20, 50, 100}

## Dependencies

This pipeline reuses modules from:
- `qe/coir/` — Data loading, dense retrieval, query expansion
- `reranker/` — Cross-encoder implementations (MMMini, XLM, mBERT)
- `sentence-transformers`, `datasets`, `google-generativeai` (for HyDE)
