"""Fetch COSQA queries to translate."""
import sys
import json
from pathlib import Path

# Add coir to path
coir_path = Path(__file__).parent / "coir"
sys.path.insert(0, str(coir_path))

from coir.data_loader import load_data_from_hf

# Load data
corpus, queries, qrels = load_data_from_hf("cosqa")

# Save to JSON
output = {
    "corpus": corpus,
    "queries": queries,
    "qrels": qrels
}

output_file = Path(__file__).parent / "cosqa_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved to {output_file}")
print(f"Queries: {len(queries)}")
print(f"Corpus: {len(corpus)}")
print(f"Qrels: {len(qrels)}")

# Print queries for translation
print("\n=== QUERIES ===")
for qid, qtext in sorted(queries.items(), key=lambda x: int(x[0][1:])):
    print(f"{qid}|{qtext}")
