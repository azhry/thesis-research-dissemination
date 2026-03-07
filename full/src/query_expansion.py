"""
Query Expansion Module for the full pipeline.

Provides multiple expansion strategies:
- translation: Direct Indonesian→English term translation (targeted, no noise)
- embedding: Cross-lingual embedding expansion (original, uses similarity search)
- hyde/technical/cot: LLM-based expansion (requires API)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _setup_coir_path():
    """Add the coir module path to sys.path."""
    qe_path = Path(__file__).parent.parent.parent / "qe"
    coir_path = qe_path / "coir"
    if str(qe_path) not in sys.path:
        sys.path.insert(0, str(qe_path))
    return coir_path


@dataclass
class QEResult:
    """Result of query expansion."""
    original_query: str
    expanded_query: str
    expansion_terms: List[str]
    method: str
    metadata: Dict[str, Any]


# ── Comprehensive Indonesian→English technical term mapping ──
INDONESIAN_TO_ENGLISH = {
    # Basic programming
    "fungsi": "function", "kelas": "class", "metode": "method",
    "variabel": "variable", "parameter": "parameter",
    "mengembalikan": "return", "kembalikan": "return",
    "perulangan": "loop", "loop": "loop",
    "kondisi": "condition", "percabangan": "branch",
    "tipe": "type", "nilai": "value", "kunci": "key",
    
    # Data structures
    "array": "array", "daftar": "list", "kamus": "dictionary",
    "teks": "string", "string": "string",
    "bilangan": "integer", "angka": "number",
    "boolean": "boolean", "objek": "object",
    "tuple": "tuple", "himpunan": "set",
    
    # File operations
    "baca": "read", "tulis": "write", "buka": "open", "tutup": "close",
    "file": "file", "berkas": "file",
    "direktori": "directory", "folder": "directory",
    
    # Web/Database
    "data": "data", "database": "database", "basis data": "database",
    "tabel": "table", "baris": "row", "kolom": "column",
    "query": "query", "permintaan": "request",
    "tanggapan": "response", "respon": "response",
    
    # Common programming actions
    "simpan": "save", "ambil": "get", "hapus": "delete",
    "ubah": "update", "tambah": "add", "tambahkan": "append",
    "cari": "search", "temukan": "find",
    "urut": "sort", "urutkan": "sort",
    "filter": "filter", "saring": "filter",
    "cetak": "print", "tampilkan": "print", "tampil": "display",
    "konversi": "convert", "ubah": "convert",
    "pisahkan": "split", "gabungkan": "join", "gabung": "merge",
    "periksa": "check", "validasi": "validate",
    "hitung": "count", "jumlah": "sum", "rata-rata": "average",
    "bandingkan": "compare",
    "salin": "copy", "pindahkan": "move",
    "buat": "create", "inisialisasi": "initialize",
    "kosong": "empty", "kosongkan": "clear",
    "deklarasikan": "declare", "mendeklarasikan": "declare",
    "definisikan": "define", "mendefinisikan": "define",
    "iterasi": "iterate", "akses": "access",
    
    # Error handling
    "error": "error", "kesalahan": "error",
    "peringatan": "warning", "pengecualian": "exception",
    "tangani": "handle", "tangkap": "catch",
    
    # Data types & operations
    "karakter": "character", "panjang": "length",
    "indeks": "index", "irisan": "slice",
    "acak": "random", "pengurutan": "sorting",
    "rekursif": "recursive", "rekursi": "recursion",
    
    # Testing
    "uji": "test", "pengujian": "test",
    
    # Common modifiers
    "semua": "all", "setiap": "each",
    "pertama": "first", "terakhir": "last",
    "terbalik": "reverse", "berdasarkan": "by",
    "dalam": "in", "dari": "from", "ke": "to",
    "apakah": "whether", "cara": "how",
    "menggunakan": "using", "dengan": "with",
    "tanpa": "without", "antara": "between",
    "hanya": "only", "saja": "only",
}


class TranslationExpander:
    """
    Translation-only query expander.
    
    Identifies Indonesian terms in the query and adds their English
    translations. This is targeted and adds no noise — only terms that
    are direct translations of identified Indonesian words.
    """
    
    def __init__(self):
        # Build a set of known Indonesian terms for efficient lookup
        self._indo_terms = set(INDONESIAN_TO_ENGLISH.keys())
        # Pre-compute multi-word terms (sorted by length, longest first)
        self._multi_word_terms = sorted(
            [t for t in self._indo_terms if ' ' in t],
            key=len, reverse=True,
        )
        # Single word terms
        self._single_word_terms = [t for t in self._indo_terms if ' ' not in t]
    
    def expand(self, query: str) -> QEResult:
        """Expand by translating identified Indonesian terms."""
        query_lower = query.lower()
        translations_added = []
        
        # 1. Check multi-word terms first
        for term in self._multi_word_terms:
            if term in query_lower:
                eng = INDONESIAN_TO_ENGLISH[term]
                if eng not in translations_added:
                    translations_added.append(eng)
        
        # 2. Check single-word terms
        query_words = set(query_lower.split())
        for term in self._single_word_terms:
            if term in query_words:
                eng = INDONESIAN_TO_ENGLISH[term]
                if eng not in translations_added:
                    translations_added.append(eng)
        
        # Build expanded query: original + translated terms
        if translations_added:
            expanded = f"{query} {' '.join(translations_added)}"
        else:
            expanded = query
        
        return QEResult(
            original_query=query,
            expanded_query=expanded,
            expansion_terms=translations_added,
            method="translation",
            metadata={"indonesian_terms_found": [
                t for t in self._single_word_terms if t in query_words
            ]},
        )


class QueryExpander:
    """
    Unified Query Expansion interface.
    
    Supports multiple expansion methods:
    - translation: Direct ID→EN term translation (recommended, no noise)
    - embedding: Cross-lingual embedding expansion (original, uses similarity)
    - hyde: HyDE-style LLM expansion (requires API)
    - technical: Technical enrichment via LLM (requires API)
    - cot: Chain-of-thought via LLM (requires API)
    """
    
    def __init__(
        self,
        method: str = "translation",
        num_terms: int = 5,
        embedding_model: str = "intfloat/multilingual-e5-small",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 512,
        device: str = "cpu",
    ):
        """
        Initialize the query expander.
        
        Args:
            method: Expansion method (translation, embedding, hyde, technical, cot)
            num_terms: Number of expansion terms (for embedding/LLM methods)
            embedding_model: Model for embedding-based expansion
            llm_provider: LLM provider for LLM-based methods
            llm_model: LLM model name
            llm_temperature: LLM temperature
            llm_max_tokens: LLM max tokens
            device: Device for embedding model
        """
        self.method = method
        self.num_terms = num_terms
        self.device = device
        
        self._expander = None
        
        if method == "translation":
            self._expander = TranslationExpander()
            logger.info("Initialized translation-based QE (no noise, ID→EN only)")
        elif method == "embedding":
            _setup_coir_path()
            self._init_embedding_expander(embedding_model)
        elif method in ("hyde", "technical", "cot"):
            _setup_coir_path()
            self._init_llm_expander(llm_provider, llm_model, llm_temperature, llm_max_tokens)
        else:
            raise ValueError(f"Unknown QE method: {method}")
    
    def _init_embedding_expander(self, model_name: str):
        """Initialize embedding-based expander."""
        from coir.embedding_expander import CrossLingualEmbeddingExpander
        
        logger.info(f"Initializing embedding-based QE with model: {model_name}")
        self._expander = CrossLingualEmbeddingExpander(
            model_name=model_name,
            device=self.device,
        )
    
    def _init_llm_expander(self, provider: str, model: str, temperature: float, max_tokens: int):
        """Initialize LLM-based expander."""
        from coir.llm_expander import LLMExpander
        
        logger.info(f"Initializing LLM-based QE with provider={provider}, model={model}")
        self._expander = LLMExpander(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def expand(self, query: str) -> QEResult:
        """
        Expand a single query.
        
        Args:
            query: Input query (typically Indonesian)
        
        Returns:
            QEResult with expanded query
        """
        if self.method == "translation":
            return self._expander.expand(query)
        elif self.method == "embedding":
            result = self._expander.expand(query, num_terms=self.num_terms)
        else:
            # LLM-based methods
            result = self._expander.expand(query, method=self.method, num_terms=self.num_terms)
        
        return QEResult(
            original_query=result.original_query,
            expanded_query=result.expanded_query,
            expansion_terms=result.expansion_terms,
            method=result.method,
            metadata=result.metadata,
        )
    
    def expand_batch(self, queries: Dict[str, str], show_progress: bool = True) -> Dict[str, QEResult]:
        """
        Expand a batch of queries.
        
        Args:
            queries: Dict mapping query_id -> query_text
            show_progress: Show progress bar
        
        Returns:
            Dict mapping query_id -> QEResult
        """
        results = {}
        items = list(queries.items())
        
        for qid, query in tqdm(items, desc=f"QE ({self.method})", disable=not show_progress):
            results[qid] = self.expand(query)
        
        return results
