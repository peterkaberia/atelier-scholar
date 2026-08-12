import logging

from functools import lru_cache
from typing import Dict, List

import numpy as np

try:
    from sentence_transformers import SparseEncoder
except ImportError:
    SparseEncoder = None

from core.config import STUDY_TYPE_WEIGHTS
from core.utils import normalize_space, citation_bonus, year_bonus, classify_study_type
from database.models import Record

logger = logging.getLogger(__name__)


# ==========================================
# EMBEDDING INFRASTRUCTURE
# ==========================================

@lru_cache(maxsize=2)
def get_sparse_encoder(model_name: str):
    """
    Loads the SPLADE/Transformer model ONCE and keeps it in memory.
    The @lru_cache decorator ensures that subsequent calls return the 
    already-loaded model instantly, preventing massive lag.
    """
    if SparseEncoder is None: 
        raise ImportError("SparseEncoder module is not available.")
        
    logger.info(f"Loading SparseEncoder '{model_name}' into memory for the first time...")
    return SparseEncoder(model_name)

# ==========================================
# SCORING LOGIC
# ==========================================

def _sparse_dot(query_dict: Dict[str, float], doc_dict: Dict[str, float]) -> float:
    """Cheap pure-Python sparse dot-product - a single query-vs-single-doc
    comparison. For scoring MANY docs against one query, use
    sparse_batch_dot instead - see its docstring for why."""
    return sum(weight * doc_dict.get(token, 0) for token, weight in query_dict.items())


def sparse_batch_dot(query_dict: Dict[str, float], doc_dicts: List[Dict[str, float]]) -> List[float]:
    """
    Scores MANY documents against one SPLADE query vector in a single
    vectorized numpy operation, instead of len(doc_dicts) separate
    Python-level dict-iteration dot products (the previous approach,
    still used by _sparse_dot for genuine one-off comparisons). Used
    anywhere a query gets compared against a whole batch at once -
    AtelierRepository.search_papers_by_vector (every cached paper in the
    local DB), .get_top_chunks_for_paper (every chunk of one paper), and
    compute_scores's cached-paper branch.

    Builds a LOCAL token->index map from just the query's own tokens (not
    the full ~30K SPLADE vocab) - a document's tokens that aren't in the
    query can never contribute to its dot product against that query, so
    there's no need to index them. That keeps the matrix small (bounded by
    however many tokens the query itself has, typically a few dozen)
    regardless of how large the underlying SPLADE vocabulary is.

    Returns a plain list of floats, same order as doc_dicts, so callers
    don't need to know this is numpy under the hood.
    """
    if not doc_dicts:
        return []
    if not query_dict:
        return [0.0] * len(doc_dicts)

    vocab = {token: i for i, token in enumerate(query_dict.keys())}
    query_vec = np.zeros(len(vocab), dtype=np.float32)
    for token, weight in query_dict.items():
        query_vec[vocab[token]] = weight

    doc_matrix = np.zeros((len(doc_dicts), len(vocab)), dtype=np.float32)
    for i, doc_dict in enumerate(doc_dicts):
        for token, weight in (doc_dict or {}).items():
            idx = vocab.get(token)
            if idx is not None:
                doc_matrix[i, idx] = weight

    return (doc_matrix @ query_vec).tolist()


def compute_scores(records: List[Record], topic: str, model_name: str) -> None:
    """
    Calculates Step 1 Neural Sparse Scores (SPLADE) and Step 2 Heuristic Scores.
    Modifies the Record objects in place to append `step2_score_100`.

    Records that already carry a `sparse_vector` (persisted from a previous
    run - see AtelierRepository.search_papers_by_vector, which now loads it
    back onto every locally-recalled Record) skip the neural forward pass
    entirely and are scored via the same cheap sparse dot-product the local
    vector-search/chunk-RAG paths already use. Only genuinely new records pay
    for a fresh model.encode_document() call - on a search that overlaps a
    lot with previous ones, this can skip embedding most of the candidate
    pool instead of re-running the model on every paper every time.
    """
    try:
        # 1. Fetch the cached model (Instantaneous after first load!)
        model = get_sparse_encoder(model_name)

        query_vector = model.encode_query([topic])
        query_dict = dict(model.decode(query_vector, top_k=None)[0])

        to_embed = [r for r in records if not r.sparse_vector]
        cached = [r for r in records if r.sparse_vector]

        # Batched, not one _sparse_dot call per paper - see
        # sparse_batch_dot's docstring. Matters most here since a search
        # that heavily overlaps prior ones can have most of its candidate
        # pool land in `cached`.
        if cached:
            scores = sparse_batch_dot(query_dict, [r.sparse_vector for r in cached])
            for rec, score in zip(cached, scores):
                rec.sparse_score = round(score, 6)

        if to_embed:
            docs = [normalize_space(f"{r.title}\n{r.abstract}") for r in to_embed]
            doc_vectors = model.encode_document(docs)
            decoded = model.decode(doc_vectors, top_k=128)
            similarities = model.similarity(query_vector, doc_vectors)[0].tolist()

            for idx, rec in enumerate(to_embed):
                rec.sparse_vector = dict(decoded[idx])
                rec.sparse_score = round(float(similarities[idx]), 6)

        if cached:
            logger.info(f"Sparse scoring: reused {len(cached)} cached vectors, freshly embedded {len(to_embed)}.")

    except Exception as e:
        logger.warning(f"⚠️ Sparse scoring failed: {e}. Falling back to heuristic defaults.")
        # Fallback pseudo-score if local Transformer models fail
        for rec in records:
            rec.sparse_score = len(rec.abstract) / 1000.0

    # 3. Calculate Final Scores
    max_sparse = max((r.sparse_score for r in records), default=1.0) or 1.0
    
    for rec in records:
        # study_type is never set by any fetcher/extraction step - classify
        # it here from title/abstract text so STUDY_TYPE_WEIGHTS actually has
        # real signal to act on. Previously every record's study_type was
        # the dataclass default "" (not "unspecified"), which always missed
        # the dict lookup AND always tripped the old "unknown but non-empty"
        # fallback bump - every paper silently got the exact same +1.0
        # regardless of its real study design. See classify_study_type's
        # docstring.
        if not rec.study_type:
            rec.study_type = classify_study_type(rec.title, rec.abstract)

        st_weight = STUDY_TYPE_WEIGHTS.get(rec.study_type.lower(), 0.0)

        rec.heuristic_score = citation_bonus(rec.citation_count) + year_bonus(rec.year) + st_weight + (max(0, len(rec.all_sources) - 1) * 1.25)
        rec.step2_score_100 = ((rec.sparse_score / max_sparse) * 100.0) + (rec.heuristic_score * 2)


# ==========================================
# FULL-TEXT CHUNK ENCODING (for RAG retrieval)
# ==========================================

def encode_chunks(chunks: List[str], model_name: str, top_k: int = 128) -> List[Dict[str, float]]:
    """
    Encodes text passages into SPLADE sparse dicts (e.g. {"cell": 1.2}) ready
    for JSON storage on PaperChunk. Capped to the top_k highest-weighted
    tokens per chunk to keep stored vectors small.
    """
    if not chunks:
        return []
    model = get_sparse_encoder(model_name)
    doc_vectors = model.encode_document(chunks)  # 2D tensor: (len(chunks), vocab_size)
    decoded = model.decode(doc_vectors, top_k=top_k)  # list[list[(token, weight)]], one per chunk
    return [dict(pairs) for pairs in decoded]


@lru_cache(maxsize=256)
def encode_query_sparse(query: str, model_name: str) -> Dict[str, float]:
    """
    Encodes a single query string into a SPLADE sparse dict for chunk
    scoring. Cached (same pattern as get_sparse_encoder above) - an
    investigation agent calling this repeatedly with overlapping/identical
    queries during one run shouldn't pay for a fresh neural forward pass
    each time. Safe to cache despite returning a dict: every caller
    (compute_scores, sparse_batch_dot, get_top_chunks_for_paper,
    search_papers_by_vector) only reads it, never mutates it in place.
    """
    model = get_sparse_encoder(model_name)
    query_vector = model.encode_query([query])  # 2D tensor: (1, vocab_size)
    decoded = model.decode(query_vector, top_k=None)  # list[list[(token, weight)]], one entry
    return dict(decoded[0])