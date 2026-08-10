import logging

from functools import lru_cache
from typing import Dict, List

try:
    from sentence_transformers import SparseEncoder
except ImportError:
    SparseEncoder = None

from core.config import STUDY_TYPE_WEIGHTS
from core.utils import normalize_space, citation_bonus, year_bonus, classify_study_type
from database import Record

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
    """Cheap pure-Python sparse dot-product - same approach already used by
    AtelierRepository.search_papers_by_vector/get_top_chunks_for_paper."""
    return sum(weight * doc_dict.get(token, 0) for token, weight in query_dict.items())


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

        for rec in cached:
            rec.sparse_score = round(_sparse_dot(query_dict, rec.sparse_vector), 6)

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


def encode_query_sparse(query: str, model_name: str) -> Dict[str, float]:
    """Encodes a single query string into a SPLADE sparse dict for chunk scoring."""
    model = get_sparse_encoder(model_name)
    query_vector = model.encode_query([query])  # 2D tensor: (1, vocab_size)
    decoded = model.decode(query_vector, top_k=None)  # list[list[(token, weight)]], one entry
    return dict(decoded[0])