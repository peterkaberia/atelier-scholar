import logging

from functools import lru_cache
from typing import List

try:
    from sentence_transformers import SparseEncoder
except ImportError:
    SparseEncoder = None

from core.config import STUDY_TYPE_WEIGHTS
from core.utils import normalize_space, citation_bonus, year_bonus
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

def compute_scores(records: List[Record], topic: str, model_name: str) -> None:
    """
    Calculates Step 1 Neural Sparse Scores (SPLADE) and Step 2 Heuristic Scores.
    Modifies the Record objects in place to append `step2_score_100`.
    """
    try:
        # 1. Fetch the cached model (Instantaneous!)
        model = get_sparse_encoder(model_name)
        
        docs = [normalize_space(f"{r.title}\n{r.abstract}") for r in records]

        # 2. Encode the query and the documents SEPARATELY
        query_vector = model.encode_query([topic])
        doc_vectors = model.encode_document(docs)
        
        # 2. Compute Similarities
        similarities = model.similarity(query_vector, doc_vectors)
        scores_list = similarities[0].tolist() 
        
        for idx, rec in enumerate(records): 
            rec.sparse_score = round(float(scores_list[idx]), 6)
            rec.doc_vector = doc_vectors[idx]
            
    except Exception as e:
        logger.warning(f"⚠️ Sparse scoring failed: {e}. Falling back to heuristic defaults.")
        # Fallback pseudo-score if local Transformer models fail
        for rec in records: 
            rec.sparse_score = len(rec.abstract) / 1000.0

    # 3. Calculate Final Scores
    max_sparse = max((r.sparse_score for r in records), default=1.0) or 1.0
    
    for rec in records:
        st_weight = STUDY_TYPE_WEIGHTS.get(rec.study_type.lower(), 0.0)
        if st_weight == 0.0 and rec.study_type != "unspecified": 
            st_weight = 1.0  # Give a slight bump for knowing the study type at all
            
        rec.heuristic_score = citation_bonus(rec.citation_count) + year_bonus(rec.year) + st_weight + (max(0, len(rec.all_sources) - 1) * 1.25)
        rec.step2_score_100 = ((rec.sparse_score / max_sparse) * 100.0) + (rec.heuristic_score * 2)