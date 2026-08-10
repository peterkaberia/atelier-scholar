import math
import re
from typing import Optional, Any, List

def normalize_space(text: str) -> str:
    """Removes extra whitespace and newlines from a string."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def doi_key(doi: str) -> str:
    """Normalizes a DOI string for use as a deduplication dictionary key."""
    return normalize_space(doi).lower().replace("https://doi.org/", "")

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    doi = clean_text(doi)
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    return doi.strip().lower()


def strict_title_key(title: str) -> str:
    """Strips punctuation and spaces, returning the first 60 chars for aggressive deduplication."""
    return re.sub(r'[^a-z0-9]', '', str(title).lower())[:60]


def parse_year_from_date(text: str) -> Optional[int]:
    """Extracts a 4-digit year (19xx or 20xx) from a raw date string."""
    m = re.search(r"(19|20)\d{2}", text or "")
    return int(m.group(0)) if m else None

    
def safe_int(value: Any) -> Optional[int]:
    """Safely converts a value to an integer, returning None on failure."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        match = re.search(r"\b(18|19|20)\d{2}\b", str(value))
        return int(match.group(0)) if match else None


def truncate(text: str, max_chars: int) -> str:
    """Truncates text to a specified character limit, appending an ellipsis if necessary."""
    text = normalize_space(text)
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Splits full text into overlapping, word-boundary-respecting passages for
    independent embedding. Used to turn a whole fetched paper (which can run
    to tens of thousands of characters) into RAG-retrievable chunks instead
    of one giant blob.
    """
    text = normalize_space(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Don't cut mid-word - back off to the last space in this window.
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # guarantee forward progress
    return chunks


def citation_bonus(citation_count: Optional[int]) -> float:
    """Calculates a logarithmic bonus score based on citation count, capped at 3.0."""
    return min(3.0, math.log1p(citation_count) * 0.5) if citation_count and citation_count > 0 else 0.0


def year_bonus(year: Optional[int]) -> float:
    """Calculates a linear recency bonus for papers published between 2005 and 2026."""
    if not year:
        return 0.0
    return 1.0 * ((max(2005, min(2026, year)) - 2005) / 21)


# Checked in order - most specific/highest-evidence-tier categories first, so
# e.g. "a systematic review of randomized controlled trials" classifies as
# the review, not the RCT. Keys match core/config.py's STUDY_TYPE_WEIGHTS
# exactly (Consensus.app-style: a paper's evidence tier feeds its ranking,
# not just topical relevance).
_STUDY_TYPE_PATTERNS: List[tuple] = [
    ("systematic review / meta-analysis", re.compile(
        r"\bsystematic review\b|\bmeta-analys[ie]s\b|\bmeta analys[ie]s\b", re.I)),
    ("guideline / consensus", re.compile(
        r"\bclinical practice guideline\b|\b(?:consensus|position|policy) statement\b|\bguideline[s]?\b", re.I)),
    ("randomized controlled trial", re.compile(
        r"\brandomi[sz]ed(?:[\s-]controlled|[\s-]clinical)?\s+trial\b|\brct\b|\bplacebo-controlled\b|\bdouble-blind\b", re.I)),
    ("cohort study", re.compile(
        r"\b(?:prospective|retrospective)?\s*cohort\s+stud(?:y|ies)\b|\blongitudinal\s+stud(?:y|ies)\b", re.I)),
    ("case-control study", re.compile(r"\bcase-control\b|\bcase\s+control\s+stud(?:y|ies)\b", re.I)),
    ("cross-sectional study", re.compile(r"\bcross-sectional\b|\bcross\s+sectional\b", re.I)),
    ("qualitative study", re.compile(
        r"\bqualitative\s+(?:study|research|analysis)\b|\bthematic\s+analysis\b|\bfocus\s+group\b|\bin-depth\s+interview", re.I)),
    ("mixed-methods study", re.compile(r"\bmixed[\s-]methods\b", re.I)),
    ("case report / case series", re.compile(r"\bcase\s+report\b|\bcase\s+series\b", re.I)),
    ("observational study", re.compile(r"\bobservational\s+stud(?:y|ies)\b", re.I)),
]


_YES_NO_STARTERS = re.compile(
    r"^\s*(does|do|did|is|are|was|were|can|could|should|shall|will|would|has|have|had|may|might|must)\b",
    re.I,
)


def is_yes_no_question(text: str) -> bool:
    """
    Cheap heuristic gate for whether a query is phrased as a yes/no research
    question (e.g. "Does metformin reduce cardiovascular risk?") - mirrors
    Consensus.app's own framing ("Ask a clear yes-or-no research question
    and the Consensus Meter instantly shows...").

    Deliberately just a regex on the opening word(s), not an LLM call: this
    runs on every synthesis regardless of topic, so it needs to be free.
    It's intentionally permissive (a plain string match, not full grammar) -
    false positives just mean llm.engine.generate_consensus_meter's own LLM
    call gets invoked and can still decline via its "applicable": false
    output; false negatives (missing a real yes/no question phrased
    unusually) just mean no meter shows, the same as any non-yes/no query.
    """
    return bool(_YES_NO_STARTERS.match(normalize_space(text or "")))


def classify_study_type(title: str, abstract: str) -> str:
    """
    Cheap regex-based evidence-tier classifier over a paper's title/abstract -
    deliberately NOT an LLM call, since this runs across every raw search
    candidate (hundreds of papers) before extraction ever narrows the field,
    where an LLM call per paper would be far too slow/expensive. Feeds
    search/sparse_encoder.py's STUDY_TYPE_WEIGHTS lookup, which previously
    had nothing populating Record.study_type and so silently gave every
    paper the exact same flat bonus regardless of actual study design.

    Returns "unspecified" (matches STUDY_TYPE_WEIGHTS's own key) when no
    pattern hits - most preprints/general papers won't self-describe with
    one of these terms, and that's fine: they just get no bonus/penalty
    rather than a wrong guess.
    """
    text = f"{title or ''} {abstract or ''}"
    for label, pattern in _STUDY_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "unspecified"

def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def nested_get(obj: Any, *path: str) -> Any:
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def normalize_pmid(pmid: Optional[str]) -> Optional[str]:
    pmid = clean_text(pmid)
    if not pmid:
        return None
    pmid = pmid.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip("/")
    pmid = pmid.replace("pmid:", "")
    return pmid.strip()


def normalize_pmcid(pmcid: Optional[str]) -> Optional[str]:
    pmcid = clean_text(pmcid)
    if not pmcid:
        return None
    pmcid = pmcid.upper().replace("PMCID:", "")
    return pmcid.strip()

def normalize_arxivid(arxiv_id: Optional[str]) -> Optional[str]:
    arxiv_id = clean_text(arxiv_id)
    if not arxiv_id:
        return None
    arxiv_id = arxiv_id.upper().replace("ARXIV ID:", "")
    return arxiv_id.strip()


def make_fingerprint(
    *,
    doi: Optional[str],
    pmid: Optional[str],
    pmcid: Optional[str],
    arxiv_id: Optional[str],
    fallback_source: str,
    fallback_id: Optional[str],
) -> str:
    doi = normalize_doi(doi)
    pmid = normalize_pmid(pmid)
    pmcid = normalize_pmcid(pmcid)

    if doi: 
        return f"doi:{doi}"
    elif pmid: 
        return f"pmid:{pmid}"
    elif pmcid: 
        return f"pmcid:{pmcid}"
    elif arxiv_id:
       return f"arxiv:{arxiv_id}"
    else:
        return f"{fallback_source}:{fallback_id}"


def split_author_string(author_string: Optional[str]) -> list[str]:
    author_string = clean_text(author_string)
    if not author_string:
        return []
    return [part.strip() for part in author_string.split(",") if part.strip()]

