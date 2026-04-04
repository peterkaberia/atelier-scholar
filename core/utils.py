import math
import re
from typing import Optional, Any

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


def citation_bonus(citation_count: Optional[int]) -> float:
    """Calculates a logarithmic bonus score based on citation count, capped at 3.0."""
    return min(3.0, math.log1p(citation_count) * 0.5) if citation_count and citation_count > 0 else 0.0


def year_bonus(year: Optional[int]) -> float:
    """Calculates a linear recency bonus for papers published between 2005 and 2026."""
    if not year: 
        return 0.0
    return 1.0 * ((max(2005, min(2026, year)) - 2005) / 21)

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

