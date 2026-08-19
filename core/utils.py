import math
import re
from typing import Any, List, Optional, Tuple

def normalize_space(text: str) -> str:
    """Removes extra whitespace and newlines from a string."""
    return re.sub(r"\s+", " ", (text or "")).strip()


# Generic English function words plus a few research-phrasing filler words
# ("factors", "affecting", "role", "impact"...) that carry no topical
# meaning on their own - stripped so pipeline.nodes.broaden_query_node's
# keyword fallback ANDs together only the actual subject-matter terms.
_QUERY_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with", "is", "are", "was", "were",
    "this", "that", "these", "those", "by", "at", "as", "it", "its", "be", "been", "being", "from",
    "does", "do", "did", "can", "could", "should", "would", "will", "shall", "has", "have", "had",
    "affecting", "affect", "affects", "factors", "factor", "related", "regarding", "about", "among",
    "role", "impact", "effect", "effects", "between", "into", "what", "how", "why", "which",
}


def extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """
    Strips stopwords/punctuation from a natural-language query, returning
    the remaining significant keyword tokens in their original order
    (duplicates removed) - used by pipeline.nodes.broaden_query_node to
    build a field-tagged fallback query for PubMed/Europe PMC instead of
    sending them an untagged raw sentence (see that function's docstring
    for why: their parsers handle a bag of tagged keywords far more
    reliably than a whole phrase, especially one with a typo or unusual
    phrasing that trips up phrase-matching heuristics).
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]*", text or "")
    seen = set()
    out = []
    for w in words:
        lw = w.lower()
        if len(lw) < min_len or lw in _QUERY_STOPWORDS or lw in seen:
            continue
        seen.add(lw)
        out.append(lw)
    return out


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

_ABSOLUTE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def normalize_url(url: Optional[str]) -> Optional[str]:
    """
    Normalizes a pdf_url/full_text_url pulled from an external API (Europe
    PMC, OpenAlex, Semantic Scholar, Crossref, arXiv - search/paper.py) into
    something a browser can actually resolve. Called from
    database.models.Record.__post_init__ so every Record gets this
    regardless of which fetcher built it.

    Some upstream APIs occasionally hand back a URL missing its scheme -
    a protocol-relative "//host/paper.pdf" or a bare "example.com/paper.pdf".
    Passed straight through to an <a href> or the in-app PDF viewer's
    `iframe.src = url` (ui/layouts/main.py's index_string), either resolves
    against the CURRENT page's own origin instead of the intended host,
    producing a link that looks plausible but 404s or silently never loads -
    confirmed as a distinct failure mode from the already-expected
    403/404s publishers return for a genuinely correct, reachable URL.
    Recovers a scheme when one is clearly recoverable; otherwise discards
    the value entirely (None) so the UI falls back to its normal "no PDF
    available" state instead of showing a link that's guaranteed to fail.
    """
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    if _ABSOLUTE_URL_RE.match(url):
        return url
    if url.startswith("//"):
        return f"https:{url}"
    # A bare domain-like string ("example.com/paper.pdf", "www.example.com")
    # - a dot in the part before the first slash, and no whitespace, reads
    # as a host that just lost its scheme rather than an already-broken
    # path fragment.
    head = url.split("/", 1)[0]
    if "." in head and " " not in url:
        return f"https://{url}"
    return None

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


# Matches a grouped citation bracket like "[doi:A; doi:B]" or "[1, 2]" -
# two or more marker-shaped tokens (a colon-delimited fingerprint, or a
# bare digit) joined by a comma/semicolon inside ONE pair of brackets. See
# order_citations_by_appearance's use of this for why it needs splitting
# before that function's own exact-marker matching runs.
_MULTI_MARKER_RE = re.compile(
    r"\[((?:[a-zA-Z0-9]+:[^\];,\s]+|\d+)(?:\s*[;,]\s*(?:[a-zA-Z0-9]+:[^\];,\s]+|\d+))+)\]"
)

def order_citations_by_appearance(text: str, candidates: List[Tuple[str, Any]]) -> Tuple[str, List[Any]]:
    """
    Renumbers a text's [marker] citations so [1] is whichever candidate is
    FIRST cited reading top-to-bottom, [2] is the next NEW one encountered,
    and so on - instead of whatever order the candidates were pre-assigned
    in before the text was even written.

    Why this matters: generate_copilot_synthesis/chat_with_literature both
    pre-assign [1]/[2]/... to valid_records in RELEVANCE-SCORE order before
    the LLM writes a word, and the LLM cites using those pre-assigned
    numbers - so a synthesis routinely cites [4] before [1] simply because
    the 4th-most-relevant paper happened to fit better into the opening
    sentence. Confirmed live and reported directly: readers expect [1] to
    be whatever's cited first, not "whatever scored highest." Called on
    each turn's own output independently (not any shared session-wide
    order), so two turns citing an overlapping paper can freely give it
    different numbers, matching THEIR OWN reading order.

    Args:
        text: the raw output containing "[marker]" citations - marker is
            whatever string was used when the text was generated (a
            1-based index like "1", "2", ... for generate_copilot_synthesis
            /chat_with_literature, or a paper fingerprint like
            "doi:10.1234/xyz" for the investigation agent - anything that
            appears literally as "[marker]" in the text works).
        candidates: [(marker, record), ...] in whatever order they were
            available in (NOT necessarily citation order) - every record
            actually cited in `text` gets kept; anything not cited is
            dropped, same "only truly-cited papers get attached" behavior
            the callers already relied on before this renumbering existed.

    Returns:
        (rewritten_text, ordered_records): rewritten_text has every
        [marker] replaced with its new 1-based position; ordered_records
        is in that SAME new order, so ordered_records[i] is exactly what
        [i+1] refers to in rewritten_text - callers can pass this directly
        wherever a synthesis's valid_records/citations are expected.
    """
    # Split a grouped citation like "[doi:A; doi:B]" or "[1, 2]" into
    # separate "[doi:A][doi:B]" brackets FIRST - the matching below looks
    # for an EXACT "[marker]" substring per candidate, so a grouped
    # bracket doesn't equal any single candidate's marker and was left
    # completely un-rewritten even when every marker inside it was
    # individually valid and already cited elsewhere in the text
    # (confirmed live: an investigation-agent report that grouped two
    # already-cited fingerprints together this way left exactly that one
    # bracket as raw, unclickable text in an otherwise fully-renumbered
    # synthesis).
    text = _MULTI_MARKER_RE.sub(
        lambda m: "".join(f"[{t}]" for t in re.split(r"\s*[;,]\s*", m.group(1))),
        text or "",
    )

    first_seen = []
    for marker, record in candidates:
        idx = text.find(f"[{marker}]")
        if idx != -1:
            first_seen.append((idx, marker, record))
    first_seen.sort(key=lambda entry: entry[0])

    if not first_seen:
        return text, []

    marker_to_new = {}
    ordered_records = []
    for new_num, (_, old_marker, record) in enumerate(first_seen, start=1):
        # A marker could legitimately map to more than one candidate only
        # if callers pass duplicates - first one wins, matching dict
        # insertion behavior generally expected here.
        marker_to_new.setdefault(old_marker, new_num)
        ordered_records.append(record)

    # Single regex pass, not sequential .replace() calls: renumbering
    # routinely produces overlapping old/new numbers (e.g. old [2] -> new
    # [1] while old [1] -> new [2]), and replacing them one at a time risks
    # a second replace() matching text a PRIOR replace() just inserted,
    # silently corrupting the result. re.sub with a lookup callback applies
    # every substitution against the ORIGINAL text in one pass instead.
    pattern = re.compile(r"\[(" + "|".join(re.escape(m) for m in marker_to_new.keys()) + r")\]")
    rewritten = pattern.sub(lambda m: f"[{marker_to_new[m.group(1)]}]", text)

    return rewritten, ordered_records


_FINGERPRINT_MARKER_RE = re.compile(r"\[([a-zA-Z0-9]+:[^\]\s]+)\]")

def extract_bracket_fingerprints(text: str) -> List[str]:
    """
    Finds every distinct "[prefix:value]" bracket marker in `text` that's
    SHAPED like a Record fingerprint (make_fingerprint always produces
    "doi:...", "pmid:...", "pmcid:...", "arxiv:...", or a
    "{source}:{source_id}" fallback - always colon-delimited) - used by
    pipeline.agent's investigation agent (see its system prompt: "cite
    papers using their fingerprint in brackets, e.g. [doi:10.1234/xyz]") to
    find every paper it MIGHT have cited, as candidates for
    order_citations_by_appearance beyond whatever a narrower DB query
    already found (see run_investigation's docstring in
    ui/callbacks/chat.py for why that narrower query alone isn't enough).

    Callers are expected to look each result up and silently discard
    anything that isn't a real, known fingerprint - this only recognizes
    the SHAPE of a marker, not whether one genuinely exists, so a markdown
    link written as "[Some Report](url)" or a stray "[Note: see above]"
    can produce a false-positive candidate that a DB lookup then filters
    out harmlessly.
    """
    return sorted(set(_FINGERPRINT_MARKER_RE.findall(text or "")))

