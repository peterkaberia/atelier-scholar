"""
Research Pipeline Nodes Module

Each function is a LangGraph node: it receives the current ResearchState and
returns a partial dict of updates. Routing functions inspect the state and
return the name of the next node.

Node graph shape:

    ensure_session -> plan_queries -> execute_search -*-> score_and_save -> extract -*-> END
                                            ^                                    |
                                            |                                    |
                                            +---------------- broaden_query <----+
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import RunnableConfig

from core.config import DEFAULT_SPARSE_MODEL
from core.utils import chunk_text, extract_keywords, truncate
from database import AtelierRepository
from llm.engine import AtelierAIEngine
from search import AtelierAcademicSearch
from search.sparse_encoder import compute_scores, encode_chunks, encode_query_sparse

from .state import ResearchState

logger = logging.getLogger(__name__)

# Character budget for extraction payloads - matches the abstract-only budget
# the pipeline already used, so full-text RAG doesn't blow past token limits.
EXTRACTION_CONTENT_BUDGET = 12000

# How many of a paper's most relevant passages to feed into extraction.
CHUNKS_PER_PAPER = 5

# How many papers go into a single batched extraction LLM call (see
# extract_node). A smaller, separate content budget applies per paper INSIDE
# a batch - batch_size * BATCH_EXTRACTION_CONTENT_BUDGET has to stay within
# a reasonable prompt size for whatever model the user has configured
# (including smaller local models with limited context windows), whereas
# EXTRACTION_CONTENT_BUDGET above assumes one paper has the whole budget to
# itself.
EXTRACTION_BATCH_SIZE = 4
BATCH_EXTRACTION_CONTENT_BUDGET = 4000

# Character budget for the single short passage carried through to
# generate_copilot_synthesis (llm/engine.py) per paper - deliberately much
# smaller than the extraction budgets above: this one gets duplicated once
# per cited paper INTO the final synthesis prompt (up to ~20 papers), so it
# has to stay small per-paper to keep that prompt's total size reasonable.
SYNTHESIS_PASSAGE_BUDGET = 400

# How many genuinely viable (real abstract, or a fetchable full-text
# source) candidates extraction tries to gather, and how large a
# relevance-ranked pool to pull from the DB to backfill from if some of the
# highest-ranked entries turn out unusable (e.g. a full_text_url that's
# just a paywalled DOI-resolver link, not real fetchable content). A fixed
# "top 20" SQL LIMIT with no backfill meant losing even half of a real
# top-20 to unusable entries silently shrank the effective candidate pool
# well below 20 - confirmed live against a real session (196 candidates,
# only 10 of the top 20 had any usable content at all).
EXTRACTION_TARGET_COUNT = 20
EXTRACTION_CANDIDATE_POOL = 60

# A Query must surface at least this many relevant papers before we accept it.
MIN_RELEVANT_PAPERS = 3

# Total number of times we're willing to broaden the query and retry the
# search/extraction loop, whether triggered by zero results or too few
# relevant papers. Shared across both triggers so the graph can't loop forever.
MAX_RETRIES = 2


# ==========================================
# 1. SESSION & PLANNING NODES
# ==========================================

def ensure_session_node(state: ResearchState) -> Dict[str, Any]:
    """Creates the session record on first entry; a no-op on retries."""
    if not AtelierRepository.session_exists(state["session_id"]):
        AtelierRepository.create_session(
            session_id=state["session_id"],
            topic=state["topic"],
            summary="",
        )
    return {"status": "Session Ready"}


def plan_queries_node(state: ResearchState) -> Dict[str, Any]:
    """Asks the LLM to plan per-engine boolean/keyword search strings."""
    ai = AtelierAIEngine(model_choice=state["model_choice"])
    plan = ai.plan_topic_queries(state["topic"])
    return {"query_plan": plan, "status": "Queries Planned"}


# ==========================================
# 2. SEARCH & RETRY NODES
# ==========================================

def execute_search_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Fans the query plan out across the local DB + all academic engines, run in parallel."""
    searcher = AtelierAcademicSearch(db_repository=AtelierRepository)
    records = searcher.run_comprehensive_search(
        state["query_plan"], current_topic=state["topic"], progress_callback=_get_progress_callback(config)
    )
    return {
        "candidate_records": records,
        "search_warnings": searcher.last_warnings,
        "status": "Search Executed",
    }


def broaden_query_node(state: ResearchState) -> Dict[str, Any]:
    """
    Fallback handler: relaxes the boolean/field-tagged query plan to widen
    recall after the LLM's structured plan returned zero candidates across
    every engine (usually because it produced an invalid/nonexistent MeSH
    descriptor or malformed field-tag syntax - a real risk with smaller/
    local models, which don't reliably know NLM's exact controlled
    vocabulary the way a large hosted model does).

    OpenAlex/Semantic Scholar/Crossref/arXiv are plain relevance-ranked
    keyword search with no special syntax to get wrong, so the raw topic
    sentence works fine for them as-is. PubMed and Europe PMC are NOT -
    dropping them to the SAME raw, untagged sentence (confirmed live: a
    topic with a typo returned 0 PubMed results and only 2 from Europe PMC,
    while the other four engines returned dozens) hands their parsers an
    unstructured phrase, and PubMed's Automatic Term Mapping in particular
    can fail outright on a phrase that doesn't cleanly match rather than
    gracefully degrading. Building an explicit OR-of-keywords[tiab]/
    TITLE:/ABSTRACT: query instead (via core.utils.extract_keywords) keeps
    them field-tagged and predictable - no MeSH-exact-match requirement
    (the thing that just failed), and OR rather than AND means one bad
    keyword (a typo, an odd word choice) can't zero out the whole query the
    same way one bad MeSH term just did.
    """
    fallback = state["topic"]
    keywords = extract_keywords(fallback)

    # OR, not AND: this is the LAST-RESORT recall-widening step, and a
    # single bad keyword (a typo, an odd word choice) inside an AND chain
    # zeroes out the entire query just like the MeSH mismatch that got us
    # here in the first place - confirmed live against PubMed's real API
    # with a genuinely typo'd topic. OR only needs ONE keyword to match.
    pubmed_fallback = " OR ".join(f'"{kw}"[tiab]' for kw in keywords) if keywords else fallback
    europe_pmc_fallback = " OR ".join(f'(TITLE:"{kw}" OR ABSTRACT:"{kw}")' for kw in keywords) if keywords else fallback

    plan = {
        "pubmed_query": pubmed_fallback,
        "europe_pmc_query": europe_pmc_fallback,
        "openalex_query": fallback,
        "semantic_scholar_query": fallback,
        "crossref_query": fallback,
        "arxiv_query": fallback,
    }
    next_retry = state.get("retry_count", 0) + 1
    logger.warning(f"Broadening query (retry {next_retry}/{MAX_RETRIES}): pubmed='{pubmed_fallback}', europe_pmc='{europe_pmc_fallback}', others='{fallback}'")
    return {
        "query_plan": plan,
        "retry_count": next_retry,
        "status": "Query Broadened",
    }


def route_after_search(state: ResearchState) -> str:
    """Decides whether to score the candidate pool, retry, or give up empty."""
    if state.get("candidate_records"):
        return "score_and_save"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "broaden_query"
    logger.warning("Pipeline aborting: no records found across any database after retries.")
    return "end"


# ==========================================
# 3. SCORING, PERSISTENCE & EXTRACTION NODES
# ==========================================

def score_and_save_node(state: ResearchState) -> Dict[str, Any]:
    """Computes SPLADE + heuristic scores and persists the candidate pool."""
    records = state["candidate_records"]
    if records:
        compute_scores(records, state["topic"], DEFAULT_SPARSE_MODEL)
        AtelierRepository.save_bulk_records(state["session_id"], records)
    return {"status": "Scored and Saved"}


def _fetch_and_encode_full_text(rec: Any, searcher: AtelierAcademicSearch):
    """
    Pure fetch + encode, with NO database writes: downloads a paper's full
    text and turns it into SPLADE chunk vectors, entirely in memory.

    Deliberately split out from persistence (see save step in
    _ensure_full_text_chunks/_prefetch_full_text below) so this half - the
    actual network I/O and model inference - can safely run from multiple
    threads at once, while every database WRITE stays confined to a single
    thread. SQLite allows only one writer at a time; when _prefetch_full_text
    ran fetch-encode-AND-persist together inside each of up to 8 worker
    threads, several threads could end up contending for that single writer
    slot simultaneously, which produced a silent, near-zero-CPU hang during
    live testing (a real, reproducible bug, not a fluke) - not from the ORM
    choice specifically (Peewee's default SQLite config isn't safe for
    concurrent multi-thread writes either), just from writing to SQLite from
    several threads at once at all.

    Returns None if this paper doesn't need fetching at all (already cached,
    per AtelierRepository.needs_full_text_fetch). Otherwise returns
    (fingerprint, chunks, vectors) - chunks/vectors are empty lists if the
    paper has no fetchable full text or the fetch/encode failed, so the
    caller still knows to mark it attempted.
    """
    if not AtelierRepository.needs_full_text_fetch(rec.fingerprint):
        return None
    if not (getattr(rec, "pmcid", None) or getattr(rec, "pdf_url", None) or getattr(rec, "full_text_url", None)):
        return (rec.fingerprint, [], [])

    try:
        full_text = searcher.get_full_text(
            pmcid=rec.pmcid or "",
            pdf_url=rec.pdf_url or "",
            full_text_url=getattr(rec, "full_text_url", "") or "",
        )
        if not full_text:
            return (rec.fingerprint, [], [])
        chunks = chunk_text(full_text)
        vectors = encode_chunks(chunks, DEFAULT_SPARSE_MODEL)
        return (rec.fingerprint, chunks, vectors)
    except Exception as e:
        logger.warning(f"Full-text fetch/encode failed for {rec.fingerprint}: {e}")
        return (rec.fingerprint, [], [])


def _persist_full_text_result(result) -> None:
    """The single-threaded write half of the fetch/encode split above."""
    if result is None:
        return
    fingerprint, chunks, vectors = result
    if chunks:
        AtelierRepository.save_paper_chunks(fingerprint, chunks, vectors)
    AtelierRepository.mark_full_text_fetched(fingerprint)


def _ensure_full_text_chunks(rec: Any, searcher: AtelierAcademicSearch) -> None:
    """
    Synchronous fetch+persist for a single paper - used by the main
    extraction loop as a cheap cache-check fallback (the real work already
    happened in _prefetch_full_text for anything that went through it).
    """
    _persist_full_text_result(_fetch_and_encode_full_text(rec, searcher))


def _build_extraction_content(
    rec: Any,
    topic_vector: Optional[Dict[str, float]],
    budget: int = EXTRACTION_CONTENT_BUDGET,
) -> str:
    """
    Prefers the paper's most relevant full-text passages (real RAG) over its
    abstract, falling back to the abstract when no chunks are available or
    scoring fails for any reason.

    Args:
        budget: Character cap for the returned content - smaller when this
            paper is one of several sharing a single batched extraction
            call (see extract_node/BATCH_EXTRACTION_CONTENT_BUDGET), since
            the prompt has to fit ALL of them, not just one.
    """
    if topic_vector:
        try:
            top_chunks = AtelierRepository.get_top_chunks_for_paper(
                rec.fingerprint, topic_vector, top_k=CHUNKS_PER_PAPER
            )
            if top_chunks:
                return truncate("\n\n".join(top_chunks), budget)
        except Exception as e:
            logger.warning(f"Chunk retrieval failed for {rec.fingerprint}, falling back to abstract: {e}")

    return truncate(rec.abstract, budget)


def _get_progress_callback(config: Optional[RunnableConfig]) -> Optional[Callable[[str], None]]:
    """
    Pulls an optional progress-reporting callback out of a node's
    RunnableConfig (config={"configurable": {"progress_callback": fn}},
    set by pipeline/orchestrator.py). Used for the extraction loop's
    per-paper updates - node-to-node progress is already visible for free
    via graph.stream(), but that only yields once a whole node finishes, so
    extract_node (which loops over up to 20 papers, each a network call)
    needs its own finer-grained reporting to avoid a single long silent gap.
    """
    if not config:
        return None
    return (config.get("configurable") or {}).get("progress_callback")


def _has_extractable_content(rec: Any) -> bool:
    """
    True if this paper has SOME usable text to work with - an abstract, or
    a fetchable full-text source (pmcid/pdf_url/full_text_url).

    Gating on "has an abstract" alone (the previous check) was too strict:
    SPLADE/heuristic scoring (search/sparse_encoder.py's compute_scores)
    ranks on title+abstract, but a paper can still score highly on title
    alone even with an EMPTY abstract - confirmed live, exactly half of a
    real top-20 had no abstract at all, yet every one of them had a
    full_text_url. The old "if not rec.abstract: continue" check threw all
    of those out before extraction ever got a chance to fetch their real
    content, silently shrinking the effective candidate pool well below 20
    even once relevance ranking itself was fixed and correct.
    """
    return bool(rec.abstract) or bool(getattr(rec, "pmcid", None)) or bool(getattr(rec, "pdf_url", None)) or bool(getattr(rec, "full_text_url", None))


def _prefetch_full_text(top_records: list, searcher: AtelierAcademicSearch, report: Optional[Callable[[str], None]]) -> None:
    """
    Fetches/chunks/embeds full text for every candidate paper CONCURRENTLY,
    ahead of the (necessarily sequential, one shared LLM endpoint) extraction
    loop below. Each paper's fetch is pure network I/O against a different
    host (Europe PMC or a publisher's own PDF URL) plus a local SPLADE
    encode - independent work with nothing to serialize on - so doing this
    upfront removes that network wait from the extraction loop's critical
    path entirely, instead of paying for it inline once per paper right
    before that paper's LLM call. Papers already cached (see
    AtelierRepository.needs_full_text_fetch) return instantly here, so this
    is a no-op cost on any paper that's been through it before.

    Only the fetch+encode half (_fetch_and_encode_full_text) runs inside the
    worker threads; every database write happens back here in the main
    thread, one at a time, as each future completes - see
    _fetch_and_encode_full_text's docstring for why writing from multiple
    threads at once caused a real, silent hang during testing.
    """
    # Skip records extraction will skip anyway (no usable content, or
    # already permanently answered from a prior session) - no point
    # fetching text for a paper we're not about to send to the LLM.
    candidates = [
        rec for rec in top_records
        if _has_extractable_content(rec) and not (getattr(rec, "answer", None) and rec.answer != "-")
    ]
    if not candidates:
        return

    if report:
        report(f"Fetching full text for {len(candidates)} papers in parallel...")

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        futures = {pool.submit(_fetch_and_encode_full_text, rec, searcher): rec for rec in candidates}
        for future in as_completed(futures):
            rec = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.warning(f"Full-text prefetch failed for {rec.fingerprint}: {e}")
                continue
            # Write, sequentially, still in the main thread.
            _persist_full_text_result(result)


def extract_records(
    session_id: str,
    topic: str,
    model_choice: str,
    limit: int = EXTRACTION_TARGET_COUNT,
    pool_size: int = EXTRACTION_CANDIDATE_POOL,
    exclude_fingerprints: Optional[list] = None,
    report: Optional[Callable[[str], None]] = None,
) -> list:
    """
    Pulls the top-scored (and not-yet-excluded) papers for this session and
    runs AI extraction on each one not already permanently cached, keeping
    only papers the LLM marks relevant. Where full text is available (Europe
    PMC XML or a direct PDF), extraction is grounded in the most relevant
    passages via chunk-level RAG rather than a truncated abstract.

    Factored out of extract_node so ui/callbacks/library.py's load_more
    callback can run the exact same extraction routine against a session
    that's ALREADY been searched, fetching the next batch of previously-
    unshown papers (via exclude_fingerprints) instead of the initial top-20
    - same quality bar, same caching behavior, no duplicated logic to drift
    out of sync between the two call sites.
    """
    ai = AtelierAIEngine(model_choice=model_choice)
    searcher = AtelierAcademicSearch(db_repository=AtelierRepository)

    # Pull a larger relevance-ranked pool than we actually need, then trim
    # to the top `limit` that are genuinely viable - this is the backfill
    # described in EXTRACTION_TARGET_COUNT's docstring above. Order is
    # preserved (still relevance_score DESC from the SQL query), so this
    # always prefers the highest-ranked usable papers first.
    candidate_pool = AtelierRepository.get_top_unprocessed_records(session_id, limit=pool_size, exclude_fingerprints=exclude_fingerprints)
    top_records = [rec for rec in candidate_pool if _has_extractable_content(rec)][:limit]

    processed: list[dict] = []

    _prefetch_full_text(top_records, searcher, report)

    # Encode the topic once and reuse it to score every paper's chunks -
    # re-encoding per paper would be wasted, identical work.
    try:
        topic_vector = encode_query_sparse(topic, DEFAULT_SPARSE_MODEL)
    except Exception as e:
        logger.warning(f"Topic query encoding failed; full-text RAG disabled for this run: {e}")
        topic_vector = None

    def _attach_top_passage(rec: Any) -> None:
        """
        Sets rec.top_passage to a short, topic-relevant full-text excerpt
        (see Record.top_passage's docstring) - a no-op if this paper has no
        stored full-text chunks. Applied uniformly to every processed
        record, reused-from-cache or freshly-extracted, so
        generate_copilot_synthesis (llm/engine.py) gets real passage-level
        RAG grounding for the final synthesis, not just each paper's
        1-sentence "answer" summary.
        """
        if not topic_vector:
            return
        try:
            top = AtelierRepository.get_top_chunks_for_paper(rec.fingerprint, topic_vector, top_k=1)
            if top:
                rec.top_passage = truncate(top[0], SYNTHESIS_PASSAGE_BUDGET)
        except Exception as e:
            logger.warning(f"Top-passage retrieval failed for {rec.fingerprint}: {e}")

    def _finalize(rec: Any, ai_data: dict) -> None:
        """Shared success path for both the reused-cache and freshly-extracted branches."""
        AtelierRepository.update_record_ai_data(session_id, rec.pmid, ai_data)
        rec.answer = str(ai_data.get("answer", "-"))
        _attach_top_passage(rec)
        rec_dict = rec.__dict__
        rec_dict["_db_id"] = getattr(rec, "_db_id", None)
        processed.append(rec_dict)

    # Partition up front: papers already permanently answered in a prior
    # session are pure reuse (no LLM call at all); the rest actually need
    # extraction and get grouped into batches below.
    needs_extraction: list[Any] = []
    for rec in top_records:
        if not _has_extractable_content(rec):
            continue
        if getattr(rec, "answer", None) and rec.answer != "-":
            _attach_top_passage(rec)
            rec_dict = rec.__dict__
            rec_dict["_db_id"] = getattr(rec, "_db_id", None)
            processed.append(rec_dict)
            continue
        needs_extraction.append(rec)

    total_batches = (len(needs_extraction) + EXTRACTION_BATCH_SIZE - 1) // EXTRACTION_BATCH_SIZE

    for batch_num, start in enumerate(range(0, len(needs_extraction), EXTRACTION_BATCH_SIZE), start=1):
        batch = needs_extraction[start:start + EXTRACTION_BATCH_SIZE]

        if report:
            first_title = batch[0].title[:50]
            more = f" +{len(batch) - 1} more" if len(batch) > 1 else ""
            report(f"Extracting findings — batch {batch_num} of {total_batches}: {first_title}{more}")

        # Already fetched/chunked/embedded by _prefetch_full_text above -
        # this is now just a cheap cache-check per paper, not a network call.
        for rec in batch:
            _ensure_full_text_chunks(rec, searcher)

        payloads = [
            {
                "paper_index": i,
                "title": rec.title,
                "content": _build_extraction_content(rec, topic_vector, budget=BATCH_EXTRACTION_CONTENT_BUDGET),
            }
            for i, rec in enumerate(batch)
        ]

        try:
            results = ai.extract_papers_batch(topic=topic, papers=payloads)
        except Exception as e:
            logger.warning(f"Batch extraction failed for batch {batch_num}/{total_batches}: {e}")
            continue

        for rec, ai_data in zip(batch, results):
            try:
                if ai_data.get("is_relevant"):
                    _finalize(rec, ai_data)
            except Exception as e:
                logger.warning(f"Failed to save AI data for paper {rec.pmid}: {e}")

    return processed


def extract_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Graph node wrapper around extract_records for the initial search pass -
    see extract_records's docstring for the actual extraction logic.
    """
    processed = extract_records(
        session_id=state["session_id"],
        topic=state["topic"],
        model_choice=state["model_choice"],
        limit=EXTRACTION_TARGET_COUNT,
        pool_size=EXTRACTION_CANDIDATE_POOL,
        report=_get_progress_callback(config),
    )
    return {
        "processed_records": processed,
        "relevant_count": len(processed),
        "status": f"Extraction Complete ({len(processed)} relevant)",
    }


def route_after_extraction(state: ResearchState) -> str:
    """Loops back to broaden the query if too few relevant papers surfaced."""
    if state.get("relevant_count", 0) < MIN_RELEVANT_PAPERS and state.get("retry_count", 0) < MAX_RETRIES:
        logger.info(
            f"Only {state.get('relevant_count', 0)} relevant papers "
            f"(need {MIN_RELEVANT_PAPERS}) - broadening and retrying."
        )
        return "broaden_query"
    return "end"
