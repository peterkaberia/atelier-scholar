"""
Investigation Agent Tools Module

LangChain @tool wrappers around Atelier's existing search/extraction/RAG
functionality, for pipeline.agent's ReAct investigation agent to call.
Nothing here is new logic - each tool delegates to the same
AtelierRepository/AtelierAcademicSearch/AtelierAIEngine/pipeline.nodes
functions the deterministic pipeline already uses, so the agent gets
identical caching, persistence, and correctness guarantees.

Session/model context is bound via closure (build_investigation_tools),
never exposed to the LLM as a parameter it would have to supply itself -
this keeps the agent from being able to (accidentally or otherwise) touch
a different session's data, and matches how a small/local model's
tool-calling is often unreliable about passing through IDs correctly.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Set

from langchain_core.tools import tool

from core.config import DEFAULT_SPARSE_MODEL
from database.models import Record
from database.repository import AtelierRepository
from llm.engine import AtelierAIEngine
from llm.model_catalog import full_text_tool_budget, rag_chunk_count, session_paper_list_count
from pipeline.nodes import _fetch_and_encode_full_text, _persist_full_text_result
from pipeline.orchestrator import ResearchOrchestrator
from search.fetchers import AtelierAcademicSearch
from search.sparse_encoder import encode_query_sparse

logger = logging.getLogger(__name__)


def _resolve_full_text(fingerprint: str, record_dict: dict, searcher: AtelierAcademicSearch) -> str:
    """
    Returns a paper's full text, fetching+chunking+persisting it first if
    that hasn't happened yet. Reuses pipeline.nodes' own fetch/persist
    split (not a second, parallel implementation) so chunks end up in
    EXACTLY the same place get_top_chunks_for_paper already reads from -
    a later rag_search_chunks call on this same paper hits the cache
    instead of re-fetching.

    _fetch_and_encode_full_text returns chunks, not one contiguous string
    (chunk_text's windows overlap), so this joins them back together rather
    than fetching the document a second time just to get an unsplit
    version - the minor duplicated text in overlap regions is a fine
    trade-off against a wasted network round-trip.
    """
    rec = Record(**record_dict)
    result = _fetch_and_encode_full_text(rec, searcher)
    if result is not None:
        _persist_full_text_result(result)
        _, chunks, _ = result
        if chunks:
            return "\n\n".join(chunks)

    # Either already fetched in a prior call (result is None - see
    # needs_full_text_fetch), or genuinely had nothing to fetch. Either way,
    # whatever's actually persisted is the source of truth now.
    cached = AtelierRepository.get_all_chunk_texts(fingerprint)
    if cached:
        return "\n\n".join(cached)

    return record_dict.get("abstract") or ""


def build_investigation_tools(session_id: str, model_choice: str, touched_fingerprints: Set[str]) -> List[Any]:
    """
    Builds the investigation agent's tool set, bound to one session.

    touched_fingerprints: a set the CALLER owns and inspects after the agent
    finishes running - every tool that meaningfully engages with a specific
    paper adds its fingerprint here, so the calling callback knows which
    papers to attach as citations on the investigation's own saved turn
    without needing the agent to explicitly report that itself (unreliable
    to ask of a tool-calling LLM, easy to just track as a side effect).
    """

    @tool
    def list_session_papers() -> str:
        """List the papers already available in this research session, with their titles and one-line takeaways. ALWAYS call this first before searching for new papers or analyzing a specific one - the answer may already be at hand."""
        # session_paper_list_count, not a fixed limit - scales with the
        # selected model's context window, see its docstring.
        records = AtelierRepository.get_session_processed_papers(session_id, limit=session_paper_list_count(model_choice))
        if not records:
            return "No papers have been processed in this session yet."
        return "\n".join(f"[{r.fingerprint}] {r.title} - {r.answer}" for r in records)

    @tool
    def search_more_papers(query: str) -> str:
        """Searches academic databases (PubMed, Europe PMC, OpenAlex, Semantic Scholar, Crossref, arXiv) for NEW papers on a specific query, extracts their key findings, and adds them to this session. Use this only when list_session_papers doesn't already cover what's needed - this is the slowest tool available, since it runs a full search+extraction pass."""
        try:
            orchestrator = ResearchOrchestrator(session_id=session_id, topic=query, model_choice=model_choice)
            processed = orchestrator.execute_full_search_pipeline()
        except Exception as e:
            logger.warning(f"Investigation agent's search_more_papers failed for '{query}': {e}")
            return f"Search failed: {e}"

        for rec in processed:
            fp = rec.get("fingerprint")
            if fp:
                touched_fingerprints.add(fp)

        if not processed:
            return "No new relevant papers were found for that query."
        lines = [f"[{r.get('fingerprint')}] {r.get('title')} - {r.get('answer')}" for r in processed]
        return f"Found and extracted {len(processed)} new papers:\n" + "\n".join(lines)

    @tool
    def get_full_text(fingerprint: str) -> str:
        """Fetches the full text of one paper (not just its abstract/takeaway) by fingerprint, so it can be read in more detail. Use this when a paper's summary isn't specific enough to answer the question."""
        record_dict = AtelierRepository.get_paper_by_fingerprint(fingerprint)
        if not record_dict:
            return f"No paper found with fingerprint '{fingerprint}'."

        touched_fingerprints.add(fingerprint)
        searcher = AtelierAcademicSearch(db_repository=AtelierRepository)
        text = _resolve_full_text(fingerprint, record_dict, searcher)
        if not text:
            return f"No full text is available for '{record_dict.get('title')}', and it has no abstract either."
        # Scaled to the SELECTED model's real context window
        # (llm.model_catalog.full_text_tool_budget), not a fixed constant -
        # a full paper can be tens of thousands of characters, which would
        # blow past a smaller model's window on its own, let alone
        # alongside the rest of the conversation and other tool results
        # already in play; a larger-context model gets proportionally more
        # of the paper instead of the same fixed slice every model got.
        return text[:full_text_tool_budget(model_choice)]

    @tool
    def rag_search_chunks(fingerprint: str, query: str) -> str:
        """Searches within ONE paper's full text for the passages most relevant to a specific question, instead of reading the whole thing - much faster than get_full_text for a targeted question. Fetches the paper's full text automatically if it hasn't been fetched yet."""
        record_dict = AtelierRepository.get_paper_by_fingerprint(fingerprint)
        if not record_dict:
            return f"No paper found with fingerprint '{fingerprint}'."

        touched_fingerprints.add(fingerprint)
        searcher = AtelierAcademicSearch(db_repository=AtelierRepository)
        # Ensures chunks exist (fetches+persists if this is the first call
        # for this paper) before scoring against them - see
        # _resolve_full_text's docstring.
        _resolve_full_text(fingerprint, record_dict, searcher)

        try:
            query_vector = encode_query_sparse(query, DEFAULT_SPARSE_MODEL)
        except Exception as e:
            return f"Couldn't encode the query for retrieval: {e}"

        # rag_chunk_count, not a fixed top_k - scales (modestly - this is a
        # TARGETED retrieval, not a full-text read) with the selected
        # model's context window, see its docstring.
        passages = AtelierRepository.get_top_chunks_for_paper(fingerprint, query_vector, top_k=rag_chunk_count(model_choice))
        if not passages:
            return f"No full text available to search for '{record_dict.get('title')}' - try analyze_papers on its abstract instead."
        return "\n---\n".join(passages)

    def _analyze_one(fingerprint: str, focus: str) -> str:
        record_dict = AtelierRepository.get_paper_by_fingerprint(fingerprint)
        if not record_dict:
            return f"[{fingerprint}] No paper found with this fingerprint."

        touched_fingerprints.add(fingerprint)
        engine = AtelierAIEngine(model_choice=model_choice)
        summary = engine.generate_paper_summary(topic=focus, record=record_dict)
        return f"[{fingerprint}] {record_dict.get('title')}\n{summary}"

    @tool
    def analyze_papers(fingerprints: List[str], focus: str) -> str:
        """Gets a focused analysis of one or more papers answering a SPECIFIC question or angle (e.g. 'what statistical methods did they use', 'how does this compare to X') rather than their generic summaries. Grounded only in each paper's own data. ALWAYS pass every paper you want analyzed for the SAME focus in one call (as a list), not one call per paper - analyzing 5 papers takes about the same wall-clock time as analyzing 1, since they run concurrently, but only when requested together."""
        if not fingerprints:
            return "No fingerprints provided."
        # Genuinely concurrent, not just "the model happened to batch tool
        # calls this turn" - see this tool's docstring. Guarantees the
        # speedup regardless of whether the underlying model reliably
        # requests multiple tool calls per turn (smaller/local models
        # especially don't always), rather than relying on
        # langgraph.prebuilt.ToolNode's own turn-level parallelism (which
        # IS real - confirmed live - but only helps when the model actually
        # asks for several tool calls at once).
        with ThreadPoolExecutor(max_workers=min(8, len(fingerprints))) as pool:
            futures = {pool.submit(_analyze_one, fp, focus): fp for fp in fingerprints}
            results = {futures[f]: f.result() for f in as_completed(futures)}
        # Preserve the order the agent asked for, not completion order.
        return "\n\n".join(results[fp] for fp in fingerprints)

    return [list_session_papers, search_more_papers, get_full_text, rag_search_chunks, analyze_papers]
