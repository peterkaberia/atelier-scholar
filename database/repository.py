"""
Database Repository Module

Handles all interactions with the local SQLite database. This class abstracts
away SQLAlchemy ORM logic, providing a clean, method-based API for the
Orchestrator to save records, retrieve top scored papers, and perform local
vector searches.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from datetime import datetime

from sqlalchemy import select, update, delete, func, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from core.secrets import decrypt, encrypt
from search.sparse_encoder import sparse_batch_dot

from .database import engine, get_db_session
from .models import Base, QueryCitation, QueryModel, SessionModel, PaperModel, PaperChunk, PaperSummaryModel, SessionPaperLink, Record, SettingModel

logger = logging.getLogger(__name__)

# In-process cache of (chunk_text, sparse_vector) pairs per paper
# fingerprint - see AtelierRepository._get_cached_chunks. Deliberately just
# a plain module-level dict with no TTL/eviction: DiskcacheManager (see
# ui/callbacks/background.py) spawns a fresh OS process per background job
# on Windows, so this cache's lifetime is naturally bounded to one job's
# run - it can't grow stale across requests or leak memory across runs,
# since the whole process (and this dict with it) is thrown away when the
# job finishes.
_CHUNK_CACHE: Dict[str, List[tuple]] = {}


class AtelierRepository:
    """
    A static repository class managing all database CRUD operations and
    local vector similarity searches for the Atelier application.
    """

    @staticmethod
    def initialize_db() -> None:
        """
        Creates tables safely if they do not already exist (SQLAlchemy's
        create_all is idempotent - a no-op for tables that already exist,
        same guarantee Peewee's create_tables(safe=True) gave us).

        NOTE: this runs in EVERY process that imports app.py, not just the
        real server process - DiskcacheManager (ui/callbacks/background.py)
        spawns a fresh child process per background job, and each one
        re-imports app.py from scratch (Windows multiprocessing uses
        'spawn'), re-running every module-level statement including this
        call. Anything here must be safe to run many times per second
        across many concurrent processes - see reap_orphaned_sessions()'s
        docstring for why that logic deliberately does NOT live here.
        """
        Base.metadata.create_all(engine)

        # create_all only creates tables that don't exist yet - it never
        # alters an EXISTING table to add a column a newer model gained
        # (e.g. QueryModel.consensus_meter, added after "queries" already
        # existed on disk). Self-heal that here instead of a one-off manual
        # ALTER TABLE, since this project has no Alembic migration system.
        # Cheap (single PRAGMA query) and safe to run from every spawned
        # process every time - ADD COLUMN is skipped if already present, and
        # the rare concurrent-process race is caught and ignored below.
        AtelierRepository._ensure_column("queries", "consensus_meter", "TEXT")
        AtelierRepository._ensure_column("query_citations", "citation_index", "INTEGER")
        AtelierRepository._ensure_column("sessions", "model_used", "TEXT")

        logger.info("Local SQLite database initialized successfully.")

    @staticmethod
    def _ensure_column(table: str, column: str, column_type_sql: str) -> None:
        """Adds `column` to `table` if it isn't already there. See initialize_db's docstring."""
        try:
            with engine.connect() as conn:
                existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
                if column not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type_sql}"))
                    conn.commit()
        except Exception as e:
            # "duplicate column name" from a concurrent process winning the
            # same race is harmless - anything else is logged but not fatal,
            # since the app should still boot even if this self-heal fails.
            logger.debug(f"_ensure_column({table}.{column}) skipped/failed: {e}")

    @staticmethod
    def reap_orphaned_sessions() -> int:
        """
        Flips any session still STATUS_IN_PROGRESS to STATUS_FAILED. Call
        this EXACTLY ONCE, only from the true top-level server process
        (app.py's `if __name__ == '__main__':` block) - never from
        initialize_db(), which runs in every spawned background-job child
        process too (see its docstring). Calling this from initialize_db()
        was a real bug: the moment a session's status was set back to
        in_progress (e.g. by the Retry button), the very next background
        step for that SAME session would spawn its own child process, which
        would re-run initialize_db() and immediately reap the status right
        back to failed before the pipeline had even resumed - silently
        undoing the retry from the user's perspective within the same
        click, even though the pipeline kept running underneath.

        A session still in_progress when the REAL server boots, though, is
        genuinely orphaned: nothing in a fresh server process could possibly
        already have a live background job for a pre-existing session, so
        its previous run's process is simply gone (crash, restart, killed).
        Left alone, its feed page would poll forever showing a stale
        "still working" message with no process left to ever finish it.
        """
        with get_db_session() as db:
            result = db.execute(
                update(SessionModel)
                .where(SessionModel.status == SessionModel.STATUS_IN_PROGRESS)
                .values(status=SessionModel.STATUS_FAILED)
            )
            orphaned = result.rowcount

        if orphaned:
            logger.warning(f"Marked {orphaned} orphaned in-progress session(s) as failed on startup (interrupted by a previous restart).")
        return orphaned

    # ==========================================
    # 1. SESSION MANAGEMENT
    # ==========================================

    @staticmethod
    def session_exists(session_id: str) -> bool:
        """
        Checks if a search session already exists in the database.

        Args:
            session_id (str): The unique ID of the session.

        Returns:
            bool: True if the session exists, False otherwise.
        """
        with get_db_session() as db:
            return db.get(SessionModel, session_id) is not None

    @staticmethod
    def create_session(session_id: str, topic: str, summary: str = "", model_used: str = "") -> None:
        """
        Creates a new search session to track user history and linked papers.
        Starts in STATUS_IN_PROGRESS - see set_session_status().

        Args:
            session_id (str): The unique ID of the session.
            topic (str): The overarching research question.
            summary (str): An optional summary of the session findings.
            model_used (str): Whichever model was selected on Home when this
                session was created - see SessionModel.model_used's own
                docstring for why this exists separately from
                QueryModel.model_used.
        """
        with get_db_session() as db:
            db.add(SessionModel(id=session_id, topic=topic, summary=summary, model_used=model_used or None))
        logger.info(f"Created new database session: {session_id}")

    @staticmethod
    def get_session_topic(session_id: str) -> str:
        """Returns a session's original question, or "" if it doesn't exist."""
        with get_db_session() as db:
            session = db.get(SessionModel, session_id)
            return session.topic if session else ""

    @staticmethod
    def get_session_status(session_id: str) -> str:
        """
        Returns a session's current status (SessionModel.STATUS_*), or
        STATUS_COMPLETED if the session doesn't exist - callers checking
        "is this still running" should treat an unknown session as done,
        not as perpetually in progress.
        """
        with get_db_session() as db:
            session = db.get(SessionModel, session_id)
            return session.status if session else SessionModel.STATUS_COMPLETED

    @staticmethod
    def set_session_status(session_id: str, status: str) -> None:
        """Updates a session's status - drives the sidebar/history status icons."""
        with get_db_session() as db:
            session = db.get(SessionModel, session_id)
            if session is None:
                logger.warning(f"Could not set status: session {session_id} not found.")
                return
            session.status = status

    @staticmethod
    def set_session_progress(session_id: str, message: str) -> None:
        """
        Persists the pipeline's latest human-readable stage. Called on every
        progress_callback tick (see pipeline/orchestrator.py) alongside
        Dash's set_progress, specifically so a tab that wasn't there for the
        live update - one that navigated back, reloaded, or opened the
        session fresh - can still show the real current stage instead of a
        generic placeholder. Silently no-ops if the session is gone (e.g. a
        stale call landing after the row was somehow removed).
        """
        with get_db_session() as db:
            db.execute(
                update(SessionModel)
                .where(SessionModel.id == session_id)
                .values(progress_message=message)
            )

    @staticmethod
    def get_session_progress(session_id: str) -> str:
        """Returns the session's last-known pipeline stage, or "" if none was ever recorded."""
        with get_db_session() as db:
            session = db.get(SessionModel, session_id)
            return (session.progress_message or "") if session else ""

    @staticmethod
    def get_last_used_model(session_id: str) -> str:
        """
        Returns the model_used from this session's most recent completed
        turn, falling back to SessionModel.model_used (the model selected
        when the session was FIRST created, before any turn has actually
        finished) when there's no completed turn yet, and finally "" if
        neither exists. Used to default the feed page's model dropdown to
        whatever the user was actually using, instead of always resetting
        to the first entry in the provider list - and now also to show a
        real model name on the "still processing" placeholder for a
        session's very first turn, which previously had nothing to show at
        all (get_all_sessions_for_history's own dict doesn't carry this,
        so build_processing_placeholder's caller reaches for this
        specifically).
        """
        with get_db_session() as db:
            last_query = db.execute(
                select(QueryModel.model_used)
                .where(QueryModel.session_id == session_id)
                .order_by(QueryModel.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if last_query:
                return last_query
            session = db.get(SessionModel, session_id)
            return (session.model_used if session else "") or ""

    @staticmethod
    def get_last_query_id(session_id: str) -> Optional[int]:
        """
        Returns this session's most recent QueryModel id, or None if it has
        no turns yet - used by ui/callbacks/chat.py's route_intent as a DB
        fallback when store-current-topic/store-processed-records (memory-
        only Dash Stores) are empty, which happens on every fresh page load
        regardless of whether the session actually has prior context. See
        route_intent's docstring for why trusting those Stores alone caused
        every follow-up submitted after a reload to be force-routed to
        SEARCH.
        """
        with get_db_session() as db:
            return db.execute(
                select(QueryModel.id)
                .where(QueryModel.session_id == session_id)
                .order_by(QueryModel.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()

    # ==========================================
    # 2. BULK SAVING & FINGERPRINTING
    # ==========================================

    @staticmethod
    def save_bulk_records(session_id: str, records: List[Record]) -> None:
        """
        Saves raw papers to the database securely. If a paper already exists
        (matched by unique fingerprint), it gracefully updates any missing links
        but DOES NOT overwrite existing AI-extracted permanent data.

        Args:
            session_id (str): The active search session linking these papers.
            records (List[Record]): The list of newly fetched Record objects.
        """
        with get_db_session() as db:
            for rec in records:
                fp = rec.fingerprint
                if not fp:
                    continue

                # 1. Insert or gently Update the Paper globally
                paper = db.execute(
                    select(PaperModel).where(PaperModel.fingerprint == fp)
                ).scalar_one_or_none()

                if paper is None:
                    paper = PaperModel(
                        fingerprint=fp,
                        title=rec.title,
                        abstract=rec.abstract,
                        authors=rec.authors,
                        journal=rec.journal,
                        year=rec.year,
                        doi=rec.doi,
                        pmid=rec.pmid,
                        pmcid=rec.pmcid,
                        pdf_url=rec.pdf_url,
                        full_text_url=rec.full_text_url,
                        citation_count=rec.citation_count,
                        sparse_vector=rec.sparse_vector,
                    )
                    db.add(paper)
                else:
                    # If it already existed, update any missing URLs or Vectors
                    if not paper.pdf_url and rec.pdf_url:
                        paper.pdf_url = rec.pdf_url
                    if not paper.full_text_url and rec.full_text_url:
                        paper.full_text_url = rec.full_text_url
                    if not paper.pmcid and rec.pmcid:
                        paper.pmcid = rec.pmcid
                    if not paper.sparse_vector and rec.sparse_vector:
                        paper.sparse_vector = rec.sparse_vector

                # 2. Link this specific paper to the user's current session
                # Note: A paper can be linked to multiple sessions if multiple users search for it.
                # (session_id, paper_fingerprint) has no DB-level unique
                # constraint (matches the original Peewee schema), so a
                # broaden-and-retry loop re-saving an overlapping candidate
                # pool for the same session could in principle find more
                # than one existing link here - .first() tolerates that
                # instead of raising, same reasoning as update_record_ai_data.
                existing_link = db.execute(
                    select(SessionPaperLink).where(
                        SessionPaperLink.session_id == session_id,
                        SessionPaperLink.paper_fingerprint == fp,
                    ).limit(1)
                ).scalars().first()
                if existing_link is None:
                    db.add(SessionPaperLink(
                        session_id=session_id,
                        paper_fingerprint=fp,
                        # NOTE: search/sparse_encoder.py's compute_scores
                        # sets rec.step2_score_100, NOT rec.step2_score -
                        # this getattr's default-0 fallback was silently
                        # firing for every single paper, meaning every
                        # session's SessionPaperLink.relevance_score has
                        # been a constant 0 the whole time. That made
                        # get_top_unprocessed_records's "ORDER BY
                        # relevance_score DESC" a no-op - the "top 20 by
                        # relevance" sent to extraction was actually
                        # whatever arbitrary order SQLite happened to
                        # return, not a real ranking, letting off-topic
                        # candidates crowd out genuinely relevant ones.
                        relevance_score=getattr(rec, "step2_score_100", 0),
                    ))

    # ==========================================
    # 3. AI EXTRACTION STORAGE
    # ==========================================

    @staticmethod
    def get_top_unprocessed_records(session_id: str, limit: int = 20, exclude_fingerprints: Optional[List[str]] = None) -> List[Record]:
        """
        Retrieves the highest-scored papers for a session, actively prioritizing
        papers that have Open Access full text available (via PMCID or URLs).

        Converts the DB rows back into the lightweight Record dataclass.

        Args:
            session_id (str): The active search session.
            limit (int): The maximum number of papers to retrieve.
            exclude_fingerprints (list[str], optional): Papers to skip -
                used by ui/callbacks/library.py's "Load More" callback to
                fetch the NEXT batch (whatever's already cited by the
                current turn, via AtelierRepository.get_query_citations)
                without relying on OFFSET/pagination staying stable across
                separate queries when relevance_score ties exist.

        Returns:
            List[Record]: The top-scored papers ready for LLM extraction.
        """
        with get_db_session() as db:
            query = (
                select(PaperModel, SessionPaperLink.relevance_score)
                .join(SessionPaperLink, PaperModel.fingerprint == SessionPaperLink.paper_fingerprint)
                .where(SessionPaperLink.session_id == session_id)
                # Force the DB to only return papers we can effectively analyze
                .where(
                    (PaperModel.pmcid.is_not(None))
                    | (PaperModel.full_text_url.is_not(None))
                    | (PaperModel.pdf_url.is_not(None))
                    | (PaperModel.abstract.is_not(None))
                )
            )
            if exclude_fingerprints:
                query = query.where(PaperModel.fingerprint.notin_(exclude_fingerprints))

            rows = db.execute(
                query.order_by(SessionPaperLink.relevance_score.desc()).limit(limit)
            ).all()

            out = []
            for row, _relevance_score in rows:
                rec = Record(
                    fingerprint=row.fingerprint,
                    title=row.title,
                    abstract=row.abstract,
                    year=row.year,
                    journal=row.journal,
                    doi=row.doi,
                    pmid=row.pmid,
                    pmcid=row.pmcid,
                    pdf_url=row.pdf_url,
                    full_text_url=row.full_text_url,
                    authors=row.authors,
                    citation_count=row.citation_count,

                    # PRE-LOAD PERMANENT AI FIELDS (Prevents duplicate LLM calls)
                    answer=row.answer,
                    population=row.population,
                    methods=row.methods,
                    results=row.results,
                    outcomes=row.outcomes,
                    sample_size=row.sample_size,
                    duration=row.duration,
                    country=row.country,
                    is_relevant=row.is_relevant,
                    _db_id=row.id,
                )
                out.append(rec)

            return out

    @staticmethod
    def get_session_processed_papers(session_id: str, limit: int = 30) -> List[Record]:
        """
        Returns papers already EXTRACTED for this session (answer != "-"),
        ordered by relevance - the mirror image of
        get_top_unprocessed_records, which deliberately looks for papers
        NOT yet extracted. Used by the investigation agent's
        list_session_papers tool: "what's already known" needs the papers
        that already have real findings attached, not the ranked-but-
        unexamined candidate pool.
        """
        with get_db_session() as db:
            rows = db.execute(
                select(PaperModel, SessionPaperLink.relevance_score)
                .join(SessionPaperLink, PaperModel.fingerprint == SessionPaperLink.paper_fingerprint)
                .where(SessionPaperLink.session_id == session_id)
                .where(PaperModel.answer.is_not(None))
                .where(PaperModel.answer != "-")
                .order_by(SessionPaperLink.relevance_score.desc())
                .limit(limit)
            ).all()

            out = []
            for row, _relevance_score in rows:
                rec = Record(
                    fingerprint=row.fingerprint, title=row.title, abstract=row.abstract,
                    year=row.year, journal=row.journal, doi=row.doi, pmid=row.pmid,
                    pmcid=row.pmcid, pdf_url=row.pdf_url, full_text_url=row.full_text_url,
                    authors=row.authors, citation_count=row.citation_count,
                    answer=row.answer, population=row.population, methods=row.methods,
                    results=row.results, outcomes=row.outcomes, sample_size=row.sample_size,
                    duration=row.duration, country=row.country, is_relevant=row.is_relevant,
                    _db_id=row.id,
                )
                out.append(rec)
            return out

    @staticmethod
    def update_record_ai_data(session_id: str, fingerprint: str, ai_data: dict) -> None:
        """
        Permanently saves the LLM's expensive extraction work to the database.
        These parameters are immutable facts about the paper.

        Args:
            session_id (str): The active session (used for logging context).
            fingerprint (str): The paper's fingerprint - the DB's own unique
                dedup key (doi:/pmid:/pmcid:/arxiv:/{source}:{source_id}, see
                core.utils.make_fingerprint). NOT pmid: this used to look up
                by PaperModel.pmid, which is genuinely None for most
                non-PubMed-sourced papers (Crossref, arXiv, OpenAlex, and
                Semantic Scholar records routinely have no PubMed
                cross-reference at all) - confirmed live against the real
                DB that every successfully-saved extraction had a non-null
                pmid, meaning extraction for EVERY paper without one was
                silently discarded here the whole time (the lookup found
                nothing, logged a warning, and returned) even though the
                LLM call itself had already succeeded. fingerprint is always
                set and is the actual unique column, so this can never
                ambiguously match more than one row the way pmid could.
            ai_data (dict): The extracted JSON metadata from the AI Engine.
        """
        with get_db_session() as db:
            paper = db.execute(
                select(PaperModel).where(PaperModel.fingerprint == fingerprint)
            ).scalar_one_or_none()

            if paper is None:
                logger.warning(f"Could not save AI data: Paper with fingerprint {fingerprint} not found in DB.")
                return

            # Map the JSON schema to the database columns
            paper.is_relevant = ai_data.get("is_relevant")
            paper.answer = str(ai_data.get("answer", "-"))
            paper.population = str(ai_data.get("population", "-"))
            paper.methods = str(ai_data.get("methods", "-"))
            paper.results = str(ai_data.get("results", "-"))
            paper.outcomes = str(ai_data.get("outcomes", "-"))
            paper.sample_size = str(ai_data.get("sample_size", "-"))
            paper.study_count = str(ai_data.get("study_count", "-"))
            paper.duration = str(ai_data.get("duration", "-"))
            paper.country = str(ai_data.get("country", "-"))

            title_preview = paper.title[:30]
        logger.info(f"Permanently saved AI metadata for paper: {title_preview}...")

    # ==========================================
    # 4. SETTINGS (UI-ENTERED, ENCRYPTED API KEYS)
    # ==========================================

    @staticmethod
    def get_setting(key: str) -> str:
        """
        Returns the decrypted value for a UI-entered setting, or "" if the
        key was never set (or fails to decrypt, e.g. after a key rotation).
        """
        with get_db_session() as db:
            row = db.get(SettingModel, key)
            return decrypt(row.encrypted_value) if row else ""

    @staticmethod
    def set_setting(key: str, value: str) -> None:
        """Encrypts and upserts a setting. An empty value deletes the key."""
        if not value or not value.strip():
            AtelierRepository.delete_setting(key)
            return

        encrypted = encrypt(value.strip())
        now = datetime.now()
        with get_db_session() as db:
            stmt = sqlite_insert(SettingModel).values(
                key=key, encrypted_value=encrypted, updated_at=now
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[SettingModel.key],
                set_={"encrypted_value": encrypted, "updated_at": now},
            )
            db.execute(stmt)

    @staticmethod
    def delete_setting(key: str) -> None:
        with get_db_session() as db:
            db.execute(delete(SettingModel).where(SettingModel.key == key))

    @staticmethod
    def get_configured_setting_keys() -> List[str]:
        """
        Returns which setting keys currently have a value, WITHOUT exposing
        the values themselves - used to render "configured" badges in the UI.
        """
        with get_db_session() as db:
            return list(db.execute(select(SettingModel.key)).scalars().all())

    # ==========================================
    # 5. FULL-TEXT RAG (PAPER CHUNKS)
    # ==========================================

    @staticmethod
    def needs_full_text_fetch(fingerprint: str) -> bool:
        """
        True if we haven't yet attempted a full-text fetch for this Paper.
        Checked before hitting Europe PMC/pdf_url so a paper with no
        available full text isn't re-requested on every single run.
        """
        with get_db_session() as db:
            paper = db.execute(
                select(PaperModel).where(PaperModel.fingerprint == fingerprint)
            ).scalar_one_or_none()
            return (not paper.full_text_fetched) if paper else False

    @staticmethod
    def mark_full_text_fetched(fingerprint: str) -> None:
        """Records that a full-text fetch was attempted, success or not."""
        with get_db_session() as db:
            paper = db.execute(
                select(PaperModel).where(PaperModel.fingerprint == fingerprint)
            ).scalar_one_or_none()
            if paper is None:
                logger.warning(f"Could not mark full_text_fetched: paper {fingerprint} not found.")
                return
            paper.full_text_fetched = True

    @staticmethod
    def save_paper_chunks(fingerprint: str, chunks: List[str], vectors: List[Dict[str, float]]) -> None:
        """
        Replaces any existing chunks for a Paper with a freshly parsed/embedded
        set. Deleting first keeps re-fetches idempotent instead of accumulating
        duplicate passages.
        """
        with get_db_session() as db:
            db.execute(delete(PaperChunk).where(PaperChunk.paper_fingerprint == fingerprint))

            rows = [
                PaperChunk(paper_fingerprint=fingerprint, chunk_index=i, text=text, sparse_vector=vector)
                for i, (text, vector) in enumerate(zip(chunks, vectors))
            ]
            if rows:
                db.add_all(rows)
                logger.info(f"Saved {len(rows)} full-text chunks for paper {fingerprint}.")

    @staticmethod
    def get_top_chunks_for_paper(fingerprint: str, query_vector: Dict[str, float], top_k: int = 5) -> List[str]:
        """
        Scores every stored chunk for a Paper against a SPLADE query vector via
        sparse dot-product (same scoring approach as search_papers_by_vector)
        and returns the top-k passages, in relevance order.
        """
        chunks = AtelierRepository._get_cached_chunks(fingerprint)
        if not chunks:
            return []

        scores = sparse_batch_dot(query_vector, [c[1] for c in chunks])
        scored = list(zip(scores, [c[0] for c in chunks]))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    @staticmethod
    def get_all_chunk_texts(fingerprint: str) -> List[str]:
        """
        Returns every stored chunk's text for a paper, in storage order (not
        scored/ranked against any query) - used by pipeline.agent_tools to
        reconstruct something close to a paper's full text from its already-
        chunked/persisted form, without re-fetching the source document.
        """
        return [text for text, _ in AtelierRepository._get_cached_chunks(fingerprint)]

    @staticmethod
    def _get_cached_chunks(fingerprint: str) -> List[tuple]:
        """
        Returns [(chunk_text, sparse_vector_dict), ...] for one paper,
        cached in-process (see _CHUNK_CACHE's module-level docstring) so
        repeated calls for the SAME paper within one run - e.g. an
        investigation agent calling get_top_chunks_for_paper several times
        while digging into one paper - skip the DB round-trip + JSON decode
        after the first call.
        """
        cached = _CHUNK_CACHE.get(fingerprint)
        if cached is not None:
            return cached

        with get_db_session() as db:
            chunks = db.execute(
                select(PaperChunk).where(PaperChunk.paper_fingerprint == fingerprint)
            ).scalars().all()
            result = [(c.text, c.sparse_vector or {}) for c in chunks]

        _CHUNK_CACHE[fingerprint] = result
        return result

    # ==========================================
    # 6. VECTOR SEARCH CLIENT
    # ==========================================

    @staticmethod
    def search_papers_by_vector(query_data: Any, limit: int = 50) -> List[Record]:
        """
        Turns the local database into a vector search client.
        Computes the dot-product similarity between the incoming SPLADE query vector
        and all previously saved paper vectors to instantly recall known literature.

        Args:
            query_data: A raw topic string (SQL substring fallback) or a
                SPLADE sparse vector dict (real semantic search).
            limit (int): Max papers to return.

        Returns:
            List[Record]: Highly relevant papers pulled directly from local memory.
        """
        scored_papers = []

        with get_db_session() as db:
            if isinstance(query_data, str):
                # --- FALLBACK: Standard SQLite Text Match ---
                rows = db.execute(
                    select(PaperModel)
                    .where(
                        PaperModel.title.contains(query_data)
                        | PaperModel.abstract.contains(query_data)
                    )
                    .limit(limit)
                ).scalars().all()

                for row in rows:
                    scored_papers.append((1.0, row))  # Dummy score for exact text matches

            elif isinstance(query_data, dict):
                # --- ADVANCED: Sparse Vector Dot-Product Search ---
                # Batched, not one dot product per paper - this scans EVERY
                # paper in the local DB with a stored vector, potentially
                # the largest-scale case of the three sparse_batch_dot call
                # sites - see its docstring.
                rows = db.execute(
                    select(PaperModel).where(PaperModel.sparse_vector.is_not(None))
                ).scalars().all()
                scores = sparse_batch_dot(query_data, [row.sparse_vector for row in rows])
                for row, score in zip(rows, scores):
                    if score > 0.5:
                        scored_papers.append((score, row))

            # Sort by highest score
            scored_papers.sort(key=lambda x: x[0], reverse=True)

            out = []
            for score, row in scored_papers[:limit]:
                rec = Record(
                    fingerprint=row.fingerprint, title=row.title, abstract=row.abstract,
                    year=row.year, journal=row.journal, doi=row.doi,
                    pmid=row.pmid, pmcid=row.pmcid, pdf_url=row.pdf_url,
                    full_text_url=row.full_text_url, authors=row.authors,
                    citation_count=row.citation_count, answer=row.answer,
                    population=row.population, methods=row.methods, results=row.results,
                    outcomes=row.outcomes, sample_size=row.sample_size,
                    duration=row.duration, country=row.country, is_relevant=row.is_relevant,
                    # Carry the already-computed SPLADE vector forward so
                    # compute_scores (search/sparse_encoder.py) can reuse it via
                    # a cheap dot-product instead of re-running the model on a
                    # paper we've already embedded in a previous session.
                    sparse_vector=row.sparse_vector or {},
                    _db_id=row.id,
                )
                rec.all_sources = ["atelier_db"]
                out.append(rec)

            return out

    @staticmethod
    def get_all_sessions_for_history() -> List[Dict[str, Any]]:
        """
        Retrieves all search sessions ordered by most recent first, each with
        its status and precomputed query/source counts - used by both the
        sidebar history list and the /history archive page. Returned as
        plain dicts (not ORM instances), so callers must use dict access
        (session["topic"]), not attribute access.

        Returns:
            List[Dict]: A list of session dictionaries.
        """
        try:
            with get_db_session() as db:
                sessions = db.execute(
                    select(SessionModel).order_by(SessionModel.created_at.desc())
                ).scalars().all()

                out = []
                for session in sessions:
                    query_count = db.execute(
                        select(func.count()).select_from(QueryModel).where(QueryModel.session_id == session.id)
                    ).scalar_one()
                    # SessionPaperLink links by a plain session_id string, not
                    # a real relationship/backref, so this has to be its own query.
                    source_count = db.execute(
                        select(func.count()).select_from(SessionPaperLink).where(SessionPaperLink.session_id == session.id)
                    ).scalar_one()
                    out.append({
                        "session_id": session.id,
                        "topic": session.topic,
                        "summary": session.summary,
                        "status": session.status,
                        "query_count": query_count,
                        "source_count": source_count,
                        "created_at": session.created_at,
                    })
                return out
        except Exception as e:
            logger.error(f"Failed to fetch session history: {e}")
            return []

    @staticmethod
    def get_query_citations(query_id: int) -> List[Dict[str, Any]]:
        """
        Fetches the full Record data for every paper cited by one specific
        query/turn - used by the reference-export callback (always sources
        from the DB by query_id rather than a live Store, so export works
        identically for the just-generated turn and any historical one) and
        by load-more's "what have I already shown" check.
        """
        try:
            with get_db_session() as db:
                # Ordered by citation_index - see get_session_chat_history's
                # docstring for why a plain "fingerprint IN (...)" fetch
                # silently scrambles citation order on every call.
                fingerprints = list(db.execute(
                    select(QueryCitation.paper_fingerprint)
                    .where(QueryCitation.query_id == query_id)
                    .order_by(QueryCitation.citation_index.is_(None), QueryCitation.citation_index.asc(), QueryCitation.id.asc())
                ).scalars().all())
                if not fingerprints:
                    return []

                papers = db.execute(
                    select(PaperModel).where(PaperModel.fingerprint.in_(fingerprints))
                ).scalars().all()
                papers_by_fp = {p.fingerprint: p for p in papers}

                out = []
                for fp in fingerprints:
                    p = papers_by_fp.get(fp)
                    if p is None:
                        continue
                    rec = Record(
                        fingerprint=p.fingerprint, title=p.title, abstract=p.abstract,
                        year=p.year, journal=p.journal, authors=p.authors,
                        answer=p.answer, population=p.population, methods=p.methods,
                        sample_size=p.sample_size, outcomes=p.outcomes, pdf_url=p.pdf_url,
                        citation_count=p.citation_count, doi=p.doi, pmid=p.pmid,
                    )
                    out.append(rec.__dict__)
                return out
        except Exception as e:
            logger.error(f"Failed to fetch citations for query {query_id}: {e}")
            return []

    @staticmethod
    def get_paper_by_fingerprint(fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Fetches one paper's full data by its fingerprint (the DB's own
        deduplication key, unique per paper) - used by
        ui/callbacks/library.py's summarize_selected_paper, which only
        knows a fingerprint (parsed off the checked checkbox's own value)
        and needs the paper's title/abstract/etc. to summarize it.
        """
        try:
            with get_db_session() as db:
                p = db.execute(select(PaperModel).where(PaperModel.fingerprint == fingerprint)).scalars().first()
                if p is None:
                    return None
                rec = Record(
                    fingerprint=p.fingerprint, title=p.title, abstract=p.abstract,
                    year=p.year, journal=p.journal, authors=p.authors,
                    answer=p.answer, population=p.population, methods=p.methods,
                    results=p.results, outcomes=p.outcomes, sample_size=p.sample_size,
                    pdf_url=p.pdf_url, citation_count=p.citation_count,
                    doi=p.doi, pmid=p.pmid,
                )
                return rec.__dict__
        except Exception as e:
            logger.error(f"Failed to fetch paper {fingerprint}: {e}")
            return None

    @staticmethod
    def get_query_by_id(query_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches a single query row's own fields - used by
        ui/callbacks/library.py's load_more (session_id/prompt/model_used,
        to run more extraction the same way the initial search did) and
        export_pdf (also needs synthesis - the actual answer text - to
        render a per-turn PDF).
        """
        try:
            with get_db_session() as db:
                q = db.get(QueryModel, query_id)
                if q is None:
                    return None
                return {
                    "session_id": q.session_id,
                    "prompt": q.prompt,
                    "model_used": q.model_used,
                    "synthesis": q.synthesis,
                }
        except Exception as e:
            logger.error(f"Failed to fetch query {query_id}: {e}")
            return None

    @staticmethod
    def add_citations_to_query(query_id: int, records: List[Any]) -> None:
        """
        Links additional papers to an EXISTING query/turn, without touching
        its QueryModel row or any citations already linked - used by
        ui/callbacks/library.py's load_more to attach newly-extracted papers
        to the turn whose "Load More" button was clicked.

        New rows continue citation_index from the current max rather than
        leaving it null - these papers are APPENDED to the end of the
        reference list (matching how the UI actually shows them: existing
        cards first, new ones after), so they should sort after everything
        already there instead of relying on citation_index's null-sorts-
        last fallback to merely happen to work.
        """
        try:
            with get_db_session() as db:
                existing_rows = db.execute(
                    select(QueryCitation.paper_fingerprint, QueryCitation.citation_index)
                    .where(QueryCitation.query_id == query_id)
                ).all()
                existing = {row[0] for row in existing_rows}
                next_index = max((row[1] for row in existing_rows if row[1] is not None), default=-1) + 1

                for rec in records:
                    fingerprint = rec.get("fingerprint") if isinstance(rec, dict) else getattr(rec, "fingerprint", None)
                    if fingerprint and fingerprint not in existing:
                        db.add(QueryCitation(query_id=query_id, paper_fingerprint=fingerprint, citation_index=next_index))
                        existing.add(fingerprint)
                        next_index += 1
        except Exception as e:
            logger.error(f"Failed to add citations to query {query_id}: {e}")

    @staticmethod
    def get_cached_paper_summary(fingerprint: str, session_id: str) -> Optional[str]:
        """
        Returns a previously-generated llm.engine.generate_paper_summary
        result for this (paper, session) pair, or None on a cache miss -
        see PaperSummaryModel's docstring for why this key is stable.
        """
        try:
            with get_db_session() as db:
                return db.execute(
                    select(PaperSummaryModel.summary)
                    .where(PaperSummaryModel.paper_fingerprint == fingerprint, PaperSummaryModel.session_id == session_id)
                    .limit(1)
                ).scalars().first()
        except Exception as e:
            logger.error(f"Failed to read cached paper summary for {fingerprint}/{session_id}: {e}")
            return None

    @staticmethod
    def save_paper_summary(fingerprint: str, session_id: str, summary: str) -> None:
        """Caches a freshly-generated paper summary - see get_cached_paper_summary/PaperSummaryModel."""
        try:
            with get_db_session() as db:
                existing = db.execute(
                    select(PaperSummaryModel)
                    .where(PaperSummaryModel.paper_fingerprint == fingerprint, PaperSummaryModel.session_id == session_id)
                    .limit(1)
                ).scalars().first()
                if existing:
                    existing.summary = summary
                else:
                    db.add(PaperSummaryModel(paper_fingerprint=fingerprint, session_id=session_id, summary=summary))
        except Exception as e:
            logger.error(f"Failed to cache paper summary for {fingerprint}/{session_id}: {e}")

    @staticmethod
    def get_session_chat_history(session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the exact chat history for a session, formatting it perfectly
        for the UI feed layout.
        """
        try:
            with get_db_session() as db:
                # Fetch all queries for this session, oldest first (chronological order)
                queries = db.execute(
                    select(QueryModel).where(QueryModel.session_id == session_id).order_by(QueryModel.created_at.asc())
                ).scalars().all()

                history = []
                for q in queries:
                    # Ordered by citation_index, NOT the plain "WHERE
                    # fingerprint IN (...)" fetch this used to be - SQL
                    # gives no ordering guarantee for an IN-clause lookup,
                    # so papers previously came back in whatever order
                    # SQLite's query planner picked (in practice
                    # PaperModel.id order, unrelated to this query's own
                    # [N] citation numbers), silently scrambling which card
                    # matched which bracket in the already-written
                    # synthesis text on every reload. is_(None) first
                    # sorts null citation_index (rows written before this
                    # column existed) after every properly-indexed row,
                    # falling back to insertion order (id) as a
                    # best-effort approximation for those legacy rows,
                    # which is the most we can recover since their true
                    # original order was never captured.
                    citation_rows = db.execute(
                        select(QueryCitation.paper_fingerprint)
                        .where(QueryCitation.query_id == q.id)
                        .order_by(QueryCitation.citation_index.is_(None), QueryCitation.citation_index.asc(), QueryCitation.id.asc())
                    ).scalars().all()
                    fingerprints_ordered = list(citation_rows)

                    cited_papers = []
                    if fingerprints_ordered:
                        papers = db.execute(
                            select(PaperModel).where(PaperModel.fingerprint.in_(fingerprints_ordered))
                        ).scalars().all()
                        papers_by_fp = {p.fingerprint: p for p in papers}
                        # Convert ORM rows back to UI-friendly Records, in
                        # the citation-index order fetched above (NOT
                        # `papers`' own arbitrary fetch order).
                        for fp in fingerprints_ordered:
                            p = papers_by_fp.get(fp)
                            if p is None:
                                continue
                            rec = Record(
                                fingerprint=p.fingerprint, title=p.title, abstract=p.abstract,
                                year=p.year, journal=p.journal, authors=p.authors,
                                answer=p.answer, population=p.population, methods=p.methods,
                                sample_size=p.sample_size, outcomes=p.outcomes, pdf_url=p.pdf_url,
                                citation_count=p.citation_count, doi=p.doi, pmid=p.pmid,
                            )
                            cited_papers.append(rec.__dict__)

                    # Pack it into the format `feed.py` expects. query_id lets
                    # the UI build collision-safe pattern-matching ids scoped
                    # to this specific turn's paper cards - see
                    # save_query_with_citations's docstring.
                    history.append({
                        "query_id": q.id,
                        "prompt": q.prompt,
                        "model_used": q.model_used,
                        "synthesis": q.synthesis,
                        "citations": cited_papers,
                        "consensus_meter": json.loads(q.consensus_meter) if q.consensus_meter else None,
                    })

                return history

        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return []

    @staticmethod
    def get_session_chat_history_as_messages(session_id: str) -> List[Dict[str, str]]:
        """
        Flattens get_session_chat_history's QueryModel-shaped rows into the
        plain {"role": "user"/"assistant", "content": ...} message list
        llm.engine.chat_with_literature / pipeline.agent.run_investigation
        actually expect.

        Used as a DB fallback wherever store-chat-history (a memory-only
        Dash Store - see ui/layouts/main.py's serve_layout) doesn't yet
        reflect the session's real history. That Store has two independent
        gaps, not just one: it resets to [] on every reload (already
        handled elsewhere the same way route_intent falls back to the DB
        for current_topic/processed_records), AND - more subtly - it never
        gets the session's very FIRST turn written into it at all, reload
        or not: ui/callbacks/search.py's generate_synth (the initial
        SEARCH turn) never writes chat_history, only run_chat/
        run_investigation/run_upload do, and only once one of THOSE has
        already run at least once. Confirmed live: a user's first-ever
        follow-up right after an initial search ("rewrite the above
        with...") got an LLM response acting as if no prior turn existed,
        because store-chat-history genuinely was still [] at that exact
        point - not stale, never populated in the first place.
        """
        history = AtelierRepository.get_session_chat_history(session_id)
        messages: List[Dict[str, str]] = []
        for turn in history:
            prompt = turn.get('prompt')
            synthesis = turn.get('synthesis')
            if prompt:
                messages.append({"role": "user", "content": prompt})
            if synthesis:
                messages.append({"role": "assistant", "content": synthesis})
        return messages

    @staticmethod
    def save_query_with_citations(session_id: str, prompt: str, synthesis: str, model_used: str, cited_records: List[Record], consensus_meter: dict = None) -> Optional[int]:
        """
        Saves a single chat turn and links the specific papers used to generate the answer.

        consensus_meter (dict, optional): The output of
            llm.engine.generate_atelier_meter (the Atelier Meter), JSON-encoded before storage
            - None (the common case, non-yes/no queries) stores NULL.

        Returns:
            Optional[int]: The new QueryModel row's id, or None if the save
                failed. Callers use this to build collision-safe
                pattern-matching component ids for that turn's paper cards
                (checkboxes/load-more button/etc.) - see build_paper_cards's
                docstring: without a per-turn scope, the same paper cited in
                two different turns on one page would produce two DOM
                elements sharing the exact same {'type':..., 'index': pmid}
                id, which Dash disallows.
        """
        try:
            with get_db_session() as db:
                # 1. Save the chat message
                session = db.get(SessionModel, session_id)
                if session is None:
                    logger.warning(f"Session {session_id} not found when trying to save query.")
                    return None

                query = QueryModel(
                    session_id=session_id,
                    prompt=prompt,
                    synthesis=synthesis,
                    model_used=model_used,
                    consensus_meter=json.dumps(consensus_meter) if consensus_meter else None,
                )
                db.add(query)
                db.flush()  # assigns query.id without waiting for the outer commit

                # 2. Link the citations used in this specific message.
                # citation_index=i matches EXACTLY how
                # llm.engine.generate_copilot_synthesis numbers its [N]
                # brackets (enumerate(valid_records), 0-based here,
                # 1-based in the bracket text) - cited_records is the same
                # processed_records list, in the same order, passed to
                # both call sites, so this index is the source of truth
                # for reconstructing that order later. See
                # get_session_chat_history/get_query_citations, which both
                # ORDER BY this column instead of trusting an unordered
                # "fingerprint IN (...)" fetch.
                for i, rec in enumerate(cited_records):
                    fingerprint = rec.get("fingerprint") if isinstance(rec, dict) else getattr(rec, "fingerprint", None)
                    if fingerprint:
                        db.add(QueryCitation(query_id=query.id, paper_fingerprint=fingerprint, citation_index=i))

                new_query_id = query.id

            logger.info(f"Saved query '{prompt[:20]}...' to session {session_id}")
            return new_query_id

        except Exception as e:
            logger.error(f"Failed to save query and citations: {e}")
            return None

    @staticmethod
    def update_session_summary(session_id: str, summary: str) -> None:
        """
        Updates the final generated synthesis summary for a specific session.
        This allows the history feed to instantly reload the AI's final conclusion.

        Args:
            session_id (str): The unique ID of the session.
            summary (str): The raw markdown generated by the LLM.
        """
        try:
            with get_db_session() as db:
                session = db.get(SessionModel, session_id)
                if session is None:
                    logger.warning(f"Session {session_id} not found when trying to update summary.")
                    return
                session.summary = summary
            logger.info(f"Successfully saved AI synthesis summary for session {session_id}")
        except Exception as e:
            logger.error(f"Error updating session summary: {e}")

    @staticmethod
    def delete_session(session_id: str) -> bool:
        """
        Permanently deletes a session and everything scoped to it - its
        chat turns and their citation links, and its paper-relevance links
        - but NOT the underlying PaperModel rows, which are shared/global
        across every session that ever surfaced them.

        SQLite doesn't enforce FK-level ON DELETE CASCADE unless
        PRAGMA foreign_keys=ON is set per-connection (it deliberately isn't
        here - see database.py), so this deletes children explicitly, in
        dependency order, rather than relying on the ondelete="CASCADE"
        declarations in models.py to do it automatically.

        Returns:
            bool: True if a session was found and deleted, False if it
                didn't exist (nothing to do).
        """
        with get_db_session() as db:
            session = db.get(SessionModel, session_id)
            if session is None:
                return False

            query_ids = list(db.execute(
                select(QueryModel.id).where(QueryModel.session_id == session_id)
            ).scalars().all())
            if query_ids:
                db.execute(delete(QueryCitation).where(QueryCitation.query_id.in_(query_ids)))
            db.execute(delete(QueryModel).where(QueryModel.session_id == session_id))
            db.execute(delete(SessionPaperLink).where(SessionPaperLink.session_id == session_id))
            db.execute(delete(SessionModel).where(SessionModel.id == session_id))

        logger.info(f"Deleted session {session_id} and its associated data.")
        return True
