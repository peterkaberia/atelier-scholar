"""
Research Orchestrator Module

This module serves as the central command for the Atelier pipeline. It wraps
the LangGraph research graph (graph.py/nodes.py) behind the same public
interface the Dash callbacks already depend on: construct with
(session_id, topic, model_choice), then call execute_full_search_pipeline().

The actual workflow - AI query planning, parallel API fetching, neural
scoring, database persistence, and AI data extraction, with retry/broaden
control-flow when a search comes back empty or under-relevant - lives in
nodes.py as graph nodes.
"""

import hashlib
import logging
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_research_graph
from .state import ResearchState

logger = logging.getLogger(__name__)

# Separate from atelier_history.db (the permanent record store) - this file
# holds only in-flight/transient LangGraph node-level checkpoints, keyed by
# thread_id (== session_id). Its own file so it can be wiped independently
# (e.g. to force every session to start fresh) without touching real data.
CHECKPOINT_DB_PATH = "atelier_checkpoints.db"

# Friendly, user-facing labels for each node-transition boundary. Keyed by
# node name (not the state["status"] strings nodes.py sets, which are meant
# for logs) so this stays decoupled from internal state semantics. extract's
# label is a fallback only - extract_node reports its own finer-grained
# per-paper progress via the same callback while it's actually running.
_NODE_PROGRESS_LABELS = {
    "ensure_session": "Preparing your session...",
    "plan_queries": "Planning optimized search queries...",
    "execute_search": "Searching academic databases...",
    "broaden_query": "No results yet - broadening the search...",
    "score_and_save": "Scoring and saving candidate papers...",
    "extract": "Extracting findings from the most relevant papers...",
}


class ResearchOrchestrator:
    """
    Manages the end-to-end pipeline of generating a literature consensus.

    Drives the compiled LangGraph research graph to fulfill a user's
    research request, preserving the same constructor/method shape the
    previous imperative orchestrator exposed.
    """

    def __init__(self, session_id: str, topic: str, model_choice: str):
        """
        Initializes the Orchestrator.

        Args:
            session_id (str): The unique identifier for the current user session.
            topic (str): The natural language research question asked by the user.
            model_choice (str): The LLM requested by the user (e.g., 'gpt-4o').
        """
        self.session_id = session_id
        self.topic = topic
        self.model_choice = model_choice

        # Own SQLite connection per orchestrator instance: background jobs
        # each run in their own spawned process (see
        # ui/callbacks/background.py's DiskcacheManager), so a shared
        # module-level connection couldn't be reused across them anyway -
        # opening fresh here, once per run, is correct. check_same_thread is
        # relaxed since a Dash background=True job may hand this off between
        # threads internally; WAL matches atelier_history.db's own setting so
        # concurrent sessions (see ui/layouts/main.py's per-tab stores) don't
        # lock each other out.
        self._checkpoint_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        self._checkpoint_conn.execute("PRAGMA journal_mode=WAL")
        self.checkpointer = SqliteSaver(self._checkpoint_conn)
        self.graph = build_research_graph(checkpointer=self.checkpointer)

        # Populated after execute_full_search_pipeline() runs: human-readable
        # labels of any academic engine that failed during the search, so
        # the caller (ui/callbacks/search.py) can surface a "results are
        # missing a source" notice instead of silently under-reporting.
        self.search_warnings: List[str] = []

    def execute_full_search_pipeline(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the entire research workflow: Plan -> Fetch -> Score -> Save
        -> Extract, with automatic query-broadening retries.

        Args:
            progress_callback: If given, called with a human-readable status
                string as the pipeline moves through each stage (and, during
                extraction, once per paper) - lets the UI show real progress
                instead of one opaque "working..." spinner for the whole run.
                See ui/callbacks/search.py for how this feeds a Dash
                background=True callback's set_progress.

        Returns:
            List[Dict[str, Any]]: A list of cleanly extracted paper dictionaries
                                  ready to be rendered by the UI.
        """
        logger.info(f"Starting graph pipeline for session '{self.session_id}' | Topic: '{self.topic}'")

        initial_state: ResearchState = {
            "session_id": self.session_id,
            "topic": self.topic,
            "model_choice": self.model_choice,
            "query_plan": {},
            "candidate_records": [],
            "search_warnings": [],
            "processed_records": [],
            "relevant_count": 0,
            "retry_count": 0,
            "status": "Initializing",
            "errors": [],
        }

        # thread_id is LangGraph's checkpoint key. Was session_id alone,
        # meaning every run under this session_id shared ONE checkpoint
        # history - correct for "resume THIS session's search if it was
        # interrupted," but it silently broke a second scenario this
        # checkpoint design never accounted for: ui/callbacks/chat.py's
        # route_intent can ALSO dispatch a brand new SEARCH for a
        # DIFFERENT topic on a session that already has one COMPLETED
        # search (a follow-up the router decided needs genuinely new
        # external results). Once a session_id's checkpoint reaches
        # "complete," the early-return below fires unconditionally and
        # returns the FIRST search's stale processed_records - the new
        # self.topic was never even looked at. Confirmed live: a
        # follow-up's own distinct topic triggered a full pipeline log
        # line, then immediately hit "already complete; skipping re-run"
        # and returned only the original search's papers.
        #
        # Mixing the topic into thread_id fixes this while preserving the
        # original resume behavior: the SAME topic on the SAME session
        # (an actual retry, or a coincidental repeat) still hashes to the
        # same thread_id and correctly resumes/short-circuits, while a
        # genuinely DIFFERENT follow-up topic gets its own fresh
        # checkpoint history instead of being silently absorbed into an
        # unrelated completed one.
        topic_key = hashlib.sha1(self.topic.strip().lower().encode("utf-8")).hexdigest()[:10]
        thread_id = f"{self.session_id}:{topic_key}"
        run_config = {"configurable": {"progress_callback": progress_callback, "thread_id": thread_id}}

        try:
            # If a previous run for this exact session_id was interrupted
            # mid-flight (server restart, killed process, a retry after
            # STATUS_FAILED), the checkpointer already has its last-completed
            # -node state saved under this thread_id. get_state() returns an
            # empty snapshot (falsy .values) for a thread that's never been
            # run, so this same check correctly covers a genuinely fresh run
            # too - no separate "is this a retry" flag needed anywhere else.
            snapshot = self.graph.get_state(run_config)

            if snapshot.values and not snapshot.next:
                # Fully completed under this thread_id already (e.g. retry
                # button clicked on a session that had actually finished) -
                # nothing left to run.
                logger.info(f"Session '{self.session_id}' already complete per checkpoint; skipping re-run.")
                processed_records = snapshot.values.get("processed_records", [])
                self.search_warnings = snapshot.values.get("search_warnings", [])
                return processed_records

            resuming = bool(snapshot.values) and bool(snapshot.next)
            stream_input = None if resuming else initial_state
            final_state: Dict[str, Any] = dict(snapshot.values) if resuming else dict(initial_state)

            if resuming:
                logger.info(f"Resuming session '{self.session_id}' from checkpoint - pending: {snapshot.next}.")
                if progress_callback:
                    progress_callback("Resuming from where this search left off...")

            # graph.stream() (rather than invoke()) yields the state update
            # after EVERY node completes, which is what makes node
            # -transition-level progress possible at all - invoke() only
            # ever returns once, at the very end, with no visibility into
            # what happened in between. Passing stream_input=None on a
            # resume is what tells LangGraph to continue from the persisted
            # checkpoint instead of restarting at "ensure_session".
            for step in self.graph.stream(stream_input, config=run_config):
                for node_name, state_update in step.items():
                    final_state.update(state_update)
                    if progress_callback:
                        label = _NODE_PROGRESS_LABELS.get(node_name)
                        if label:
                            progress_callback(label)

            processed_records = final_state.get("processed_records", [])
            self.search_warnings = final_state.get("search_warnings", [])

            logger.info(f"Pipeline complete. Yielded {len(processed_records)} relevant, extracted papers.")
            return processed_records
        finally:
            # Matters most under CeleryManager (long-lived worker processes
            # that run many jobs over time, unlike DiskcacheManager's
            # one-subprocess-per-job model) - without this, connections would
            # accumulate for the life of the worker instead of per-run.
            self._checkpoint_conn.close()
