"""
Research Pipeline State Module

Defines the shared state object that flows through the LangGraph research
graph. Every node reads from and returns a partial update to this shape.
"""

from typing import Any, Dict, List, Optional, TypedDict


class ResearchState(TypedDict):
    """The state threaded through every node of the research graph."""

    # --- Session Identity ---
    session_id: str
    topic: str
    model_choice: str

    # --- Search Planning & Execution ---
    query_plan: Dict[str, str]
    candidate_records: List[Any]  # database.models.Record instances
    # Human-readable labels of any academic engine that errored out during
    # execute_search_node (rate limit, timeout, etc.) - lets the UI tell the
    # user a source is missing instead of silently under-reporting results.
    search_warnings: List[str]

    # --- Extraction Output ---
    processed_records: List[Dict[str, Any]]
    relevant_count: int

    # --- Control-Flow Guards ---
    retry_count: int

    # --- Observability ---
    status: str
    errors: List[str]
