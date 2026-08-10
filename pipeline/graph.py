"""
Research Graph Assembly Module

Wires the node functions in nodes.py into a compiled LangGraph StateGraph.
Kept separate from orchestrator.py so the graph shape can be inspected/tested
independently of the class that drives it.
"""

from langgraph.graph import END, StateGraph

from . import nodes as N
from .state import ResearchState


def build_research_graph(checkpointer=None):
    """
    Assembles and compiles the research pipeline graph.

    Args:
        checkpointer: An optional LangGraph checkpointer (e.g.
            langgraph.checkpoint.sqlite.SqliteSaver). When given, the graph
            persists its state after every node completes, keyed by the
            thread_id in each run's RunnableConfig - see
            pipeline/orchestrator.py for how this makes an interrupted run
            (server restart, killed process) resumable from wherever it left
            off instead of needing to start over from "ensure_session".
    """
    graph = StateGraph(ResearchState)

    graph.add_node("ensure_session", N.ensure_session_node)
    graph.add_node("plan_queries", N.plan_queries_node)
    graph.add_node("execute_search", N.execute_search_node)
    graph.add_node("broaden_query", N.broaden_query_node)
    graph.add_node("score_and_save", N.score_and_save_node)
    graph.add_node("extract", N.extract_node)

    graph.set_entry_point("ensure_session")
    graph.add_edge("ensure_session", "plan_queries")
    graph.add_edge("plan_queries", "execute_search")

    graph.add_conditional_edges(
        "execute_search",
        N.route_after_search,
        {
            "score_and_save": "score_and_save",
            "broaden_query": "broaden_query",
            "end": END,
        },
    )
    graph.add_edge("broaden_query", "execute_search")
    graph.add_edge("score_and_save", "extract")

    graph.add_conditional_edges(
        "extract",
        N.route_after_extraction,
        {
            "broaden_query": "broaden_query",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
