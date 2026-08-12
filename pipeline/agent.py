"""
Investigation Agent Module

A LangGraph ReAct agent (langgraph.prebuilt.create_react_agent) that
investigates a follow-up question using the tool set in agent_tools.py.
Used for follow-ups routed to the INVESTIGATE intent (see
llm.engine.followup_chat / ui/callbacks/chat.py) - multi-step questions
that need more than a single plain-RAG chat answer, but don't necessarily
need a whole new deterministic search pipeline run.

Deliberately NOT the default path for every follow-up: tool-calling
reliability varies a lot across models, especially smaller/local ones this
app also supports, so this only runs when the router specifically decides
a follow-up needs it, and falls back to the existing plain
chat_with_literature path on any failure - see run_investigation's
docstring.
"""

import hashlib
import logging
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from database.repository import AtelierRepository
from llm.engine import AtelierAIEngine

from .agent_tools import build_investigation_tools
from .orchestrator import CHECKPOINT_DB_PATH

logger = logging.getLogger(__name__)

# Hard cap on tool-call round-trips. Confirmed live that a genuinely
# thorough multi-paper comparison can legitimately need ~15-20 tool calls
# (list papers, then analyze/search several on each side of a comparison),
# so this is deliberately generous rather than tight - a bigger cost is
# cutting off a real investigation partway through. recursion_limit is
# derived from this with a wide margin (not a tight 2x) since
# create_react_agent's real steps-per-tool-call ratio in practice ran
# higher than the naively-expected "one LLM turn + one tool execution" -
# see run_investigation's docstring for what happens if the budget is
# still exceeded (create_react_agent degrades gracefully with a canned
# message rather than raising, so that's handled explicitly, not just via
# GraphRecursionError).
MAX_TOOL_CALLS = 15

_SYSTEM_PROMPT = """You are Atelier's research investigation assistant. The user has asked a follow-up question that needs active digging, not just a summary of what's already on screen.

You have tools to: see what papers are already in this session, search academic databases for new ones, fetch a paper's full text, search within a paper's full text for specific passages, and get a focused analysis of one or more papers on a specific angle.

Rules:
1. ALWAYS call list_session_papers first - the answer may already be sitting in papers already gathered, and searching again wastes time.
2. Only call search_more_papers if the existing papers genuinely don't cover the question - it's the slowest tool available.
3. Prefer rag_search_chunks or analyze_papers over get_full_text when you have a specific question about a paper - they're faster and more focused than reading the whole thing.
4. IMPORTANT for speed: analyze_papers accepts MULTIPLE fingerprints in one call and analyzes them concurrently - if you need to look at several papers for the same angle (e.g. comparing how 4 papers each treat a topic), pass ALL of their fingerprints in ONE analyze_papers call, never one call per paper. Same for get_full_text/rag_search_chunks when you already know you need several specific papers - request what you can up front rather than one at a time when the results don't depend on each other.
5. Cite papers using their fingerprint in brackets, e.g. [doi:10.1234/xyz], so the user can trace every claim back to a real source.
6. Give your final answer as clear, direct prose - not a tool-call log. Once you have enough to answer, stop calling tools and answer.
"""


def run_investigation(
    session_id: str,
    query: str,
    model_choice: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs the ReAct investigation agent for one follow-up question.

    Args:
        progress_callback: called with a human-readable label each time the
            agent invokes a tool, e.g. "Using tool: rag_search_chunks..." -
            same live-progress mechanism run_search's report() already uses.

    Returns:
        {"answer": str, "cited_fingerprints": List[str], "fell_back": bool}
        fell_back=True means the agent itself failed or hit MAX_TOOL_CALLS
        without producing a final answer, and `answer` came from the
        existing plain chat_with_literature path instead - callers should
        treat this identically to a normal CHAT response (see run_chat in
        ui/callbacks/chat.py), just with a different citation set.
    """
    touched: set = set()
    tools = build_investigation_tools(session_id, model_choice, touched)
    engine = AtelierAIEngine(model_choice=model_choice)

    checkpoint_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpoint_conn.execute("PRAGMA journal_mode=WAL")

    try:
        # _get_llm is "internal" to AtelierAIEngine only by convention -
        # pipeline/ already depends tightly on llm/engine.py throughout
        # (nodes.py does the same), and this is the one place that needs a
        # raw LangChain chat model instance rather than one of
        # AtelierAIEngine's own higher-level methods.
        llm = engine._get_llm(temperature=0.1)
        checkpointer = SqliteSaver(checkpoint_conn)
        agent = create_react_agent(llm, tools, checkpointer=checkpointer, prompt=_SYSTEM_PROMPT)

        messages = []
        for turn in (chat_history or [])[-6:]:
            if turn.get("role") == "user":
                messages.append(HumanMessage(content=turn.get("content", "")))
            else:
                messages.append(AIMessage(content=turn.get("content", "")))
        messages.append(HumanMessage(content=query))

        # thread_id keyed by query text (not just session_id) so a
        # DIFFERENT follow-up in the same session gets its own checkpoint
        # history instead of resuming/short-circuiting into an unrelated
        # completed investigation - same reasoning as
        # ResearchOrchestrator.execute_full_search_pipeline's thread_id.
        query_key = hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()[:10]
        config = {
            "configurable": {"thread_id": f"agent:{session_id}:{query_key}"},
            # Wide margin, not a tight 2x: confirmed live that
            # create_react_agent's actual steps-per-tool-call ratio runs
            # higher than "one LLM turn + one tool execution" in practice
            # (a 19-tool-call investigation exhausted an 18-step budget) -
            # see MAX_TOOL_CALLS's docstring.
            "recursion_limit": (MAX_TOOL_CALLS * 4) + 4,
        }

        final_messages = None
        for step in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            final_messages = step["messages"]
            last = final_messages[-1] if final_messages else None
            tool_calls = getattr(last, "tool_calls", None) if last else None
            if progress_callback and tool_calls:
                for tc in tool_calls:
                    progress_callback(f"Investigating — using {tc['name']}...")

        if not final_messages:
            raise RuntimeError("Investigation agent produced no output.")

        answer = final_messages[-1].content
        if not answer or not isinstance(answer, str):
            raise RuntimeError("Investigation agent's final message had no usable text.")

        # create_react_agent degrades gracefully when its internal step
        # budget runs low: it returns THIS exact canned message as a normal
        # final AI message instead of raising GraphRecursionError (this is
        # documented behavior, not a guess - see create_react_agent's own
        # docstring in langgraph/prebuilt/chat_agent_executor.py). Confirmed
        # live: a genuinely complex investigation hit this and would have
        # been silently accepted as a real, if useless, answer without this
        # check - GraphRecursionError below never fires for this case.
        if answer.strip() == "Sorry, need more steps to process this request.":
            logger.warning(f"Investigation agent exhausted its step budget for session {session_id}; falling back to plain chat.")
            return _fallback_chat(engine, session_id, query, chat_history)

        return {"answer": answer, "cited_fingerprints": list(touched), "fell_back": False}

    except GraphRecursionError:
        logger.warning(f"Investigation agent hit its {MAX_TOOL_CALLS}-tool-call cap for session {session_id}; falling back to plain chat.")
        return _fallback_chat(engine, session_id, query, chat_history)
    except Exception as e:
        logger.warning(f"Investigation agent failed for session {session_id}: {e}; falling back to plain chat.")
        return _fallback_chat(engine, session_id, query, chat_history)
    finally:
        checkpoint_conn.close()


def _fallback_chat(engine: AtelierAIEngine, session_id: str, query: str, chat_history: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
    """Graceful degradation path - see run_investigation's docstring."""
    context_records = AtelierRepository.get_session_processed_papers(session_id, limit=20)
    fallback_history = (chat_history or []) + [{"role": "user", "content": query}]
    answer = engine.chat_with_literature(
        chat_history=fallback_history,
        context_records=[r.__dict__ for r in context_records],
    )
    return {
        "answer": answer,
        "cited_fingerprints": [r.fingerprint for r in context_records],
        "fell_back": True,
    }
