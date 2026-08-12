import html
import dash
import logging
from dash import html, ALL, Input, Output, State, callback, no_update
from core.utils import order_citations_by_appearance
from database.repository import AtelierRepository
from database.models import SessionModel
from llm import AtelierAIEngine
from pipeline.agent import run_investigation as run_investigation_agent
from pipeline.nodes import extract_records
from ui.layouts import build_flow_header, build_loading_skeleton, build_paper_cards, build_synthesis_body
from .background import background_manager

logger = logging.getLogger(__name__)

@callback(
    Output('trigger-router', 'data', allow_duplicate=True),
    Output('flow-container', 'children', allow_duplicate=True),
    Input({'type': 'retry-search-btn', 'index': ALL}, 'n_clicks'),
    State('llm-dropdown', 'value'),
    prevent_initial_call=True,
)
def retry_failed_search(n_clicks_list, selected_llm):
    """
    Re-runs a STATUS_FAILED session's pipeline for the SAME session_id
    rather than starting a brand new one. Thanks to LangGraph's SQLite
    checkpointer (pipeline/orchestrator.py) plus the permanent per-paper
    caching already in place (extracted answers, SPLADE vectors, full-text
    chunks all persist as they complete, independent of which run produced
    them), this isn't a blind do-over - if the interrupted run had gotten
    partway through, the graph resumes from its last completed node instead
    of re-planning/re-searching/re-extracting everything from zero.

    The target session_id comes from the CLICKED BUTTON's own pattern
    -matching index (via callback_context), not from a separate
    State('current-session-id') read - that Store is set once when
    layout_feed() renders and is NOT guaranteed to still match whichever
    failed placeholder is actually on screen (e.g. a stale
    poll_session_status tick from a previous page can still land after
    navigating elsewhere). Trusting a sibling Store instead of the trigger
    itself was a real bug: it could retry a completely different session
    than the one the user actually clicked Retry on.
    """
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered = dash.callback_context.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise dash.exceptions.PreventUpdate
    session_id = triggered.get("index")
    if not session_id:
        raise dash.exceptions.PreventUpdate

    topic = AtelierRepository.get_session_topic(session_id)
    if not topic:
        raise dash.exceptions.PreventUpdate

    AtelierRepository.set_session_status(session_id, SessionModel.STATUS_IN_PROGRESS)
    new_flows = [build_loading_skeleton(topic, "Retrying — resuming where this search left off...")]
    return {"query": topic, "llm": selected_llm, "session_id": session_id, "existing_flows": new_flows}, new_flows


@callback(
    Output('trigger-router', 'data', allow_duplicate=True), 
    Output('flow-container', 'children', allow_duplicate=True),
    Output('search-input', 'value', allow_duplicate=True),
    Output('store-pending-search', 'data', allow_duplicate=True), # Clears clipboard
    Input('current-session-id', 'data'), # Fires when the feed page mounts
    Input('search-btn', 'n_clicks'),     # Fires from the bottom bar
    State('store-pending-search', 'data'),
    State('search-input', 'value'),
    State('llm-dropdown', 'value'),
    State('flow-container', 'children'),
    prevent_initial_call=True
)
def handle_feed_interactions(session_id, bottom_clicks, pending_search, bottom_text, bottom_llm, existing_flows):
    """
    Acts as the entry point for AI generation within the Feed View.
    Handles two scenarios:
    A) The page just loaded and there is a query waiting in the clipboard.
    B) The user typed a follow-up directly into the bottom chat bar.

    Deliberately NOT background=True: this does zero I/O (just reads Store
    state and appends a skeleton), so backgrounding it only adds dispatch
    overhead - the actual slow work happens in route_intent/run_search
    (still background=True), triggered synchronously from here.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not existing_flows: 
        existing_flows = []

    # SCENARIO A: The Feed Page just loaded from a Home Page handoff
    if triggered_id == 'current-session-id':
        if pending_search and pending_search.get("session_id") == session_id:
            query = pending_search["query"]
            llm = pending_search["llm"]

            # A brand-new session's first render (this scenario only fires
            # for one) - layout_feed() may have already rendered its own
            # "still processing" placeholder in existing_flows, since the
            # session row now exists (create_new_session creates it
            # immediately) before this turn's real skeleton is ready.
            # Replace it rather than appending, or the user sees two
            # stacked loading indicators for a moment.
            existing_flows = [build_loading_skeleton(query, "Analyzing query intent...")]
            # Trigger Router, Update flow, Clear bottom input, Clear clipboard
            router_payload = {"query": query, "llm": llm, "session_id": session_id, "existing_flows": existing_flows}
            return router_payload, existing_flows, "", None
        return [no_update] * 4

    # SCENARIO B: User typed a follow-up query in the bottom bar
    elif triggered_id == 'search-btn':
        if not bottom_clicks or not bottom_text or not bottom_text.strip():
            return [no_update] * 4

        # Inject Skeleton Loader
        existing_flows.append(build_loading_skeleton(bottom_text, "Analyzing query intent..."))
        # Trigger Router, Update flow, Clear bottom input, Do nothing to clipboard
        #
        # existing_flows is handed forward explicitly in the router payload
        # (not left for route_intent to re-read from State('flow-container'))
        # because route_intent runs in a background job that can dispatch
        # before this callback's own flow-container write has actually been
        # applied by the client - see route_intent's docstring.
        router_payload = {"query": bottom_text, "llm": bottom_llm, "session_id": session_id, "existing_flows": existing_flows}
        return router_payload, existing_flows, "", no_update

    return [no_update] * 4

@callback(
    Output('trigger-search', 'data', allow_duplicate=True),
    Output('trigger-chat', 'data', allow_duplicate=True),
    Output('trigger-investigate', 'data', allow_duplicate=True),
    Output('pending-flow-update', 'data', allow_duplicate=True),
    Input('trigger-router', 'data'),
    State('store-current-topic', 'data'),
    State('store-processed-records', 'data'),
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def route_intent(router_data, current_topic, processed_records):
    """
    Context-Aware Intent Router.
    Determines if the user's query requires fetching new documents (SEARCH)
    or just chatting with the existing context (CHAT).

    Writes flow-container through pending-flow-update, not directly - see
    ui/callbacks/ui_extras.py's gate_flow_update docstring. trigger-search/
    trigger-chat/trigger-investigate stay direct (ungated): those drive the
    actual pipeline work (search, extraction, persistence), which must
    still complete and save even if the user has since navigated away from
    this session's page - only the VISUAL update should be suppressed in
    that case, not the underlying work.

    existing_flows comes from router_data (handed off by whichever
    callback triggered this), NOT from State('flow-container', 'children').
    This callback is background=True, dispatched the instant trigger-router
    changes - at that exact moment there's no guarantee the triggering
    callback's own flow-container write has actually landed on the client
    yet (it may itself be going through the async pending-flow-update gate).
    Reading State here raced against that write and could silently observe
    a stale/earlier flow-container, which is how the "[Semantic Scholar
    skipped] warning banner disappears after a follow-up" bug happened -
    a downstream stage's existing_flows.pop()/append() operated on a list
    that didn't actually include what the previous stage had just added.

    current_topic/processed_records fall back to the DB when the State
    Stores are empty - store-current-topic/store-processed-records are
    memory-only (see ui/layouts/main.py's serve_layout), so they reset to
    empty on EVERY fresh page load/reload, regardless of whether the
    session actually has prior context. Without this fallback, ANY
    follow-up typed after simply reloading the page hit the "no context"
    branch below and got force-routed to SEARCH - confirmed live: a
    follow-up asking to "write an abstract with the info you have"
    triggered a full new pipeline run instead of ever reaching CHAT.
    """
    query = router_data.get('query')
    selected_llm = router_data.get('llm')
    session_id = router_data.get('session_id')
    existing_flows = router_data.get('existing_flows') or []

    if not current_topic:
        current_topic = AtelierRepository.get_session_topic(session_id)
    if not processed_records:
        last_query_id = AtelierRepository.get_last_query_id(session_id)
        if last_query_id:
            processed_records = AtelierRepository.get_query_citations(last_query_id)

    if not current_topic or not processed_records:
        route = "SEARCH"
        search_query = query
    else:
        # LLM analyzes intent based on the query and current context state
        engine = AtelierAIEngine(model_choice=selected_llm)
        parsed_data = engine.followup_chat(query=query, current_topic=current_topic, processed_records=processed_records)
        route = parsed_data.get("intent", "CHAT").upper()
        search_query = parsed_data.get("standalone_query", query)

        if not search_query or search_query.lower() == "null":
            search_query = f"{current_topic} {query}"

        # Before committing to a full external re-search, check whether
        # papers already fetched during the ORIGINAL search but never
        # extracted (ranked beyond the initial top-20 cutoff -
        # pipeline/nodes.py's EXTRACTION_TARGET_COUNT) happen to cover this
        # follow-up. Often does: a follow-up frequently drills into an
        # angle the first pass simply didn't prioritize highly enough to
        # make the cut, not one genuinely absent from the search results
        # altogether. Cheaper and much faster than a fresh multi-engine
        # search, and only spent when the cheap classification above
        # already leans SEARCH - unaffected follow-ups pay nothing extra.
        if "SEARCH" in route:
            existing_fingerprints = [r.get('fingerprint') for r in processed_records if r.get('fingerprint')]
            extra_records = extract_records(
                session_id=session_id,
                topic=search_query,
                model_choice=selected_llm,
                limit=10,
                pool_size=30,
                exclude_fingerprints=existing_fingerprints,
            )
            if extra_records:
                expanded_records = processed_records + extra_records
                recheck = engine.followup_chat(query=query, current_topic=current_topic, processed_records=expanded_records)
                recheck_intent = recheck.get("intent", "SEARCH").upper()
                if recheck_intent in ("CHAT", "INVESTIGATE"):
                    route = recheck_intent
                    processed_records = expanded_records

    existing_flows.pop() # Remove routing skeleton

    if "SEARCH" in route:
        # run_search (ui/callbacks/search.py) takes over from here and
        # replaces this skeleton with live per-stage progress.
        existing_flows.append(build_loading_skeleton(search_query, "Planning your search..."))
        pending = {"session_id": session_id, "updates": {"flow_container": existing_flows}}
        search_trigger = {"query": search_query, "llm": selected_llm, "session_id": session_id, "existing_flows": existing_flows}
        return search_trigger, no_update, no_update, pending

    elif "INVESTIGATE" in route:
        # run_investigation (this file, below) takes over - a ReAct agent
        # (pipeline/agent.py) that digs into the existing papers more
        # actively than plain CHAT, falling back to plain CHAT itself if
        # the agent fails - see run_investigation's docstring.
        existing_flows.append(build_loading_skeleton(query, "Investigating your question...", icon="travel_explore", spin=False))
        pending = {"session_id": session_id, "updates": {"flow_container": existing_flows}}
        investigate_trigger = {"query": query, "llm": selected_llm, "session_id": session_id, "existing_flows": existing_flows}
        return no_update, no_update, investigate_trigger, pending

    else:
        existing_flows.append(build_loading_skeleton(query, "Deep diving into context...", icon="psychology", spin=False))
        pending = {"session_id": session_id, "updates": {"flow_container": existing_flows}}
        # processed_records here is the EXPANDED set when local pool
        # expansion above found extra papers and flipped the route to CHAT -
        # carried through the trigger payload (not left for run_chat to read
        # State('store-processed-records')) for the same race-condition
        # reason existing_flows already is - see this function's own
        # docstring.
        chat_trigger = {"query": query, "llm": selected_llm, "session_id": session_id, "existing_flows": existing_flows, "processed_records": processed_records}
        return no_update, chat_trigger, no_update, pending

@callback(
    Output('pending-flow-update', 'data', allow_duplicate=True),
    Input('trigger-chat', 'data'),
    State('store-chat-history', 'data'),
    # No `running=[...]` here (deliberately - see below) and `cancel` is
    # still safe: an Input that simply can't fire while its component is
    # unmounted just means the cancel trigger is unavailable, not an error.
    background=True,
    manager=background_manager(),
    cancel=[Input("search-btn", "n_clicks")],
    prevent_initial_call=True
)
def run_chat(chat_data, chat_history):
    """
    Context-aware RAG implementation. Generates responses to follow-up
    questions using the currently loaded literature abstracts.

    Writes flow-container/store-chat-history through pending-flow-update,
    not directly - see ui/callbacks/ui_extras.py's gate_flow_update
    docstring.

    existing_flows AND processed_records both come from chat_data (handed
    off by route_intent), not State('flow-container', 'children') /
    State('store-processed-records', 'data') - see route_intent's docstring
    for why reading State here is racy and previously dropped content.
    processed_records specifically may be a LARGER set than what's in the
    Store: route_intent's local-pool-expansion step (see its own docstring)
    can fold in a few extra papers before routing here, and that expanded
    set - not the original Store snapshot - is what this turn should
    actually answer from and cite.

    No `running=[...]` toggling search-btn/search-btn-icon (previously
    disabled the send button + spun its icon while this job was in
    flight): those ids only exist inside layout_feed()'s bottom bar, but
    this callback is triggered by trigger-chat, a globally-mounted Store
    route_intent can write from ANY page - and since this is a
    background=True job, there's a real window between dispatch and
    completion where the user can navigate away from the feed page
    entirely. Dash's `running` outputs apply unconditionally at job
    start/end with no equivalent to the pending-flow-update gate used
    elsewhere in this file, so if the target page isn't mounted when they
    fire, the client throws "nonexistent object was used in an Output" -
    the same bug class fixed in ui/callbacks/ui_extras.py's
    confirm_session_delete. Losing the button's disabled/spin state during
    processing is a minor UX gap; a hard crash isn't an acceptable trade
    for it.
    """
    if not chat_data or not chat_data.get('query'):
        raise dash.exceptions.PreventUpdate

    query = chat_data.get('query')
    selected_llm = chat_data.get('llm')
    session_id = chat_data.get('session_id')
    existing_flows = chat_data.get('existing_flows') or []
    processed_records = chat_data.get('processed_records') or []

    # Ensure memory structures exist
    if chat_history is None: chat_history = []
    if existing_flows is None: existing_flows = []

    # 2. Append user message to memory
    chat_history.append({"role": "user", "content": query})

    try:
        # 3. Boot up the AI Engine and generate the response
        logger.info(f"Generating follow-up chat for session '{session_id}'...")
        engine = AtelierAIEngine(model_choice=selected_llm)

        ai_response = engine.chat_with_literature(
            chat_history=chat_history,
            context_records=processed_records
        )

        # Renumber [N] citations by order of first appearance in the reply,
        # not the pre-assigned context_records order - same reasoning as
        # generate_synth's identical step in ui/callbacks/search.py, see
        # core.utils.order_citations_by_appearance's docstring. Also means
        # this reply's own reference numbering doesn't need to (and won't)
        # match any other turn's, even for the same overlapping papers.
        ai_response, processed_records = order_citations_by_appearance(
            ai_response, [(str(i + 1), r) for i, r in enumerate(processed_records)]
        )

        # 4. Save the interaction and citations to the SQLite database
        new_query_id = AtelierRepository.save_query_with_citations(
            session_id=session_id,
            prompt=query,
            synthesis=ai_response,
            model_used=selected_llm,
            cited_records=processed_records
        )

        # 5. Append AI response to memory
        chat_history.append({"role": "assistant", "content": ai_response})

        # 6. Build the new visual block for the Feed UI
        #
        # Includes build_paper_cards(), matching what layout_feed's
        # historical reconstruction renders for every chat turn on reload -
        # a prior version of this omitted it here specifically to avoid
        # duplicating the library table on every follow-up, but that made
        # the Results/Evidence Library section for a follow-up genuinely
        # absent from the LIVE update and only ever visible after a reload
        # (confirmed live - the "results section only visible after
        # reload" bug). ui/layouts/main.py's index_string collapses every
        # flow-block's Results section except the newest one by default
        # (results-accordion-body), so repeating it per turn no longer
        # means repeating it VISIBLY per turn either.
        new_flow = html.Div(className="flow-block border-t border-slate-100", children=[
            build_flow_header(query, selected_llm, f"{len(processed_records)} Context Papers", query_id=new_query_id),
            build_synthesis_body(ai_response, valid_records=processed_records),
            build_paper_cards(processed_records, query_id=new_query_id, session_id=session_id),
        ])

        # Remove route_intent's "Deep diving into context..." skeleton
        # (the last element of existing_flows - see route_intent's
        # docstring) before appending the real answer - without this the
        # skeleton was never replaced, just left permanently stranded
        # above the finished turn (confirmed live via screenshot: both the
        # spinner block AND the completed answer block visible at once,
        # forever - run_search/generate_synth already pop their own
        # skeleton the same way, this just never got the same treatment).
        if existing_flows:
            existing_flows.pop()
        existing_flows.append(new_flow)
        logger.info("Chat follow-up generated and saved successfully.")

        return {"session_id": session_id, "updates": {"flow_container": existing_flows, "chat_history": chat_history}}

    except Exception as e:
        logger.error(f"Chat generation failed: {e}")

        # Graceful UI degradation if the API fails or the user cancels
        error_flow = html.Div(className="py-12 px-4 md:px-12 bg-red-50/30 border-t border-red-100", children=[
            html.Div(className="max-w-5xl mx-auto flex items-center space-x-3 text-red-600", children=[
                html.Span("error", className="material-symbols-outlined text-2xl"),
                html.Span(f"Request interrupted or failed: {str(e)}", className="font-bold")
            ])
        ])
        if existing_flows:
            existing_flows.pop()  # same skeleton-removal reasoning as the success path above
        existing_flows.append(error_flow)

        # chat_history still gets the user's message even though the
        # assistant's reply failed - matches the previous (pre-gating)
        # behavior of always returning chat_history here.
        return {"session_id": session_id, "updates": {"flow_container": existing_flows, "chat_history": chat_history}}


@callback(
    Output('pending-flow-update', 'data', allow_duplicate=True),
    Input('trigger-investigate', 'data'),
    State('store-chat-history', 'data'),
    background=True,
    manager=background_manager(),
    progress=[Output('pending-flow-update', 'data', allow_duplicate=True)],
    cancel=[Input("search-btn", "n_clicks")],
    prevent_initial_call=True
)
def run_investigation(set_progress, investigate_data, chat_history):
    """
    Runs pipeline.agent's ReAct investigation agent for a follow-up routed
    to INVESTIGATE (see route_intent). Live per-tool-call progress via
    set_progress mirrors run_search's report() closure - the agent can take
    a while (multiple tool calls, each potentially its own LLM round trip),
    so this avoids sitting on one opaque spinner the whole time.

    existing_flows comes from investigate_data (handed off by route_intent),
    not State('flow-container', 'children') - same race-condition reasoning
    as run_chat/run_search - see route_intent's docstring.

    Citation format conversion: the agent cites papers by fingerprint (e.g.
    "[doi:10.1234/xyz]" - see pipeline/agent.py's system prompt), but the
    rest of Atelier's citation UI (build_synthesis_body's hover-popup JS)
    expects sequential [1]/[2] brackets positionally indexed into
    valid_records. Rewritten below so investigation answers get the exact
    same interactive citation treatment as a normal synthesis/chat answer,
    not plain unclickable bracket text.
    """
    if not investigate_data or not investigate_data.get('query'):
        raise dash.exceptions.PreventUpdate

    query = investigate_data.get('query')
    selected_llm = investigate_data.get('llm')
    session_id = investigate_data.get('session_id')
    existing_flows = investigate_data.get('existing_flows') or []

    if chat_history is None:
        chat_history = []
    chat_history.append({"role": "user", "content": query})

    def report(label: str):
        live_flows = existing_flows[:-1] + [build_loading_skeleton(query, label, icon="travel_explore", spin=False)]
        set_progress([{"session_id": session_id, "updates": {"flow_container": live_flows}}])

    try:
        logger.info(f"Running investigation agent for session '{session_id}'...")
        result = run_investigation_agent(
            session_id=session_id,
            query=query,
            model_choice=selected_llm,
            chat_history=chat_history,
            progress_callback=report,
        )

        # Citations are determined by scanning the answer TEXT for
        # [fingerprint] markers against every paper in this session, not
        # from result['cited_fingerprints'] (a live side-effect of tools
        # actually executing) - confirmed live that a checkpoint-resumed
        # agent run can answer entirely from tool results already recorded
        # in a prior checkpointed run without re-invoking any tools this
        # time, leaving cited_fingerprints empty even though the answer
        # genuinely cites specific papers. Scanning the text itself is
        # correct regardless of whether tools freshly ran or the agent
        # resumed from checkpoint.
        #
        # order_citations_by_appearance also fixes the renumbering itself:
        # this used to assign [1]/[2]/... by iterating session papers in
        # relevance-score order, not by where each fingerprint actually
        # FIRST appears in the answer, and did the substitution via
        # sequential .replace() calls - both wrong, see that function's
        # docstring for why (citation order should follow the text, and
        # sequential replace() can corrupt overlapping renumberings).
        candidates = [
            (rec.fingerprint, rec.__dict__)
            for rec in AtelierRepository.get_session_processed_papers(session_id, limit=100)
        ]
        answer_text, valid_records = order_citations_by_appearance(result["answer"], candidates)

        new_query_id = AtelierRepository.save_query_with_citations(
            session_id=session_id,
            prompt=query,
            synthesis=answer_text,
            model_used=selected_llm,
            cited_records=valid_records,
        )

        chat_history.append({"role": "assistant", "content": answer_text})

        ref_label = f"{len(valid_records)} Papers Investigated" if not result.get("fell_back") else f"{len(valid_records)} Context Papers"
        # build_paper_cards() included for the same reason as run_chat's
        # identical addition above - see its comment. Historical
        # reconstruction (layout_feed) already rendered this on reload;
        # this was the gap that made it look reload-only live.
        new_flow = html.Div(className="flow-block border-t border-slate-100", children=[
            build_flow_header(query, selected_llm, ref_label, query_id=new_query_id),
            build_synthesis_body(answer_text, title="Investigation", valid_records=valid_records),
            build_paper_cards(valid_records, query_id=new_query_id, session_id=session_id),
        ])
        # Remove route_intent's "Investigating your question..." skeleton
        # (the last element of existing_flows) before appending the real
        # answer - report()'s set_progress calls above only ever pushed a
        # LOCAL live_flows copy, never mutated existing_flows itself, so
        # without this pop the skeleton was left permanently stranded
        # above the finished turn - same bug and fix as run_chat's
        # identical addition above, see its comment for the full story.
        if existing_flows:
            existing_flows.pop()
        existing_flows.append(new_flow)
        logger.info(f"Investigation complete for session '{session_id}' (fell_back={result.get('fell_back')}).")

        return {"session_id": session_id, "updates": {"flow_container": existing_flows, "chat_history": chat_history}}

    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        error_flow = html.Div(className="py-12 px-4 md:px-12 bg-red-50/30 border-t border-red-100", children=[
            html.Div(className="max-w-5xl mx-auto flex items-center space-x-3 text-red-600", children=[
                html.Span("error", className="material-symbols-outlined text-2xl"),
                html.Span(f"Investigation interrupted or failed: {str(e)}", className="font-bold")
            ])
        ])
        if existing_flows:
            existing_flows.pop()  # same skeleton-removal reasoning as the success path above
        existing_flows.append(error_flow)
        return {"session_id": session_id, "updates": {"flow_container": existing_flows, "chat_history": chat_history}}