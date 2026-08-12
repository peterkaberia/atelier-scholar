import dash
import logging
import uuid
from dash import html, Input, Output, State, callback, no_update

from core.logger import setup_global_logging
from core.utils import is_yes_no_question, order_citations_by_appearance
from database import AtelierRepository, Record
from database.models import SessionModel
from llm import AtelierAIEngine
from pipeline import ResearchOrchestrator
from ui.layouts import build_atelier_meter, build_failed_placeholder, build_flow_header, build_loading_skeleton, build_paper_cards, build_search_warning_banner, build_synthesis_body
from .background import background_manager

logger = logging.getLogger(__name__)

@callback(
    Output('url', 'pathname'),
    Output('store-pending-search', 'data'),
    Output('store-current-topic', 'data', allow_duplicate=True),
    Output('store-processed-records', 'data', allow_duplicate=True),
    Output('store-chat-history', 'data', allow_duplicate=True),
    Input('hero-search-btn', 'n_clicks'),
    State('hero-search-input', 'value'),
    State('hero-llm-dropdown', 'value'),
    prevent_initial_call=True
)
def create_new_session(hero_clicks, hero_text, hero_llm):
    """
    Fires when a user initiates a search from the Home Page. Creates the
    session row immediately (STATUS_IN_PROGRESS) - not backgrounded, since
    this does zero I/O and redirecting instantly matters more here than
    background-callback semantics - so it shows up in the sidebar/history
    right away instead of only appearing once the pipeline's first node
    finally gets around to it, and writes the new query to the pending
    clipboard, then redirects to the new Feed view.
    """
    if not hero_clicks or not hero_text or not hero_text.strip():
        return [no_update] * 5

    new_session_id = str(uuid.uuid4())[:8]
    topic = hero_text.strip()
    AtelierRepository.create_session(new_session_id, topic, "")

    trigger_data = {"query": topic, "llm": hero_llm, "session_id": new_session_id}

    return f"/history/{new_session_id}", trigger_data, "", [], []

@callback(
    Output('trigger-synthesis', 'data', allow_duplicate=True),
    Output('pending-flow-update', 'data', allow_duplicate=True),
    Input('trigger-search', 'data'),
    background=True,
    manager=background_manager(),
    progress=[Output('pending-flow-update', 'data', allow_duplicate=True)],
    prevent_initial_call=True
)
def run_search(set_progress, search_data):
    """
    Executes external API calls (PubMed, Europe PMC, OpenAlex, Semantic
    Scholar, Crossref, arXiv - run in parallel, see
    AtelierAcademicSearch.run_comprehensive_search), merges data, calculates
    SPLADE scores, and performs LLM extraction on the top K papers -
    reporting real progress through each stage (and each paper during
    extraction) rather than sitting on one opaque spinner for the whole run.

    flow-container/store-current-topic/store-processed-records all go
    through pending-flow-update, not directly - see
    ui/callbacks/ui_extras.py's gate_flow_update docstring: a background
    job keeps running (and eventually writes its result) even after the
    user has navigated to a DIFFERENT session's page in this same tab, and
    Dash Outputs target component IDs regardless of what the user is
    currently looking at, so an unguarded direct write would silently
    overwrite whatever the user navigated to.

    trigger-synthesis stays direct (ungated): generate_synth still needs to
    run, save its results, and mark the session complete even if the user
    has since navigated away - only the VISUAL update should be withheld in
    that case, not the underlying pipeline work. It carries
    processed_records directly in its own payload rather than making
    generate_synth depend on store-processed-records, since that store
    might be withheld by the gate while generate_synth's own downstream
    step still needs the real data regardless of what's on screen.

    existing_flows comes from search_data (handed off by route_intent),
    NOT from State('flow-container', 'children'). route_intent's own
    flow-container write goes through the async pending-flow-update gate,
    and this callback (also background=True) can start running before that
    gate has actually applied the update on the client - reading State here
    would race and could observe an earlier flow-container, silently
    dropping whatever route_intent had just added (this is confirmed to
    have happened - see route_intent's docstring).
    """
    # Dash's DiskcacheManager runs every background=True callback in a
    # spawned subprocess (via the `multiprocess`/dill library, NOT stdlib
    # multiprocessing) that unpickles this function's closure directly -
    # confirmed by direct inspection that app.py never gets imported there,
    # so app.py's own module-level setup_global_logging() call (which only
    # ever runs in the real server process and the Werkzeug reloader's
    # child) never reaches this process. Without this, the root logger here
    # has zero handlers and sits at the default WARNING level, silently
    # dropping every logger.info() call in this function AND everything it
    # calls into (search/paper.py, pipeline/nodes.py, database/repository.py
    # ...) - confirmed as the cause of a live "logging isn't logging"
    # report, where background-job output (the vast majority of the app's
    # actual logging) simply never appeared anywhere. Idempotent (guards on
    # `if not root_logger.handlers`), so calling it on every invocation is
    # harmless.
    setup_global_logging()

    query = search_data.get('query')
    selected_llm = search_data.get('llm')
    session_id = search_data.get("session_id")
    existing_flows = search_data.get('existing_flows') or []

    def report(label: str):
        # Persisted (not just pushed live) so a tab that wasn't watching -
        # one that navigated back, reloaded, or opened the session fresh -
        # can still show the real current stage. See
        # AtelierRepository.set_session_progress's docstring.
        AtelierRepository.set_session_progress(session_id, label)
        live_flows = existing_flows[:-1] + [build_loading_skeleton(query, label)]
        set_progress([{"session_id": session_id, "updates": {"flow_container": live_flows}}])

    try:
        orchestrator = ResearchOrchestrator(
            session_id=session_id,
            topic=query,
            model_choice=selected_llm
        )
        processed_records = orchestrator.execute_full_search_pipeline(progress_callback=report)
    except Exception as e:
        logger.error(f"Search pipeline failed for session {session_id}: {e}")
        AtelierRepository.set_session_status(session_id, SessionModel.STATUS_FAILED)
        existing_flows.pop()  # remove the routing skeleton
        existing_flows.append(build_failed_placeholder(query, session_id))
        pending = {"session_id": session_id, "updates": {"flow_container": existing_flows, "current_topic": query, "processed_records": []}}
        # trigger-synthesis stays no_update: no processed_records to synthesize from.
        return no_update, pending

    existing_flows.pop()  # Remove search skeleton

    # Disclose any source that errored out mid-search (rate limit, timeout,
    # etc.) rather than letting the paper count silently look complete -
    # see AtelierAcademicSearch.last_warnings / ResearchState.search_warnings.
    warning_banner = build_search_warning_banner(orchestrator.search_warnings)
    if warning_banner:
        existing_flows.append(warning_banner)

    existing_flows.append(build_loading_skeleton(query, "Synthesizing AI consensus...", icon="auto_awesome"))

    # Pass processed_records AND existing_flows forward so generate_synth
    # doesn't have to rely on store-processed-records or a fresh
    # State('flow-container') read (see docstrings above).
    synth_trigger = {**search_data, "processed_records": processed_records, "existing_flows": existing_flows}
    pending = {"session_id": session_id, "updates": {"flow_container": existing_flows, "current_topic": query, "processed_records": processed_records}}
    return synth_trigger, pending

@callback(
    Output('pending-flow-update', 'data', allow_duplicate=True),
    Input('trigger-synthesis', 'data'),
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def generate_synth(synth_data):
    """
    Generates the final structured synthesis markdown document based on the
    extracted paper logic. Completes the SEARCH flow, marking the session
    STATUS_COMPLETED (or STATUS_FAILED if something goes wrong here) so the
    sidebar/history status icons and the feed's resume-on-navigate logic
    reflect reality.

    Writes flow-container/store-synthesis-markdown through
    pending-flow-update, not directly - see run_search's docstring and
    ui/callbacks/ui_extras.py's gate_flow_update.
    """
    if not synth_data:
        raise dash.exceptions.PreventUpdate

    # See run_search's identical call/comment above - this callback is its
    # own separate spawned subprocess too.
    setup_global_logging()

    query = synth_data.get('query')
    selected_llm = synth_data.get('llm')
    session_id = synth_data.get('session_id')
    # Carried directly in the trigger payload by run_search, not read from
    # store-processed-records/State('flow-container') - see run_search's
    # docstring: this callback is background=True and can dispatch before
    # run_search's own pending-flow-update gate has actually applied, so a
    # fresh State read here was racy and previously dropped content (e.g.
    # the search-warning banner run_search had just appended).
    processed_records = synth_data.get('processed_records') or []
    existing_flows = synth_data.get('existing_flows') or []

    try:
        engine = AtelierAIEngine(model_choice=selected_llm)
        raw_md = engine.generate_copilot_synthesis(topic=query, valid_records=processed_records)

        # Renumber [N] citations by the order they actually appear in the
        # text, not the relevance-rank order they were pre-assigned in
        # before the LLM wrote anything (generate_copilot_synthesis cites
        # using indices matching processed_records' existing order, and the
        # LLM's narrative doesn't necessarily follow that same order) - see
        # core.utils.order_citations_by_appearance's docstring. Reorders
        # processed_records to match, BEFORE generate_atelier_meter below,
        # since the meter's own paper_index positions need to line up with
        # whatever order actually ships to the UI/DB.
        raw_md, processed_records = order_citations_by_appearance(
            raw_md, [(str(i + 1), r) for i, r in enumerate(processed_records)]
        )

        # Atelier Meter: only attempted for queries that read as yes/no
        # research questions (cheap local check first - see
        # core.utils.is_yes_no_question) - the LLM call inside
        # generate_atelier_meter has its own second-layer "applicable"
        # check and returns None for anything that doesn't cleanly resolve
        # to a yes/no verdict, so meter is None (and the UI renders nothing)
        # for the common case.
        meter = engine.generate_atelier_meter(topic=query, valid_records=processed_records) if is_yes_no_question(query) else None

        # 1. Save the Query and its Citations to the Database
        new_query_id = AtelierRepository.save_query_with_citations(
            session_id=session_id,
            prompt=query,
            model_used=selected_llm,
            synthesis=raw_md,
            cited_records=processed_records,
            consensus_meter=meter,
        )

        # Small inline check to avoid querying the DB if we don't have to
        if " SEARCH " not in existing_flows[0]['props'].get('children', str(existing_flows)):
            display_summary = raw_md.split('\n')[0].replace('#', '').strip()[:300]  # Simple heuristic for summary
            AtelierRepository.update_session_summary(session_id, display_summary)

        AtelierRepository.set_session_status(session_id, SessionModel.STATUS_COMPLETED)

    except Exception as e:
        logger.error(f"Synthesis failed for session {session_id}: {e}")
        AtelierRepository.set_session_status(session_id, SessionModel.STATUS_FAILED)
        existing_flows.pop()
        existing_flows.append(build_failed_placeholder(query, session_id))
        return {"session_id": session_id, "updates": {"flow_container": existing_flows}}

    existing_flows.pop()
    existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
        build_flow_header(query, selected_llm, f"{len(processed_records)} References", query_id=new_query_id),
        build_atelier_meter(meter),
        build_synthesis_body(raw_md, title="Abstract", valid_records=processed_records),
        build_paper_cards(processed_records, query_id=new_query_id, session_id=session_id)
    ]))
    return {"session_id": session_id, "updates": {"flow_container": existing_flows, "synthesis_markdown": raw_md}}
