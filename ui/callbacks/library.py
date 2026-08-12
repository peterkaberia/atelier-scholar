import logging

import dash
from dash import ALL, MATCH, Input, Output, State, callback, dcc, html, no_update

from core.export import build_bibtex, build_ris
from core.pdf_export import build_turn_pdf
from database.repository import AtelierRepository
from llm import AtelierAIEngine
from llm.utils import get_model_choices
from pipeline.nodes import extract_records
from ui.layouts.feed import _build_paper_card_articles
from .background import background_manager

logger = logging.getLogger(__name__)

_PANEL_OPEN = {"transform": "translateX(0%)", "transition": "transform 0.25s ease-out"}
_PANEL_CLOSED = {"transform": "translateX(100%)", "transition": "transform 0.25s ease-out"}


def _selected_fingerprints(checkbox_values: list, scope: str) -> set:
    """
    Flattens dcc.Checklist pattern-matching State values (a list of
    zero-or-one-item lists, one per checkbox on the page - see
    ui/layouts/feed.py's build_paper_cards) down to the set of paper
    fingerprints checked WITHIN one specific scope (query_id), stripping the
    "scope:" prefix each checkbox value carries so unrelated flow-blocks'
    selections on the same page never bleed into this one's.
    """
    checked = {v for vals in checkbox_values for v in (vals or [])}
    prefix = f"{scope}:"
    return {v[len(prefix):] for v in checked if v.startswith(prefix)}


@callback(
    Output('download-references', 'data'),
    Input({'type': 'export-btn', 'query_id': ALL, 'format': ALL}, 'n_clicks'),
    State({'type': 'paper-checkbox', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def export_references(n_clicks_list, checkbox_values):
    """
    Downloads the checked papers (or every paper in this turn, if none are
    checked - exporting nothing on an empty selection would be a confusing
    dead-end) as a .bib or .ris file, sourced fresh from the DB by query_id
    rather than a live Store - works identically for a just-generated turn
    and a historical one reloaded from disk.
    """
    triggered = dash.callback_context.triggered_id
    trigger_value = dash.callback_context.triggered[0]['value'] if dash.callback_context.triggered else None
    if not triggered or not isinstance(triggered, dict) or not trigger_value:
        raise dash.exceptions.PreventUpdate

    query_id = triggered.get('query_id')
    fmt = triggered.get('format')

    try:
        qid_int = int(query_id)
    except (TypeError, ValueError):
        logger.warning(f"Export requested with a non-integer query_id ({query_id!r}) - likely an unsaved/failed turn.")
        raise dash.exceptions.PreventUpdate

    records = AtelierRepository.get_query_citations(qid_int)
    if not records:
        raise dash.exceptions.PreventUpdate

    selected = _selected_fingerprints(checkbox_values, str(query_id))
    if selected:
        records = [r for r in records if r.get('fingerprint') in selected]

    content = build_bibtex(records) if fmt == 'bib' else build_ris(records)
    filename = f"atelier-references-{query_id}.{fmt}"
    return dcc.send_string(content, filename)


@callback(
    Output('download-references', 'data', allow_duplicate=True),
    Input({'type': 'export-single-btn', 'fingerprint': ALL, 'format': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def export_single_reference(n_clicks_list):
    """
    Downloads just the ONE paper currently open in the paper-detail panel
    (see _build_paper_detail_content's export_buttons) - a quicker path
    than checking that paper's box and using the bulk export buttons up in
    the turn's header when you only want this single reference.
    """
    triggered = dash.callback_context.triggered_id
    trigger_value = dash.callback_context.triggered[0]['value'] if dash.callback_context.triggered else None
    if not triggered or not isinstance(triggered, dict) or not trigger_value:
        raise dash.exceptions.PreventUpdate

    fingerprint = triggered.get('fingerprint')
    fmt = triggered.get('format')
    if not fingerprint:
        raise dash.exceptions.PreventUpdate

    record = AtelierRepository.get_paper_by_fingerprint(fingerprint)
    if not record:
        raise dash.exceptions.PreventUpdate

    content = build_bibtex([record]) if fmt == 'bib' else build_ris([record])
    filename = f"atelier-reference.{fmt}"
    return dcc.send_string(content, filename)


@callback(
    Output({'type': 'paper-cards-list', 'query_id': MATCH}, 'children'),
    Output({'type': 'load-more-status', 'query_id': MATCH}, 'children'),
    Output({'type': 'ref-count-text', 'query_id': MATCH}, 'children'),
    Input({'type': 'load-more-btn', 'query_id': MATCH}, 'n_clicks'),
    running=[
        (Output({'type': 'load-more-btn', 'query_id': MATCH}, 'disabled'), True, False),
        (Output({'type': 'load-more-btn', 'query_id': MATCH}, 'children'), "Searching for more…", "Load More Results"),
    ],
    background=True,
    manager=background_manager(),
    prevent_initial_call=True,
)
def load_more(n_clicks):
    """
    Fetches the next batch of ranked-but-unshown papers for this turn's
    session, extracts them the same way the initial search did (see
    pipeline.nodes.extract_records), persists them as additional citations
    on this SAME query/turn, and appends their cards to the list - a real
    "load more" rather than the previously-dead button.

    Deliberately simpler than the main pipeline in one respect: no
    full-text RAG passage re-fetch beyond what extract_records already does
    identically - this IS the same routine, just with a narrower candidate
    pool (whatever's left after excluding what's already shown) and a
    smaller target count. Kept as a single MATCH'd background callback
    (not per-flow-block wiring) since {'type':..., 'query_id':...} pattern-
    matching IDs already scope everything correctly per turn.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    triggered = dash.callback_context.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise dash.exceptions.PreventUpdate

    try:
        qid_int = int(triggered.get('query_id'))
    except (TypeError, ValueError):
        raise dash.exceptions.PreventUpdate

    query_info = AtelierRepository.get_query_by_id(qid_int)
    if not query_info:
        return no_update, "Couldn't load more — this turn is no longer available.", no_update

    existing_records = AtelierRepository.get_query_citations(qid_int)
    existing_fingerprints = [r.get('fingerprint') for r in existing_records if r.get('fingerprint')]

    new_records = extract_records(
        session_id=query_info['session_id'],
        topic=query_info['prompt'],
        model_choice=query_info['model_used'],
        limit=20,
        pool_size=60,
        exclude_fingerprints=existing_fingerprints,
    )

    if not new_records:
        return no_update, "No more relevant papers found for this search.", no_update

    AtelierRepository.add_citations_to_query(qid_int, new_records)

    all_records = existing_records + new_records
    articles = _build_paper_card_articles(all_records, str(qid_int))
    plural = "s" if len(new_records) != 1 else ""
    status = f"Added {len(new_records)} more paper{plural}."
    return articles, status, f"{len(all_records)} References"


def _scope_to_session_id(scope: str) -> str:
    """
    scope is a query_id (int-as-string) when one was available at the
    paper card's render time, otherwise a bare session_id (or "unscoped")
    - see build_paper_cards's docstring for the fallback chain.
    """
    try:
        query_info = AtelierRepository.get_query_by_id(int(scope))
        return query_info["session_id"] if query_info else None
    except (TypeError, ValueError):
        return scope if scope != "unscoped" else None


_LOADING_CONTENT = html.Div(className="flex items-center justify-center h-40", children=[
    html.Span("autorenew", className="material-symbols-outlined animate-spin text-slate-300 text-3xl"),
])


def _build_paper_detail_content(record: dict, topic: str, selected_llm: str, session_id: str, fingerprint: str) -> "html.Div":
    """
    Full detail view: all authors, journal link (via DOI), PDF link
    (in-app viewer), AI summary.

    The summary is cached per (fingerprint, session_id) - see
    PaperSummaryModel's docstring - so re-opening the same paper in the
    same session is instant on every click after the first, instead of
    re-running the LLM call every time.
    """
    cached_summary = AtelierRepository.get_cached_paper_summary(fingerprint, session_id) if session_id else None
    if cached_summary:
        summary_text = cached_summary
    else:
        engine = AtelierAIEngine(model_choice=selected_llm)
        summary_text = engine.generate_paper_summary(topic=topic, record=record)
        if session_id:
            AtelierRepository.save_paper_summary(fingerprint, session_id, summary_text)

    authors = record.get('authors') or ['Unknown']
    doi = record.get('doi')
    pdf_url = record.get('pdf_url')

    links = []
    if doi:
        links.append(html.A("View Journal Page ↗", href=f"https://doi.org/{doi}", target="_blank",
                             className="text-xs font-bold text-primary hover:underline"))
    if pdf_url:
        links.append(html.Button("View PDF", className="pdf-viewer-btn text-xs font-bold text-primary hover:underline",
                                  **{"data-pdf-url": pdf_url}))

    fp_key = fingerprint or record.get('fingerprint') or ''
    export_buttons = html.Div(className="flex items-center gap-3 mb-4", children=[
        html.Button(
            [html.Span("download", className="material-symbols-outlined text-sm mr-1"), "Export .bib"],
            id={'type': 'export-single-btn', 'fingerprint': fp_key, 'format': 'bib'}, n_clicks=0,
            className="flex items-center text-[11px] font-bold px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-700 hover:border-primary transition-colors",
        ),
        html.Button(
            [html.Span("download", className="material-symbols-outlined text-sm mr-1"), "Export .ris"],
            id={'type': 'export-single-btn', 'fingerprint': fp_key, 'format': 'ris'}, n_clicks=0,
            className="flex items-center text-[11px] font-bold px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-700 hover:border-primary transition-colors",
        ),
    ])

    return html.Div(children=[
        html.H3(record.get('title', 'Untitled'), className="text-base font-bold text-slate-900 mb-1"),
        html.P(", ".join(authors), className="text-xs text-slate-500 font-medium mb-1"),
        html.P(f"{record.get('journal', '')} • {record.get('year', 'N.D.')}", className="text-xs text-slate-400 font-semibold mb-3"),
        html.Div(className="flex items-center gap-4 mb-4", children=links) if links else None,
        export_buttons,
        html.Hr(className="border-slate-100 mb-4"),
        html.Div(className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2", children="AI Summary"),
        dcc.Markdown(summary_text, className="prose prose-sm max-w-none"),
    ])


@callback(
    Output('paper-summary-panel', 'style'),
    Output('paper-summary-content', 'children'),
    Input({'type': 'paper-checkbox', 'index': ALL}, 'value'),
    Input({'type': 'paper-detail-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def open_paper_summary_panel(checkbox_values, detail_clicks):
    """
    Opens the panel with a loading spinner (or closes it) INSTANTLY -
    deliberately NOT background=True. summarize_selected_paper below does
    the actual work (fetch record, hit the LLM if uncached) as a
    background=True job, which on this app spawns a brand new OS process
    per invocation (Windows multiprocessing 'spawn' - ui/callbacks/
    background.py) that has to re-import this entire app - torch,
    sentence-transformers, langchain, sqlalchemy, everything - before a
    single line of the callback body runs. Confirmed live: that spawn+
    import cost alone was ~15 seconds, ALL of it before the callback's own
    set_progress call (an earlier attempt at "show a spinner immediately")
    ever got a chance to fire - set_progress can only run once the process
    already exists. This callback shares the exact same Inputs but runs
    synchronously in the main process, so it reaches the browser in the
    same round trip as the click itself, with no spawn cost to wait behind.

    Both callbacks fire from the same click - Dash allows multiple
    callbacks on the same Input freely. This one always resolves in
    milliseconds, so in practice it always reaches the client and paints
    before the background job even finishes spawning, let alone
    completing - there's no real race despite both eventually touching
    paper-summary-content.
    """
    triggered = dash.callback_context.triggered_id

    if isinstance(triggered, dict) and triggered.get('type') == 'paper-detail-btn':
        if not dash.callback_context.triggered[0]['value']:
            raise dash.exceptions.PreventUpdate  # guards the n_clicks=0 render-time firing
        return _PANEL_OPEN, _LOADING_CONTENT

    checked = [v for vals in checkbox_values for v in (vals or [])]
    if len(checked) != 1:
        return _PANEL_CLOSED, no_update
    return _PANEL_OPEN, _LOADING_CONTENT


@callback(
    Output('paper-summary-content', 'children', allow_duplicate=True),
    Input({'type': 'paper-checkbox', 'index': ALL}, 'value'),
    Input({'type': 'paper-detail-btn', 'index': ALL}, 'n_clicks'),
    background=True,
    manager=background_manager(),
    prevent_initial_call=True,
)
def summarize_selected_paper(checkbox_values, detail_clicks):
    """
    Does the actual work for the paper-detail panel (open_paper_summary_panel
    above already opened it with a spinner) - fetches the paper, resolves
    session/topic/model, and generates (or reuses a cached) AI summary, then
    replaces the spinner with the real content. See
    open_paper_summary_panel's docstring for why this is split out as its
    own background=True callback instead of one callback doing both.

    A single shared panel (not one per flow-block) deliberately - unlike
    load_more/export, this doesn't need per-turn scoping to be correct: it
    just needs to know WHICH paper (by fingerprint) and WHAT topic, both
    derivable from the trigger's own scope prefix (see below) without
    needing per-block pattern-matched Outputs for something that's
    fundamentally a single "here's the detail view" surface at a time.

    Deliberately has NO State('current-session-id', ...) or
    State('llm-dropdown', ...) - both are page-scoped ids that only exist
    in layout_feed()'s render. Both of this callback's Inputs are
    pattern-matching wildcards with no page restriction of their own (they
    validate fine with zero matches on any page), so pairing either with a
    plain-id State that ISN'T always mounted throws "nonexistent object was
    used in a State" the moment Dash evaluates this callback's dependencies
    against a page that isn't the feed view (confirmed live on the home
    page) - the same underlying issue as the Output-side bug fixed in
    ui/callbacks/ui_extras.py's confirm_session_delete, just on the State
    side instead. Session/model are derived from the trigger's own
    "scope:fingerprint" value instead (scope is a query_id, or a bare
    session_id/"unscoped" when no query_id was available - see
    build_paper_cards's docstring), which carries no page dependency at all.
    """
    triggered = dash.callback_context.triggered_id

    if isinstance(triggered, dict) and triggered.get('type') == 'paper-detail-btn':
        if not dash.callback_context.triggered[0]['value']:
            raise dash.exceptions.PreventUpdate  # guards the n_clicks=0 render-time firing
        scope_fp = triggered.get('index', '')
    else:
        checked = [v for vals in checkbox_values for v in (vals or [])]
        if len(checked) != 1:
            raise dash.exceptions.PreventUpdate  # closing case - open_paper_summary_panel already handled it
        scope_fp = checked[0]

    # "scope:fingerprint" (see build_paper_cards).
    scope, _, fingerprint = scope_fp.partition(":")

    record = AtelierRepository.get_paper_by_fingerprint(fingerprint)
    if not record:
        return html.Div("Couldn't find this paper's data.", className="text-red-500")

    session_id = _scope_to_session_id(scope)
    topic = AtelierRepository.get_session_topic(session_id) if session_id else ""
    if not topic:
        return html.Div("No active topic to summarize against.", className="text-slate-400")

    last_used_model = AtelierRepository.get_last_used_model(session_id)
    available_models = get_model_choices()
    selected_llm = last_used_model if last_used_model in available_models else (available_models[0] if available_models else None)

    return _build_paper_detail_content(record, topic, selected_llm, session_id, fingerprint)


@callback(
    Output('download-pdf', 'data'),
    Input({'type': 'download-pdf-btn', 'query_id': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def export_pdf(n_clicks_list):
    """
    Downloads one turn (a synthesis, chat reply, or investigation answer -
    whatever query_id refers to) as a standalone, well-formatted PDF - see
    core.pdf_export.build_turn_pdf. Always sourced fresh from the DB by
    query_id, same as export_references, so it works identically for a
    just-generated turn and a historical one reloaded from disk.
    """
    triggered = dash.callback_context.triggered_id
    trigger_value = dash.callback_context.triggered[0]['value'] if dash.callback_context.triggered else None
    if not triggered or not isinstance(triggered, dict) or not trigger_value:
        raise dash.exceptions.PreventUpdate

    try:
        qid_int = int(triggered.get('query_id'))
    except (TypeError, ValueError):
        raise dash.exceptions.PreventUpdate

    query_info = AtelierRepository.get_query_by_id(qid_int)
    if not query_info:
        raise dash.exceptions.PreventUpdate

    records = AtelierRepository.get_query_citations(qid_int)
    pdf_bytes = build_turn_pdf(
        title=query_info["prompt"],
        model_name=query_info["model_used"],
        markdown_text=query_info["synthesis"],
        records=records,
    )

    filename = f"atelier-{qid_int}.pdf"
    return dcc.send_bytes(pdf_bytes, filename)
