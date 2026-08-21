import logging
from datetime import date, timedelta
import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
from core.pdf_export import derive_document_title, extract_section_headings, strip_leading_title
from database import AtelierRepository
from database.models import SessionModel
from ui.layouts import (
    build_failed_placeholder,
    build_flow_header,
    build_loading_skeleton,
    build_paper_cards,
    build_synthesis_body,
)
from ui.layouts.home import get_suggested_prompts

logger = logging.getLogger(__name__)

# Sidebar history date buckets, in display order - same convention as most
# chat-history sidebars (ChatGPT, Claude, etc.), which is itself what
# Stitch's own "Yesterday / Last 7 days / Last 30 days / This year" grouping
# is drawing from. A session's created_at falls into the first bucket whose
# predicate matches, checked in this order (Today before Yesterday before
# the 7/30-day windows, "Older" as the catch-all).
def _sidebar_date_buckets(today: date):
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    return [
        ("Today", lambda d: d == today),
        ("Yesterday", lambda d: d == yesterday),
        ("Previous 7 Days", lambda d: week_ago <= d < yesterday),
        ("Previous 30 Days", lambda d: month_ago <= d < week_ago),
        ("Older", lambda d: d < month_ago),
    ]

@callback(
    Output('selected-papers-container', 'children'), 
    Input({'type': 'paper-checkbox', 'index': ALL}, 'value'), 
    State('store-processed-records', 'data'), 
    prevent_initial_call=True
)
def update_pills(checkbox_values, processed_records):
    """Dynamically builds UI pills for explicitly selected papers in the evidence library."""
    selected = [val[0] for val in checkbox_values if val] 
    if not selected or not processed_records: return []
    
    return [
        html.Div(className="flex items-center bg-primary/5 hover:bg-primary/10 transition-all rounded-xl py-2 px-3 md:px-4 border border-primary/10 group shrink-0", children=[
            html.Span("description", className="material-symbols-outlined text-primary text-[18px] mr-2"), 
            html.Span(rec.get('title', '')[:35]+'...', className="text-primary text-[10px] font-extrabold uppercase"), 
            html.Button("close", id={'type':'remove-paper-btn', 'index':str(rec.get('pmid'))}, className="material-symbols-outlined text-primary/40 text-[14px] ml-2 hover:text-primary")
        ]) for rec in processed_records if str(rec.get('pmid')) in selected
    ]


@callback(
    Output('hero-search-input', 'value'),
    Input({'type': 'suggestion-chip', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def fill_suggestion_prompt(n_clicks_list):
    """
    Clicking one of Home's quick-start pills (ui/layouts/home.py's
    get_suggested_prompts - tailored to the user's own recent research
    once there's any, not the same static examples forever) drops its
    full text straight into the textarea, same as if the user had typed
    it - doesn't submit on its own, so they can still edit it first.
    Recomputes get_suggested_prompts() fresh rather than reading back
    whatever layout_home rendered, since nothing round-trips the actual
    chip labels (which are truncated for display) back to this callback -
    see that function's own docstring for why calling it twice like this
    is an acceptable trade rather than needing a Store in between.
    """
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered = dash.callback_context.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise dash.exceptions.PreventUpdate

    idx = triggered.get('index')
    prompts = get_suggested_prompts()
    if idx is None or idx >= len(prompts):
        raise dash.exceptions.PreventUpdate

    return prompts[idx]

# Small icon + tint per status, standing in for a real thumbnail (Atelier
# sessions have no rendered preview image the way a Stitch design project
# does) - gives each sidebar row the same "icon tile + title + date" shape
# Stitch's own project cards use, just without an actual picture.
_STATUS_ICON = {
    SessionModel.STATUS_IN_PROGRESS: ("autorenew", "text-amber-600 bg-amber-50"),
    SessionModel.STATUS_COMPLETED: ("description", "text-primary bg-primary/10"),
    SessionModel.STATUS_FAILED: ("error_outline", "text-red-600 bg-red-50"),
}


def _sidebar_history_row(s: dict):
    session_id = s.get('session_id')
    title = s.get('topic') or 'Untitled Search'
    status = s.get('status')
    created = s.get('created_at')
    date_str = created.strftime('%b %d, %Y') if created else ''
    icon, icon_class = _STATUS_ICON.get(status, _STATUS_ICON[SessionModel.STATUS_COMPLETED])
    pulse = " animate-pulse" if status == SessionModel.STATUS_IN_PROGRESS else ""

    # The delete button is a SIBLING of the Link, not nested inside it -
    # an interactive element nested inside a Dash dcc.Link (an <a> tag)
    # would still trigger navigation when clicked (click bubbles up to
    # the anchor) since there's no straightforward way to stopPropagation
    # from a plain Dash callback. Keeping them side-by-side avoids that
    # entirely instead of fighting it.
    return html.Li(className="group flex items-center gap-1", children=[
        dcc.Link(
            className="flex-1 min-w-0 flex items-center gap-2.5 px-2 py-2 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors",
            href=f"/session/{session_id}",
            children=[
                html.Span(
                    html.Span(icon, className="material-symbols-outlined text-[16px]"),
                    className=f"w-7 h-7 rounded-md flex items-center justify-center shrink-0{pulse} {icon_class}",
                ),
                html.Div(className="min-w-0 flex-1", children=[
                    html.P(title, className="text-xs font-semibold text-slate-700 truncate leading-snug m-0"),
                    html.P(date_str, className="text-[10px] text-slate-400 leading-snug m-0"),
                ]),
            ]
        ),
        html.Button(
            # A trash icon, not a kebab (more_vert) - this button has
            # exactly one action (delete), so a "more options" glyph
            # implying a menu was misleading. Reported directly.
            html.Span("delete", className="material-symbols-outlined text-[16px]"),
            id={'type': 'delete-session-btn', 'index': session_id},
            title="Delete this search",
            # Always visible (not opacity-0/group-hover:opacity-100)
            # - confirmed earlier in this app that Tailwind's
            # group-hover:opacity/visible variants don't reliably
            # take effect via this Tailwind CDN runtime (see
            # ui/layouts/main.py's index_string, which had to
            # replace the same pattern with real JS event listeners
            # for the citation hover popup). A subtle
            # always-visible icon sidesteps that risk entirely.
            className="shrink-0 p-1 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded transition-colors",
        )
    ])


def _build_sidebar_history_items(filter_text: str = ""):
    """
    Builds the sidebar's date-grouped recent-history list (Today/Yesterday/
    Previous 7 Days/Previous 30 Days/Older - see _sidebar_date_buckets),
    modeled on Stitch's own grouped project list. Extracted so both
    update_sidebar_history (below) and confirm_session_delete's post-delete
    refresh can produce the exact same markup without duplicating it.

    filter_text: case-insensitive substring match against each session's
    topic - powers the sidebar's search box (see layout_sidebar) the same
    way Stitch's own "Search projects" input does, entirely server-side.
    """
    # Unbounded - there's no separate '/history' archive page to defer to
    # anymore, so this IS the full list now. The Nav wrapping this in
    # layout_sidebar already scrolls internally (flex-1 min-h-0
    # overflow-y-auto), so an arbitrarily long history is a scroll, not a
    # layout problem.
    sessions = AtelierRepository.get_all_sessions_for_history()

    if filter_text:
        needle = filter_text.strip().lower()
        sessions = [s for s in sessions if needle in (s.get('topic') or '').lower()]

    if not sessions:
        empty_message = "No matching searches." if filter_text else "No searches yet."
        return [html.Li(empty_message, className="px-3 py-2 text-xs text-slate-400 italic")]

    buckets = _sidebar_date_buckets(date.today())
    grouped = {label: [] for label, _matches in buckets}
    for s in sessions:
        created = s.get('created_at')
        session_date = created.date() if created else date.today()  # no created_at (shouldn't happen) - bucket as Today rather than crash
        for label, matches in buckets:
            if matches(session_date):
                grouped[label].append(s)
                break

    groups = []
    for label, _matches in buckets:
        items = grouped[label]
        if not items:
            continue
        groups.append(html.Div(className="mb-1", children=[
            html.H4(label, className="px-2.5 pt-3 pb-1 text-[10px] font-bold text-slate-400 uppercase tracking-[0.1em]"),
            html.Ul(className="space-y-0.5", children=[_sidebar_history_row(s) for s in items]),
        ]))

    return groups


@callback(
    Output("sidebar-recent-history", "children"),
    Input("url", "pathname"),
    Input("sidebar-status-poll", "n_intervals"),
    Input("sidebar-history-search", "value"),
)
def update_sidebar_history(pathname, n_intervals, search_value):
    """
    Updates the sidebar history list on navigation, on a periodic poll (see
    layout_sidebar's dcc.Interval - navigation alone missed any status
    change or summary update for a background job finishing while the user
    stayed put on one page), and live as the user types into the sidebar's
    own search box.

    Empty while viewing one session's own detail page (/session/<id>) -
    "other sessions" (everything this list shows) shouldn't appear there
    at all, reported directly; that's update_session_nav's panel to show
    instead (this session's own prompts/sections). The search box that
    filters this list is hidden the same way, client-side - see
    toggleSidebarSearchBox in app.py/main.py.
    """
    if pathname and pathname.startswith('/session/'):
        return []
    return _build_sidebar_history_items(search_value or "")


def _build_session_nav_items(session_id: str):
    """
    A per-session table of contents: one entry per turn (its own short
    derived title - same derive_document_title logic build_flow_header
    uses - linking straight to that turn's flow-block via #turn-{query_id}),
    with that turn's own top-level "##" section headings nested underneath
    it (#turn-{query_id}-h-{i}, matching the ids ui/layouts/main.py's
    assignHeadingIds assigns client-side by the same position). Lets a
    long multi-turn session, or a single long structured report, be jumped
    around instead of only ever scrolled through top-to-bottom - reported
    directly: the sidebar should surface links to a session's own prompts
    and sections once you're looking at one.

    Returns None for a session with no turns yet (still mid-search, or an
    invalid id) - the caller renders nothing rather than an empty panel.
    """
    chat_history = AtelierRepository.get_session_chat_history(session_id)
    if not chat_history:
        return None

    items = []
    for chat in chat_history:
        query_id = chat.get('query_id')
        prompt = chat.get('prompt') or ''
        synthesis = chat.get('synthesis') or ''
        label = derive_document_title(synthesis, prompt)
        headings = extract_section_headings(strip_leading_title(synthesis))

        sub_items = [
            html.Li(html.A(
                h, href=f"#turn-{query_id}-h-{i}",
                className="block px-2 py-1 text-[11px] text-slate-500 hover:text-primary truncate",
            ))
            for i, h in enumerate(headings)
        ]

        items.append(html.Li(className="mb-0.5", children=[
            html.A(
                label, href=f"#turn-{query_id}", title=prompt,
                className="block px-2 py-1.5 text-xs font-semibold text-slate-700 hover:text-primary hover:bg-slate-50 rounded-md truncate",
            ),
            html.Ul(sub_items, className="ml-3 border-l border-slate-100 pl-2") if sub_items else None,
        ]))

    return html.Div(className="mb-4 pb-4 border-b border-slate-100", children=[
        html.H4("This Session", className="px-2.5 pb-1 text-[10px] font-bold text-slate-400 uppercase tracking-[0.1em]"),
        html.Ul(items, className="space-y-0.5"),
    ])


@callback(
    Output('sidebar-session-nav', 'children'),
    Input('url', 'pathname'),
    Input('sidebar-status-poll', 'n_intervals'),
)
def update_session_nav(pathname, n_intervals):
    """
    Populates the sidebar's per-session table of contents whenever a
    session detail page is showing - empty on Home/Settings, where there's
    no single session's prompts/sections to jump around. Shares
    sidebar-status-poll with update_sidebar_history so a turn completing
    while the user's still on the page adds its entry here too, not just
    updating the global history list.
    """
    if not pathname or not pathname.startswith('/session/'):
        return None
    session_id = pathname.rsplit('/', 1)[-1]
    if not session_id:
        return None
    return _build_session_nav_items(session_id)


@callback(
    Output('pending-delete-session', 'data'),
    Output('confirm-delete-session-dialog', 'displayed'),
    Input({'type': 'delete-session-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def request_session_delete(n_clicks_list):
    """
    Step 1 of the two-step delete flow: a click on any delete-session-btn
    (sidebar list or /history page - both use the same pattern-matching
    type) records WHICH session via callback_context.triggered_id and
    opens the shared confirmation dialog (ui/layouts/sidebar.py). Nothing
    is deleted yet - see confirm_session_delete for the actual delete,
    gated on the dialog's own confirm click.
    """
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered = dash.callback_context.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise dash.exceptions.PreventUpdate

    session_id = triggered.get('index')
    if not session_id:
        raise dash.exceptions.PreventUpdate

    return session_id, True


@callback(
    Output('sidebar-recent-history', 'children', allow_duplicate=True),
    Output('url', 'pathname', allow_duplicate=True),
    Input('confirm-delete-session-dialog', 'submit_n_clicks'),
    State('pending-delete-session', 'data'),
    State('url', 'pathname'),
    State('sidebar-history-search', 'value'),
    prevent_initial_call=True,
)
def confirm_session_delete(submit_n_clicks, session_id, current_pathname, search_value):
    """
    Step 2: fires only when the user clicks the ConfirmDialog's own
    "OK"/confirm button (submit_n_clicks), never on cancel/dismiss - Dash's
    dcc.ConfirmDialog only increments submit_n_clicks on an actual confirm.
    Deletes the session, refreshes the sidebar list, and redirects home if
    the user was looking at the very session just deleted (there's no
    standalone history/archive page to fall back to anymore - the sidebar
    IS the history list, and it's mounted on every route already).
    """
    if not submit_n_clicks or not session_id:
        raise dash.exceptions.PreventUpdate

    deleted = AtelierRepository.delete_session(session_id)
    if not deleted:
        logger.warning(f"Delete requested for session {session_id}, but it was already gone.")

    was_viewing_deleted = current_pathname == f"/session/{session_id}"
    new_pathname = "/" if was_viewing_deleted else no_update

    return _build_sidebar_history_items(search_value or ""), new_pathname

@callback(
    Output('flow-container', 'children', allow_duplicate=True),
    Output('feed-status-poll', 'disabled'),
    Input('feed-status-poll', 'n_intervals'),
    State('current-session-id', 'data'),
    prevent_initial_call=True,
)
def poll_session_status(n_intervals, session_id):
    """
    Only mounted (and only enabled) when layout_feed() rendered the
    "still processing" placeholder - i.e. the user opened/returned to a
    session whose search is genuinely still running server-side
    (background=True keeps it alive independent of this browser tab).
    Checks back every few seconds and swaps in the real content the moment
    the session leaves STATUS_IN_PROGRESS, then stops polling.
    """
    if not session_id:
        return no_update, True

    if not AtelierRepository.session_exists(session_id):
        # Row not created yet (a narrow race right after Home's redirect,
        # since create_new_session's write and this page's first render
        # happen in separate requests) - keep polling rather than treating
        # "not found" as "done", which would strand the user on the
        # placeholder forever.
        return no_update, False

    status = AtelierRepository.get_session_status(session_id)
    if status == SessionModel.STATUS_IN_PROGRESS:
        # Still running - refresh the displayed stage from the DB (not just
        # skip the update) so a resumed/reloaded tab, which never received
        # the live set_progress pushes, still advances roughly every poll
        # interval instead of sitting frozen on whatever it first showed.
        progress_message = AtelierRepository.get_session_progress(session_id)
        if progress_message:
            topic = AtelierRepository.get_session_topic(session_id)
            # See layout_feed's identical fallback for why this can show a
            # real model name even before this session's first turn finishes.
            processing_model = AtelierRepository.get_last_used_model(session_id)
            return [build_loading_skeleton(topic or "Still working on this...", progress_message, model_name=processing_model)], False
        return no_update, False

    chat_history = AtelierRepository.get_session_chat_history(session_id)
    if chat_history:
        flows = [
            html.Div(id=f"turn-{chat.get('query_id')}", className="flow-block border-t border-slate-100", children=[
                build_flow_header(chat.get('prompt'), chat.get('model_used'), f"{len(chat.get('citations', []))} References", query_id=chat.get('query_id'), synthesis_text=chat.get('synthesis')),
                build_synthesis_body(chat.get('synthesis'), valid_records=chat.get('citations', []), query_id=chat.get('query_id')),
                build_paper_cards(chat.get('citations', []), query_id=chat.get('query_id'), session_id=session_id),
            ])
            for chat in chat_history
        ]
        return flows, True

    if status == SessionModel.STATUS_FAILED:
        topic = AtelierRepository.get_session_topic(session_id)
        return [build_failed_placeholder(topic, session_id)], True

    # Completed but somehow still no history (shouldn't normally happen) -
    # stop polling either way rather than spinning forever.
    return no_update, True


@callback(
    Output({'type': 'paper-checkbox', 'index': MATCH}, 'value'),
    Input({'type': 'remove-paper-btn', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def remove_pill(n_clicks):
    """Unchecks the paper checkbox when the corresponding pill's close button is clicked."""
    if n_clicks and n_clicks > 0:
        return []
    return no_update


@callback(
    Output('flow-container', 'children', allow_duplicate=True),
    Output('store-current-topic', 'data', allow_duplicate=True),
    Output('store-processed-records', 'data', allow_duplicate=True),
    Output('store-chat-history', 'data', allow_duplicate=True),
    Output('store-synthesis-markdown', 'data', allow_duplicate=True),
    Input('pending-flow-update', 'data'),
    State('current-session-id', 'data'),
    prevent_initial_call=True,
)
def gate_flow_update(pending, current_session_id):
    """
    The single point where background-job output actually reaches the
    screen. route_intent/run_search/generate_synth/run_chat (chat.py,
    search.py) never write flow-container or the follow-up context stores
    directly - they all write here instead, tagged with the session_id
    they were working on. This callback is a normal (non-background,
    synchronous) one, so State('current-session-id') is read FRESH, live,
    at the moment each pending update actually arrives - exactly what's
    needed to catch "the user has since navigated to a different session's
    page in this same tab" (background=True jobs run for minutes; State
    values captured when a background job STARTS are long stale by the
    time it finishes, so gating inside the background callback itself
    can't work - only a freshly-triggered callback like this one can).

    A mismatched or missing session_id means the job's output belongs to a
    session the user isn't looking at anymore - every Output stays
    no_update, so whatever the user actually navigated to is left alone.
    """
    if not pending or not isinstance(pending, dict):
        raise dash.exceptions.PreventUpdate
    if pending.get('session_id') != current_session_id:
        raise dash.exceptions.PreventUpdate

    updates = pending.get('updates') or {}
    return (
        updates.get('flow_container', no_update),
        updates.get('current_topic', no_update),
        updates.get('processed_records', no_update),
        updates.get('chat_history', no_update),
        updates.get('synthesis_markdown', no_update),
    )
