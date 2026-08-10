import html
import logging
import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
from database import AtelierRepository
from database.models import SessionModel
from ui.layouts import (
    build_failed_placeholder,
    build_flow_header,
    build_loading_skeleton,
    build_paper_cards,
    build_status_dot,
    build_synthesis_body,
)

logger = logging.getLogger(__name__)

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

def _build_sidebar_history_items():
    """
    Builds the sidebar's recent-history <li> list. Extracted so both
    update_sidebar_history (below) and confirm_session_delete's post-delete
    refresh can produce the exact same markup without duplicating it.
    """
    recent_sessions = AtelierRepository.get_all_sessions_for_history()[:5]

    links = []
    for s in recent_sessions:
        session_id = s.get('session_id')
        title = s.get('topic', 'Untitled Search')
        status = s.get('status')

        # The delete button is a SIBLING of the Link, not nested inside it -
        # an interactive element nested inside a Dash dcc.Link (an <a> tag)
        # would still trigger navigation when clicked (click bubbles up to
        # the anchor) since there's no straightforward way to stopPropagation
        # from a plain Dash callback. Keeping them side-by-side avoids that
        # entirely instead of fighting it.
        links.append(
            html.Li(className="group flex items-center gap-1", children=[
                dcc.Link(
                    className="flex-1 min-w-0 flex items-center space-x-3 px-3 py-2 text-slate-500 hover:bg-slate-50 hover:text-slate-900 rounded-md transition-colors",
                    href=f"/history/{session_id}",
                    children=[
                        build_status_dot(status),
                        html.Span(title, className="text-xs font-semibold truncate w-full")
                    ]
                ),
                html.Button(
                    html.Span("more_vert", className="material-symbols-outlined text-[16px]"),
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
        )

    return links


@callback(
    Output("sidebar-recent-history", "children"),
    Input("url", "pathname"),
    Input("sidebar-status-poll", "n_intervals"),
)
def update_sidebar_history(pathname, n_intervals):
    """
    Updates the sidebar history links on navigation AND on a periodic
    poll (see layout_sidebar's dcc.Interval) - navigation alone missed any
    status change (in_progress -> completed/failed) or summary update for a
    background job finishing while the user stayed put on one page.
    """
    return _build_sidebar_history_items()


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
    prevent_initial_call=True,
)
def confirm_session_delete(submit_n_clicks, session_id, current_pathname):
    """
    Step 2: fires only when the user clicks the ConfirmDialog's own
    "OK"/confirm button (submit_n_clicks), never on cancel/dismiss - Dash's
    dcc.ConfirmDialog only increments submit_n_clicks on an actual confirm.
    Deletes the session, refreshes the sidebar list, and redirects away if
    the user was looking at the very session just deleted.

    Does NOT target history-list-container. This dialog is opened from
    request_session_delete, whose delete-session-btn exists both in the
    sidebar (mounted on every page) and on /history itself - so this
    callback's Input can fire while the user is on any page. An
    Output('history-list-container', ...) previously sat alongside the
    sidebar Output here on the (wrong) assumption that Dash "harmlessly
    ignores" an Output whose target isn't in the currently rendered tree -
    it doesn't; it throws "ReferenceError: nonexistent object" the instant
    this fires from anywhere other than /history, which is the common case
    (deleting from the sidebar while looking at a feed page). /history's
    own list is always rebuilt fresh from the DB on every navigation to it
    (see the router's display_page), so it self-corrects without needing a
    direct patch here - the only rough edge is that deleting a session
    while already sitting on /history won't remove its card until the next
    navigation to that route, which is a minor, acceptable gap next to a
    hard crash on every other page.
    """
    if not submit_n_clicks or not session_id:
        raise dash.exceptions.PreventUpdate

    deleted = AtelierRepository.delete_session(session_id)
    if not deleted:
        logger.warning(f"Delete requested for session {session_id}, but it was already gone.")

    was_viewing_deleted = current_pathname == f"/history/{session_id}"
    new_pathname = "/history" if was_viewing_deleted else no_update

    return _build_sidebar_history_items(), new_pathname

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
            return [build_loading_skeleton(topic or "Still working on this...", progress_message)], False
        return no_update, False

    chat_history = AtelierRepository.get_session_chat_history(session_id)
    if chat_history:
        flows = [
            html.Div(className="flow-block border-t border-slate-100", children=[
                build_flow_header(chat.get('prompt'), chat.get('model_used'), f"{len(chat.get('citations', []))} References"),
                build_synthesis_body(chat.get('synthesis'), valid_records=chat.get('citations', [])),
                build_paper_cards(chat.get('citations', [])),
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
