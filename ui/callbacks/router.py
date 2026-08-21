from dash import Input, Output, State, callback
from database import AtelierRepository
from llm import get_model_choices
from ui.layouts import layout_home, layout_feed, layout_404, layout_no_llm, layout_settings

@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    State('store-pending-search', 'data')
)
def display_page(pathname, pending_search):
    """
    Master Router: Swaps out the main layout based on the current URL.
    """
    # Settings must be reachable even with zero keys configured - otherwise
    # there's no way to ever escape the "no LLM keys" screen through the UI.
    if pathname == '/settings':
        return layout_settings()

    # Resolved per navigation (not at import time) so a key saved via
    # Settings unlocks the app on the very next click, without a restart.
    if not get_model_choices():
        return layout_no_llm()

    if pathname is None or pathname == '/':
        return layout_home()

    # No standalone '/history' archive page anymore - the sidebar's own
    # date-grouped, searchable session list (ui/layouts/sidebar.py) already
    # covers that job, always visible rather than a separate page to visit.
    elif pathname and pathname.startswith('/session/'):
        session_id = pathname.split('/')[-1] 

        if session_id and session_id.strip():
            is_valid = False
            if (pending_search and pending_search.get("session_id") == session_id) or AtelierRepository.session_exists(session_id):
                is_valid = True
            if is_valid:
                return layout_feed(session_id, pending_search=pending_search)
            else:
                return layout_404("Session Not Found.")
        else:
            return layout_404("Invalid session ID.")
    else: 
        return layout_404("Page Not Found.")