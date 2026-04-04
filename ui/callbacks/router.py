from dash import Input, Output, State, callback
from database import AtelierRepository
from llm import get_model_choices
from ui.layouts import layout_home, layout_history, layout_feed, layout_404, layout_no_llm

available_llms = get_model_choices()

@callback(
    Output('page-content', 'children'), 
    Input('url', 'pathname'), 
    State('store-user-history', 'data'),
    State('store-pending-search', 'data')
)
def display_page(pathname, user_history, pending_search):
    """
    Master Router: Swaps out the main layout based on the current URL.
    """
    if not available_llms: 
        return layout_no_llm()
    
    if pathname is None or pathname == '/': 
        return layout_home()
    
    elif pathname == '/history': 
        return layout_history()
    
    elif pathname and pathname.startswith('/history/'):
        session_id = pathname.split('/')[-1] 

        if session_id and session_id.strip():
            is_valid = False
            if (pending_search and pending_search.get("session_id") == session_id) or AtelierRepository.session_exists(session_id):
                is_valid = True
            if is_valid:
                return layout_feed(session_id)
            else:
                return layout_404("Session Not Found.")
        else:
            return layout_404("Invalid session ID.")
    else: 
        return layout_404("Page Not Found.")