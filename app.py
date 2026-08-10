"""
Atelier Main Application Module.

This file serves as the entry point for the Dash Single Page Application (SPA).
It handles:
1. UI Layouts & Tailwind CSS Injection.
2. Dynamic Routing (/home vs /history vs /history/<id>).
3. The AI "Flow" Pipeline (Routing Intent -> Search/Extract -> Synthesize).
4. State management across sessions using local/session storage.
"""

import dash
from dash import Input, Output, clientside_callback, ClientsideFunction

from core.logger import setup_global_logging
setup_global_logging()

from database import AtelierRepository
AtelierRepository.initialize_db()

from ui.layouts import index_string, serve_layout

# Initialize Dash App
app = dash.Dash(__name__, title="Atelier", suppress_callback_exceptions=True)

app.index_string = index_string

# Bind the auto-resize JS to the search text area
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='resizeTextarea'),
    Output('search-input', 'style'),
    Input('search-input', 'value')
)

# Mobile drawer sidebar: hamburger opens it, its own close button or a
# backdrop tap closes it (all three toggle the same drawer state).
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='toggleMobileMenu'),
    Output('mobile-menu-toggle-dummy', 'data'),
    Input('mobile-menu-btn', 'n_clicks'),
    Input('mobile-menu-close-btn', 'n_clicks'),
    Input('mobile-menu-backdrop', 'n_clicks'),
    prevent_initial_call=True,
)

# Force the drawer closed whenever the route changes, so navigating via a
# sidebar link doesn't leave it open over the newly-loaded page.
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='closeMobileMenuOnNav'),
    Output('mobile-menu-nav-dummy', 'data'),
    Input('url', 'pathname'),
    prevent_initial_call=True,
)

# The Settings "Add Provider" dialogs (open/close/close-on-save) are wired
# entirely in ui/callbacks/settings.py, as regular Python callbacks driven
# by a dcc.Store - deliberately NOT clientside, and deliberately not mixed
# with any direct DOM manipulation of the dialog element. See that module's
# docstring for why (a mixed imperative+declarative approach on the same
# element caused both a spurious-open-on-load bug and a doesn't-close-on-save bug).

app.layout = serve_layout()

import ui.callbacks.chat
import ui.callbacks.library
import ui.callbacks.router
import ui.callbacks.search
import ui.callbacks.settings
import ui.callbacks.ui_extras

if __name__ == '__main__':
    # Only the true server process ever reaches this guard - a spawned
    # background-job child process re-imports this whole module (see
    # AtelierRepository.initialize_db()'s docstring) but never satisfies
    # __name__ == '__main__' itself, so this runs exactly once per real
    # server start. See reap_orphaned_sessions()'s docstring for why it
    # must NOT be called from module level / initialize_db() instead.
    AtelierRepository.reap_orphaned_sessions()

    # threaded=True matters here, not just for perf: Flask's dev server
    # otherwise handles one HTTP request at a time, so two browser tabs each
    # running a search serialize through that single worker - one tab's
    # background-callback polling/trigger requests visibly stall while the
    # other has the server's attention, making a concurrent search look
    # stuck rather than progressing.
    app.run(debug=True, port=8050, threaded=True)