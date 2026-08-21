"""
Atelier Main Application Module.

This file serves as the entry point for the Dash Single Page Application (SPA).
It handles:
1. UI Layouts & Tailwind CSS Injection.
2. Dynamic Routing (/home vs /history vs /session/<id>).
3. The AI "Flow" Pipeline (Routing Intent -> Search/Extract -> Synthesize).
4. State management across sessions using local/session storage.
"""

import os

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

# Bind the auto-resize JS to both the feed's follow-up textarea and the
# home page's hero textarea - same clientside function (ui/layouts/main.py's
# resizeTextarea reads WHICH element triggered it), two separate bindings
# since a Dash Output can only be one specific component. Previously only
# ever bound to search-input, so the home page's textarea never
# auto-resized at all.
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='resizeTextarea'),
    Output('search-input', 'style'),
    Input('search-input', 'value')
)
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='resizeTextarea'),
    Output('hero-search-input', 'style'),
    Input('hero-search-input', 'value')
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

# App bar back button: shown on every page except Home (Settings included -
# it used to only cover session detail views, which left Settings a dead
# end), and does a real browser-history back on click - see
# ui/layouts/main.py's toggleAppBarBackButton/goBack for why each is its
# own separate clientside callback (one keyed on the URL, one on the
# click - genuinely different triggers, not the same event).
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='toggleAppBarBackButton'),
    Output('app-bar-back-btn', 'style'),
    Input('url', 'pathname'),
)
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='toggleAppBarSettingsIcon'),
    Output('app-bar-settings-link', 'style'),
    Input('url', 'pathname'),
)
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='toggleSidebarSearchBox'),
    Output('sidebar-search-box', 'style'),
    Input('url', 'pathname'),
)
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='goBack'),
    Output('app-bar-back-dummy', 'data'),
    Input('app-bar-back-btn', 'n_clicks'),
    prevent_initial_call=True,
)

# Mobile's own back button - the desktop app bar is `hidden md:flex`, so
# small screens had no back option at all until now. Same rules as above,
# just against the mobile topbar's own button id (ui/layouts/sidebar.py's
# layout_mobile_topbar).
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='toggleMobileBackButton'),
    Output('mobile-back-btn', 'style'),
    Input('url', 'pathname'),
)
clientside_callback(
    ClientsideFunction(namespace='ui', function_name='mobileGoBack'),
    Output('mobile-back-dummy', 'data'),
    Input('mobile-back-btn', 'n_clicks'),
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
import ui.callbacks.uploads

if __name__ == '__main__':
    DEBUG_MODE = True

    # Only the true server process ever reaches this guard - a spawned
    # background-job child process re-imports this whole module (see
    # AtelierRepository.initialize_db()'s docstring) but never satisfies
    # __name__ == '__main__' itself, so this runs exactly once per real
    # server start. See reap_orphaned_sessions()'s docstring for why it
    # must NOT be called from module level / initialize_db() instead.
    AtelierRepository.reap_orphaned_sessions()

    # Preload the SPLADE model once here (in a background thread - doesn't
    # delay this process's own startup) and keep it warm for background
    # jobs to reuse over a loopback socket instead of each paying its own
    # ~20s cold load - see search/sparse_server.py's docstring for the full
    # reasoning. Guarded to run in exactly ONE process: with DEBUG_MODE on,
    # Werkzeug's dev-reloader re-executes this whole module twice - once in
    # a watcher parent (which never serves real requests, and wouldn't set
    # WERKZEUG_RUN_MAIN) and once in the actual serving child (which does) -
    # starting the warm server in both would double-pay the cold load AND
    # crash the second attempt trying to bind a port the first already
    # holds. DEBUG_MODE off has no separate watcher process, so this
    # process is unconditionally the one to start it.
    if not DEBUG_MODE or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        from core.config import DEFAULT_SPARSE_MODEL
        from search.sparse_server import start_warm_encoder_server
        start_warm_encoder_server(DEFAULT_SPARSE_MODEL)

    # threaded=True matters here, not just for perf: Flask's dev server
    # otherwise handles one HTTP request at a time, so two browser tabs each
    # running a search serialize through that single worker - one tab's
    # background-callback polling/trigger requests visibly stall while the
    # other has the server's attention, making a concurrent search look
    # stuck rather than progressing.
    app.run(debug=DEBUG_MODE, port=8050, threaded=True)