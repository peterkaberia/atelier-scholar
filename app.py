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

app.layout = serve_layout()

import ui.callbacks.chat
import ui.callbacks.router
import ui.callbacks.search
import ui.callbacks.ui_extras

if __name__ == '__main__':
    app.run(debug=True, port=8050)