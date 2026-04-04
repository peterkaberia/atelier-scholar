import html
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
from database import AtelierRepository

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
    Output("sidebar-recent-history", "children"),
    Input("url", "pathname")
)
def update_sidebar_history(pathname):
    """Updates the sidebar history links whenever the user navigates."""
    
    # Fetch up to 5 most recent sessions
    recent_sessions = AtelierRepository.get_all_sessions_for_history()[:5]
    
    links = []
    for s in recent_sessions:
        session_id = s.get('session_id')
        title = s.get('topic', 'Untitled Search')
        
        links.append(
            html.Li(dcc.Link(
                className="flex items-center space-x-3 px-3 py-2 text-slate-500 hover:bg-slate-50 hover:text-slate-900 rounded-md transition-colors group", 
                href=f"/history/{session_id}", 
                children=[
                    html.Span("chat_bubble", className="material-symbols-outlined text-[16px] group-hover:text-primary transition-colors"), 
                    html.Span(title, className="text-xs font-semibold truncate w-full")
                ]
            ))
        )
        
    return links

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
