import html
import dash
import logging
from dash import html, Input, Output, State, callback, no_update
from database.repository import AtelierRepository
from llm import AtelierAIEngine
from ui.layouts import build_flow_header, build_synthesis_body
from .background import background_manager

logger = logging.getLogger(__name__)

@callback(
    Output('trigger-router', 'data', allow_duplicate=True), 
    Output('flow-container', 'children', allow_duplicate=True),
    Output('search-input', 'value', allow_duplicate=True),
    Output('store-pending-search', 'data', allow_duplicate=True), # Clears clipboard
    Input('current-session-id', 'data'), # Fires when the feed page mounts
    Input('search-btn', 'n_clicks'),     # Fires from the bottom bar
    State('store-pending-search', 'data'),
    State('search-input', 'value'),
    State('llm-dropdown', 'value'),
    State('flow-container', 'children'),
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def handle_feed_interactions(session_id, bottom_clicks, pending_search, bottom_text, bottom_llm, existing_flows):
    """
    Acts as the entry point for AI generation within the Feed View. 
    Handles two scenarios: 
    A) The page just loaded and there is a query waiting in the clipboard.
    B) The user typed a follow-up directly into the bottom chat bar.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not existing_flows: 
        existing_flows = []

    # SCENARIO A: The Feed Page just loaded from a Home Page handoff
    if triggered_id == 'current-session-id':
        if pending_search and pending_search.get("session_id") == session_id:
            query = pending_search["query"]
            llm = pending_search["llm"]
            
            # Inject Skeleton Loader
            existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
                build_flow_header(query, llm, "Scanning Intent..."),
                html.Section(className="py-12 px-4 md:px-12 bg-[#F8FAFF]", children=[
                    html.Div(className="max-w-5xl mx-auto text-center animate-pulse", children=[
                        html.Span("autorenew", className="material-symbols-outlined text-primary text-4xl animate-spin mb-4"),
                        html.P("Analyzing query intent...", className="text-sm font-bold text-slate-500 uppercase tracking-widest")
                    ])
                ])
            ]))
            # Trigger Router, Update flow, Clear bottom input, Clear clipboard
            return {"query": query, "llm": llm, "session_id": session_id}, existing_flows, "", None
        return [no_update] * 4

    # SCENARIO B: User typed a follow-up query in the bottom bar
    elif triggered_id == 'search-btn':
        if not bottom_clicks or not bottom_text or not bottom_text.strip(): 
            return [no_update] * 4
        
        # Inject Skeleton Loader
        existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
            build_flow_header(bottom_text, bottom_llm, "Scanning Intent..."),
            html.Section(className="py-12 px-4 md:px-12 bg-[#F8FAFF]", children=[
                html.Div(className="max-w-5xl mx-auto text-center animate-pulse", children=[
                    html.Span("autorenew", className="material-symbols-outlined text-primary text-4xl animate-spin mb-4"),
                    html.P("Analyzing query intent...", className="text-sm font-bold text-slate-500 uppercase tracking-widest")
                ])
            ])
        ]))
        # Trigger Router, Update flow, Clear bottom input, Do nothing to clipboard
        return {"query": bottom_text, "llm": bottom_llm, "session_id": session_id}, existing_flows, "", no_update
        
    return [no_update] * 4

@callback(
    Output('trigger-search', 'data', allow_duplicate=True), 
    Output('trigger-chat', 'data', allow_duplicate=True),
    Output('flow-container', 'children', allow_duplicate=True),
    Input('trigger-router', 'data'),
    State('store-current-topic', 'data'), 
    State('store-processed-records', 'data'), 
    State('flow-container', 'children'), 
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def route_intent(router_data, current_topic, processed_records, existing_flows):
    """
    Context-Aware Intent Router.
    Determines if the user's query requires fetching new documents (SEARCH) 
    or just chatting with the existing context (CHAT).
    """
    query = router_data.get('query')
    selected_llm = router_data.get('llm')
    session_id = router_data.get('session_id')

    if not current_topic or not processed_records:
        route = "SEARCH"
        search_query = query
    else:
        # LLM analyzes intent based on the query and current context state
        engine = AtelierAIEngine(model_choice=selected_llm)
        parsed_data = engine.followup_chat(query=query, current_topic=current_topic, processed_records=processed_records)
        route = parsed_data.get("intent", "CHAT").upper()
        search_query = parsed_data.get("standalone_query", query)
        
        if not search_query or search_query.lower() == "null": 
            search_query = f"{current_topic} {query}"
    
    existing_flows.pop() # Remove routing skeleton
    
    if "SEARCH" in route:
        existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
            build_flow_header(search_query, selected_llm, "Fetching Database..."), 
            html.Section(className="py-12 px-4 md:px-12 bg-[#F8FAFF]", children=[
                html.Div(className="max-w-5xl mx-auto text-center animate-pulse", children=[
                    html.Span("travel_explore", className="material-symbols-outlined text-primary text-4xl animate-pulse mb-4"), 
                    html.P("Extracting literature...", className="text-sm font-bold text-slate-500 uppercase tracking-widest")
                ])
            ])
        ]))
        return {"query": search_query, "llm": selected_llm, "session_id": session_id}, no_update, existing_flows
        
    else:
        existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
            build_flow_header(query, selected_llm, "Analyzing Context..."), 
            html.Section(className="py-12 px-4 md:px-12 bg-[#F8FAFF]", children=[
                html.Div(className="max-w-5xl mx-auto text-center animate-pulse", children=[
                    html.Span("psychology", className="material-symbols-outlined text-primary text-4xl animate-pulse mb-4"), 
                    html.P("Deep diving into context...", className="text-sm font-bold text-slate-500 uppercase tracking-widest")
                ])
            ])
        ]))
        return no_update, {"query": query, "llm": selected_llm, "session_id": session_id}, existing_flows

@callback(
    Output('flow-container', 'children', allow_duplicate=True), 
    Output('store-chat-history', 'data', allow_duplicate=True),
    Input('trigger-chat', 'data'), 
    State('store-processed-records', 'data'), 
    State('store-chat-history', 'data'), 
    State('flow-container', 'children'), 
    running=[
        (Output("search-btn-icon", "children"), "stop", "arrow_upward"),
        (Output("search-btn", "className"), "w-10 h-10 md:w-11 md:h-11 bg-slate-400 rounded-2xl text-white flex items-center justify-center cursor-not-allowed", "w-10 h-10 md:w-11 md:h-11 bg-primary hover:bg-primary/90 hover:scale-105 active:scale-95 rounded-2xl text-white shadow-xl shadow-primary/20 transition-all flex items-center justify-center"),
    ],
    background=True,
    manager=background_manager(),
    cancel=[Input("search-btn", "n_clicks")],
    prevent_initial_call=True
)
def run_chat(chat_data, processed_records, chat_history, existing_flows):
    """
    Context-aware RAG implementation. Generates responses to follow-up questions
    using the currently loaded literature abstracts.
    """
    if not chat_data or not chat_data.get('query'):
        return existing_flows, chat_history

    query = chat_data.get('query')
    selected_llm = chat_data.get('llm')
    session_id = chat_data.get('session_id')

    # Ensure memory structures exist
    if chat_history is None: chat_history = []
    if existing_flows is None: existing_flows = []
    if processed_records is None: processed_records = []

    # 2. Append user message to memory
    chat_history.append({"role": "user", "content": query})

    try:
        # 3. Boot up the AI Engine and generate the response
        logger.info(f"Generating follow-up chat for session '{session_id}'...")
        engine = AtelierAIEngine(model_choice=selected_llm)
        
        ai_response = engine.chat_with_literature(
            chat_history=chat_history, 
            context_records=processed_records
        )

        # 4. Save the interaction and citations to the SQLite database
        AtelierRepository.save_query_with_citations(
            session_id=session_id,
            prompt=query,
            synthesis=ai_response,
            model_used=selected_llm,
            cited_records=processed_records 
        )

        # 5. Append AI response to memory
        chat_history.append({"role": "assistant", "content": ai_response})

        # 6. Build the new visual block for the Feed UI
        new_flow = html.Div(className="flow-block border-t border-slate-100", children=[
            build_flow_header(query, selected_llm, f"{len(processed_records)} Context Papers"), 
            build_synthesis_body(ai_response, valid_records=processed_records)
            # Note: We omit build_paper_cards() here so we don't duplicate the 
            # library table every single time the user asks a follow-up question.
        ])

        existing_flows.append(new_flow)
        logger.info("Chat follow-up generated and saved successfully.")

        return existing_flows, chat_history

    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        
        # Graceful UI degradation if the API fails or the user cancels
        error_flow = html.Div(className="py-12 px-4 md:px-12 bg-red-50/30 border-t border-red-100", children=[
            html.Div(className="max-w-5xl mx-auto flex items-center space-x-3 text-red-600", children=[
                html.Span("error", className="material-symbols-outlined text-2xl"),
                html.Span(f"Request interrupted or failed: {str(e)}", className="font-bold")
            ])
        ])
        existing_flows.append(error_flow)
        
        return existing_flows, chat_history