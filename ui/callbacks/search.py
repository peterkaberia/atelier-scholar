import uuid
from dash import html, Input, Output, State, callback, no_update

from database import AtelierRepository, Record
from llm import AtelierAIEngine
from pipeline import ResearchOrchestrator
from ui.layouts import build_flow_header, build_paper_cards, build_synthesis_body
from .background import background_manager

@callback(
    Output('url', 'pathname'), 
    Output('store-pending-search', 'data'), 
    Output('store-current-topic', 'data', allow_duplicate=True),
    Output('store-processed-records', 'data', allow_duplicate=True),
    Output('store-chat-history', 'data', allow_duplicate=True),
    Input('hero-search-btn', 'n_clicks'),
    State('hero-search-input', 'value'), 
    State('hero-llm-dropdown', 'value'),
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def create_new_session(hero_clicks, hero_text, hero_llm):
    """
    Fires when a user initiates a search from the Home Page. 
    It clears legacy session data, writes the new query to the pending clipboard, 
    and redirects the URL to generate a new Feed view.
    """
    if not hero_clicks or not hero_text or not hero_text.strip(): 
        return [no_update] * 5
        
    new_session_id = str(uuid.uuid4())[:8] 
    trigger_data = {"query": hero_text, "llm": hero_llm, "session_id": new_session_id}
    
    return f"/history/{new_session_id}", trigger_data, "", [], []

@callback(
    Output('store-all-records', 'data', allow_duplicate=True), 
    Output('store-processed-records', 'data', allow_duplicate=True), 
    Output('store-current-topic', 'data', allow_duplicate=True),
    Output('trigger-synthesis', 'data', allow_duplicate=True), 
    Output('flow-container', 'children', allow_duplicate=True), 
    Output('store-user-history', 'data', allow_duplicate=True),
    Input('trigger-search', 'data'), 
    State('store-all-records', 'data'), 
    State('store-processed-records', 'data'), 
    State('store-user-history', 'data'), 
    State('flow-container', 'children'), 
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def run_search(search_data, all_records, processed_records, user_history, existing_flows):
    """
    Executes external API calls (PubMed, Europe PMC), merges data, calculates 
    SPLADE scores, and performs LLM extraction on the top K papers.
    """
    query = search_data.get('query')
    selected_llm = search_data.get('llm')
    session_id = search_data.get("session_id")

    orchestrator = ResearchOrchestrator(
        session_id=session_id,
        topic=query,
        model_choice=selected_llm
    )

    processed_records = orchestrator.execute_full_search_pipeline()
        
    existing_flows.pop() # Remove search skeleton
    existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
        build_flow_header(query, selected_llm, f"{len(processed_records)} Extracted"), 
        html.Section(className="py-12 px-4 md:px-12 bg-[#F8FAFF]", children=[
            html.Div(className="max-w-5xl mx-auto text-center animate-pulse", children=[
                html.Span("auto_awesome", className="material-symbols-outlined text-primary text-4xl animate-spin mb-4"), 
                html.P("Synthesizing AI Consensus...", className="text-sm font-bold text-slate-500 uppercase tracking-widest")
            ])
        ])
    ]))
    
    # Pass search_data forward so generate_synth has the session_id
    return [], processed_records, query, search_data, existing_flows, user_history

@callback(
    Output('flow-container', 'children', allow_duplicate=True), 
    Output('store-synthesis-markdown', 'data', allow_duplicate=True),
    Input('trigger-synthesis', 'data'), 
    State('store-processed-records', 'data'), 
    State('flow-container', 'children'), 
    background=True,
    manager=background_manager(),
    prevent_initial_call=True
)
def generate_synth(synth_data, processed_records, existing_flows):
    """
    Generates the final structured synthesis markdown document based on the 
    extracted paper logic. Completes the SEARCH flow.
    """
    query = synth_data.get('query')
    selected_llm = synth_data.get('llm')
    session_id = synth_data.get('session_id')

    engine = AtelierAIEngine(model_choice=selected_llm)
    raw_md = engine.generate_copilot_synthesis(topic=query, valid_records=processed_records)
    
    # 1. Save the Query and its Citations to the Database
    AtelierRepository.save_query_with_citations(
        session_id=session_id,
        prompt=query,
        model_used=selected_llm,
        synthesis=raw_md,
        cited_records=processed_records
    )
    
    # Small inline check to avoid querying the DB if we don't have to
    if " SEARCH " not in existing_flows[0]['props'].get('children', str(existing_flows)):
        display_summary = raw_md.split('\n')[0].replace('#', '').strip()[:300]  # Simple heuristic for summary
        AtelierRepository.update_session_summary(session_id, display_summary)
        
    existing_flows.pop() 
    existing_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
        build_flow_header(query, selected_llm, f"{len(processed_records)} References"), 
        build_synthesis_body(raw_md, valid_records=processed_records), 
        build_paper_cards(processed_records)
    ]))
    return existing_flows, raw_md
