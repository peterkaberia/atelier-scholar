import re
from dash import html, dcc
from database.repository import AtelierRepository
from llm.utils import get_model_choices

available_llms = get_model_choices()
default_llm = available_llms[0] if available_llms else "default-model"

def layout_feed(session_id: str):
    """Route: '/history/<id>' - The Follow-up Feed Interface"""

    historical_flows = []
    chat_history = []

    # Check if this session exists in the new DB Repository
    if session_id and AtelierRepository.session_exists(session_id):
        # Note: Ensure you have a method like get_session_chat_history in your repo!
        chat_history = AtelierRepository.get_session_chat_history(session_id) 
        
        for chat in chat_history:
            historical_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
                build_flow_header(chat.get('prompt'), chat.get('model_used'), f"{len(chat.get('citations', []))} References"), 
                build_synthesis_body(chat.get('synthesis'), valid_records=chat.get('citations', [])), 
                build_paper_cards(chat.get('citations', []))
            ]))

    return [

        dcc.Store(id='current-session-id', data=session_id),

        html.Div(id="flow-container", className="flex-1 overflow-y-auto pb-64 no-scrollbar", children=historical_flows),

        html.Div(id="bottom-bar-container", className="fixed bottom-6 left-1/2 md:left-[calc(50%+128px)] -translate-x-1/2 w-full max-w-4xl px-4 md:px-6 z-[100]", children=[
            html.Div(className="glass-chat-bar rounded-3xl p-4 md:p-6 flex flex-col transition-all duration-300", children=[
                html.Div(id="selected-papers-container", className="flex items-center mb-4 px-1 gap-2 md:gap-3 overflow-x-auto no-scrollbar empty:hidden"),
                html.Div(className="flex items-center w-full px-1 mb-3", children=[
                    dcc.Textarea(
                        id="search-input", 
                        className="w-full bg-transparent border-none text-slate-800 placeholder-slate-400 focus:ring-0 text-[16px] md:text-[17px] font-medium py-1 outline-none resize-none min-h-[24px] max-h-[120px] leading-relaxed", 
                        placeholder="Ask follow up or request a deep dive...", rows=1
                    )
                ]),
                
                html.Div(className="flex items-center justify-between px-1", children=[
                    html.Div(className="flex items-center space-x-1", children=[
                        html.Button(html.Span("attach_file", className="material-symbols-outlined text-2xl"), className="p-2 text-slate-400 hover:text-primary transition-colors hover:bg-slate-100/50 rounded-lg")
                    ]),
                    html.Div(className="flex items-center space-x-3", children=[
                        html.Div(className="relative group", children=[
                            dcc.Dropdown(
                                id='llm-dropdown', 
                                options=[{'label': html.Span(m, className="text-slate-600 text-[10px] font-extrabold mr-2 uppercase tracking-widest"), 'value': m} for m in available_llms], 
                                value=default_llm, clearable=False, searchable=False, 
                                className="hidden sm:flex items-center bg-slate-100/80 border border-slate-200/50 rounded-xl ps-4 pe-2 py-2 cursor-pointer hover:bg-white transition-all group"
                            )
                        ]),
                        html.Button(id="search-btn", n_clicks=0, className="w-10 h-10 md:w-11 md:h-11 bg-primary hover:bg-primary/90 hover:scale-105 active:scale-95 rounded-2xl text-white shadow-xl shadow-primary/20 transition-all flex items-center justify-center", children=[
                            html.Span("arrow_upward", id="search-btn-icon", className="material-symbols-outlined text-xl md:text-2xl font-bold")
                        ])
                    ])
                ])
            ])
        ])
    ]

def build_flow_header(query: str, model_name: str, ref_text: str):
    return html.Section(className="py-12 px-4 md:px-12", children=[
        html.Div(className="max-w-3xl mx-auto", children=[
            html.Div(className="flex flex-col space-y-3 max-w-3xl", children=[
                html.H1(query, className="text-3xl font-extrabold text-slate-900 tracking-tight"),
                html.Div(className="flex items-center space-x-6", children=[
                    html.Div(className="flex items-center space-x-2", children=[
                        html.Span("precision_manufacturing", className="material-symbols-outlined text-primary text-lg"), 
                        html.Span(model_name, className="text-[11px] font-extrabold text-slate-500 uppercase tracking-widest")
                    ]),
                    html.Div(className="flex items-center space-x-2", children=[
                        html.Span("description", className="material-symbols-outlined text-accent text-lg"), 
                        html.Span(ref_text, className="text-[11px] font-extrabold text-slate-500 uppercase tracking-widest")
                    ])
                ])
            ])
        ])
    ])

def build_synthesis_body(raw_markdown: str, title: str = "Synthesis Summary", valid_records: list = None):
    if valid_records is None: 
        valid_records = []
        
    def replace_cite(match):
        # match.group(1) catches [5], match.group(2) catches 5<strong
        raw_num = match.group(1) or match.group(2)
        idx = int(raw_num) - 1
        
        if 0 <= idx < len(valid_records):
            rec = valid_records[idx]
            author = rec.get('authors', ['Unknown'])[0] + " et al." if rec.get('authors') else 'Unknown'
            year = rec.get('year', 'N.D.')
            journal = rec.get('journal', 'Unknown')
            title = rec.get('title', 'Untitled Document')
            snippet = rec.get('answer', '') or rec.get('abstract', f"Authored by {author}")

            html_string = f'''
            <span class="group citation relative inline-block cursor-help text-primary font-bold hover:underline mx-1">
                [{idx+1}]
                <span class="citation-tooltip absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 p-5 glass-chat-bar text-slate-900 text-xs rounded-2xl z-50 pointer-events-none font-sans invisible opacity-0 translate-y-2 group-hover:visible group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300 shadow-2xl flex flex-col">
                    <span class="flex justify-between items-start mb-2 w-full">
                        <span class="bg-blue-50 text-blue-700 px-2 py-0.5 rounded text-[10px] font-bold uppercase shrink-0">SOURCE {idx+1}</span>
                        <span class="text-slate-400 italic text-[10px] text-right ml-2 line-clamp-1">{year} • {journal}</span>
                    </span>
                    <strong class="block text-sm leading-snug mb-2 font-bold text-primary line-clamp-2 text-left w-full">{title}</strong>
                    <span class="text-slate-500 line-clamp-2 mb-3 text-left font-normal block w-full">{snippet}</span>
                </span>
            </span>
            '''
            return re.sub(r'\s+', ' ', html_string).strip()
            
        return match.group(0)
        
    synthesis_html = re.sub(r'\[(\d+)\]', replace_cite, raw_markdown)
    
    return html.Section(className="py-0 px-4 md:px-12", children=[
        html.Div(className="max-w-3xl mx-auto", children=[
            html.Article(className="relative", children=[
                html.Div(className="relative z-10 space-y-4", children=[
                    html.Div(className="pb-6", children=[
                        html.H2(title, className="text-4xl font-extrabold text-slate-900 tracking-tight leading-tight")
                    ])
                ]), 
                html.Div(dcc.Markdown(synthesis_html, dangerously_allow_html=True), className="prose max-w-none")
            ])
        ])
    ])

def build_paper_cards(records: list):
    """
    Builds the Evidence Library section containing cards for each parsed paper,
    now including the structured AI metadata.
    """
    articles = []
    for i, rec in enumerate(records, 1):
        num_str = f"{i:02d}" 
        
        articles.append(
            html.Article(className="atelier-result py-8 flex gap-6", children=[
                html.Div(className="flex flex-col items-center", children=[
                    html.Span(num_str, className="text-3xl font-extrabold text-slate-200 mb-2"), 
                    dcc.Checklist(
                        id={'type': 'paper-checkbox', 'index': str(rec.get('pmid', i))}, 
                        options=[{'label': '', 'value': str(rec.get('pmid', i))}], 
                        value=[], 
                        className="w-5 h-5 rounded border-slate-300 text-primary focus:ring-primary"
                    )
                ]),
                html.Div(className="flex-1", children=[
                    html.Div(className="flex justify-between items-start mb-1", children=[
                        html.H3(rec.get('title', ''), className="text-lg font-bold text-primary hover:underline cursor-pointer leading-snug")
                    ]),
                    html.Div(className="text-sm text-slate-500 mb-3 font-medium", children=[
                        html.Span(f"{rec.get('authors', ['Unknown'])[0]} et al.", className="text-slate-900"), 
                        f" • {rec.get('year', 'N.D.')}",
                        f" • {rec.get('citation_count', 0)} Citations • ",
                        html.Em(rec.get('journal', ''), className="text-slate-400 not-italic")
                    ]),
                    
                    # 1-Sentence Takeaway
                    html.P(rec.get('answer', ''), className="text-sm text-slate-600 mb-5 leading-relaxed"),
                    
                    html.Div(className="flex items-center space-x-6", children=[
                        html.A(className="flex items-center space-x-2 text-xs font-bold text-primary hover:text-accent transition-colors", href=rec.get('pdf_url', '#'), target="_blank" if rec.get('pdf_url') else "_self", children=[
                            html.Span("picture_as_pdf", className="material-symbols-outlined text-lg"), 
                            html.Span("DOWNLOAD PDF")
                        ]), 
                        html.Button(className="flex items-center space-x-2 text-xs font-bold text-slate-400 hover:text-slate-900 transition-colors ml-auto", children=[
                            html.Span("bookmark", className="material-symbols-outlined text-lg"), 
                            html.Span("SAVE")
                        ])
                    ])
                ])
            ])
        )
        
    return html.Section(className="py-12 px-4 md:px-12", children=[
        html.Div(className="max-w-3xl mx-auto", children=[
            html.Div(className="space-y-8", children=[
                html.Div(className="flex items-center justify-between border border-slate-200 p-2 rounded-2xl shadow-sm", children=[
                    html.Div(className="flex items-center space-x-4 pl-4", children=[
                        html.H2("Results", className="text-lg font-extrabold text-slate-900 tracking-tight")
                    ]), 
                    html.Div(className="flex items-center gap-2", children=[
                        html.Button(className="flex items-center space-x-2 text-[11px] font-bold px-4 py-2 bg-white border border-slate-200 rounded-xl text-slate-700 shadow-sm", children=[
                            html.Span("download", className="material-symbols-outlined text-lg"), 
                            html.Span("Export CSV")
                        ]),
                        html.Button(className="flex items-center bg-white border border-slate-200 rounded-xl p-1 shadow-sm", children=[
                           html.Button("LIST", className="px-3 py-1.5 rounded-lg bg-primary text-white font-bold text-[10px]"),
                           html.Button("TABLE", className="px-3 py-1.5 rounded-lg text-slate-400 font-bold text-[10px]")
                        ])
                    ])
                ]), 
                html.Div(className="space-y-0 divide-y divide-slate-200/40", children=articles), 
                html.Div(className="pt-0 pb-4 flex justify-center", children=[
                    html.Button("Load More Results", id="load-more-btn", n_clicks=0, className="px-8 py-3 bg-white border border-slate-200 rounded-full text-sm font-bold text-primary hover:border-primary hover:shadow-md transition-all")
                 ])
                
            ])
        ])
    ])