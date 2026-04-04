from dash import html, dcc
from llm import get_model_choices

available_llms = get_model_choices()
default_llm = available_llms[0]

def layout_home():
    """Route: '/' - The Initial Search Page (Hero Interface)"""
    return html.Div(className="flex-1 flex flex-col items-center justify-center w-full h-full pb-32 px-8 bg-[#FDFDFD] dot-grid", children=[
        html.Section(className="text-center mb-16 space-y-4", children=[
            html.H1("Welcome to Atelier.", className="text-4xl md:text-6xl font-extrabold tracking-tighter text-slate-900 font-sans animate-fade-in"),
            html.P("What shall we research today?", className="text-xl md:text-2xl font-serif italic text-slate-500 leading-relaxed max-w-2xl mx-auto animate-fade-in", style={"animationDelay": "0.1s"}),
        ]),
        html.Div(className="relative z-10 w-full max-w-3xl mx-auto animate-fade-in", style={"animationDelay": "0.2s"}, children=[
            html.Div(className="glass-panel p-2 rounded-2xl shadow-2xl shadow-primary/10 border border-white/60 ring-1 ring-black/5 transition-all duration-500 focus-within:ring-primary/40", children=[
                html.Div(className="flex flex-col md:flex-row gap-2", children=[
                    
                    html.Div(className="flex items-center px-4 py-3 bg-slate-50/80 rounded-xl border border-transparent hover:border-slate-200 transition-all cursor-pointer group min-w-[180px]", children=[
                        dcc.Dropdown(
                            id='hero-llm-dropdown', 
                            options=[{'label': html.Span(m, className="text-slate-600 text-[10px] font-extrabold uppercase tracking-widest"), 'value': m} for m in available_llms], 
                            value=default_llm, 
                            clearable=False, searchable=False, className="w-full"
                        )
                    ]),
                    html.Div(className="flex-grow flex items-center px-4", children=[
                        dcc.Input(
                            id="hero-search-input", 
                            className="w-full bg-transparent border-none focus:ring-0 text-slate-800 placeholder:text-slate-400 font-sans py-4 text-xl outline-none", 
                            placeholder="Ask a question or synthesize research...", type="text"
                        )
                    ]),
                    html.Button(id="hero-search-btn", n_clicks=0, className="scholar-gradient text-white px-8 py-4 rounded-xl flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 transition-all shadow-lg shadow-primary/20", children=[
                        html.Span("Synthesize", className="font-bold text-sm tracking-widest uppercase"),
                        html.Span("arrow_forward", className="material-symbols-outlined")
                    ])
                    
                ])
            ])
        ])
    ])