from dash import html, dcc
from llm import get_model_choices

def layout_home():
    """Route: '/' - The Initial Search Page (Hero Interface)"""
    # Resolved per render (not at import time) so a key saved via Settings
    # shows up in the dropdown without an app restart.
    available_llms = get_model_choices()
    default_llm = available_llms[0] if available_llms else "default-model"

    return html.Div(className="flex-1 flex flex-col items-center justify-center w-full h-full pb-32 px-8 bg-[#FDFDFD] dot-grid", children=[
        html.Section(className="text-center mb-16 space-y-4", children=[
            html.H1("Welcome to Atelier.", className="text-4xl md:text-6xl font-extrabold tracking-tighter text-slate-900 font-sans animate-fade-in"),
            html.P("What shall we research today?", className="text-xl md:text-2xl font-serif italic text-slate-500 leading-relaxed max-w-2xl mx-auto animate-fade-in", style={"animationDelay": "0.1s"}),
        ]),
        # Same composer shape as the feed's bottom bar (question row on top,
        # controls row below) for visual/responsive consistency between the
        # first-question and follow-up entry points - centered here instead
        # of docked to the bottom, since this is the landing page.
        html.Div(className="relative z-10 w-full max-w-3xl mx-auto animate-fade-in px-4 md:px-0", style={"animationDelay": "0.2s"}, children=[
            html.Div(className="glass-chat-bar rounded-3xl p-4 md:p-6 flex flex-col transition-all duration-300", children=[
                # Same id as ui/layouts/feed.py's identical chip row -
                # home/feed are mutually exclusive routes, so reusing it
                # lets ui/callbacks/uploads.py's stage_uploads/
                # remove_pending_upload callbacks work unmodified on
                # whichever page is actually mounted.
                html.Div(id="pending-uploads-container", className="flex items-center mb-4 px-1 gap-2 md:gap-3 overflow-x-auto no-scrollbar empty:hidden"),
                html.Div(className="flex items-center w-full px-1 mb-3", children=[
                    dcc.Textarea(
                        id="hero-search-input",
                        className="w-full bg-transparent border-none text-slate-800 placeholder-slate-400 focus:ring-0 text-[16px] md:text-[18px] font-medium py-1 outline-none resize-none min-h-[28px] max-h-[45vh] leading-relaxed overflow-y-auto",
                        placeholder="Ask a question or synthesize research...", rows=1
                    )
                ]),
                html.Div(className="flex items-center justify-between px-1 gap-2", children=[
                    # dcc.Upload, not a plain button - see
                    # ui/layouts/feed.py's identical control for why (a
                    # plain html.Button can't open a file picker or
                    # receive file data). Same id as feed's - see this
                    # Div's own comment above.
                    dcc.Upload(
                        id='file-upload',
                        multiple=True,
                        accept=".pdf,.txt,.md,.markdown",
                        className="inline-flex",
                        children=html.Div(
                            html.Span("attach_file", className="material-symbols-outlined text-2xl"),
                            className="p-2 text-slate-400 hover:text-primary transition-colors hover:bg-slate-100/50 rounded-lg cursor-pointer flex-shrink-0",
                        ),
                    ),
                    html.Div(className="flex items-center gap-2 md:gap-3", children=[
                        html.Div(className="relative group", children=[
                            dcc.Dropdown(
                                id='hero-llm-dropdown',
                                options=[{'label': html.Span(m, className="text-slate-600 text-[10px] font-extrabold uppercase tracking-widest"), 'value': m} for m in available_llms],
                                value=default_llm,
                                clearable=False, searchable=False,
                                className="hidden sm:flex items-center bg-slate-100/80 border border-slate-200/50 rounded-xl ps-4 pe-2 py-2 cursor-pointer hover:bg-white transition-all group min-w-[160px]"
                            )
                        ]),
                        html.Button(id="hero-search-btn", n_clicks=0, className="scholar-gradient text-white h-10 md:h-11 px-5 md:px-6 rounded-2xl flex items-center justify-center gap-2 hover:opacity-90 active:scale-95 transition-all shadow-xl shadow-primary/20 flex-shrink-0", children=[
                            html.Span("Synthesize", className="font-bold text-xs md:text-sm tracking-widest uppercase"),
                            html.Span("arrow_forward", className="material-symbols-outlined text-lg")
                        ])
                    ])
                ])
            ])
        ])
    ])