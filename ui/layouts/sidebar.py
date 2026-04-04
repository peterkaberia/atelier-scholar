from dash import html, dcc

def layout_sidebar():
    return html.Aside(className="w-64 flex-shrink-0 border-r border-border-light bg-surface-light flex flex-col h-full hidden md:flex relative z-50", children=[
            html.Div(className="p-6 flex flex-col h-full", children=[
                html.Div(className="flex items-center space-x-2 text-primary font-extrabold text-xl mb-8 tracking-tight", children=[
                    html.Span("school", className="material-symbols-outlined text-3xl"), html.Span("Atelier")
                ]),
                dcc.Link(href="/", children=[
                    html.Button(id="new-synthesis-btn-2", className="w-full flex items-center justify-center space-x-2 bg-primary text-white py-2.5 rounded-lg font-semibold hover:opacity-90 transition-all mb-8 shadow-sm", children=[
                        html.Span("add", className="material-symbols-outlined text-xl"), html.Span("New Search")
                    ])
                ]),
                html.Nav(className="space-y-6 flex-1", children=[
                    html.Div([
                        html.H3("Search Tools", className="text-[11px] font-bold text-slate-400 uppercase tracking-[0.1em] mb-3 px-3"), 
                        html.Ul(className="space-y-1", children=[
                            html.Li(dcc.Link(className="flex items-center space-x-3 px-3 py-2 bg-slate-50 text-primary rounded-md font-semibold", href="/", children=[
                                html.Span("search", className="material-symbols-outlined text-[20px]"), html.Span("Home")
                            ])),
                            html.Li(className="flex flex-col", children=[
                                dcc.Link(className="flex items-center space-x-3 px-3 py-2 text-slate-500 hover:bg-slate-50 hover:text-slate-900 rounded-md transition-colors", href="/history", children=[
                                    html.Span("history", className="material-symbols-outlined text-[20px]"), html.Span("History")
                                ]),
                                html.Ul(id="sidebar-recent-history", className="mt-1 ml-7 pl-2 border-l-2 border-slate-100 space-y-0.5 overflow-y-auto max-h-[30vh] no-scrollbar") 
                                
                            ])

                        ])
                    ]),
                ])
            ])
        ])