from datetime import date, timedelta
from dash import html, dcc
from database import AtelierRepository

def layout_history():
    """Route: '/history' - The Archive List backed by SQLite and new Timeline UI"""
    
    # Fetch all sessions from the database
    sessions = AtelierRepository.get_all_sessions_for_history()
    
    # Group sessions by relative date
    today = date.today()
    yesterday = today - timedelta(days=1)
    grouped_sessions = {}
    
    for s in sessions:
        created_str = s.get('created_at', '')
        s_date = created_str.split(' ')[0] if created_str else "Unknown Date"
        if s_date == today:
            group_name = "Today"
        elif s_date == yesterday:
            group_name = "Yesterday"
        else:
            # Format older dates like "October 24, 2023"
            group_name = created_str.strftime("%B %d, %Y")
            
        if group_name not in grouped_sessions:
            grouped_sessions[group_name] = []
        grouped_sessions[group_name].append(s)

    # Build the dynamic UI elements
    history_groups_ui = []

    
    for group_name, group_sessions in grouped_sessions.items():
        cards_ui = []
        
        for s in group_sessions:
            # Dynamically fetch the contextual stats using our SQLite schema
            query_count = s.queries.count()
            source_count = s.paper_links.count()
            time_str = created_str.strftime("%I:%M %p")
            
            card = dcc.Link(
                href=f"/history/{s.id}", 
                className="group block cursor-pointer mb-2",
                children=[
                    html.Div(className="flex flex-col md:flex-row gap-6 p-4 rounded-xl hover:bg-surface-container-low transition-all duration-300 border border-transparent hover:border-outline-variant/30 hover:shadow-sm", children=[
                        html.Div(className="flex-1", children=[
                            html.Div(className="flex items-center justify-between mb-2", children=[
                                html.H3(s.topic, className="text-xl font-headline font-bold text-on-surface group-hover:text-primary transition-colors"),
                                html.Span(f"{group_name}, {time_str}", className="text-xs font-headline font-semibold text-outline-variant bg-surface-container px-2 py-1 rounded")
                            ]),
                            html.P(f'"{s.summary}"', className="font-body text-lg text-on-surface-variant leading-relaxed mb-4 line-clamp-2"),
                            
                            html.Div(className="flex items-center gap-4", children=[
                                html.Div(className="flex items-center gap-1.5 text-xs font-headline font-semibold text-primary", children=[
                                    html.Span("query_stats", className="material-symbols-outlined text-sm"),
                                    f"{query_count} {'query' if query_count == 1 else 'queries'}"
                                ]),
                                html.Div(className="flex items-center gap-1.5 text-xs font-headline font-semibold text-on-secondary-container bg-secondary-container px-2 py-0.5 rounded-full", children=[
                                    html.Span("article", className="material-symbols-outlined text-sm"),
                                    f"{source_count} {'Source' if source_count == 1 else 'Sources'}"
                                ])
                            ])
                        ])
                    ])
                ]
            )
            cards_ui.append(card)
            
        # Build the Group Wrapper (The timeline separator)
        group_block = html.Div(className="relative pt-6 first:pt-0", children=[
            html.Div(className="flex items-center gap-4 mb-6", children=[
                html.Span(group_name, className="text-[10px] uppercase tracking-[0.2em] font-headline font-extrabold text-outline"),
                html.Div(className="h-px flex-1 bg-outline-variant/20")
            ]),
            html.Div(className="space-y-4", children=cards_ui)
        ])
        
        history_groups_ui.append(group_block)

    # Return the full layout container
    return html.Div(className="w-full max-w-4xl mx-auto px-6 py-12", children=[
        
        # Top Header Section
        html.Section(className="mb-16", children=[
            html.H1("History", className="text-4xl font-bold font-headline tracking-tight text-on-surface mb-6"),
            html.Div(className="max-w-2xl relative group", children=[
                html.Span("filter_list", className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline group-focus-within:text-primary transition-colors"),
                dcc.Input(
                    type="text",
                    placeholder="Filter through your past insights...",
                    className="w-full bg-surface-container-high border-none py-4 pl-12 pr-6 rounded focus:bg-surface-container-highest focus:ring-1 focus:ring-primary/30 transition-all font-body text-lg italic outline-none ring-0"
                )
            ])
        ]),
        
        # Chronological List
        html.Div(className="space-y-11", children=history_groups_ui) if history_groups_ui else html.Div(),
        
        # Empty State Footer
        html.Div(className="mt-20 text-center max-w-4xl py-12 bg-surface-container-low/50 rounded-2xl border border-dashed border-outline-variant/30", children=[
            html.Span("auto_awesome", className="material-symbols-outlined text-outline-variant text-4xl mb-3"),
            html.P("The archive is complete.", className="font-headline font-semibold text-on-surface-variant"),
            html.P('"An investment in knowledge always pays the best interest."', className="font-body italic text-outline")
        ])
    ])