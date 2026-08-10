from datetime import date, timedelta
from dash import html, dcc
from database import AtelierRepository
from database.models import SessionModel


def build_status_dot(status: str, *, with_label: bool = False):
    """
    A small colored status indicator - orange while a session's search is
    still running, green once it's completed, red if it failed. Shared
    between the History archive page and the sidebar's recent-history list.
    """
    tone_map = {
        SessionModel.STATUS_IN_PROGRESS: ("bg-amber-400", "text-amber-700", "Active"),
        SessionModel.STATUS_COMPLETED: ("bg-emerald-500", "text-emerald-700", "Completed"),
        SessionModel.STATUS_FAILED: ("bg-red-500", "text-red-700", "Failed"),
    }
    dot_class, text_class, label = tone_map.get(status, tone_map[SessionModel.STATUS_COMPLETED])

    dot = html.Span(
        className=f"inline-block w-2 h-2 rounded-full {dot_class} flex-shrink-0"
        + (" animate-pulse" if status == SessionModel.STATUS_IN_PROGRESS else ""),
        title=label,
    )
    if not with_label:
        return dot

    return html.Span(className="inline-flex items-center gap-1.5", children=[
        dot,
        html.Span(label, className=f"text-[10px] font-bold uppercase tracking-widest {text_class}"),
    ])


def build_history_list():
    """
    Builds the grouped, dated list of session cards, rendered into
    'history-list-container' below. Nothing outside this page's own render
    writes to that id directly anymore - see
    ui/callbacks/ui_extras.py's confirm_session_delete docstring for why: a
    callback whose Output targets an id that isn't in the CURRENTLY
    RENDERED DOM throws a runtime "nonexistent object" error the moment it
    fires, regardless of app.py's suppress_callback_exceptions=True (that
    flag only relaxes Dash's startup-time callback-graph validation, not
    the browser-side check that a response's target element actually
    exists right now). /history rebuilds this list fresh from the DB on
    every navigation to it, so it self-corrects without needing a direct
    external patch.
    """
    # Fetch all sessions from the database (list of dicts - see
    # AtelierRepository.get_all_sessions_for_history)
    sessions = AtelierRepository.get_all_sessions_for_history()

    # Group sessions by relative date
    today = date.today()
    yesterday = today - timedelta(days=1)
    grouped_sessions = {}

    for s in sessions:
        created = s.get('created_at')
        s_date = created.date() if created else None

        if s_date == today:
            group_name = "Today"
        elif s_date == yesterday:
            group_name = "Yesterday"
        elif created:
            group_name = created.strftime("%B %d, %Y")
        else:
            group_name = "Unknown Date"

        grouped_sessions.setdefault(group_name, []).append(s)

    # Build the dynamic UI elements
    history_groups_ui = []

    for group_name, group_sessions in grouped_sessions.items():
        cards_ui = []

        for s in group_sessions:
            created = s.get('created_at')
            time_str = created.strftime("%I:%M %p") if created else ""
            query_count = s.get('query_count', 0)
            source_count = s.get('source_count', 0)
            status = s.get('status', SessionModel.STATUS_COMPLETED)
            summary = s.get('summary') or (
                "Still researching..." if status == SessionModel.STATUS_IN_PROGRESS else "No summary yet."
            )

            # The delete button is a SIBLING of the Link (not nested inside
            # it) - a click on an element nested inside a dcc.Link's <a>
            # tag would still trigger navigation (the click bubbles up to
            # the anchor), and there's no simple stopPropagation from a
            # plain Dash callback. Same reasoning as the sidebar's history
            # list (ui/callbacks/ui_extras.py's update_sidebar_history).
            card = html.Div(className="group flex items-start gap-2 mb-2", children=[
                dcc.Link(
                    href=f"/history/{s['session_id']}",
                    className="flex-1 min-w-0 cursor-pointer",
                    children=[
                        html.Div(className="flex flex-col md:flex-row gap-6 p-4 rounded-xl hover:bg-surface-container-low transition-all duration-300 border border-transparent hover:border-outline-variant/30 hover:shadow-sm", children=[
                            html.Div(className="flex-1 min-w-0", children=[
                                html.Div(className="flex items-center justify-between mb-2 gap-3", children=[
                                    html.Div(className="flex items-center gap-2.5 min-w-0", children=[
                                        build_status_dot(status),
                                        html.H3(s['topic'], className="text-xl font-headline font-bold text-on-surface group-hover:text-primary transition-colors truncate"),
                                    ]),
                                    html.Span(f"{group_name}, {time_str}", className="text-xs font-headline font-semibold text-outline-variant bg-surface-container px-2 py-1 rounded flex-shrink-0"),
                                ]),
                                html.P(f'"{summary}"', className="font-body text-lg text-on-surface-variant leading-relaxed mb-4 line-clamp-2"),

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
                ),
                html.Button(
                    html.Span("more_vert", className="material-symbols-outlined text-lg"),
                    id={'type': 'delete-session-btn', 'index': s['session_id']},
                    title="Delete this search",
                    className="shrink-0 mt-4 p-1.5 text-outline hover:text-red-500 hover:bg-red-50 rounded transition-colors",
                ),
            ])
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

    return history_groups_ui


def layout_history():
    """Route: '/history' - The Archive List backed by SQLite and new Timeline UI"""

    history_groups_ui = build_history_list()

    # Return the full layout container
    return html.Div(className="w-full max-w-4xl mx-auto px-4 md:px-6 py-12", children=[

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
        html.Div(id="history-list-container", className="space-y-11", children=history_groups_ui) if history_groups_ui else html.Div(id="history-list-container"),

        # Empty State Footer
        html.Div(className="mt-20 text-center py-12 bg-surface-container-low/50 rounded-2xl border border-dashed border-outline-variant/30", children=[
            html.Span("auto_awesome", className="material-symbols-outlined text-outline-variant text-4xl mb-3"),
            html.P("The archive is complete." if history_groups_ui else "No searches yet.", className="font-headline font-semibold text-on-surface-variant"),
            html.P('"An investment in knowledge always pays the best interest."', className="font-body italic text-outline")
        ]) if not history_groups_ui else html.Div(),
    ])
