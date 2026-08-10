from dash import html, dcc

def layout_sidebar():
    """
    The nav column. On desktop (>=md) it's a static column. On mobile it's a
    slide-in drawer, hidden off-screen by default (-translate-x-full) and
    toggled via the hamburger button in layout_mobile_topbar() - see the
    toggleMobileMenu/closeMobileMenuOnNav clientside callbacks in app.py.
    """
    return html.Aside(id="app-sidebar", className="w-64 flex-shrink-0 border-r border-border-light bg-surface-light flex flex-col h-full fixed md:relative inset-y-0 left-0 z-[200] -translate-x-full md:translate-x-0 transition-transform duration-300 ease-in-out", children=[
            # Shared delete-session confirmation, used by both the sidebar's
            # own history list and the /history archive page - the sidebar
            # is always mounted (part of serve_layout()), so this is
            # reachable regardless of which page triggered it. See
            # ui/callbacks/ui_extras.py's request_session_delete/
            # confirm_session_delete for the two-step (click menu item ->
            # confirm) flow this drives.
            dcc.Store(id="pending-delete-session", data=None),
            dcc.ConfirmDialog(
                id="confirm-delete-session-dialog",
                message="Delete this search permanently? This can't be undone.",
            ),
            # Keeps the sidebar's status dots/summaries live without
            # requiring a navigation. update_sidebar_history (ui/callbacks/
            # ui_extras.py) previously only refreshed on Input('url',
            # 'pathname') - correct for showing a BRAND NEW session the
            # moment it's created, but it meant a session's status dot
            # (in_progress -> completed/failed) or summary text only ever
            # updated once the user happened to navigate somewhere, not
            # when the underlying background job actually finished. The
            # sidebar is part of the persistent shell (always mounted), so
            # this poll runs regardless of which page is showing.
            dcc.Interval(id='sidebar-status-poll', interval=5000),
            html.Div(className="p-6 flex flex-col h-full", children=[
                html.Div(className="flex items-center justify-between mb-8", children=[
                    html.Div(className="flex items-center space-x-2 text-primary font-extrabold text-xl tracking-tight", children=[
                        html.Span("school", className="material-symbols-outlined text-3xl"), html.Span("Atelier")
                    ]),
                    html.Button(id="mobile-menu-close-btn", className="md:hidden p-1.5 text-slate-400 hover:text-slate-900 hover:bg-slate-50 rounded-lg transition-colors", children=[
                        html.Span("close", className="material-symbols-outlined text-xl")
                    ])
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
                ]),
                html.Div(className="pt-4 border-t border-slate-100", children=[
                    dcc.Link(className="flex items-center space-x-3 px-3 py-2 text-slate-500 hover:bg-slate-50 hover:text-slate-900 rounded-md transition-colors", href="/settings", children=[
                        html.Span("settings", className="material-symbols-outlined text-[20px]"), html.Span("Settings")
                    ])
                ])
            ])
        ])


def layout_mobile_topbar():
    """Sticky top bar shown only below the md breakpoint - the sidebar's replacement when it's off-screen."""
    return html.Div(id="mobile-topbar", className="md:hidden flex items-center justify-between px-4 py-3 border-b border-border-light bg-surface-light sticky top-0 z-[100] flex-shrink-0", children=[
        html.Div(className="flex items-center space-x-2 text-primary font-extrabold text-lg tracking-tight", children=[
            html.Span("school", className="material-symbols-outlined text-2xl"), html.Span("Atelier")
        ]),
        html.Button(id="mobile-menu-btn", className="p-2 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors", children=[
            html.Span("menu", className="material-symbols-outlined text-2xl")
        ])
    ])


def layout_mobile_backdrop():
    """Tap-to-close overlay behind the open drawer. Hidden by default; toggled alongside the drawer itself."""
    return html.Div(id="mobile-menu-backdrop", className="hidden fixed inset-0 bg-black/40 z-[150] md:hidden")
