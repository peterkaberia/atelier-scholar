from dash import html, dcc

def layout_sidebar():
    """
    The nav column. On desktop (>=md) it's a static column. On mobile it's a
    slide-in drawer, hidden off-screen by default (-translate-x-full) and
    toggled via the hamburger button in layout_mobile_topbar() - see the
    toggleMobileMenu/closeMobileMenuOnNav clientside callbacks in app.py.
    """
    # w-80 (320px), not the old w-64 (256px) - Stitch's own sidebar reads
    # noticeably roomier than Atelier's previous one; this is the width
    # change requested to get closer to that feel.
    return html.Aside(id="app-sidebar", className="w-80 flex-shrink-0 border-r border-border-light bg-surface-light flex flex-col h-full fixed md:relative inset-y-0 left-0 z-[200] -translate-x-full md:translate-x-0 transition-transform duration-300 ease-in-out", children=[
            # Shared delete-session confirmation for the sidebar's history
            # list - the sidebar is always mounted (part of serve_layout()),
            # so this is reachable regardless of which page is showing. See
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
                    # The logo IS the "go home" action now - a New Search
                    # button and a separate Home nav link both pointed at
                    # "/" too, three ways to do the same thing. Clicking the
                    # wordmark to go home is a near-universal enough web
                    # convention that it doesn't need its own label either.
                    dcc.Link(href="/", className="flex items-center space-x-2 text-primary font-extrabold text-xl tracking-tight", children=[
                        html.Span("school", className="material-symbols-outlined text-3xl"), html.Span("Atelier")
                    ]),
                    html.Button(id="mobile-menu-close-btn", className="md:hidden p-1.5 text-slate-400 hover:text-slate-900 hover:bg-slate-50 rounded-lg transition-colors", children=[
                        html.Span("close", className="material-symbols-outlined text-xl")
                    ])
                ]),
                # Server-side filter over the grouped history list below -
                # see ui/callbacks/ui_extras.py's update_sidebar_history.
                # Mirrors Stitch's own "Search projects" input, pill-shaped
                # to match. Hidden while viewing one session's own detail
                # page (see toggleSidebarSearchBox in app.py/main.py) -
                # "other sessions" (what this searches) shouldn't appear
                # there at all, reported directly, so a search box with
                # nothing left to search is hidden right along with them.
                html.Div(id="sidebar-search-box", className="relative mb-3", children=[
                    html.Span("search", className="material-symbols-outlined text-[18px] text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"),
                    dcc.Input(
                        id="sidebar-history-search", type="text", placeholder="Search sessions", debounce=True,
                        className="w-full bg-slate-50 border border-slate-200/70 rounded-full pl-9 pr-3 py-2 text-xs font-medium text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/30 transition-all",
                    ),
                ]),
                html.Nav(className="flex-1 min-h-0 overflow-y-auto no-scrollbar -mx-1 px-1", children=[
                    # Only populated while viewing a session's own detail
                    # page (/session/<id>) - empty everywhere else. See
                    # ui/callbacks/ui_extras.py's update_session_nav/
                    # _build_session_nav_items. Sits ABOVE the global
                    # history list so "jump to a section of what I'm
                    # looking at right now" is the first thing in view,
                    # ahead of "browse everything else."
                    html.Div(id="sidebar-session-nav", children=[]),
                    html.Div(id="sidebar-recent-history", children=[]),
                ]),
                # md:hidden - on desktop, Settings lives in the persistent
                # top-right icon instead (see main.py's serve_layout), same
                # placement Stitch uses for its own account/settings-type
                # icons. Kept here for mobile, where that top-right icon is
                # itself hidden to avoid colliding with the hamburger menu
                # button in the same corner (layout_mobile_topbar).
                html.Div(className="pt-4 border-t border-slate-100 md:hidden", children=[
                    dcc.Link(className="flex items-center space-x-3 px-3 py-2 text-slate-500 hover:bg-slate-50 hover:text-slate-900 rounded-md transition-colors", href="/settings", children=[
                        html.Span("settings", className="material-symbols-outlined text-[20px]"), html.Span("Settings")
                    ])
                ])
            ])
        ])


def layout_mobile_topbar():
    """Sticky top bar shown only below the md breakpoint - the sidebar's replacement when it's off-screen."""
    return html.Div(id="mobile-topbar", className="md:hidden flex items-center justify-between px-4 py-3 border-b border-border-light bg-surface-light sticky top-0 z-[100] flex-shrink-0", children=[
        html.Div(className="flex items-center gap-2", children=[
            # Mobile's own back button - the desktop app bar's (see
            # main.py's serve_layout/toggleAppBarBackButton) is `hidden
            # md:flex`, so small screens had no equivalent at all,
            # reported directly ("also consider the small screen back
            # screen"). Same rule (hidden on Home, shown everywhere else)
            # and same real-browser-back behavior, via mobileGoBack below -
            # a separate function/Output rather than reusing the desktop
            # one, since a clientside callback's Output can't be shared
            # across two different button ids.
            html.Button(id="mobile-back-btn", style={"display": "none"}, className="p-1 -ml-1 text-slate-500 hover:text-primary transition-colors", children=[
                html.Span("arrow_back", className="material-symbols-outlined text-2xl"),
            ]),
            # Same "logo IS the home link" convention as the desktop sidebar -
            # see layout_sidebar's own comment.
            dcc.Link(href="/", className="flex items-center space-x-2 text-primary font-extrabold text-lg tracking-tight", children=[
                html.Span("school", className="material-symbols-outlined text-2xl"), html.Span("Atelier")
            ]),
        ]),
        html.Button(id="mobile-menu-btn", className="p-2 text-slate-600 hover:bg-slate-50 rounded-lg transition-colors", children=[
            html.Span("menu", className="material-symbols-outlined text-2xl")
        ])
    ])


def layout_mobile_backdrop():
    """Tap-to-close overlay behind the open drawer. Hidden by default; toggled alongside the drawer itself."""
    return html.Div(id="mobile-menu-backdrop", className="hidden fixed inset-0 bg-black/40 z-[150] md:hidden")
