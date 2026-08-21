from dash import html, dcc
from core.utils import truncate
from database import AtelierRepository
from llm import get_model_choices

# Fallback quick-start prompts for a brand-new user with no history yet to
# draw from - Atelier's equivalent of Stitch's own suggestion pills ("A
# trip packing checklist app that sugge...", etc.), just scoped to what
# THIS app actually does (literature synthesis, not app design). Once
# there IS history, get_suggested_prompts below prefers that instead -
# "it can start static but specialise," reported directly.
_DEFAULT_SUGGESTED_PROMPTS = [
    "Does metformin reduce cardiovascular risk in type 2 diabetics?",
    "Summarize the evidence on urban green space and mental health",
    "Write a literature review on microplastics in drinking water",
]


def get_suggested_prompts() -> list[str]:
    """
    Tailored to the user's own past research once there's any to draw
    from - their most recent distinct session topics, so returning to
    Home suggests picking up a thread they actually care about instead of
    the same three generic examples forever. Falls back to
    _DEFAULT_SUGGESTED_PROMPTS for a brand-new user with nothing yet.

    Deliberately not an LLM call (e.g. "generate 3 NEW related questions
    from this history") - Home has to stay fast to load, and a raw recent
    topic is already a genuinely useful, zero-latency suggestion: clicking
    it re-opens that same research thread, which is exactly what "continue
    where I left off" means for a tool like this.

    Called from BOTH layout_home (to render the chips) and
    ui/callbacks/ui_extras.py's fill_suggestion_prompt (to resolve which
    full prompt a click maps to) - kept as one shared function rather than
    a value cached at render time, so the two agree without needing a
    Store in between; the DB rarely changes in the few seconds between a
    page rendering and a chip actually being clicked, and if it does, the
    worst case is a chip's fill text quietly refers to a slightly
    different topic instead of an error.
    """
    seen = set()
    tailored = []
    for s in AtelierRepository.get_all_sessions_for_history():
        topic = (s.get('topic') or '').strip()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        tailored.append(topic)
        if len(tailored) == 3:
            break
    return tailored or _DEFAULT_SUGGESTED_PROMPTS


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
                        # Icon-only, matching the feed's own follow-up
                        # submit button (ui/layouts/feed.py's search-btn)
                        # exactly - a text-labeled "SYNTHESIZE" button here
                        # was the odd one out next to that, and Stitch's
                        # own composer button is icon-only too. See
                        # ui/callbacks/search.py's create_new_session for
                        # the matching running=[...] spinner swap.
                        html.Button(id="hero-search-btn", n_clicks=0, className="w-10 h-10 md:w-11 md:h-11 bg-primary hover:bg-primary/90 hover:scale-105 active:scale-95 rounded-2xl text-white shadow-xl shadow-primary/20 transition-all flex items-center justify-center flex-shrink-0", children=[
                            html.Span("arrow_upward", id="hero-search-btn-icon", className="material-symbols-outlined text-xl md:text-2xl font-bold"),
                        ])
                    ])
                ])
            ]),
            # Quick-start pills, Stitch-style - see get_suggested_prompts
            # above. A plain flex-wrap row (not a carousel like Stitch's
            # "Need inspiration?" strip below the fold): three short
            # prompts fit on one line at this width, so a horizontal-
            # scroll rig would just be unused complexity for how little
            # content there is here. Truncated for display (a real recent
            # session topic can run to several sentences) - title carries
            # the full text on hover, and clicking still fills in the FULL
            # prompt (ui/callbacks/ui_extras.py's fill_suggestion_prompt
            # re-resolves it by index, not from this truncated label).
            html.Div(className="flex flex-wrap items-center justify-center gap-2 mt-4 px-4 animate-fade-in", style={"animationDelay": "0.3s"}, children=[
                html.Button(
                    truncate(prompt, 70), id={'type': 'suggestion-chip', 'index': i}, n_clicks=0, title=prompt,
                    className="px-4 py-2 rounded-full border border-slate-200 bg-white text-slate-600 text-xs font-semibold hover:border-primary/30 hover:bg-primary/5 hover:text-primary transition-colors shadow-sm",
                )
                for i, prompt in enumerate(get_suggested_prompts())
            ]),
        ])
    ])