import json
import re
from typing import Dict
from dash import html, dcc
from core.utils import classify_study_type
from database.repository import AtelierRepository
from database.models import SessionModel
from llm.utils import get_model_choices

def layout_feed(session_id: str, pending_search: dict = None):
    """Route: '/history/<id>' - The Follow-up Feed Interface"""

    # Resolved per render (not at import time) so a key saved via Settings
    # shows up in the dropdown without an app restart.
    available_llms = get_model_choices()
    # Default to whatever model this session actually used last, not just
    # the first provider in the list - falls back to the first available
    # choice for a brand-new session (no turns yet) or if the last-used
    # model is no longer configured (e.g. its provider was removed in
    # Settings since).
    last_used_model = AtelierRepository.get_last_used_model(session_id) if session_id else ""
    # A BRAND NEW session has no persisted turns yet (get_last_used_model
    # returns "") - the model the user actually picked on Home's hero
    # dropdown only exists in the pending-search handoff at this point
    # (ui/callbacks/search.py's create_new_session), not the DB. Without
    # this, the feed page's dropdown silently reverted to the first
    # provider in the list on every new session regardless of what was
    # chosen on Home - the dropdown wasn't wired to persist the choice
    # across that one handoff at all.
    pending_model = pending_search.get('llm') if pending_search and pending_search.get('session_id') == session_id else ""
    if last_used_model and last_used_model in available_llms:
        default_llm = last_used_model
    elif pending_model and pending_model in available_llms:
        default_llm = pending_model
    else:
        default_llm = available_llms[0] if available_llms else "default-model"

    historical_flows = []
    chat_history = []
    # ui/callbacks/router.py only ever renders this page for a session_id
    # that's either already real or was just handed off from Home as a
    # pending search - so "doesn't exist in the DB yet" here means "about to
    # be created any moment", not "gone" - default to in_progress, not
    # completed, so the processing placeholder below covers that gap too
    # (in practice create_new_session already creates the row immediately,
    # so this default is mostly a safety net for the brief window before
    # that write commits).
    status = SessionModel.STATUS_IN_PROGRESS

    # Check if this session exists in the new DB Repository
    if session_id and AtelierRepository.session_exists(session_id):
        status = AtelierRepository.get_session_status(session_id)
        chat_history = AtelierRepository.get_session_chat_history(session_id)

        for chat in chat_history:
            historical_flows.append(html.Div(className="flow-block border-t border-slate-100", children=[
                build_flow_header(chat.get('prompt'), chat.get('model_used'), f"{len(chat.get('citations', []))} References", query_id=chat.get('query_id')),
                build_atelier_meter(chat.get('consensus_meter')),
                build_synthesis_body(chat.get('synthesis'), valid_records=chat.get('citations', [])),
                build_paper_cards(chat.get('citations', []), query_id=chat.get('query_id'), session_id=session_id)
            ]))

    # A search that's genuinely still running (background=True callbacks keep
    # executing server-side regardless of whether this browser tab is even
    # open) has no persisted QueryModel yet, so historical_flows is empty -
    # without this, navigating away mid-search and back showed a blank page.
    # feed-status-poll (see ui/callbacks/ui_extras.py) refreshes this from
    # the DB every few seconds until the session's status leaves in_progress.
    still_processing = not historical_flows and status == SessionModel.STATUS_IN_PROGRESS
    if still_processing:
        topic = AtelierRepository.get_session_topic(session_id)
        # The real current stage, if one was ever recorded (see
        # AtelierRepository.set_session_progress) - falls back to the
        # generic "resuming" message only for a session that hasn't
        # reported any progress yet at all.
        progress_message = AtelierRepository.get_session_progress(session_id)
        historical_flows = [
            build_loading_skeleton(topic or "Still working on this...", progress_message)
            if progress_message
            else build_processing_placeholder(topic)
        ]
    elif not historical_flows and status == SessionModel.STATUS_FAILED:
        historical_flows = [build_failed_placeholder(AtelierRepository.get_session_topic(session_id), session_id)]

    return [

        dcc.Store(id='current-session-id', data=session_id),

        dcc.Interval(id='feed-status-poll', interval=3000, disabled=not still_processing),

        html.Div(id="flow-container", className="flex-1 overflow-y-auto pb-64 no-scrollbar", children=historical_flows),

        # absolute (not fixed) relative to page-content (see main.py's
        # serve_layout - it's the nearest position:relative ancestor). This
        # box previously used position:fixed with left-1/2 PLUS a fixed
        # md:left-[calc(50%+128px)] desktop offset to visually dodge the
        # 256px-wide sidebar - fixed positions relative to the VIEWPORT, so
        # that +128px offset was computed against the full window width,
        # not the actual space page-content has available next to the
        # sidebar. At viewport widths near the md breakpoint (e.g. a
        # resized/narrower desktop window), the math overflowed both
        # edges at once: the left side slid under the sidebar while the
        # right side pushed past the edge of the screen - confirmed live
        # ("merges with the edges" on a smaller screen). page-content
        # itself is a flex sibling of the sidebar, so its own box already
        # correctly excludes the sidebar's width at every viewport size -
        # positioning absolutely within it needs no separate desktop
        # offset at all, and self-corrects at any width.
        html.Div(id="bottom-bar-container", className="absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-4xl px-4 md:px-6 z-[100]", children=[
            html.Div(className="glass-chat-bar rounded-3xl p-4 md:p-6 flex flex-col transition-all duration-300", children=[
                html.Div(id="selected-papers-container", className="flex items-center mb-4 px-1 gap-2 md:gap-3 overflow-x-auto no-scrollbar empty:hidden"),
                # Files staged via the paperclip control below but not yet
                # sent - see ui/callbacks/uploads.py's stage_uploads for
                # why staging is a separate step from actually processing
                # them.
                html.Div(id="pending-uploads-container", className="flex items-center mb-4 px-1 gap-2 md:gap-3 overflow-x-auto no-scrollbar empty:hidden"),
                html.Div(className="flex items-center w-full px-1 mb-3", children=[
                    dcc.Textarea(
                        id="search-input", 
                        className="w-full bg-transparent border-none text-slate-800 placeholder-slate-400 focus:ring-0 text-[16px] md:text-[17px] font-medium py-1 outline-none resize-none min-h-[24px] max-h-[45vh] leading-relaxed overflow-y-auto",
                        placeholder="Ask follow up or request a deep dive...", rows=1
                    )
                ]),
                
                html.Div(className="flex items-center justify-between px-1", children=[
                    html.Div(className="flex items-center space-x-1", children=[
                        # dcc.Upload, not a plain button - a plain
                        # html.Button has no way to actually open a file
                        # picker/receive file data. Styled to look exactly
                        # like the icon-only button it replaces; the
                        # dashed-drop-zone look that's dcc.Upload's default
                        # is opted out of entirely by fully overriding
                        # `children` (see this component's `style` prop is
                        # deliberately left unset - CSS module default
                        # padding/border would otherwise show through).
                        # multiple=True: uploading several files in one
                        # picker use is expected UX (comparing multiple
                        # documents at once), not one-at-a-time.
                        dcc.Upload(
                            id='file-upload',
                            multiple=True,
                            accept=".pdf,.txt,.md,.markdown",
                            className="inline-flex",
                            children=html.Div(
                                html.Span("attach_file", className="material-symbols-outlined text-2xl"),
                                className="p-2 text-slate-400 hover:text-primary transition-colors hover:bg-slate-100/50 rounded-lg cursor-pointer",
                            ),
                        )
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

def build_loading_skeleton(title: str, status_text: str, icon: str = "autorenew", spin: bool = True):
    """
    A reusable 'flow-block' skeleton: title + spinning icon + one line of
    live status text. Shared by every in-progress state in the app - the
    pipeline's per-stage progress updates (ui/callbacks/search.py), the
    resumed-in-progress placeholder below, and the router's intent-scanning
    skeletons (ui/callbacks/chat.py) - so a single visual language covers
    "something is happening" everywhere instead of one-off inline markup.
    """
    return html.Div(className="flow-block border-t border-slate-100", children=[
        html.Section(className="py-12 px-4 md:px-12", children=[
            html.Div(className="max-w-3xl mx-auto", children=[
                html.H1(title, className="text-3xl font-extrabold text-slate-900 tracking-tight"),
            ])
        ]),
        html.Section(className="py-12 px-4 md:px-12 bg-[#F8FAFF]", children=[
            html.Div(className="max-w-5xl mx-auto text-center", children=[
                html.Span(icon, className=f"material-symbols-outlined text-primary text-4xl mb-4 {'animate-spin' if spin else 'animate-pulse'}"),
                html.P(status_text, className="text-sm font-bold text-slate-500 uppercase tracking-widest animate-pulse")
            ])
        ])
    ])


def build_processing_placeholder(topic: str = ""):
    """Shown when the feed page loads with no completed turns yet, but the session's search is still genuinely running server-side (background=True keeps it alive across navigation)."""
    return build_loading_skeleton(
        topic or "Still working on this...",
        "Resuming — this search is still running in the background. This page will update automatically.",
        icon="autorenew",
    )


def build_search_warning_banner(failed_sources: list[str]):
    """
    A small inline notice appended above the synthesis when one or more
    academic engines errored out mid-search (rate limit, timeout, etc.).
    Distinct from build_failed_placeholder - the run itself still succeeded,
    this just discloses that its results are missing a source rather than
    letting the paper count silently look complete.
    """
    if not failed_sources:
        return None

    label = failed_sources[0] if len(failed_sources) == 1 else f"{', '.join(failed_sources[:-1])} and {failed_sources[-1]}"
    return html.Div(className="flow-block border-t border-amber-100", children=[
        html.Div(className="py-4 px-4 md:px-12", children=[
            html.Div(className="max-w-3xl mx-auto flex items-center gap-3 text-amber-700 bg-amber-50/60 rounded-xl px-4 py-3", children=[
                html.Span("warning", className="material-symbols-outlined text-xl shrink-0"),
                html.Span(f"{label} didn't respond and {'was' if len(failed_sources) == 1 else 'were'} skipped for this search - results below are from the other sources.", className="text-sm font-semibold"),
            ])
        ])
    ])


def build_failed_placeholder(topic: str = "", session_id: str = ""):
    """
    Shown when the feed page loads and the session's last run ended in
    STATUS_FAILED with no completed turns. Includes a Retry button (wired in
    ui/callbacks/chat.py's retry_failed_search) that re-runs the SAME
    session_id rather than starting a brand new one - thanks to the
    permanent per-paper caching already in place (extracted answers, SPLADE
    vectors, full-text chunks all persist as they complete, independent of
    which run produced them), a retry is cheap for whatever had already
    finished before the failure and only genuinely redoes the incomplete
    tail, instead of paying for everything again from zero.
    """
    return html.Div(className="flow-block border-t border-red-100", children=[
        html.Div(className="py-12 px-4 md:px-12", children=[
            html.Div(className="max-w-3xl mx-auto flex items-center justify-between gap-4 flex-wrap", children=[
                html.Div(className="flex items-center gap-3 text-red-600", children=[
                    html.Span("error", className="material-symbols-outlined text-2xl"),
                    html.Span(f"'{topic}' failed to complete." if topic else "This search failed to complete.", className="font-bold"),
                ]),
                html.Button(
                    [html.Span("refresh", className="material-symbols-outlined text-lg"), html.Span("Retry")],
                    id={'type': 'retry-search-btn', 'index': session_id},
                    n_clicks=0,
                    className="flex items-center gap-2 px-4 py-2 bg-red-50 hover:bg-red-100 text-red-700 text-xs font-bold rounded-xl transition-colors shrink-0",
                ) if session_id else None,
            ])
        ])
    ])


def build_flow_header(query: str, model_name: str, ref_text: str, query_id=None):
    """
    query_id (optional): when given, wraps the reference-count span in a
    pattern-matching id so ui/callbacks/library.py's load_more callback can
    update the visible count live after adding more papers, instead of it
    staying stale until the next full page reload. Also gates the "Download
    PDF" button (ui/callbacks/library.py's export_pdf) - both need a real
    saved QueryModel row to work from, so both are simply absent if a save
    ever failed and no id came back (rare, but see save_query_with_citations's
    docstring - None is a real possible return).
    """
    ref_span = html.Span(ref_text, className="text-[11px] font-extrabold text-slate-500 uppercase tracking-widest")
    pdf_button = None
    if query_id is not None:
        ref_span = html.Span(ref_text, id={'type': 'ref-count-text', 'query_id': str(query_id)}, className="text-[11px] font-extrabold text-slate-500 uppercase tracking-widest")
        pdf_button = html.Button(
            [html.Span("picture_as_pdf", className="material-symbols-outlined text-lg"), html.Span("Download PDF")],
            id={'type': 'download-pdf-btn', 'query_id': str(query_id)},
            n_clicks=0,
            className="flex items-center space-x-2 text-[11px] font-extrabold text-slate-400 hover:text-primary uppercase tracking-widest transition-colors",
        )

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
                        ref_span,
                    ]),
                    pdf_button,
                ])
            ])
        ])
    ])

def build_atelier_meter(meter_data: dict):
    """
    The Atelier Meter: an agreement meter for yes/no-shaped research
    questions - see llm.engine.generate_atelier_meter for how meter_data
    is produced (None for non-yes/no topics, so this is only ever called
    when there's a real verdict to show). A stacked Yes/Possibly/No bar,
    weighted by each paper's evidence tier and citation count rather than a
    flat per-paper count - so 9 high-quality RCTs agreeing reads very
    differently from 9 case reports agreeing (inspired by Consensus.app's
    own Consensus Meter 2.0 fix for the same flattening problem).
    """
    if not meter_data:
        return html.Div()

    yes_pct = meter_data.get("yes_pct", 0)
    no_pct = meter_data.get("no_pct", 0)
    possibly_pct = meter_data.get("possibly_pct", 0)
    counts = meter_data.get("counts", {})

    verdict_colors = {
        "Likely Yes": "text-emerald-600",
        "Possibly Yes": "text-emerald-600",
        "Likely No": "text-red-600",
        "Possibly No": "text-red-600",
        "Mixed": "text-amber-600",
    }
    verdict_class = verdict_colors.get(meter_data.get("verdict"), "text-slate-600")

    def segment(pct, color_class):
        if pct <= 0:
            return None
        return html.Div(
            className=f"h-full {color_class} first:rounded-l-full last:rounded-r-full",
            style={"width": f"{pct}%"},
            title=f"{pct}%",
        )

    return html.Section(className="py-0 px-4 md:px-12", children=[
        html.Div(className="max-w-3xl mx-auto", children=[
            html.Div(className="rounded-2xl border border-slate-200 bg-slate-50/60 px-6 py-5 mb-6", children=[
                html.Div(className="flex items-center justify-between mb-3", children=[
                    html.Div(className="flex items-center space-x-2", children=[
                        html.Span("balance", className="material-symbols-outlined text-primary text-xl"),
                        html.Span("Atelier Meter", className="text-[11px] font-extrabold text-slate-500 uppercase tracking-widest"),
                    ]),
                    html.Span(meter_data.get("verdict", ""), className=f"text-lg font-extrabold {verdict_class}"),
                ]),
                html.Div(className="w-full h-3 bg-slate-200 rounded-full overflow-hidden flex", children=[
                    s for s in [
                        segment(yes_pct, "bg-emerald-500"),
                        segment(possibly_pct, "bg-amber-400"),
                        segment(no_pct, "bg-red-500"),
                    ] if s is not None
                ]),
                html.Div(className="flex items-center space-x-4 mt-3 text-xs font-semibold text-slate-500", children=[
                    html.Span(f"Yes ({counts.get('yes', 0)})", className="text-emerald-600"),
                    html.Span(f"Possibly ({counts.get('possibly', 0)})", className="text-amber-600"),
                    html.Span(f"No ({counts.get('no', 0)})", className="text-red-600"),
                    html.Span("Weighted by study design & citations", className="text-slate-400 font-medium normal-case tracking-normal ml-auto"),
                ]),
            ]),
        ]),
    ])


def build_synthesis_body(raw_markdown: str, title: str = "Synthesis Summary", valid_records: list = None):
    """
    No separate references section: the Evidence Library (build_paper_cards,
    further down the page) already lists every cited paper in full, so a
    second list here would just duplicate it. Instead, the [N] markers
    already present in the AI's prose become the hover targets directly, in
    place, exactly where they appear.

    That inline-hover behavior can't be built by embedding styled HTML
    citations into raw_markdown and relying on dcc.Markdown to preserve it -
    confirmed by direct inspection that its renderer doesn't reliably keep
    custom classes/structure even on a single clean pass (a wrapping <span>
    became a bare anchor tag; a bare "[1]" gets auto-linkified into a dead
    empty-href link). So [N] is left as escaped, literal bracket text here
    (dcc.Markdown renders it as plain "[1]" - guaranteed correct, nothing
    fancy attempted), and a citation payload is attached as a data-citations
    JSON attribute on the wrapping Div instead. A clientside script
    (ui/layouts/main.py's index_string) walks the ALREADY-RENDERED DOM after
    the fact, replacing each [N] text occurrence with an interactive marker
    and wiring its hover popup - operating on live DOM nodes via normal
    browser APIs entirely sidesteps dcc.Markdown's HTML-string sanitization,
    since nothing is fed through its parser.
    """
    if valid_records is None:
        valid_records = []

    escaped_markdown = re.sub(r'\[(\d+)\]', r'\\[\1\\]', raw_markdown or "")

    citations_payload = json.dumps([
        {
            "index": i,
            "title": rec.get("title") or "Untitled Document",
            "author": f"{rec.get('authors')[0]} et al." if rec.get("authors") else "Unknown",
            "year": rec.get("year") or "N.D.",
            "journal": rec.get("journal") or "",
            "snippet": rec.get("answer") or rec.get("abstract") or "",
        }
        for i, rec in enumerate(valid_records, start=1)
    ])

    return html.Section(className="py-0 px-4 md:px-12", children=[
        html.Div(className="max-w-3xl mx-auto", children=[
            html.Article(className="relative", children=[
                html.Div(className="relative z-10 space-y-4", children=[
                    html.Div(className="pb-6", children=[
                        html.H2(title, className="text-4xl font-extrabold text-slate-900 tracking-tight leading-tight")
                    ])
                ]),
                html.Div(
                    dcc.Markdown(escaped_markdown),
                    className="prose max-w-none synthesis-citations",
                    **{"data-citations": citations_payload},
                ),
            ])
        ])
    ])

def build_paper_cards(records: list, query_id=None, session_id: str = ""):
    """
    Builds the Evidence Library section containing cards for each parsed paper,
    now including the structured AI metadata.

    Adds a study-type filter chip row (Consensus.app-style "narrow by study
    design") above the card list - a client-side JS filter (see
    ui/layouts/main.py's index_string), not a Dash callback: these records
    can arrive from a background job's live flow-block OR from
    AtelierRepository.get_session_chat_history's historical reconstruction,
    and wiring a pattern-matching callback per flow-block for a purely
    visual show/hide would add real complexity for no behavior a plain
    click-delegation handler doesn't already give the rest of this app's
    hover/delete interactions.

    study_type is recomputed here via classify_study_type rather than
    trusted from the record - same reasoning as
    llm.engine._format_synthesis_entry: it's set during
    search/sparse_encoder.py's compute_scores for ranking, but both the live
    (extract_node re-fetches from the DB) and historical
    (get_session_chat_history) paths reconstruct Record objects without it,
    since it isn't a persisted column. Recomputing is cheap and deterministic.

    query_id (int, optional): the QueryModel row this card list belongs to
    (see save_query_with_citations's docstring) - used to scope every
    pattern-matching component id in this function (checkboxes, load-more
    button, the list container itself) so a paper cited in two different
    turns on one page never produces two DOM elements sharing one id, which
    Dash disallows outright. Falls back to session_id, then the literal
    string "unscoped" (best-effort only - covers the rare case where a save
    failed and no id was returned; a session can't have two truly
    *concurrent* unscoped blocks in practice).
    """
    scope = str(query_id) if query_id is not None else (session_id or "unscoped")
    _ensure_study_types(records)

    type_counts: Dict[str, int] = {}
    for rec in records:
        st = rec.get("study_type") or "unspecified"
        type_counts[st] = type_counts.get(st, 0) + 1

    filter_chips = None
    if len(type_counts) > 1:
        chip_buttons = [
            html.Button(
                f"All ({len(records)})",
                className="filter-chip-btn active px-3 py-1.5 rounded-full text-[11px] font-bold border border-primary bg-primary text-white transition-colors",
                **{"data-filter-type": "__all__"},
            )
        ]
        for st, count in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            label = "Unspecified" if st == "unspecified" else st.title()
            chip_buttons.append(html.Button(
                f"{label} ({count})",
                className="filter-chip-btn px-3 py-1.5 rounded-full text-[11px] font-bold border border-slate-200 bg-white text-slate-600 hover:border-primary transition-colors",
                **{"data-filter-type": st},
            ))
        filter_chips = html.Div(className="flex flex-wrap gap-2 pb-2", children=chip_buttons)

    articles = _build_paper_card_articles(records, scope)

    return html.Section(className="py-12 px-4 md:px-12", children=[
        html.Div(className="max-w-3xl mx-auto", children=[
            html.Div(className="space-y-8", children=[
                html.Div(className="flex items-center justify-between border border-slate-200 p-2 rounded-2xl shadow-sm", children=[
                    # Only THIS inner group is the click target for
                    # collapsing/expanding (results-accordion-toggle, see
                    # ui/layouts/main.py's index_string) - not the whole
                    # header row, which also holds the export/view-mode
                    # buttons on the right. A toggle spanning the full row
                    # would fire on every button click too, and there's no
                    # clean stopPropagation path from a plain Dash Button.
                    html.Div(className="flex items-center space-x-4 pl-4 cursor-pointer select-none results-accordion-toggle", children=[
                        html.Span("expand_more", className="material-symbols-outlined text-slate-400 accordion-chevron transition-transform duration-200"),
                        html.H2("Results", className="text-lg font-extrabold text-slate-900 tracking-tight"),
                    ]),
                    html.Div(className="flex items-center gap-2", children=[
                        # Exports whatever's checked (all shown papers if
                        # nothing's checked) via ui/callbacks/library.py's
                        # export_references. Two buttons, not a format
                        # picker: both BibTeX and RIS are universally
                        # importable by reference software (Zotero, EndNote,
                        # Mendeley) and there's no clean way to know which
                        # one a given user's tool prefers.
                        html.Button(className="flex items-center space-x-2 text-[11px] font-bold px-4 py-2 bg-white border border-slate-200 rounded-xl text-slate-700 shadow-sm hover:border-primary transition-colors", id={'type': 'export-btn', 'query_id': scope, 'format': 'bib'}, n_clicks=0, children=[
                            html.Span("download", className="material-symbols-outlined text-lg"),
                            html.Span("Export .bib")
                        ]),
                        html.Button(className="flex items-center space-x-2 text-[11px] font-bold px-4 py-2 bg-white border border-slate-200 rounded-xl text-slate-700 shadow-sm hover:border-primary transition-colors", id={'type': 'export-btn', 'query_id': scope, 'format': 'ris'}, n_clicks=0, children=[
                            html.Span("download", className="material-symbols-outlined text-lg"),
                            html.Span("Export .ris")
                        ]),
                        html.Button(className="flex items-center bg-white border border-slate-200 rounded-xl p-1 shadow-sm", children=[
                           html.Button("LIST", className="px-3 py-1.5 rounded-lg bg-primary text-white font-bold text-[10px]"),
                           html.Button("TABLE", className="px-3 py-1.5 rounded-lg text-slate-400 font-bold text-[10px]")
                        ])
                    ])
                ]),
                # Collapsible body - rendered expanded by default (server
                # side has no notion of "which flow-block is active"; that's
                # purely a client-side, per-page-load concept). JS
                # (ui/layouts/main.py's index_string, enforceActiveFlowBlock)
                # collapses every flow-block's body except the LAST one on
                # the page whenever a new turn is appended, and a click on
                # results-accordion-toggle above flips it manually.
                html.Div(className="results-accordion-body", children=[
                    filter_chips,
                    # class AND pattern-matching id: the class is for the
                    # filter-chip JS (main.py's index_string), scoped via
                    # closest() so it doesn't need a unique id. The id is for
                    # ui/callbacks/library.py's load_more callback, which DOES
                    # need to target this exact container's children with a
                    # Dash Output - a class alone can't be an Output target.
                    html.Div(
                        id={'type': 'paper-cards-list', 'query_id': scope},
                        className="space-y-0 divide-y divide-slate-200/40 paper-cards-list",
                        children=articles,
                    ),
                    html.Div(className="pt-0 pb-4 flex flex-col items-center gap-2", children=[
                        html.Button(
                            "Load More Results",
                            id={'type': 'load-more-btn', 'query_id': scope},
                            n_clicks=0,
                            className="px-8 py-3 bg-white border border-slate-200 rounded-full text-sm font-bold text-primary hover:border-primary hover:shadow-md transition-all",
                        ),
                        html.Div(id={'type': 'load-more-status', 'query_id': scope}, className="text-xs font-semibold text-slate-400"),
                    ]),
                ]),
            ])
        ])
    ])


def _ensure_study_types(records: list) -> None:
    """Shared by build_paper_cards and _build_paper_card_articles - see either's docstring."""
    for rec in records:
        if not rec.get("study_type"):
            rec["study_type"] = classify_study_type(rec.get("title", ""), rec.get("abstract", ""))


def _build_paper_card_articles(records: list, scope: str) -> list:
    """
    Builds just the html.Article card list (no surrounding section/header/
    filter-chips/load-more chrome) - factored out of build_paper_cards so
    ui/callbacks/library.py's load_more callback can rebuild just the card
    list's contents (its Output targets the list container's children
    directly) without duplicating this markup. Re-ensures study_type itself
    (cheap/idempotent) rather than trusting the caller already did it, since
    load_more calls this directly on freshly-extracted records that never
    passed through build_paper_cards's own classification pass.
    """
    _ensure_study_types(records)
    articles = []
    for i, rec in enumerate(records, 1):
        num_str = f"{i:02d}"

        articles.append(
            html.Article(className="atelier-result py-8 flex gap-6", **{"data-study-type": rec.get("study_type") or "unspecified"}, children=[
                html.Div(className="flex flex-col items-center", children=[
                    html.Span(num_str, className="text-3xl font-extrabold text-slate-200 mb-2"), 
                    dcc.Checklist(
                        # index carries scope:fingerprint together (see this
                        # function's query_id docstring). fingerprint, not
                        # pmid-or-position - it's the one identifier every
                        # paper is guaranteed to have (the DB's own
                        # deduplication key), and it's what
                        # ui/callbacks/library.py's export/load-more/
                        # per-paper-summary callbacks re-fetch records by
                        # after a checkbox is read back. pmid is often
                        # missing (non-PubMed sources), and a positional
                        # fallback wouldn't reliably map back to the same
                        # paper once records are re-fetched from the DB in a
                        # possibly different order.
                        id={'type': 'paper-checkbox', 'index': f"{scope}:{rec.get('fingerprint') or i}"},
                        options=[{'label': '', 'value': f"{scope}:{rec.get('fingerprint') or i}"}],
                        value=[],
                        className="w-5 h-5 rounded border-slate-300 text-primary focus:ring-primary"
                    )
                ]),
                html.Div(className="flex-1", children=[
                    html.Div(className="flex justify-between items-start mb-1", children=[
                        html.H3(
                            html.Button(
                                rec.get('title', ''),
                                id={'type': 'paper-detail-btn', 'index': f"{scope}:{rec.get('fingerprint') or i}"},
                                n_clicks=0,
                                className="text-left text-lg font-bold text-primary hover:underline cursor-pointer leading-snug",
                            ),
                            className="w-full",
                        )
                    ]),
                    html.Div(className="text-sm text-slate-500 mb-3 font-medium", children=[
                        # rec.get('authors', ['Unknown']) only falls back to
                        # the default when the KEY is missing entirely - a
                        # paper record with authors explicitly present but
                        # EMPTY (authors: []), which papers pulled in via a
                        # bare full_text_url with no other metadata can
                        # genuinely have, sailed straight through and
                        # crashed on [0] with an IndexError. `or ['Unknown']`
                        # catches both cases (falsy check, not "key absent").
                        html.Span(f"{(rec.get('authors') or ['Unknown'])[0]} et al.", className="text-slate-900"),
                        f" • {rec.get('year', 'N.D.')}",
                        f" • {rec.get('citation_count', 0)} Citations • ",
                        html.Em(rec.get('journal', ''), className="text-slate-400 not-italic")
                    ]),
                    
                    # 1-Sentence Takeaway
                    html.P(rec.get('answer', ''), className="text-sm text-slate-600 mb-5 leading-relaxed"),
                    
                    html.Div(className="flex items-center space-x-6", children=[
                        # A button, not a link with target=_blank/_self:
                        # opens the in-app PDF viewer modal (JS click
                        # delegation in main.py's index_string) instead of
                        # navigating away or replacing this page. Only
                        # rendered when there's actually a pdf_url to show.
                        html.Button(
                            className="pdf-viewer-btn flex items-center space-x-2 text-xs font-bold text-primary hover:text-accent transition-colors",
                            **{"data-pdf-url": rec.get('pdf_url')},
                            children=[
                                html.Span("picture_as_pdf", className="material-symbols-outlined text-lg"),
                                html.Span("VIEW PDF")
                            ]
                        ) if rec.get('pdf_url') else html.Span(
                            className="flex items-center space-x-2 text-xs font-bold text-slate-300 cursor-not-allowed",
                            children=[
                                html.Span("picture_as_pdf", className="material-symbols-outlined text-lg"),
                                html.Span("NO PDF AVAILABLE")
                            ]
                        ),
                        html.Button(className="flex items-center space-x-2 text-xs font-bold text-slate-400 hover:text-slate-900 transition-colors ml-auto", children=[
                            html.Span("bookmark", className="material-symbols-outlined text-lg"),
                            html.Span("SAVE")
                        ])
                    ])
                ])
            ])
        )

    return articles