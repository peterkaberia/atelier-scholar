"""
Document Upload Callbacks

Two-phase UX, mirroring how paper-selection pills already work (see
ui/callbacks/ui_extras.py's update_pills): picking file(s) via the
paperclip control (ui/layouts/feed.py's dcc.Upload) only STAGES them as
removable chips - nothing is parsed or sent to an LLM yet. Actual
ingestion (pipeline/uploads.py) happens once the user hits Send, handled
by ui/callbacks/chat.py's handle_feed_interactions building a trigger-upload
payload, dispatched here as its own background job.
"""

import base64
import logging

import dash
from dash import ALL, Input, Output, State, callback, html, no_update

from core.logger import setup_global_logging
from core.utils import order_citations_by_appearance
from database.repository import AtelierRepository
from llm import AtelierAIEngine
from pipeline.uploads import SUPPORTED_EXTENSIONS, ingest_uploaded_documents
from ui.layouts import build_flow_header, build_loading_skeleton, build_paper_cards, build_synthesis_body

from .background import background_manager

logger = logging.getLogger(__name__)


def _build_pending_upload_chips(pending):
    """Shared by stage_uploads/remove_pending_upload - the removable file chip row above the bottom bar's textarea."""
    if not pending:
        return []
    chips = []
    for i, f in enumerate(pending):
        name = f.get("filename") or "file"
        label = name if len(name) <= 35 else name[:35] + "..."
        chips.append(html.Div(className="flex items-center bg-accent/5 hover:bg-accent/10 transition-all rounded-xl py-2 px-3 md:px-4 border border-accent/10 group shrink-0", children=[
            html.Span("description", className="material-symbols-outlined text-accent text-[18px] mr-2"),
            html.Span(label, className="text-accent text-[10px] font-extrabold uppercase"),
            html.Button("close", id={'type': 'remove-upload-btn', 'index': i}, className="material-symbols-outlined text-accent/40 text-[14px] ml-2 hover:text-accent")
        ]))
    return chips


@callback(
    Output('store-pending-uploads', 'data', allow_duplicate=True),
    Output('pending-uploads-container', 'children', allow_duplicate=True),
    Output('file-upload', 'contents'),
    Input('file-upload', 'contents'),
    State('file-upload', 'filename'),
    State('store-pending-uploads', 'data'),
    prevent_initial_call=True,
)
def stage_uploads(contents_list, filename_list, pending):
    """
    Stages newly-picked file(s) (filename + raw base64 content) into
    store-pending-uploads and renders a removable chip per file - nothing
    is parsed or processed yet (see this module's docstring).

    dcc.Upload with multiple=True always hands back lists for both
    contents/filename (even a single file comes back as a 1-element list),
    so there's no single-vs-multi branching needed. Files with an
    unsupported extension are dropped with a warning here rather than
    staged and only failing later at Send time - pipeline/uploads.py would
    reject them anyway, but telling the user immediately (via the chip
    simply never appearing) is a tighter feedback loop than a silent
    failure minutes later.

    Resets file-upload's own `contents` back to None after staging - a
    dcc.Upload's contents/filename otherwise stay set to the LAST picked
    batch forever, which would re-stage the exact same file(s) a second
    time the next time ANY unrelated prop change happened to re-fire this
    callback's Input.
    """
    if not contents_list:
        raise dash.exceptions.PreventUpdate
    pending = list(pending or [])

    for content, filename in zip(contents_list, filename_list):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            logger.warning(f"Skipping unsupported upload '{filename}' - only PDF/.txt/.md are supported.")
            continue
        pending.append({"filename": filename, "content": content})

    return pending, _build_pending_upload_chips(pending), None


@callback(
    Output('store-pending-uploads', 'data', allow_duplicate=True),
    Output('pending-uploads-container', 'children', allow_duplicate=True),
    Input({'type': 'remove-upload-btn', 'index': ALL}, 'n_clicks'),
    State('store-pending-uploads', 'data'),
    prevent_initial_call=True,
)
def remove_pending_upload(n_clicks_list, pending):
    """Removes one staged-but-not-yet-sent file - mirrors ui/callbacks/ui_extras.py's remove_pill for paper-selection pills."""
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    triggered = dash.callback_context.triggered_id
    if not triggered or not isinstance(triggered, dict):
        raise dash.exceptions.PreventUpdate

    idx = triggered.get('index')
    pending = list(pending or [])
    if idx is None or not (0 <= idx < len(pending)):
        raise dash.exceptions.PreventUpdate

    pending.pop(idx)
    return pending, _build_pending_upload_chips(pending)


@callback(
    Output('pending-flow-update', 'data', allow_duplicate=True),
    Input('trigger-upload', 'data'),
    State('store-chat-history', 'data'),
    background=True,
    manager=background_manager(),
    progress=[Output('pending-flow-update', 'data', allow_duplicate=True)],
    cancel=[Input("search-btn", "n_clicks")],
    prevent_initial_call=True,
)
def run_upload(set_progress, upload_data, chat_history):
    """
    Ingests the file(s) attached to a follow-up (pipeline/uploads.py),
    folding each into this session's evidence pool as a first-class,
    citable Paper - the same Evidence Library treatment a search result
    gets, not a second-class "attachment." If the user also typed a
    question alongside the attachment(s), answers it using the
    newly-ingested document(s) PLUS the session's existing processed
    papers (same RAG chat path run_chat uses); if they just attached
    files with no question, this only confirms what was added instead of
    spending an unrequested extra LLM call on a summary nobody asked for.

    existing_flows comes from upload_data (handed off by
    handle_feed_interactions, ui/callbacks/chat.py), not
    State('flow-container', 'children') - same race-condition reasoning
    as run_chat/run_search - see route_intent's docstring in chat.py.

    Always pulls context from AtelierRepository.get_session_processed_papers
    (the DB), not State('store-processed-records', 'data') - that Store is
    memory-only and resets to empty on every reload regardless of whether
    the session has real prior context (same reasoning route_intent's own
    DB fallback documents), and an upload follow-up specifically needs the
    FULL session context, not just whatever the last SEARCH turn happened
    to populate the Store with.
    """
    if not upload_data or not upload_data.get('files'):
        raise dash.exceptions.PreventUpdate

    setup_global_logging()

    query = (upload_data.get('query') or "").strip()
    selected_llm = upload_data.get('llm')
    session_id = upload_data.get('session_id')
    existing_flows = upload_data.get('existing_flows') or []
    files_payload = upload_data.get('files') or []

    # Falls back to the DB-reconstructed history when store-chat-history
    # is empty - this is genuinely common, not just an after-reload edge
    # case: that Store never gets the session's very FIRST turn written
    # into it at all (only run_chat/run_investigation/run_upload write it,
    # and only once one of them has already run once - the initial SEARCH
    # turn never does), so a user's first-ever follow-up ("rewrite the
    # above with this additional context" + an attachment) otherwise hits
    # an LLM that has no idea what "the above" refers to - confirmed live,
    # this exact wording. See get_session_chat_history_as_messages's
    # docstring for the full explanation; same fallback as run_chat's.
    if not chat_history:
        chat_history = AtelierRepository.get_session_chat_history_as_messages(session_id)

    def report(label: str):
        live_flows = existing_flows[:-1] + [build_loading_skeleton(query or "Processing attachments...", label, icon="upload_file", spin=False)]
        set_progress([{"session_id": session_id, "updates": {"flow_container": live_flows}}])

    # Decode every staged file's data-URL content up front - a corrupt
    # base64 payload should fail this turn cleanly, not partway through
    # ingest_uploaded_documents after some files already processed.
    decoded_files = []
    for f in files_payload:
        filename = f.get("filename") or "file"
        content = f.get("content") or ""
        try:
            _header, b64data = content.split(",", 1)
            decoded_files.append((filename, base64.b64decode(b64data)))
        except Exception as e:
            logger.warning(f"Could not decode uploaded file '{filename}': {e}")

    try:
        # Captured BEFORE ingestion, not after - ingest_uploaded_documents
        # below persists each new file to the DB as it goes (same
        # SessionPaperLink/answer-field writes a search result gets), so a
        # read taken AFTER it returns already includes the files THIS very
        # call just added. Confirmed live via the DB: a 3-file upload
        # produced 6 saved citations - the same 3 papers each appearing
        # TWICE, because "existing" (read after) and "processed" (the
        # ingest return value) both already contained them.
        existing_processed = [r.__dict__ for r in AtelierRepository.get_session_processed_papers(session_id, limit=100)]

        topic = query or AtelierRepository.get_session_topic(session_id) or "this document"
        processed, failed = ingest_uploaded_documents(
            session_id=session_id,
            files=decoded_files,
            topic=topic,
            model_choice=selected_llm,
            report=report,
        )

        if not processed:
            raise RuntimeError(
                f"Couldn't extract any readable text from: {', '.join(failed) if failed else 'the attached file(s)'} "
                "(scanned/image-only PDFs aren't supported yet)."
            )

        combined_records = existing_processed + processed

        if query:
            engine = AtelierAIEngine(model_choice=selected_llm)
            chat_history.append({"role": "user", "content": query})
            ai_response = engine.chat_with_literature(chat_history=chat_history, context_records=combined_records)

            # Same per-turn appearance-ordering as run_chat's identical
            # step (ui/callbacks/chat.py) - see
            # core.utils.order_citations_by_appearance's docstring.
            ai_response, cited_records = order_citations_by_appearance(
                ai_response, [(str(i + 1), r) for i, r in enumerate(combined_records)]
            )
            chat_history.append({"role": "assistant", "content": ai_response})

            new_query_id = AtelierRepository.save_query_with_citations(
                session_id=session_id, prompt=query, synthesis=ai_response,
                model_used=selected_llm, cited_records=cited_records,
            )
            new_flow = html.Div(id=f"turn-{new_query_id}", className="flow-block border-t border-slate-100", children=[
                build_flow_header(query, selected_llm, f"{len(cited_records)} Context Papers", query_id=new_query_id, synthesis_text=ai_response),
                build_synthesis_body(ai_response, valid_records=cited_records, query_id=new_query_id),
                build_paper_cards(cited_records, query_id=new_query_id, session_id=session_id),
            ])
        else:
            title = f"Added {len(processed)} document{'s' if len(processed) != 1 else ''}"
            summary_lines = [f"Added **{len(processed)}** document(s) to this session's evidence library:", ""]
            summary_lines += [f"- {r.get('title')}" for r in processed]
            if failed:
                summary_lines += ["", f"*Couldn't process: {', '.join(failed)} (no readable text found).*"]
            summary_md = "\n".join(summary_lines)

            chat_history.append({"role": "user", "content": f"[Attached {len(files_payload)} document(s)]"})
            chat_history.append({"role": "assistant", "content": summary_md})

            new_query_id = AtelierRepository.save_query_with_citations(
                session_id=session_id, prompt=title, synthesis=summary_md,
                model_used=selected_llm, cited_records=processed,
            )
            new_flow = html.Div(id=f"turn-{new_query_id}", className="flow-block border-t border-slate-100", children=[
                build_flow_header(title, selected_llm, f"{len(processed)} New Papers", query_id=new_query_id, synthesis_text=summary_md),
                build_synthesis_body(summary_md, title="Documents Added", valid_records=processed, query_id=new_query_id),
                build_paper_cards(processed, query_id=new_query_id, session_id=session_id),
            ])

        if existing_flows:
            existing_flows.pop()  # remove the "Processing..." skeleton - see chat.py's run_chat for why this pop matters
        existing_flows.append(new_flow)
        return {"session_id": session_id, "updates": {"flow_container": existing_flows, "chat_history": chat_history}}

    except Exception as e:
        logger.error(f"Upload processing failed for session {session_id}: {e}")
        error_flow = html.Div(className="py-12 px-4 md:px-12 bg-red-50/30 border-t border-red-100", children=[
            html.Div(className="max-w-5xl mx-auto flex items-center space-x-3 text-red-600", children=[
                html.Span("error", className="material-symbols-outlined text-2xl"),
                html.Span(f"Couldn't process the attached file(s): {str(e)}", className="font-bold")
            ])
        ])
        if existing_flows:
            existing_flows.pop()
        existing_flows.append(error_flow)
        return {"session_id": session_id, "updates": {"flow_container": existing_flows, "chat_history": chat_history}}
