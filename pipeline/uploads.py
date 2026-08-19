"""
Document Upload Ingestion Module

Turns a user-uploaded file (PDF or plain text/Markdown - see
ui/callbacks/uploads.py's dcc.Upload wiring) into a fully first-class
Record: the same Paper/SessionPaperLink/chunk/extraction treatment a
search-sourced paper gets, via the exact same persistence/extraction
building blocks pipeline/nodes.py's search pipeline already uses
(save_bulk_records, chunk_text/encode_chunks/save_paper_chunks,
extract_papers_batch + update_record_ai_data) rather than a second,
parallel implementation. Once ingested, an uploaded document is citable,
RAG-searchable, and shows up in the Evidence Library exactly like anything
Atelier found itself - no separate code path anywhere downstream needs to
know it didn't come from a search.
"""

import hashlib
import io
import logging
from typing import Optional

from core.config import DEFAULT_SPARSE_MODEL
from core.utils import chunk_text, truncate
from database.models import Record
from database.repository import AtelierRepository
from llm.engine import AtelierAIEngine
from llm.model_catalog import full_text_tool_budget
from search.sparse_encoder import compute_scores, encode_chunks

logger = logging.getLogger(__name__)

# Hard cap on how much of an uploaded document's raw text gets kept at all
# (chunked + stored) - a very large PDF (a whole textbook, say) would
# otherwise produce hundreds of chunks and dominate every RAG retrieval for
# the rest of the session. Generous relative to a typical paper (tens of
# thousands of characters) without being unbounded.
MAX_UPLOAD_CHARS = 200_000

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")

# Shown as this paper's "answer" (1-sentence takeaway) when parsing/saving
# succeeded but the LLM extraction call itself failed (network error, rate
# limit, etc.) - keeps the document a first-class, processed, citable
# paper (get_session_processed_papers filters on answer != "-") even
# though its structured population/methods/etc. fields stay "-". The
# user's file WAS successfully ingested and is fully searchable via
# full-text RAG regardless of whether this one summarization call worked.
_EXTRACTION_UNAVAILABLE_ANSWER = "Uploaded and indexed - AI analysis is temporarily unavailable, but its full text is searchable."


def _extract_text(filename: str, raw_bytes: bytes) -> str:
    """
    Returns the plain text content of an uploaded file, or "" if the
    format isn't supported or extraction failed. PDF parsing reuses pypdf
    (already a dependency - see search/fetchers.py's identical use for
    fetched-by-URL PDFs), just against in-memory bytes instead of a
    network response.
    """
    lower = filename.lower()
    if lower.endswith((".txt", ".md", ".markdown")):
        try:
            return raw_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to decode text file '{filename}': {e}")
            return ""

    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p.strip() for p in pages if p.strip())
        except Exception as e:
            logger.warning(f"Failed to parse PDF '{filename}': {e}")
            return ""

    logger.warning(f"Unsupported upload type for '{filename}' - only PDF/.txt/.md are supported.")
    return ""


def _derive_title(filename: str, text: str) -> str:
    """
    Best-effort human title: the first plausible-looking line of the
    document (a common convention for both PDF-extracted text and plain
    markdown/text files - the title is usually the very first line), else
    the filename itself.
    """
    for line in (text or "").splitlines():
        line = line.strip().lstrip("#").strip()
        # "Plausible title": not empty, not absurdly long (a whole
        # paragraph that happens to be the first line isn't a title).
        if 3 <= len(line) <= 200:
            return line
    return filename


def ingest_uploaded_document(
    session_id: str,
    filename: str,
    raw_bytes: bytes,
    topic: str,
    model_choice: str,
) -> Optional[dict]:
    """
    Ingests one uploaded file into `session_id`'s evidence pool.

    Unlike pipeline/nodes.py's extract_records, this never gates on the
    LLM's own "is_relevant" verdict - the user explicitly chose to attach
    this document, so it's kept regardless of how relevant the model
    judges it against the current topic (the verdict still gets stored on
    the record, just not used to discard it).

    Returns None (logging why) if the file's text couldn't be extracted at
    all - a scanned/image-only PDF with no text layer, an unsupported
    extension, or a genuinely empty file. Otherwise returns the same
    record-dict shape extract_records' processed list uses, ready to fold
    straight into a turn's processed_records/citations.
    """
    text = _extract_text(filename, raw_bytes)
    if not text or not text.strip():
        logger.warning(f"No extractable text found in uploaded file '{filename}'.")
        return None
    text = text[:MAX_UPLOAD_CHARS]

    # sha1 of the raw bytes, not the extracted text - re-uploading the
    # identical file always dedupes to the same Paper row even if pypdf's
    # extraction happens to be nondeterministic in some edge case; two
    # DIFFERENT files that happen to extract to the same text (rare) still
    # correctly get treated as different uploads.
    fingerprint = f"upload:{hashlib.sha1(raw_bytes).hexdigest()[:16]}"
    title = _derive_title(filename, text)

    rec = Record(
        source="upload",
        source_id=filename,
        title=title,
        abstract=truncate(text, 500),
        authors=[],
        journal="Uploaded Document",
        fingerprint=fingerprint,
    )

    # Same scoring + persistence path a search result goes through
    # (pipeline/nodes.py's score_and_save_node) - gives this paper a real
    # SessionPaperLink.relevance_score so it sorts sensibly alongside
    # search-sourced papers instead of always landing at 0/last.
    compute_scores([rec], topic, DEFAULT_SPARSE_MODEL)
    AtelierRepository.save_bulk_records(session_id, [rec])

    chunks = chunk_text(text)
    if chunks:
        vectors = encode_chunks(chunks, DEFAULT_SPARSE_MODEL)
        AtelierRepository.save_paper_chunks(fingerprint, chunks, vectors)
    # Marks full-text-fetched regardless of whether chunking produced
    # anything - matches _persist_full_text_result's same unconditional
    # call, and prevents any later code from thinking this upload still
    # needs a (nonsensical - there's no URL to fetch) full-text fetch.
    AtelierRepository.mark_full_text_fetched(fingerprint)

    ai = AtelierAIEngine(model_choice=model_choice)
    try:
        # full_text_tool_budget, not a fixed constant - this document is
        # alone in the extraction prompt (uploads are always extracted one
        # at a time, no batch to share a budget with), same "generous
        # single-item share of the selected model's real context window"
        # reasoning as pipeline/agent_tools.py's get_full_text tool - see
        # llm.model_catalog's "CONTEXT-BUDGET SCALING" section.
        results = ai.extract_papers_batch(
            topic=topic,
            papers=[{"paper_index": 0, "title": title, "content": truncate(text, full_text_tool_budget(model_choice))}],
        )
        ai_data = results[0] if results else {}
        if not ai_data:
            raise RuntimeError("Extraction returned no results.")
    except Exception as e:
        logger.warning(f"Extraction failed for uploaded file '{filename}': {e}")
        ai_data = {"answer": _EXTRACTION_UNAVAILABLE_ANSWER, "is_relevant": True}

    AtelierRepository.update_record_ai_data(session_id, fingerprint, ai_data)

    rec.is_relevant = ai_data.get("is_relevant")
    rec.answer = str(ai_data.get("answer", "-"))
    rec.population = str(ai_data.get("population", "-"))
    rec.methods = str(ai_data.get("methods", "-"))
    rec.results = str(ai_data.get("results", "-"))
    rec.outcomes = str(ai_data.get("outcomes", "-"))
    rec.sample_size = str(ai_data.get("sample_size", "-"))
    rec.study_count = str(ai_data.get("study_count", "-"))
    rec.duration = str(ai_data.get("duration", "-"))
    rec.country = str(ai_data.get("country", "-"))

    return rec.__dict__


def ingest_uploaded_documents(
    session_id: str,
    files: list,
    topic: str,
    model_choice: str,
    report=None,
) -> "tuple[list, list]":
    """
    Ingests several uploaded files in one go (see ui/callbacks/uploads.py -
    the attach control accepts multiple files per send). Sequential, not
    parallelized like _prefetch_full_text's per-paper fetch: each file
    already does its own LLM extraction call against one shared endpoint,
    so there's nothing to gain and real risk of the same SQLite
    concurrent-write hang _fetch_and_encode_full_text's docstring
    describes if this ever grew a thread pool around it.

    Args:
        files: [(filename, raw_bytes), ...]
        report: optional progress_callback, called once per file - same
            live-progress mechanism run_search's report() closure uses.

    Returns:
        (processed, failed): processed is a list of record-dicts (same
        shape ingest_uploaded_document returns); failed is a list of
        filenames that produced no extractable text at all.
    """
    processed = []
    failed = []
    for filename, raw_bytes in files:
        if report:
            report(f"Processing '{filename}'...")
        try:
            rec_dict = ingest_uploaded_document(session_id, filename, raw_bytes, topic, model_choice)
        except Exception as e:
            logger.error(f"Unexpected error ingesting uploaded file '{filename}': {e}")
            rec_dict = None
        if rec_dict is None:
            failed.append(filename)
        else:
            processed.append(rec_dict)
    return processed, failed
