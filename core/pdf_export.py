"""
Per-turn PDF export.

Renders one flow-block turn (a synthesis, chat reply, or investigation
answer - whatever's stored as QueryModel.synthesis, markdown either way) as
a standalone, readable PDF: title, metadata, the answer body with real
typography (headers, tables, bold/italics all render properly, not just a
plain-text dump), and a numbered references section for whatever papers
were cited.

markdown -> HTML -> PDF (via the `markdown` and `xhtml2pdf` packages),
not a from-scratch PDF layout - both are pure-Python with no native/system
dependencies (unlike WeasyPrint, which needs GTK/Pango/Cairo runtime
libraries separately installed - a real pain on Windows, where this app
primarily runs), so this stays a plain `uv add`, nothing more.
"""

import html
import io
import logging
import re
from datetime import datetime
from typing import Any, List, Optional, Tuple

import markdown as md

from core.utils import truncate

logger = logging.getLogger(__name__)

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _fill_empty_table_cells(markdown_text: str) -> str:
    """
    Forward-fills empty markdown table cells with the last non-empty value
    seen in that column, within each table block.

    xhtml2pdf's table renderer has a genuine row-height/pagination bug with
    truly empty `<td></td>` cells - confirmed directly (isolated repro):
    a table using LLM-style "leave the cell blank to visually group
    repeated category rows" formatting renders with adjacent rows' text
    OVERLAPPING illegibly, while the exact same table with every cell
    filled (no truly empty ones) renders perfectly. Forward-filling instead
    of just inserting a blank placeholder also happens to better match what
    the LLM actually meant by leaving cells blank in the first place (a
    grouped/repeated value), not a loss of information.
    """
    lines = markdown_text.split("\n")
    out_lines = []
    last_values: List[str] = []
    in_table = False

    for i, line in enumerate(lines):
        is_row = bool(_TABLE_ROW_RE.match(line))
        is_separator = is_row and bool(_TABLE_SEPARATOR_RE.match(line))

        if is_row and not is_separator and not in_table:
            # Header row - the first `|...|` line, not yet inside a table
            # until the NEXT line confirms it's a separator.
            next_is_separator = (i + 1 < len(lines)) and bool(_TABLE_SEPARATOR_RE.match(lines[i + 1]))
            if next_is_separator:
                in_table = True
                last_values = []
                out_lines.append(line)
                continue

        if in_table and is_separator:
            out_lines.append(line)
            continue

        if in_table and is_row:
            cells = line.strip().strip("|").split("|")
            if not last_values:
                last_values = [""] * len(cells)
            filled = []
            for idx, cell in enumerate(cells):
                stripped = cell.strip()
                if stripped:
                    if idx < len(last_values):
                        last_values[idx] = stripped
                    filled.append(cell)
                else:
                    filled.append(f" {last_values[idx]} " if idx < len(last_values) else cell)
            out_lines.append("|" + "|".join(filled) + "|")
            continue

        # A non-`|` line ends the current table block.
        in_table = False
        out_lines.append(line)

    return "\n".join(out_lines)

## Colors pulled straight from the live app's own palette (ui/layouts/main.py's
# tailwind.config: primary "#1A237E", accent "#3B82F6"; .prose's body/table
# colors), not an unrelated navy invented for this file - previously this
# stylesheet used its own made-up #0F172A/#1a1a1a/#eef1f7 scheme that visibly
# didn't match the app's actual branding (confirmed live, reported directly:
# "the color of download report does not match"). Section headings now use
# the same indigo (#1A237E) the app uses for table headers/list bullets/links
# throughout .prose; body text uses the same slate (#334155) as .prose p.
_CSS = """
    @page { size: A4; margin: 2.2cm; }
    body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; color: #334155; line-height: 1.5; }
    h1.doc-title { font-size: 19pt; margin: 0 0 4pt 0; color: #0F172A; }
    .doc-meta { font-size: 9pt; color: #64748B; margin-bottom: 18pt; border-bottom: 1pt solid #E2E8F0; padding-bottom: 10pt; }
    h1, h2, h3 { color: #1A237E; margin-top: 16pt; margin-bottom: 6pt; }
    h3 { font-size: 13pt; }
    p { margin: 6pt 0; text-align: justify; }
    table { width: 100%; border-collapse: collapse; margin: 10pt 0; font-size: 9.5pt; }
    th, td { border: 0.5pt solid #E2E8F0; padding: 5pt 7pt; text-align: left; vertical-align: top; }
    th { background-color: #F8FAFC; color: #1A237E; font-weight: bold; }
    td { color: #334155; }
    .references { margin-top: 22pt; border-top: 1pt solid #E2E8F0; padding-top: 10pt; }
    .references h2 { font-size: 13pt; color: #1A237E; }
    .ref-item { font-size: 9.5pt; margin: 6pt 0; color: #334155; }
    .ref-num { font-weight: bold; color: #1A237E; }
    /* Matches the web UI's .ref-badge-inline citation styling (ui/layouts/main.py) -
       colored + bold, no underline so it doesn't read as a stray hyperlink
       mid-sentence, same as on screen. */
    a.cite-link { color: #1A237E; font-weight: bold; text-decoration: none; }
"""


def _get_val(rec: Any, key: str, default: str = "") -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


_LEADING_HEADING_RE = re.compile(r"^\s*#{1,3}\s+(.+?)\s*$")

# A line that's clearly a HEADING for "References"/"Bibliography"/"Works
# Cited" - either real markdown heading hashes, or a standalone bold line -
# with only optional numbering ("13.") alongside it. Deliberately does NOT
# match a bare, unformatted "13. References" line with no "#"/"**" at all:
# that shape is indistinguishable from a plain numbered table-of-contents
# ENTRY (e.g. "13. References" listed among "1. Executive Summary" etc.),
# and matching it there truncated the entire report at the ToC - confirmed
# directly against a real generated report, which listed its ToC entry as
# bare text but wrote the real section heading as "## 13. REFERENCES".
_SELF_WRITTEN_REFS_RE = re.compile(
    r"(?im)^[ \t]*(?:"
    r"#{1,4}[ \t]*(?:\d+[.)][ \t]*)?(?:references|bibliography|works cited)[ \t]*"
    r"|"
    r"\*\*[ \t]*(?:\d+[.)][ \t]*)?(?:references|bibliography|works cited)[ \t]*\**"
    r")[ \t]*$"
)


def _extract_leading_title(markdown_text: str) -> Tuple[Optional[str], str]:
    """
    Pulls a leading markdown heading (#/##/###) off the very first
    non-blank line and returns (heading_text, remaining_body) - so a
    structured-deliverable answer's own title (chat_with_literature's
    genre-detection prompt has the model open long reports with one, e.g.
    "# Hydrological and Ecosystem Impact Assessment of...") becomes the
    PDF's real title instead of being rendered a second time as the body's
    own first heading. Returns (None, text unchanged) when the first
    non-blank line isn't a heading - a plain Q&A answer, for instance,
    which the model has no reason to open with one.
    """
    lines = (markdown_text or "").split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        m = _LEADING_HEADING_RE.match(line)
        if m:
            remaining = "\n".join(lines[:i] + lines[i + 1:])
            return m.group(1).strip(), remaining
        break  # first non-blank line isn't a heading - nothing to extract
    return None, markdown_text or ""


def _strip_self_written_references(markdown_text: str) -> str:
    """
    Drops a model-written "References"/"Bibliography" section, and
    everything after it, from the synthesis body before it's rendered.

    chat_with_literature's system prompt does NOT forbid this (unlike the
    investigation agent's, which explicitly does), and even where a prompt
    does forbid it, a long "write me a full report" request routinely gets
    it ignored anyway - confirmed directly: an exported PDF for one such
    report contained the model's own numbered reference list (with
    duplicate entries and no real tie-back to the [n] citation markers
    actually used in the prose) immediately followed by Atelier's own,
    correctly-built one (_references_html below, sourced from the papers
    genuinely linked to this turn) - two different reference lists back to
    back. References is conventionally a document's last section, so
    truncating the text at the heading (rather than trying to locate just
    the list itself) is safe and also drops any trailing sign-off text the
    model appended past it (a word count, a disclaimer paragraph, etc.)
    that has no place in a formatted export either.
    """
    m = _SELF_WRITTEN_REFS_RE.search(markdown_text or "")
    if not m:
        return markdown_text or ""
    return markdown_text[: m.start()].rstrip()


def derive_document_title(markdown_text: str, fallback: str) -> str:
    """
    A proper title for this turn - NOT the user's raw prompt, which for a
    long "write me a report" request can run to several paragraphs (used
    to get dumped wholesale onto the PDF as its title - see
    build_turn_pdf's docstring for the bug this replaced). Prefers the
    answer's own opening heading when it has one, falling back to a short,
    truncated version of the prompt for genres where the model doesn't
    emit one (a plain Q&A answer has no header to pull from).

    Exposed separately from build_turn_pdf so callers building a filename
    (ui/callbacks/library.py's export_pdf) can derive the same title
    without re-implementing the extraction.
    """
    heading, _ = _extract_leading_title(markdown_text)
    return heading or truncate(fallback or "Atelier Report", 100)


_H2_HEADING_RE = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)


def extract_section_headings(markdown_text: str) -> List[str]:
    """
    Every top-level ("##") section heading in a turn's answer, in
    document order - e.g. ["1. Introduction", "2. Hydrological Setting...",
    ...] for a structured report. Used by ui/callbacks/ui_extras.py's
    sidebar table-of-contents (_build_session_nav_items) to link straight
    to a section, matching ids a clientside script assigns to the SAME
    headings at render time (ui/layouts/main.py's assignHeadingIds) purely
    by position (turn-{query_id}-h-{i}) - both sides just walk "##" lines/
    <h2> elements top-to-bottom, so they agree without needing to match on
    heading text itself. Deliberately "##" only, not "###" too - a session
    nav panel listing every subsection of every turn would get unusably
    long and deep for a ~320px sidebar; the top-level sections are the
    useful jump targets.
    """
    return [m.group(1).strip() for m in _H2_HEADING_RE.finditer(markdown_text or "")]


def strip_leading_title(markdown_text: str) -> str:
    """
    Returns markdown_text with its leading heading line removed, if it had
    one - the exact same split derive_document_title uses to pull that
    heading out as the document's title. Exposed so a caller that displays
    derive_document_title's result as its OWN separate heading element
    (ui/layouts/feed.py's build_flow_header on the web page, and
    build_synthesis_body right below it, mirroring build_turn_pdf's
    identical split for the PDF) can avoid rendering that same heading a
    second time as the body's own first line too - confirmed live via
    screenshot: the report's title appearing twice in a row, once as the
    page's H1, immediately followed by the exact same text again as the
    synthesis body's first heading.

    Safe to call unconditionally - if there was no leading heading to
    begin with, this just returns the text unchanged.
    """
    _, body = _extract_leading_title(markdown_text)
    return body


_CITATION_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def _linkify_citations(body_html: str, num_records: int) -> str:
    """
    Turns every bare "[n]" citation marker in the rendered body into a
    colored, clickable link jumping to that paper's entry in the
    References section (#ref-n, anchored by _references_html below) -
    matching the web UI's own citation styling (.ref-badge-inline in
    ui/layouts/main.py: colored + bold), which the PDF's plain black
    bracket text didn't have. Runs on the already-markdown-converted HTML,
    not the raw markdown, so it can't collide with the markdown parser's
    own handling of "[...]"  (link/reference syntax, footnotes, etc.).

    Only "n" values within [1, num_records] are linkified - chat_with_
    literature's own citations are always assigned from that exact range
    (see order_citations_by_appearance), so anything outside it is either
    a table/figure number unrelated to citations or a bracket the model
    invented while writing near a genuine reference - either way, safer
    to leave it as plain text than link it somewhere wrong.
    """
    if num_records <= 0:
        return body_html

    def _sub(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if 1 <= n <= num_records:
            return f'<a class="cite-link" href="#ref-{n}">[{n}]</a>'
        return m.group(0)

    return _CITATION_MARKER_RE.sub(_sub, body_html)


def _references_html(records: List[Any]) -> str:
    if not records:
        return ""
    items = []
    for i, rec in enumerate(records, start=1):
        authors = _get_val(rec, "authors") or []
        author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        title = _get_val(rec, "title") or "Untitled"
        journal = _get_val(rec, "journal") or ""
        year = _get_val(rec, "year") or "n.d."
        doi = _get_val(rec, "doi") or ""
        doi_str = f" doi:{doi}" if doi else ""
        items.append(
            f'<div class="ref-item" id="ref-{i}"><a name="ref-{i}"></a><span class="ref-num">[{i}]</span> '
            f'{author_str} ({year}). {title}. <i>{journal}</i>.{doi_str}</div>'
        )
    return '<div class="references"><h2>References</h2>' + "".join(items) + "</div>"


def build_turn_pdf(title: str, model_name: str, markdown_text: str, records: List[Any] = None) -> bytes:
    """
    Renders one turn as a PDF and returns its raw bytes.

    Args:
        title: the turn's prompt/question - used as a FALLBACK document
            title only, when the answer itself doesn't open with a
            heading to pull a proper title from (see derive_document_title).
            A long "write me a report" prompt used to be dumped onto the
            PDF wholesale as its title, spanning multiple pages before the
            actual report even started - confirmed directly from a live
            export - which is what derive_document_title now replaces.
            The raw prompt itself is no longer shown anywhere in the PDF
            (it used to get its own "Original request" box here too) -
            it's shown on the web page instead (ui/layouts/feed.py's
            build_flow_header), and repeating it in the PDF as well was
            reported as redundant.
        model_name: which LLM generated this answer, shown in the metadata line.
        markdown_text: the raw stored synthesis/answer text (bare [1]/[2]
            bracket citations, same text stored in QueryModel.synthesis -
            NOT the web UI's escaped-for-React-Markdown version, since this
            goes through a completely separate markdown renderer that
            doesn't share dcc.Markdown's auto-linkify quirk).
        records: the cited papers, in the same [1]/[2] order as the text -
            rendered as a numbered References section at the end.

    Returns:
        bytes: the finished PDF's raw content, ready for dcc.send_bytes.
    """
    from xhtml2pdf import pisa  # deferred: this module is only needed here, keeps app startup import-light

    records = records or []
    heading, body_markdown = _extract_leading_title(markdown_text or "")
    doc_title = heading or truncate(title or "Atelier Report", 100)
    body_markdown = _strip_self_written_references(body_markdown)
    cleaned_markdown = _fill_empty_table_cells(body_markdown)
    body_html = md.markdown(cleaned_markdown, extensions=["tables", "fenced_code", "nl2br"])
    body_html = _linkify_citations(body_html, len(records))
    generated_at = datetime.now().strftime("%B %d, %Y")
    escaped_title = html.escape(doc_title)

    # No "Original request" box here (there used to be one) - the raw
    # prompt is now shown on the web page itself instead (see
    # ui/layouts/feed.py's build_flow_header), so repeating it in the PDF
    # too was reported as redundant clutter. The PDF just gets the report
    # itself: title, metadata, body, references.
    full_html = f"""
    <html>
    <head><title>{escaped_title}</title><style>{_CSS}</style></head>
    <body>
        <h1 class="doc-title">{escaped_title}</h1>
        <div class="doc-meta">Generated by Atelier &middot; {model_name} &middot; {generated_at}</div>
        {body_html}
        {_references_html(records)}
    </body>
    </html>
    """

    buffer = io.BytesIO()
    result = pisa.CreatePDF(full_html, dest=buffer)
    if result.err:
        logger.warning(f"PDF generation reported {result.err} error(s) for '{title[:50]}' - returning best-effort output anyway.")
    return buffer.getvalue()
