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

import io
import logging
import re
from datetime import datetime
from typing import Any, List

import markdown as md

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

_CSS = """
    @page { size: A4; margin: 2.2cm; }
    body { font-family: "Times New Roman", Georgia, serif; font-size: 11pt; color: #1a1a1a; line-height: 1.5; }
    h1.doc-title { font-size: 19pt; margin: 0 0 4pt 0; color: #0F172A; }
    .doc-meta { font-size: 9pt; color: #555; margin-bottom: 18pt; border-bottom: 1pt solid #ccc; padding-bottom: 10pt; }
    h1, h2, h3 { color: #0F172A; margin-top: 16pt; margin-bottom: 6pt; }
    h3 { font-size: 13pt; }
    p { margin: 6pt 0; text-align: justify; }
    table { width: 100%; border-collapse: collapse; margin: 10pt 0; font-size: 9.5pt; }
    th, td { border: 0.5pt solid #999; padding: 5pt 7pt; text-align: left; vertical-align: top; }
    th { background-color: #eef1f7; }
    .references { margin-top: 22pt; border-top: 1pt solid #ccc; padding-top: 10pt; }
    .references h2 { font-size: 13pt; }
    .ref-item { font-size: 9.5pt; margin: 6pt 0; }
    .ref-num { font-weight: bold; }
"""


def _get_val(rec: Any, key: str, default: str = "") -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


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
            f'<div class="ref-item"><span class="ref-num">[{i}]</span> '
            f'{author_str} ({year}). {title}. <i>{journal}</i>.{doi_str}</div>'
        )
    return '<div class="references"><h2>References</h2>' + "".join(items) + "</div>"


def build_turn_pdf(title: str, model_name: str, markdown_text: str, records: List[Any] = None) -> bytes:
    """
    Renders one turn as a PDF and returns its raw bytes.

    Args:
        title: the turn's prompt/question - used as the document title.
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

    cleaned_markdown = _fill_empty_table_cells(markdown_text or "")
    body_html = md.markdown(cleaned_markdown, extensions=["tables", "fenced_code", "nl2br"])
    generated_at = datetime.now().strftime("%B %d, %Y")

    full_html = f"""
    <html>
    <head><style>{_CSS}</style></head>
    <body>
        <h1 class="doc-title">{title}</h1>
        <div class="doc-meta">Generated by Atelier &middot; {model_name} &middot; {generated_at}</div>
        {body_html}
        {_references_html(records or [])}
    </body>
    </html>
    """

    buffer = io.BytesIO()
    result = pisa.CreatePDF(full_html, dest=buffer)
    if result.err:
        logger.warning(f"PDF generation reported {result.err} error(s) for '{title[:50]}' - returning best-effort output anyway.")
    return buffer.getvalue()
