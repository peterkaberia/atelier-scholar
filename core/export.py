"""
Reference-manager export formats (BibTeX / RIS).

Both formats are universally importable by reference software (Zotero,
EndNote, Mendeley, etc.), so rather than pick one, ui/callbacks/library.py's
export_references callback offers both as separate download buttons.
"""

import re
from typing import Any, Dict, List

from core.utils import clean_text


def _get_val(rec: Any, key: str, default: str = "") -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def _bibtex_key(rec: Any, used_keys: set) -> str:
    """
    Generates a citation key in the common "AuthorYearFirstWord" convention
    (e.g. "Smith2021Metformin"), sanitized to alnum-only and de-duplicated
    against keys already used in this export (appends a/b/c... on collision)
    so a batch of similarly-titled papers never produces two identical keys
    in the same .bib file, which most reference managers reject on import.
    """
    authors = _get_val(rec, "authors") or []
    last_name = re.split(r"[\s,]+", authors[0])[-1] if authors else "Unknown"
    year = _get_val(rec, "year") or "n.d."
    title_word = next((w for w in re.split(r"\s+", _get_val(rec, "title") or "") if len(w) > 3), "")
    base = re.sub(r"[^A-Za-z0-9]", "", f"{last_name}{year}{title_word}") or "Reference"

    key = base
    suffix = "a"
    while key in used_keys:
        key = f"{base}{suffix}"
        suffix = chr(ord(suffix) + 1)
    used_keys.add(key)
    return key


def _escape_bibtex(value: str) -> str:
    return (value or "").replace("{", "\\{").replace("}", "\\}")


def build_bibtex(records: List[Any]) -> str:
    """Builds a .bib file's contents from a list of Record dicts/objects."""
    used_keys: set = set()
    entries = []

    for rec in records:
        key = _bibtex_key(rec, used_keys)
        title = _escape_bibtex(clean_text(_get_val(rec, "title")) or "Untitled")
        authors = _get_val(rec, "authors") or []
        author_field = _escape_bibtex(" and ".join(authors)) if authors else "Unknown"
        journal = _escape_bibtex(clean_text(_get_val(rec, "journal")) or "")
        year = _get_val(rec, "year") or ""
        doi = clean_text(_get_val(rec, "doi")) or ""
        url = clean_text(_get_val(rec, "url")) or clean_text(_get_val(rec, "pdf_url")) or ""

        fields = [f'  title = {{{title}}}', f'  author = {{{author_field}}}']
        if journal:
            fields.append(f'  journal = {{{journal}}}')
        if year:
            fields.append(f'  year = {{{year}}}')
        if doi:
            fields.append(f'  doi = {{{doi}}}')
        if url:
            fields.append(f'  url = {{{url}}}')

        entries.append(f"@article{{{key},\n" + ",\n".join(fields) + "\n}")

    return "\n\n".join(entries) + "\n"


def build_ris(records: List[Any]) -> str:
    """Builds a .ris file's contents from a list of Record dicts/objects."""
    entries = []
    for rec in records:
        lines = ["TY  - JOUR", f"TI  - {clean_text(_get_val(rec, 'title')) or 'Untitled'}"]
        for author in (_get_val(rec, "authors") or []):
            lines.append(f"AU  - {author}")
        journal = clean_text(_get_val(rec, "journal"))
        if journal:
            lines.append(f"JO  - {journal}")
        year = _get_val(rec, "year")
        if year:
            lines.append(f"PY  - {year}")
        doi = clean_text(_get_val(rec, "doi"))
        if doi:
            lines.append(f"DO  - {doi}")
        url = clean_text(_get_val(rec, "url")) or clean_text(_get_val(rec, "pdf_url"))
        if url:
            lines.append(f"UR  - {url}")
        abstract = clean_text(_get_val(rec, "abstract"))
        if abstract:
            lines.append(f"AB  - {abstract}")
        lines.append("ER  - ")
        entries.append("\n".join(lines))

    return "\n\n".join(entries) + "\n"
