"""
Academic Paper Search Engines Module.

This module provides the concrete implementations for querying various external 
academic databases (PubMed, Europe PMC, OpenAlex, Semantic Scholar). 
All engines inherit from a robust BasePaperEngine that handles network 
throttling, retries, and unified JSON/Text fetching.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from database import Record
from core.utils import (
    clean_text,
    make_fingerprint,
    nested_get,
    normalize_doi,
    normalize_pmcid,
    normalize_pmid,
    safe_int,
    split_author_string,
)
from .http_client import HttpClient

logger = logging.getLogger(__name__)


class BasePaperEngine(ABC):
    """
    Abstract base class for all external academic APIs. 
    
    Delegates all network calls to the robust `HttpClient` to ensure 
    global throttling, exponential backoff, retry logic, and consistent headers.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: int = 30,
        client: Optional[HttpClient] = None
    ) -> None:
        """
        Initializes the base engine.

        Args:
            api_key (Optional[str]): The API key for the service, if applicable.
            timeout (int): Network timeout in seconds. Defaults to 30.
        """
        self.api_key = api_key
        
        # Instantiate our custom Client (handles retries, backoff, and timeouts)
        self.client = client or HttpClient(timeout=timeout)
        
        # Standard headers applied to every API call
        self.default_headers = {
            "Accept": "application/json, text/xml, application/xml;q=0.9, */*;q=0.8",
        }

    def _get_json(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Wrapper to merge default headers and execute a JSON GET request.
        """
        merged_headers = {**self.default_headers, **(headers or {})}
        return self.client.get_json(url, params=params, headers=merged_headers)

    def _get_text(self, url: str, *, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> str:
        """
        Wrapper to merge default headers and execute a Text/XML GET request.
        """
        merged_headers = {**self.default_headers, **(headers or {})}
        return self.client.get_text(url, params=params, headers=merged_headers)

    @abstractmethod
    def search(self, query: str, limit: int = 200) -> List[Record]:
        """
        Executes a search against the specific academic database.

        Args:
            query (str): The boolean or keyword search string.
            limit (int): The maximum number of results to return.

        Returns:
            List[Record]: A list of normalized Record objects.
        """
        raise NotImplementedError


# ----------------------------
# PubMed Engine
# ----------------------------

class PubMedEngine(BasePaperEngine):
    """
    Search PubMed via the NCBI E-utilities API.
    Utilizes a two-step process: ESearch to find IDs, and EFetch to pull XML data.
    """
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        *,
        tool: str = "atelier",
        email: str = "dodoma700@gmail.com",
        api_key: Optional[str] = None,
        timeout: int = 30,
        client: Optional[HttpClient] = None
    ) -> None:
        super().__init__(api_key=api_key, timeout=timeout, client=client)
        self.tool = tool
        self.email = email

    def _common_params(self) -> Dict[str, Any]:
        """Builds standard NCBI identification parameters."""
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(self, query: str, limit: int = 200) -> List[Record]:
        logger.info(f"PubMed: Executing ESearch for query: '{query[:50]}...'")
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": min(limit, 10000),
        }
        
        payload = self._get_json(self.ESEARCH_URL, params=params)
        pmids = payload.get("esearchresult", {}).get("idlist", [])
        
        if not pmids:
            logger.info("PubMed: No PMIDs found for query.")
            return []
            
        logger.info(f"PubMed: Found {len(pmids)} PMIDs. Fetching XML metadata...")
        return self.fetch_by_pmids(pmids)

    def fetch_by_pmids(self, pmids: Iterable[str]) -> List[Record]:
        pmids = [str(x) for x in pmids if x]
        if not pmids: 
            return []

        params = {
            **self._common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        xml_text = self._get_text(self.EFETCH_URL, params=params)
        return self._parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml(self, xml_text: str) -> List[Record]:
        """Parses the deeply nested NCBI PubMed XML format into Record objects."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"PubMed XML Parse Error: {e}")
            return []

        out: List[Record] = []

        for rec in root.findall("PubmedArticle"):
            medline = rec.find("MedlineCitation")
            article = rec.find("./MedlineCitation/Article")
            pubmed_data = rec.find("PubmedData")

            if medline is None or article is None: 
                continue

            pmid = self._text(medline.find("PMID"))
            title = self._text(article.find("ArticleTitle"))
            
            if not title: 
                logger.debug(f"PubMed: Skipping record {pmid} due to missing title.")
                continue

            abstract = self._parse_abstract(article)
            authors = self._parse_authors(article)
            journal = (self._text(article.find("./Journal/Title")) or 
                       self._text(article.find("./Journal/ISOAbbreviation")))
            year = self._parse_year(article.find("./Journal/JournalIssue/PubDate"))

            doi = None
            for eloc in article.findall("ELocationID"):
                if eloc.attrib.get("EIdType") == "doi":
                    doi = normalize_doi(self._text(eloc))
                    if doi: break

            pmcid = None
            if pubmed_data is not None:
                for aid in pubmed_data.findall("./ArticleIdList/ArticleId"):
                    id_type = aid.attrib.get("IdType", "").lower()
                    value = self._text(aid)
                    if id_type == "doi" and not doi:
                        doi = normalize_doi(value)
                    elif id_type in {"pmc", "pmcid"}:
                        pmcid = normalize_pmcid(value)

            try:
                fingerprint = make_fingerprint(doi=doi, pmid=pmid, pmcid=pmcid, arxiv_id=None, fallback_source="pubmed", fallback_id=pmid)
                out.append(Record(
                    source="pubmed", source_id=pmid, fingerprint=fingerprint, title=title,
                    abstract=abstract, authors=authors, journal=journal, year=year,
                    doi=doi, pmid=normalize_pmid(pmid), pmcid=pmcid, raw={}
                ))
            except ValueError as e:
                logger.debug(f"PubMed: Could not generate fingerprint for record: {e}")

        return out

    @staticmethod
    def _text(el: Optional[ET.Element]) -> Optional[str]:
        if el is None: return None
        return clean_text("".join(el.itertext()))

    def _parse_abstract(self, article: ET.Element) -> Optional[str]:
        parts: List[str] = []
        for ab in article.findall("./Abstract/AbstractText"):
            label = ab.attrib.get("Label")
            txt = self._text(ab)
            if txt: 
                parts.append(f"{label}: {txt}" if label else txt)
        return "\n\n".join(parts) if parts else None

    def _parse_authors(self, article: ET.Element) -> List[str]:
        authors: List[str] = []
        for author in article.findall("./AuthorList/Author"):
            collective = self._text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue

            last_name = self._text(author.find("LastName"))
            fore_name = self._text(author.find("ForeName"))
            initials = self._text(author.find("Initials"))

            if fore_name and last_name: authors.append(f"{fore_name} {last_name}")
            elif initials and last_name: authors.append(f"{initials} {last_name}")
            elif last_name: authors.append(last_name)
        return authors

    def _parse_year(self, pubdate: Optional[ET.Element]) -> Optional[int]:
        if pubdate is None: return None
        year = self._text(pubdate.find("Year"))
        if year and year.isdigit(): return int(year)
        medline_date = self._text(pubdate.find("MedlineDate"))
        return safe_int(medline_date)


# ----------------------------
# Europe PMC Engine
# ----------------------------

class EuropePMCEngine(BasePaperEngine):
    """
    Search Europe PMC via their RESTful JSON API.
    Excellent for retrieving open access PDF URLs and PMCIDs.
    """
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(self, query: str, limit: int = 200, result_type: str = "core") -> List[Record]:
        logger.info(f"EuropePMC: Searching for '{query[:50]}...'")
        params = {
            "query": query, 
            "format": "json", 
            "pageSize": min(limit, 1000), 
            "resultType": result_type,
        }
        
        payload = self._get_json(self.BASE_URL, params=params)
        items = payload.get("resultList", {}).get("result", [])
        
        if not items:
            logger.info("EuropePMC: No results found.")
            
        return [self._normalize_item(item) for item in items if item.get("title")]

    def _normalize_item(self, item: Dict[str, Any]) -> Record:
        doi = normalize_doi(item.get("doi"))
        pmid = normalize_pmid(item.get("pmid") or (item.get("id") if item.get("source") == "MED" else None))
        pmcid = normalize_pmcid(item.get("pmcid"))

        full_text_url = None
        pdf_url = None

        ft_list = nested_get(item, "fullTextUrlList", "fullTextUrl") or []
        if isinstance(ft_list, list):
            for ft in ft_list:
                url = ft.get("url")
                if not full_text_url and url: 
                    full_text_url = url
                if ft.get("documentStyle", "").lower() == "pdf" and url: 
                    pdf_url = url

        fingerprint = make_fingerprint(doi=doi, pmid=pmid, pmcid=pmcid, arxiv_id=None, fallback_source="europepmc", fallback_id=item.get("id"))
        
        return Record(
            source="europepmc", source_id=clean_text(item.get("id")), fingerprint=fingerprint,
            title=item["title"].strip(), abstract=clean_text(item.get("abstractText")),
            authors=split_author_string(item.get("authorString")),
            journal=clean_text(item.get("journalTitle") or item.get("bookTitle")),
            year=safe_int(item.get("pubYear") or item.get("firstPublicationDate")),
            doi=doi, pmid=pmid, pmcid=pmcid, pdf_url=pdf_url, full_text_url=full_text_url,
            citation_count=safe_int(item.get("citedByCount")) or 0, raw=item
        )


# ----------------------------
# OpenAlex Engine
# ----------------------------

class OpenAlexEngine(BasePaperEngine):
    """
    Search OpenAlex API. Provides broad coverage across multiple disciplines.
    """
    BASE_URL = "https://api.openalex.org/works"

    def search(self, query: str, limit: int = 200) -> List[Record]:
        logger.info(f"OpenAlex: Searching for '{query[:50]}...'")
        params = {"search": query, "per_page": min(limit, 100)}
        if self.api_key: 
            params["api_key"] = self.api_key

        payload = self._get_json(self.BASE_URL, params=params)
        
        if not payload: 
            logger.warning("OpenAlex: API returned empty payload.")
            return []
        
        items = payload.get("results", [])
        return [self._normalize_item(item) for item in items if item.get("display_name") or item.get("title")]

    def _normalize_item(self, item: Dict[str, Any]) -> Record:
        ids = item.get("ids", {}) or {}
        doi = normalize_doi(ids.get("doi") or item.get("doi"))
        pmid = normalize_pmid(ids.get("pmid") or item.get("pmid"))
        pmcid = normalize_pmcid(ids.get("pmcid") or item.get("pmcid"))

        authors = [nested_get(a, "author", "display_name") for a in item.get("authorships", []) or [] if nested_get(a, "author", "display_name")]
        journal = (nested_get(item, "primary_location", "source", "display_name") or 
                   nested_get(item, "host_venue", "display_name"))
        landing_url = (nested_get(item, "best_oa_location", "landing_page_url") or 
                       nested_get(item, "primary_location", "landing_page_url") or 
                       nested_get(item, "open_access", "oa_url"))
        pdf_url = (nested_get(item, "best_oa_location", "pdf_url") or 
                   nested_get(item, "primary_location", "pdf_url"))

        fingerprint = make_fingerprint(doi=doi, pmid=pmid, pmcid=pmcid, arxiv_id=None, fallback_source="openalex", fallback_id=item.get("id"))

        return Record(
            source="openalex", source_id=clean_text(item.get("id")), fingerprint=fingerprint,
            title=clean_text(item.get("display_name") or item.get("title")) or "",
            abstract=None, authors=authors, journal=clean_text(journal),
            year=safe_int(item.get("publication_year")), doi=doi, pmid=pmid, pmcid=pmcid,
            pdf_url=clean_text(pdf_url), full_text_url=clean_text(landing_url),
            citation_count=safe_int(item.get("cited_by_count")) or 0, raw=item
        )


# ----------------------------
# Semantic Scholar Engine
# ----------------------------

class SemanticScholarEngine(BasePaperEngine):
    """
    Search Semantic Scholar Graph API.
    Highly effective for exact literature search and citation counts.
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: int = 30,
        mode: str = "relevance",  # "relevance" or "bulk"
        client: Optional[HttpClient] = None
    ) -> None:
        super().__init__(api_key=api_key, timeout=timeout, client=client)
        self.mode = mode

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def search(self, query: str, limit: int = 200) -> List[Record]:
        logger.info(f"SemanticScholar: Searching for '{query[:50]}...'")
        if self.mode == "bulk":
            return self._search_bulk(query=query, limit=limit)
        return self._search_relevance(query=query, limit=limit)

    def _search_relevance(self, query: str, limit: int) -> List[Record]:
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query, 
            "limit": min(limit, 100),
            "fields": ",".join(["title", "abstract", "year", "venue", "url", "authors", 
                                "externalIds", "openAccessPdf", "citationCount", 
                                "publicationDate", "publicationTypes"]),
        }
        
        payload = self._get_json(url, params=params, headers=self._headers())
        if not payload: 
            logger.warning("SemanticScholar: API returned empty payload (Possible throttle limit).")
            return []
        
        items = payload.get("data", [])
        return [self._normalize_item(item) for item in items if item.get("title")]

    def _search_bulk(self, query: str, limit: int) -> List[Record]:
        url = f"{self.BASE_URL}/paper/search/bulk"
        params = {
            "query": query,
            "fields": ",".join(["title", "year", "venue", "url", "authors", 
                                "externalIds", "openAccessPdf", "citationCount"]),
        }
        
        payload = self._get_json(url, params=params, headers=self._headers())
        if not payload: 
            logger.warning("SemanticScholar: API returned empty payload (Possible throttle limit).")
            return []
        
        items = (payload.get("data") or [])[:limit]
        return [self._normalize_item(item) for item in items if item.get("title")]

    def _normalize_item(self, item: Dict[str, Any]) -> Record:
        ext = item.get("externalIds", {}) or {}
        doi = normalize_doi(ext.get("DOI"))
        arxiv_id = ext.get("ArXiv")
        pmid = normalize_pmid(ext.get("PubMed") or ext.get("PMID"))
        pmcid = normalize_pmcid(ext.get("PubMedCentral") or ext.get("PMCID"))

        pdf_url = nested_get(item, "openAccessPdf", "url")
        full_text_url = pdf_url or item.get("url")

        source_id = clean_text(item.get("paperId"))
        fingerprint = make_fingerprint(doi=doi, pmid=pmid, pmcid=pmcid, arxiv_id=arxiv_id, fallback_source="semanticscholar", fallback_id=source_id)

        return Record(
            source="semanticscholar", source_id=source_id, fingerprint=fingerprint,
            title=clean_text(item.get("title")) or "", abstract=clean_text(item.get("abstract")),
            authors=[a.get("name") for a in item.get("authors", []) if a.get("name")],
            journal=clean_text(item.get("venue")), year=safe_int(item.get("year") or item.get("publicationDate")),
            doi=doi, pmid=pmid, pmcid=pmcid, pdf_url=clean_text(pdf_url), full_text_url=clean_text(full_text_url),
            citation_count=safe_int(item.get("citationCount")) or 0, raw=item
        )

