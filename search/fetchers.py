"""
Atelier Academic Search Module.

This module houses the AtelierAcademicSearch engine, which orchestrates 
queries across local vector databases and external APIs (PubMed, Europe PMC, 
OpenAlex, Semantic Scholar).
"""

import logging
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

from database.models import Record
from core.config import DEFAULT_SPARSE_MODEL, resolve_key
from .sparse_encoder import encode_query_sparse
from .paper import (
    ArxivEngine,
    CrossRefEngine,
    EuropePMCEngine,
    OpenAlexEngine,
    PubMedEngine,
    SemanticScholarEngine,
)
from .http_client import HttpClient

logger = logging.getLogger(__name__)


class AtelierAcademicSearch:
    """
    The unified search manager called by the Orchestrator.
    
    Routes AI-generated boolean queries to the precise OOP engines and
    seamlessly maps their distinct outputs back to the application's core 
    Database Record model.
    """

    # Human-readable labels, used for progress reporting and warnings.
    ENGINE_LABELS = {
        "pubmed": "PubMed",
        "europe_pmc": "Europe PMC",
        "openalex": "OpenAlex",
        "semantic_scholar": "Semantic Scholar",
        "crossref": "Crossref",
        "arxiv": "arXiv",
    }

    def __init__(self, db_repository: Any = None, max_results_per_source: int = 40):
        """
        Initializes the comprehensive search manager.

        Args:
            db_repository (Any, optional): The database repository for local vector search. Defaults to None.
            max_results_per_source (int, optional): The limit of papers to retrieve per engine.
                Defaults to 40 - safe to keep generous now that engines run
                in parallel (see run_comprehensive_search): a bigger raw
                candidate pool costs no extra wall-clock time and gives the
                SPLADE + heuristic reranker (search/sparse_encoder.py) more
                to choose the true top papers from, instead of only being
                able to pick among whatever each engine's own default
                ranking happened to put in its first 20 hits.
        """
        self.db_repo = db_repository
        self.max_results = max_results_per_source
        # Populated by the most recent run_comprehensive_search() call: the
        # human-readable labels of any engine that errored out, so callers
        # (pipeline/nodes.py -> ui/callbacks/search.py) can tell the user
        # "results are missing a source" instead of silently under-reporting.
        self.last_warnings: List[str] = []

        pubmed_client = HttpClient(timeout=25, request_per_second=2.5)
        epmc_client = HttpClient(timeout=25, request_per_second=4.0)
        openalex_client = HttpClient(timeout=25, request_per_second=4.0)
        s2_client = HttpClient(timeout=25, request_per_second=2.0)
        crossref_client = HttpClient(timeout=25, request_per_second=4.0)
        # arXiv's API terms of use ask for no more than one request every 3
        # seconds - only one call happens per search run so this throttle
        # never actually delays anything in practice, but it's the correct
        # limit to declare for a good API citizen.
        arxiv_client = HttpClient(timeout=25, request_per_second=1 / 3)

        # Initialize the OOP API engines with their respective keys, resolved
        # fresh each time (Settings UI takes priority over .env) - safe here
        # since AtelierAcademicSearch is instantiated per pipeline run.
        self.engines = {
            "pubmed": PubMedEngine(client=pubmed_client),
            "europe_pmc": EuropePMCEngine(client=epmc_client),
            "openalex": OpenAlexEngine(api_key=resolve_key("OPENALEX_API_KEY"), client=openalex_client),
            "semantic_scholar": SemanticScholarEngine(api_key=resolve_key("SEMANTIC_SCHOLAR_API_KEY"), client=s2_client),
            "crossref": CrossRefEngine(client=crossref_client),
            "arxiv": ArxivEngine(client=arxiv_client),
        }

    def _convert_to_db_record(self, paper: Any) -> Record:
        """
        Converts an engine-specific PaperRecord into the standard Database Record.

        Args:
            paper (Any): The PaperRecord object returned by a specific engine.

        Returns:
            Record: The standardized application database record.
        """
        return Record(
            source=paper.source,
            source_id=paper.source_id or "",
            all_sources=[paper.source],
            title=paper.title,
            abstract=paper.abstract or "",
            year=paper.year,
            journal=paper.journal or "",
            authors=paper.authors,
            doi=paper.doi or "",
            pmid=paper.pmid or "",
            pmcid=paper.pmcid or "",
            pdf_url=paper.pdf_url or "",
            full_text_url=paper.full_text_url or "",
            citation_count=paper.citation_count,
            fingerprint=paper.fingerprint
        )

    def fetch_full_text_from_epmc(self, pmcid: str) -> str:
        """
        Dedicated method for the Orchestrator to download XML full text from Europe PMC.

        Args:
            pmcid (str): The PubMed Central ID (e.g., 'PMC1234567').

        Returns:
            str: The parsed full text as a string, or an empty string if it fails.
        """
        if not pmcid:
            return ""

        try:
            # Borrow the configured HTTP client from the Europe PMC engine
            target_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            xml_text = self.engines["europe_pmc"].client.get_text(target_url)

            if not xml_text:
                return ""

            root = ET.fromstring(xml_text)
            return "\n\n".join(["".join(p.itertext()).strip() for p in root.findall(".//p") if "".join(p.itertext()).strip()])

        except Exception as e:
            logger.warning(f"Failed to fetch XML full text for {pmcid}: {e}")
            return ""

    def fetch_full_text_from_pdf(self, pdf_url: str) -> str:
        """
        Downloads an open-access PDF and extracts its text via pypdf.

        Args:
            pdf_url (str): A direct URL to a PDF file.

        Returns:
            str: The extracted full text, or an empty string if it fails.
        """
        if not pdf_url:
            return ""

        try:
            import io
            from pypdf import PdfReader

            # Borrow any already-configured, throttled HTTP client.
            client = self.engines["europe_pmc"].client
            response = client.session.get(pdf_url, timeout=client.timeout)
            response.raise_for_status()

            reader = PdfReader(io.BytesIO(response.content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p.strip() for p in pages if p.strip())

        except Exception as e:
            logger.warning(f"Failed to fetch/parse PDF full text from {pdf_url}: {e}")
            return ""

    def get_full_text(self, *, pmcid: str = "", pdf_url: str = "", full_text_url: str = "") -> str:
        """
        Best-effort full text retrieval for a paper: tries structured Europe
        PMC XML first (cleaner extraction), then a direct PDF, then
        full_text_url as a last resort.

        Args:
            pmcid (str): The paper's PubMed Central ID, if known.
            pdf_url (str): A direct PDF URL, if known.
            full_text_url (str): A general "read the full text here" link
                (e.g. OpenAlex's landing_page_url) - often an HTML article
                page rather than a direct file, so this is tried last and
                via the same PDF-extraction path as pdf_url: some sources
                do put a working direct PDF/XML link here even when
                pdf_url itself is empty, and a genuine landing page just
                fails gracefully (fetch_full_text_from_pdf already
                degrades to "" on a non-PDF response) rather than raising.

        Returns:
            str: The full text, or an empty string if no source worked.
        """
        if pmcid:
            text = self.fetch_full_text_from_epmc(pmcid)
            if text:
                return text
        if pdf_url:
            text = self.fetch_full_text_from_pdf(pdf_url)
            if text:
                return text
        if full_text_url:
            text = self.fetch_full_text_from_pdf(full_text_url)
            if text:
                return text
        return ""

    def run_comprehensive_search(self, plan: Dict[str, str], current_topic: str, progress_callback=None) -> List[Record]:
        """
        Executes the search plan across the local DB and external APIs.

        The external engines are queried CONCURRENTLY (one thread per engine)
        rather than one after another - they hit entirely different hosts
        with independent per-client rate limiters, so there's no reason a
        slow PubMed round-trip should hold up Europe PMC/OpenAlex/etc. This
        cuts total wait time to roughly the slowest single engine instead of
        the sum of all of them.

        Args:
            plan (Dict[str, str]): The JSON boolean query plan generated by the LLM.
            current_topic (str): The user's original natural language topic.
            progress_callback (Optional[Callable[[str], None]]): If given,
                called once before the parallel fetch starts, then again as
                each engine finishes (success or failure) - this is the
                specific "10 seconds of silence while it searches" gap the
                per-node graph.stream() progress can't see into on its own,
                since all engines run inside this one node.

        Returns:
            List[Record]: A deduplicated list of standardized Database Records.
                Check self.last_warnings afterwards for any engine that
                failed - a failure here never raises, it just means that
                source's results are missing from the returned list.
        """
        results: List[Record] = []
        seen: set[str] = set()
        self.last_warnings = []

        # 1. Fetch from Local Database first (Costs 0 API calls/tokens!)
        # Encode the topic into a real SPLADE sparse vector before handing it
        # to the repository - passing the raw string here used to silently
        # fall through to search_papers_by_vector's crude SQL-substring
        # fallback (isinstance(query_data, str)) instead of ever reaching its
        # actual vector dot-product search, so a session's previously-scored
        # papers were never recalled by genuine semantic similarity, only by
        # an exact substring match. This is what makes overlapping/repeat
        # topics noticeably faster: previously-seen papers surface instantly
        # from local vectors instead of needing another external API round
        # trip + a fresh model embedding in compute_scores.
        if self.db_repo:
            logger.info("Checking local database for previously stored vectors...")
            try:
                topic_vector = encode_query_sparse(current_topic, DEFAULT_SPARSE_MODEL)
            except Exception as e:
                logger.warning(f"Local vector search encoding failed, skipping local recall: {e}")
                topic_vector = None

            if topic_vector:
                local_records = self.db_repo.search_papers_by_vector(topic_vector)
                for rec in local_records:
                    if rec.fingerprint not in seen:
                        seen.add(rec.fingerprint)
                        results.append(rec)

        # 2. Map AI Plan to the appropriate engines, dropping any the LLM
        # left blank/"null" (e.g. a source it judged not useful for this topic).
        query_map = {
            "pubmed": plan.get("pubmed_query"),
            "europe_pmc": plan.get("europe_pmc_query"),
            "openalex": plan.get("openalex_query"),
            "semantic_scholar": plan.get("semantic_scholar_query"),
            "crossref": plan.get("crossref_query"),
            "arxiv": plan.get("arxiv_query"),
        }
        active = {
            name: query for name, query in query_map.items()
            if query and query.lower() != "null" and name in self.engines
        }

        if not active:
            logger.info("Comprehensive search completed. Total unique records: %d", len(results))
            return results

        if progress_callback:
            labels = ", ".join(self.ENGINE_LABELS.get(n, n) for n in active)
            progress_callback(f"Searching {len(active)} sources in parallel: {labels}...")

        # 3. Execute external API engines concurrently
        logger.info(f"Executing {len(active)} external academic engine searches in parallel...")
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures: Dict[Future, str] = {
                pool.submit(self.engines[name].search, query, self.max_results): name
                for name, query in active.items()
            }

            # as_completed() yields whichever engine finishes first - result
            # merging below runs in THIS thread only (never inside a worker),
            # so mutating `results`/`seen` here is safe without extra locking.
            for future in as_completed(futures):
                engine_name = futures[future]
                label = self.ENGINE_LABELS.get(engine_name, engine_name)
                try:
                    raw_papers = future.result()
                    added_count = 0
                    for paper in raw_papers:
                        if paper.fingerprint not in seen:
                            seen.add(paper.fingerprint)
                            results.append(self._convert_to_db_record(paper))
                            added_count += 1

                    logger.info(f"{engine_name.upper()} returned {added_count} new unique records.")
                    if progress_callback:
                        progress_callback(f"{label}: found {added_count} new paper{'s' if added_count != 1 else ''}...")

                except Exception as e:
                    logger.error(f"{engine_name.upper()} fetch failed: {e}")
                    self.last_warnings.append(label)
                    if progress_callback:
                        progress_callback(f"{label} didn't respond - continuing with the other sources...")

        logger.info(f"Comprehensive search completed. Total unique records: {len(results)}")
        return results