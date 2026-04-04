"""
Database Repository Module

Handles all interactions with the local SQLite database. This class abstracts 
away Peewee ORM logic, providing a clean, method-based API for the Orchestrator 
to save records, retrieve top scored papers, and perform local vector searches.
"""

import logging
from typing import List, Dict, Any

from .database import db
from .models import QueryCitation, QueryModel, SessionModel, PaperModel, SessionPaperLink, Record

logger = logging.getLogger(__name__)


class AtelierRepository:
    """
    A static repository class managing all database CRUD operations and 
    local vector similarity searches for the Atelier application.
    """

    @staticmethod
    def initialize_db() -> None:
        """
        Initializes the SQLite database connection and creates tables 
        safely if they do not already exist.
        """
        db.connect()
        db.create_tables([SessionModel, PaperModel, SessionPaperLink, QueryModel, QueryCitation], safe=True)
        db.close()
        logger.info("Local SQLite database initialized successfully.")

    # ==========================================
    # 1. SESSION MANAGEMENT
    # ==========================================
    
    @staticmethod
    def session_exists(session_id: str) -> bool:
        """
        Checks if a search session already exists in the database.
        
        Args:
            session_id (str): The unique ID of the session.
            
        Returns:
            bool: True if the session exists, False otherwise.
        """
        return SessionModel.select().where(SessionModel.id == session_id).exists()

    @staticmethod
    def create_session(session_id: str, topic: str, summary: str = "") -> None:
        """
        Creates a new search session to track user history and linked papers.
        
        Args:
            session_id (str): The unique ID of the session.
            topic (str): The overarching research question.
            summary (str): An optional summary of the session findings.
        """
        SessionModel.create(id=session_id, topic=topic, summary=summary)
        logger.info(f"Created new database session: {session_id}")

    # ==========================================
    # 2. BULK SAVING & FINGERPRINTING
    # ==========================================

    @staticmethod
    def save_bulk_records(session_id: str, records: List[Record]) -> None:
        """
        Saves raw papers to the database securely. If a paper already exists 
        (matched by unique fingerprint), it gracefully updates any missing links 
        but DOES NOT overwrite existing AI-extracted permanent data.
        
        Args:
            session_id (str): The active search session linking these papers.
            records (List[Record]): The list of newly fetched Record objects.
        """
        for rec in records:
            fp = rec.fingerprint 
            if not fp: 
                continue

            # 1. Insert or gently Update the Paper globally
            paper, created = PaperModel.get_or_create(
                fingerprint=fp,
                defaults={
                    'title': rec.title,
                    'abstract': rec.abstract,
                    'authors': rec.authors,
                    'journal': rec.journal,
                    'year': rec.year,
                    'doi': rec.doi,
                    'pmid': rec.pmid,
                    'pmcid': rec.pmcid,
                    'pdf_url': rec.pdf_url,
                    'full_text_url': rec.full_text_url,
                    'citation_count': rec.citation_count,
                    'sparse_vector': rec.sparse_vector
                }
            )

            # If it wasn't created, update any missing URLs or Vectors
            if not created:
                updated = False
                if not paper.pdf_url and rec.pdf_url: 
                    paper.pdf_url = rec.pdf_url
                    updated = True
                if not paper.full_text_url and rec.full_text_url: 
                    paper.full_text_url = rec.full_text_url
                    updated = True
                if not paper.pmcid and rec.pmcid: 
                    paper.pmcid = rec.pmcid
                    updated = True
                if not paper.sparse_vector and rec.sparse_vector: 
                    paper.sparse_vector = rec.sparse_vector
                    updated = True
                
                if updated:
                    paper.save()

            # 2. Link this specific paper to the user's current session
            # Note: A paper can be linked to multiple sessions if multiple users search for it
            SessionPaperLink.get_or_create(
                session_id=session_id,
                paper_fingerprint=fp,
                defaults={'relevance_score': getattr(rec, 'step2_score', 0)}
            )

    # ==========================================
    # 3. AI EXTRACTION STORAGE
    # ==========================================

    @staticmethod
    def get_top_unprocessed_records(session_id: str, limit: int = 20) -> List[Record]:
        """
        Retrieves the highest-scored papers for a session, actively prioritizing 
        papers that have Open Access full text available (via PMCID or URLs).
        
        Converts the DB Models back into the lightweight Record dataclass.
        
        Args:
            session_id (str): The active search session.
            limit (int): The maximum number of papers to retrieve.
            
        Returns:
            List[Record]: The top-scored papers ready for LLM extraction.
        """
        query = (PaperModel
                 .select(PaperModel, SessionPaperLink.relevance_score)
                 .join(SessionPaperLink, on=(PaperModel.fingerprint == SessionPaperLink.paper_fingerprint))
                 .where(SessionPaperLink.session_id == session_id)
                 # Force the DB to only return papers we can effectively analyze
                 .where(
                     PaperModel.pmcid.is_null(False) | 
                     PaperModel.full_text_url.is_null(False) | 
                     PaperModel.pdf_url.is_null(False) |
                     PaperModel.abstract.is_null(False)
                 )
                 .order_by(SessionPaperLink.relevance_score.desc())
                 .limit(limit))

        out = []
        for row in query:
            rec = Record(
                fingerprint=row.fingerprint,
                title=row.title,
                abstract=row.abstract,
                year=row.year,
                journal=row.journal,
                doi=row.doi,
                pmid=row.pmid,
                pmcid=row.pmcid,
                pdf_url=row.pdf_url,
                full_text_url=row.full_text_url,
                authors=row.authors,
                citation_count=row.citation_count,
                
                # PRE-LOAD PERMANENT AI FIELDS (Prevents duplicate LLM calls)
                answer=row.answer,
                population=row.population,
                methods=row.methods,
                results=row.results,
                outcomes=row.outcomes,
                sample_size=row.sample_size,
                duration=row.duration,
                country=row.country,
                is_relevant=row.is_relevant,
                
                _db_id=row.id
            )
            out.append(rec)
            
        return out

    @staticmethod
    def update_record_ai_data(session_id: str, pmid: str, ai_data: dict) -> None:
        """
        Permanently saves the LLM's expensive extraction work to the database.
        These parameters are immutable facts about the paper.
        
        Args:
            session_id (str): The active session (used for logging context).
            pmid (str): The PubMed ID (or source ID) of the paper.
            ai_data (dict): The extracted JSON metadata from the AI Engine.
        """
        try:
            # Locate the paper in the database
            paper = PaperModel.get(PaperModel.pmid == str(pmid))
            
            # Map the JSON schema to the database columns
            paper.is_relevant = ai_data.get('is_relevant')
            paper.answer = str(ai_data.get('answer', '-'))
            paper.population = str(ai_data.get('population', '-'))
            paper.methods = str(ai_data.get('methods', '-'))
            paper.results = str(ai_data.get('results', '-'))
            paper.outcomes = str(ai_data.get('outcomes', '-'))
            paper.sample_size = str(ai_data.get('sample_size', '-'))
            paper.study_count = str(ai_data.get('study_count', '-'))
            paper.duration = str(ai_data.get('duration', '-'))
            paper.country = str(ai_data.get('country', '-'))
            
            paper.save()
            logger.info(f"Permanently saved AI metadata for paper: {paper.title[:30]}...")
            
        except PaperModel.DoesNotExist:
            logger.warning(f"Could not save AI data: Paper with PMID {pmid} not found in DB.")

    # ==========================================
    # 4. VECTOR SEARCH CLIENT
    # ==========================================

    @staticmethod
    def search_papers_by_vector(query_data: Any, limit: int = 50) -> List[Record]:
        """
        Turns the local database into a vector search client. 
        Computes the dot-product similarity between the incoming SPLADE query vector 
        and all previously saved paper vectors to instantly recall known literature.
        
        Args:
            query_sparse_vector (Dict[str, float]): The neural representation of the user's query.
            limit (int): Max papers to return.
            
        Returns:
            List[Record]: Highly relevant papers pulled directly from local memory.
        """
        scored_papers = []

        if isinstance(query_data, str):
            # --- FALLBACK: Standard SQLite Text Match ---
            query = PaperModel.select().where(
                PaperModel.title.contains(query_data) | 
                PaperModel.abstract.contains(query_data)
            ).limit(limit)
            
            for row in query:
                scored_papers.append((1.0, row)) # Dummy score for exact text matches
                
        elif isinstance(query_data, dict):
            # --- ADVANCED: Sparse Vector Dot-Product Search ---
            query = PaperModel.select().where(PaperModel.sparse_vector.is_null(False))
            for paper in query:
                paper_vec = paper.sparse_vector
                score = sum(weight * paper_vec.get(token, 0) for token, weight in query_data.items())
                
                if score > 0.5: 
                    scored_papers.append((score, paper))

        # Sort by highest score
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        out = []
        for score, row in scored_papers[:limit]:
            rec = Record(
                fingerprint=row.fingerprint, title=row.title, abstract=row.abstract,
                year=row.year, journal=row.journal, doi=row.doi,
                pmid=row.pmid, pmcid=row.pmcid, pdf_url=row.pdf_url,
                full_text_url=row.full_text_url, authors=row.authors,
                citation_count=row.citation_count, answer=row.answer,
                population=row.population, methods=row.methods, results=row.results,
                outcomes=row.outcomes, sample_size=row.sample_size,
                duration=row.duration, country=row.country, is_relevant=row.is_relevant,
                _db_id=row.id
            )
            rec.all_sources = ["atelier_db"] 
            out.append(rec)
            
        return out
    
    @staticmethod
    def get_all_sessions_for_history() -> List[Dict[str, Any]]:
        """
        Retrieves all search sessions ordered by most recent first.
        Used to populate the sidebar history UI.
        
        Returns:
            List[Dict]: A list of session dictionaries.
        """
        try:
            # Fetch all sessions, newest first
            query = SessionModel.select().order_by(SessionModel.created_at.desc())
            
            out = []
            for session in query:
                out.append({
                    "session_id": session.id,
                    "topic": session.topic,
                    "summary": session.summary,
                    "created_at": session.created_at.strftime("%Y-%m-%d %H:%M:%S") if session.created_at else ""
                })
            return out
        except Exception as e:
            logger.error(f"Failed to fetch session history: {e}")
            return []

    @staticmethod
    def get_session_chat_history(session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the exact chat history for a session, formatting it perfectly 
        for the UI feed layout.
        """
        try:
            # Fetch all queries for this session, oldest first (chronological order)
            queries = QueryModel.select().where(QueryModel.session_id == session_id).order_by(QueryModel.created_at.asc())
            
            history = []
            for q in queries:
                # Find the actual paper data for the citations used in this query
                citation_links = QueryCitation.select().where(QueryCitation.query == q)
                fingerprints = [link.paper_fingerprint for link in citation_links]
                
                cited_papers = []
                if fingerprints:
                    papers = PaperModel.select().where(PaperModel.fingerprint.in_(fingerprints))
                    # Convert Peewee models back to UI-friendly Records
                    for p in papers:
                        rec = Record(
                            fingerprint=p.fingerprint, title=p.title, abstract=p.abstract,
                            year=p.year, journal=p.journal, authors=p.authors,
                            answer=p.answer, population=p.population, methods=p.methods, 
                            sample_size=p.sample_size, outcomes=p.outcomes, pdf_url=p.pdf_url,
                            citation_count=p.citation_count
                        )
                        cited_papers.append(rec.__dict__)
                
                # Pack it into the format `feed.py` expects
                history.append({
                    "prompt": q.prompt,
                    "model_used": q.model_used,
                    "synthesis": q.synthesis,
                    "citations": cited_papers
                })
                
            return history
            
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return []  

    @staticmethod
    def save_query_with_citations(session_id: str, prompt: str, synthesis: str, model_used: str, cited_records: List[Record]):
        """
        Saves a single chat turn and links the specific papers used to generate the answer.
        """
        try:
            # 1. Save the chat message
            session = SessionModel.get(SessionModel.id == session_id)
            query = QueryModel.create(
                session=session,
                prompt=prompt,
                synthesis=synthesis,
                model_used=model_used
            )
            
            # 2. Link the citations used in this specific message
            for rec in cited_records:
                fingerprint = rec.get("fingerprint") if isinstance(rec, dict) else getattr(rec, "fingerprint", None)
                if fingerprint:
                    QueryCitation.create(query=query, paper_fingerprint=fingerprint)
                    
            logger.info(f"Saved query '{prompt[:20]}...' to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save query and citations: {e}")

    @staticmethod
    def update_session_summary(session_id: str, summary: str) -> None:
        """
        Updates the final generated synthesis summary for a specific session.
        This allows the history feed to instantly reload the AI's final conclusion.
        
        Args:
            session_id (str): The unique ID of the session.
            summary (str): The raw markdown generated by the LLM.
        """
        try:
            session = SessionModel.get(SessionModel.id == session_id)
            session.summary = summary
            session.save()
            logger.info(f"Successfully saved AI synthesis summary for session {session_id}")
        except SessionModel.DoesNotExist:
            logger.warning(f"Session {session_id} not found when trying to update summary.")
        except Exception as e:
            logger.error(f"Error updating session summary: {e}")
