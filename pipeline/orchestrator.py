"""
Research Orchestrator Module

This module serves as the central command for the Atelier pipeline. 
It abstracts the complex workflow of AI query planning, parallel API fetching, 
neural scoring, database management, and AI data extraction into a single, 
clean execution method.
"""

import json
import logging
from typing import List, Dict, Any

from core.config import DEFAULT_SPARSE_MODEL
from core.utils import truncate
from database import AtelierRepository
from llm.engine import AtelierAIEngine
from search import AtelierAcademicSearch
from search.sparse_encoder import compute_scores

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """
    Manages the end-to-end pipeline of generating a literature consensus.
    
    This class orchestrates the various subsystems (LLM Engine, Search Fetchers, 
    and Database Repository) to fulfill a user's research request.
    """

    def __init__(self, session_id: str, topic: str, model_choice: str):
        """
        Initializes the Orchestrator and its specialized worker subsystems.

        Args:
            session_id (str): The unique identifier for the current user session.
            topic (str): The natural language research question asked by the user.
            model_choice (str): The LLM requested by the user (e.g., 'gpt-4o').
        """
        self.session_id = session_id
        self.topic = topic
        self.model_choice = model_choice
        
        # Instantiate our specialized workers
        self.ai = AtelierAIEngine(model_choice=self.model_choice)
        self.searcher = AtelierAcademicSearch(db_repository=AtelierRepository)

    def execute_full_search_pipeline(self) -> List[Dict[str, Any]]:
        """
        Executes the entire research workflow: Plan -> Fetch -> Score -> Save -> Extract.
        
        This method checks the local database for previously extracted insights 
        to save API tokens, and only invokes the LLM for new literature.
        
        Returns:
            List[Dict[str, Any]]: A list of cleanly extracted paper dictionaries 
                                  ready to be rendered by the UI.
        """
        logger.info(f"Starting full pipeline for session '{self.session_id}' | Topic: '{self.topic}'")

        # 1. Initialize session in DB if missing
        if not AtelierRepository.session_exists(self.session_id):
            AtelierRepository.create_session(
                session_id=self.session_id, 
                topic=self.topic, 
                summary=""
            )

        # 2. AI generates the optimized boolean queries
        logger.info("Phase 1: AI Query Planning")
        plan = self.ai.plan_topic_queries(self.topic)

        # 3. Fetcher retrieves and merges papers from Local DB + APIs
        logger.info("Phase 2: Comprehensive Database Search")
        merged_records = self.searcher.run_comprehensive_search(plan, current_topic=self.topic)

        if not merged_records:
            logger.warning("Pipeline aborted: No records found across any database.")
            return []

        # 4. Neural Scoring (SPLADE)
        logger.info("Phase 3: Neural Scoring & Re-ranking")
        compute_scores(merged_records, self.topic, DEFAULT_SPARSE_MODEL)

        # 5. Save all raw records to the Database
        logger.info("Phase 4: Persisting raw literature to local database")
        AtelierRepository.save_bulk_records(self.session_id, merged_records)

        # 6. Retrieve the top 20 highest-scored papers for LLM Extraction
        top_20 = AtelierRepository.get_top_unprocessed_records(self.session_id)
        processed_records = []

        # 7. AI Extraction Loop
        logger.info("Phase 5: AI Data Extraction Loop")
        for rec in top_20:
            
            # Skip if there's no abstract
            if not rec.abstract:
                continue
                
            # If the DB search fetched a paper that already has AI extraction data attached, 
            # we skip the LLM call entirely to save time and API costs.
            if getattr(rec, 'answer', None) and rec.answer != "-":
                logger.info(f"Skipping extraction for {rec.pmid}: Data exists in local DB.")
                
                # Convert to UI dictionary format
                rec_dict = rec.__dict__
                rec_dict['_db_id'] = getattr(rec, '_db_id', None)
                processed_records.append(rec_dict)
                continue

            # Otherwise, call the LLM to extract the data
            try:
                # Truncate to save tokens and avoid context limits
                paper_payload = json.dumps({
                    "title": rec.title, 
                    "content": truncate(rec.abstract, 12000)
                })
                
                ai_data = self.ai.extract_paper_data(
                    topic=self.topic, 
                    paper_json_str=paper_payload
                )
                
                # Ensure the AI deemed it highly relevant before including it
                if ai_data.get("is_relevant"):
                    
                    # Save the new AI insights to the database
                    AtelierRepository.update_record_ai_data(self.session_id, rec.pmid, ai_data)
                    
                    # Attach the 1-sentence answer for UI rendering
                    rec.answer = str(ai_data.get("answer", "-"))
                    
                    # Convert to UI dictionary format
                    rec_dict = rec.__dict__
                    rec_dict['_db_id'] = getattr(rec, '_db_id', None)
                    processed_records.append(rec_dict)
                    
            except Exception as e:
                logger.warning(f"Failed to extract AI data for paper {rec.pmid}: {e}")
                pass

        logger.info(f"Pipeline complete. Yielded {len(processed_records)} relevant, extracted papers.")
        return processed_records