"""
Database Models Module

Defines the SQLite schemas using Peewee ORM. This module includes the permanent 
AI-extracted parameters and JSON vector storage, effectively turning the local 
database into a searchable knowledge graph.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

from peewee import (
    Model, CharField, TextField, IntegerField, 
    DateTimeField, BooleanField, ForeignKeyField
)
from playhouse.sqlite_ext import JSONField
from .database import db



class BaseModel(Model):
    """Base model class that all other database models inherit from."""
    class Meta:
        database = db


# ==========================================
# 1. DATABASE ORM MODELS (Peewee)
# ==========================================

class SessionModel(BaseModel):
    """
    Represents a single research session initiated by the user.
    """
    id = CharField(primary_key=True)
    topic = TextField()
    summary = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)


class PaperModel(BaseModel):
    """
    The permanent storage for a single academic paper.
    Serves as the core entity for the Atelier local knowledge graph.
    """
    # --- CORE IDENTITY ---
    fingerprint = CharField(unique=True)  # e.g., 'doi:10.1038...' or 'pmid:12345'
    title = TextField()
    abstract = TextField(null=True)
    authors = JSONField(default=list) 
    journal = CharField(null=True)
    year = IntegerField(null=True)
    doi = CharField(null=True)
    pmid = CharField(null=True)
    pmcid = CharField(null=True)
    
    # --- SOURCE LINKS ---
    pdf_url = CharField(null=True)
    full_text_url = CharField(null=True)
    citation_count = IntegerField(default=0)
    
    # --- PERMANENT AI-EXTRACTED DATA ---
    # These fields are generated once by the LLM and saved forever.
    is_relevant = BooleanField(null=True)
    answer = TextField(default="-")
    population = CharField(default="-")
    methods = CharField(default="-")
    results = TextField(default="-")
    outcomes = CharField(default="-")
    sample_size = CharField(default="-")
    study_count = CharField(default="-")
    duration = CharField(default="-")
    country = CharField(default="-")
    
    # --- VECTOR DATA ---
    # Stores SPLADE dictionaries (e.g., {"cell": 1.2, "cancer": 2.4}) for fast local recall
    sparse_vector = JSONField(null=True) 


class SessionPaperLink(BaseModel):
    """
    A Many-to-Many junction table linking a search Session to the Papers it surfaced.
    Tracks the specific relevance score for a paper within the context of a given query.
    """
    session_id = CharField()
    paper_fingerprint = CharField()
    relevance_score = IntegerField(default=0)  # The SPLADE score for THIS specific session

class QueryModel(BaseModel):
    """
    Represents a single prompt/response interaction within a broader Session.
    This builds the actual chat history for the feed view.
    """
    session = ForeignKeyField(SessionModel, backref='queries', on_delete='CASCADE')
    prompt = TextField()
    synthesis = TextField()
    model_used = CharField()
    created_at = DateTimeField(default=datetime.now)

class QueryCitation(BaseModel):
    """
    A Many-to-Many junction table linking a specific Query to the Papers it cited.
    """
    query = ForeignKeyField(QueryModel, backref='citations', on_delete='CASCADE')
    paper_fingerprint = CharField() # Links to PaperModel.fingerprint


# ==========================================
# 2. IN-MEMORY DATACLASS
# ==========================================

@dataclass
class Record:
    """
    A lightweight, in-memory object used to pass paper data fluidly between 
    the Fetchers, the AI Engine, the Database, and the Dash UI.
    """
    # --- Search Engine Identifiers ---
    source: str = ""
    source_id: str = ""
    all_sources: List[str] = field(default_factory=list)
    
    # --- Core Metadata ---
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None

    # --- Standardized IDs ---
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None

    # --- Links ---
    url: str = ""
    pdf_url: Optional[str] = None
    full_text_url: Optional[str] = None
    citation_count: int = 0
    
    study_type: str = ""
    
    # --- AI Extracted Metadata ---
    answer: str = "-"
    population: str = "-"
    methods: str = "-"
    results: str = "-"
    outcomes: str = "-"
    sample_size: str = "-"
    study_count: str = "-"
    duration: str = "-"
    country: str = "-"
    is_relevant: Optional[bool] = None
    
    # --- System & Vector Fields ---
    fingerprint: str = ""
    sparse_vector: Dict[str, float] = field(default_factory=dict)
    _db_id: Optional[int] = None

    raw: dict[str, Any] = field(default_factory=dict)

    def to_model_dict(self) -> dict[str, Any]:
        """Shape aligned to your PaperModel fields."""
        data = asdict(self)
        data.pop("source", None)
        data.pop("source_id", None)
        data.pop("raw", None)
        return data