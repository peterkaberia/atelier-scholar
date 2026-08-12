"""
Database Models Module

Defines the SQLite schemas using SQLAlchemy's declarative ORM. This module
includes the permanent AI-extracted parameters and JSON vector storage,
effectively turning the local database into a searchable knowledge graph.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from core.utils import normalize_url

Base = declarative_base()


# ==========================================
# 1. DATABASE ORM MODELS (SQLAlchemy)
# ==========================================

class SettingModel(Base):
    """
    A key-value store for user-entered secrets (LLM + academic API keys) set
    via the Settings UI. Values are always encrypted at rest by the
    repository layer (core.secrets) before being written here - this model
    never sees or stores plaintext.
    """
    __tablename__ = "settings"

    key = Column(String, primary_key=True)  # e.g. "OPENAI_API_KEY"
    encrypted_value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.now)


class SessionModel(Base):
    """
    Represents a single research session initiated by the user.
    """
    __tablename__ = "sessions"

    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    id = Column(String, primary_key=True)
    topic = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    # Drives the sidebar/history status icons and lets the feed page tell a
    # still-running search apart from one that's actually done (or crashed)
    # when the user navigates back to it.
    status = Column(String, default=STATUS_IN_PROGRESS, nullable=False)
    # The latest human-readable pipeline stage (e.g. "Extracting findings -
    # paper 5 of 20"). Dash's background-callback set_progress only reaches
    # the ONE browser tab that started the run - nothing else can see it,
    # and it's never persisted anywhere on its own. Mirroring every progress
    # update here is what lets a returning tab (or a second tab, or a
    # reload) show the real current stage instead of a generic "still
    # working" message with no information in it.
    progress_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    queries = relationship("QueryModel", backref="session", cascade="all, delete-orphan")


class PaperModel(Base):
    """
    The permanent storage for a single academic paper.
    Serves as the core entity for the Atelier local knowledge graph.
    """
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # --- CORE IDENTITY ---
    fingerprint = Column(String, unique=True, nullable=False)  # e.g., 'doi:10.1038...' or 'pmid:12345'
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    authors = Column(JSON, default=list)
    journal = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    doi = Column(String, nullable=True)
    pmid = Column(String, nullable=True)
    pmcid = Column(String, nullable=True)

    # --- SOURCE LINKS ---
    pdf_url = Column(String, nullable=True)
    full_text_url = Column(String, nullable=True)
    citation_count = Column(Integer, default=0)

    # --- PERMANENT AI-EXTRACTED DATA ---
    # These fields are generated once by the LLM and saved forever.
    is_relevant = Column(Boolean, nullable=True)
    answer = Column(Text, default="-")
    population = Column(String, default="-")
    methods = Column(String, default="-")
    results = Column(Text, default="-")
    outcomes = Column(String, default="-")
    sample_size = Column(String, default="-")
    study_count = Column(String, default="-")
    duration = Column(String, default="-")
    country = Column(String, default="-")

    # --- VECTOR DATA ---
    # Stores SPLADE dictionaries (e.g., {"cell": 1.2, "cancer": 2.4}) for fast local recall
    sparse_vector = Column(JSON, nullable=True)

    # --- FULL TEXT RAG ---
    # True once we've attempted a full-text fetch (success or failure) so we
    # never retry a failing pdf_url/pmcid lookup on every subsequent run.
    full_text_fetched = Column(Boolean, default=False)


class PaperChunk(Base):
    """
    A passage-level slice of a Paper's full text, independently embedded so
    RAG retrieval can pull the most relevant passages instead of an entire
    (possibly huge) document. Many chunks belong to one Paper, linked by
    fingerprint - matching the plain-string-link convention SessionPaperLink
    already uses rather than a strict ForeignKey.
    """
    __tablename__ = "paper_chunks"
    __table_args__ = (
        UniqueConstraint("paper_fingerprint", "chunk_index", name="uq_paper_chunk"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_fingerprint = Column(String, index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    sparse_vector = Column(JSON, nullable=True)


class SessionPaperLink(Base):
    """
    A Many-to-Many junction table linking a search Session to the Papers it surfaced.
    Tracks the specific relevance score for a paper within the context of a given query.
    """
    __tablename__ = "session_paper_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    paper_fingerprint = Column(String, nullable=False)
    relevance_score = Column(Integer, default=0)  # The SPLADE score for THIS specific session


class QueryModel(Base):
    """
    Represents a single prompt/response interaction within a broader Session.
    This builds the actual chat history for the feed view.
    """
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    prompt = Column(Text, nullable=False)
    synthesis = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    # JSON-encoded Atelier Meter result (see llm.engine.generate_atelier_meter)
    # - null for the common case of a non-yes/no query. See
    # AtelierRepository.initialize_db's _ensure_column call: this column was
    # added after "queries" already existed on disk, so it self-heals via
    # ALTER TABLE rather than SQLAlchemy's create_all (which never alters
    # existing tables).
    consensus_meter = Column(Text, nullable=True)

    citations = relationship("QueryCitation", backref="query", cascade="all, delete-orphan")


class QueryCitation(Base):
    """
    A Many-to-Many junction table linking a specific Query to the Papers it cited.
    """
    __tablename__ = "query_citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    paper_fingerprint = Column(String, nullable=False)  # Links to PaperModel.fingerprint
    # 0-based position matching the [N] citation numbers baked into the
    # persisted synthesis markdown (citation_index i == bracket [i+1]) -
    # llm.engine.generate_copilot_synthesis numbers citations strictly by
    # each record's position in valid_records via enumerate(), and
    # build_paper_cards/build_synthesis_body's hover-citation matching both
    # depend on reconstructing that EXACT same order later. Without this,
    # AtelierRepository.get_session_chat_history/get_query_citations had to
    # re-fetch papers via "WHERE fingerprint IN (...)", which SQL does not
    # guarantee returns in the order the fingerprints were listed - on
    # reload, papers came back in whatever order SQLite's query planner
    # happened to pick (in practice, PaperModel.id order - i.e. whichever
    # session FIRST ever discovered each paper, completely unrelated to
    # THIS query's citation numbering), silently scrambling which paper
    # card corresponds to which [N] in the already-written text. Nullable
    # since existing rows written before this column existed have no way
    # to recover their original order retroactively - see
    # AtelierRepository.initialize_db's _ensure_column self-heal.
    citation_index = Column(Integer, nullable=True)


class PaperSummaryModel(Base):
    """
    Caches llm.engine.generate_paper_summary's output per (paper, session) -
    that summary is topic-focused (grounded in the session's topic, not
    just the paper itself), and a session's topic is fixed at creation
    (SessionModel.topic never changes after the fact - see
    AtelierRepository.get_session_topic), so this pair is a stable, correct
    cache key: the same paper viewed again in the same session always
    resolves to the same topic and therefore the same summary. Lets
    ui/callbacks/library.py's summarize_selected_paper skip the LLM call
    entirely on a repeat click instead of regenerating every time. A brand
    new table, unlike QueryModel.consensus_meter's ALTER TABLE self-heal -
    create_all handles new tables fine, only altering EXISTING ones needs
    that workaround.
    """
    __tablename__ = "paper_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_fingerprint = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (UniqueConstraint('paper_fingerprint', 'session_id', name='uq_paper_summary_fp_session'),)


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
    # A short, topic-relevant full-text excerpt (when full text was
    # available), set during pipeline/nodes.py's extraction pass and reused
    # by llm/engine.py's generate_copilot_synthesis to ground the final
    # synthesis in an actual passage rather than only the 1-sentence
    # "answer" summary - real RAG at the synthesis step, not just extraction.
    top_passage: str = ""
    _db_id: Optional[int] = None

    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Every Record, regardless of which fetcher (search/paper.py) or DB
        # reconstruction path built it, gets its links passed through
        # normalize_url - see that function's docstring for why a
        # scheme-less URL from an upstream API needs fixing up here rather
        # than wherever it's eventually rendered.
        self.pdf_url = normalize_url(self.pdf_url)
        self.full_text_url = normalize_url(self.full_text_url)

    def to_model_dict(self) -> dict[str, Any]:
        """Shape aligned to your PaperModel fields."""
        data = asdict(self)
        data.pop("source", None)
        data.pop("source_id", None)
        data.pop("raw", None)
        return data
