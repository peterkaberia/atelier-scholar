"""
Database Engine Module

Sets up the SQLAlchemy Engine + Session factory for the local SQLite
database. Kept separate from models.py/repository.py so the connection
config is in one obvious place.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

DB_PATH = "atelier_history.db"

# check_same_thread=False: Flask now runs threaded=True (see app.py), and
# AtelierRepository's static methods can be called from any request thread -
# SQLAlchemy's own connection pool (not this flag) is what actually keeps
# concurrent access safe; this just stops pysqlite's own same-thread check
# from raising first. Each background job additionally runs in its OWN
# spawned process (see ui/callbacks/background.py), which gets a completely
# separate Engine/pool anyway since this module is re-imported fresh there.
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """WAL journal mode for concurrent reader/writer performance (matches
    the checkpoint DB's own pragma in pipeline/orchestrator.py), plus a
    larger page cache - same settings the previous Peewee config used."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA cache_size=-65536")  # 64MB, negative = KB
    cursor.close()


# expire_on_commit=False: repository methods build and return plain dicts /
# Record dataclasses (never raw ORM instances) from within their own `with
# get_db_session()` block, but a couple of unwrapped attribute reads happen
# right at the session/commit boundary - this avoids DetachedInstanceError
# without changing any calling code's shape.
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_db_session():
    """
    Yields a SQLAlchemy Session scoped to a single `with` block, committing
    on clean exit and rolling back on any exception - the SQLAlchemy
    equivalent of Peewee's implicit per-call connection handling, made
    explicit since SQLAlchemy has no autocommit-by-default model.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
