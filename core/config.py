import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Static .env-only snapshots - kept for any call site that only ever needs
# the process-env value. Prefer resolve_key() for anything a user might set
# via the Settings UI, since these are captured once at import time and
# won't reflect keys entered after the app started.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

REDIS_URL = os.getenv("REDIS_URL")


def _is_configured(value: Optional[str]) -> bool:
    """
    True only for a value that looks like a real credential, not an unfilled
    ".env.example"-style placeholder (e.g. "your_openalex_api_key"). Mirrors
    llm/utils.py's _is_set (duplicated rather than imported - llm/utils.py
    itself imports resolve_key from this module, so importing back would be
    circular). Without this, engines that treat "any non-empty api_key" as
    real (OpenAlexEngine, SemanticScholarEngine) would send the literal
    placeholder text as credentials and get a hard 401/403 on every call,
    instead of just falling back to anonymous access as intended.
    """
    return bool(value and value.strip() and "your_" not in value)


def resolve_key(env_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Resolves a secret dynamically: a value entered via the Settings UI
    (encrypted in SQLite) takes priority; falls back to .env otherwise.
    Never returns an unfilled ".env" placeholder - see _is_configured.

    Unlike the module-level constants above, this re-checks on every call,
    so a key saved through Settings takes effect on the next request - no
    app restart needed.
    """
    try:
        # Local import: keeps core/ from taking a hard dependency on the
        # database package at import time (and sidesteps any import-order
        # issues during app boot).
        from database.repository import AtelierRepository

        db_value = AtelierRepository.get_setting(env_name)
        if _is_configured(db_value):
            return db_value
    except Exception:
        pass  # DB not ready yet, or not initialized - fall through to .env

    env_value = os.getenv(env_name, default)
    return env_value if _is_configured(env_value) else default

# ==========================================
# APPLICATION CONSTANTS
# ==========================================
# Used to identify your app to academic APIs (OpenAlex requires this)
USER_AGENT = "atelier-oss/1.0 (mailto:dodoma700@gmail.com)"

# Default Models
DEFAULT_SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"

# ==========================================
# SCORING HEURISTICS & WEIGHTS
# ==========================================
# Hierarchy of Evidence Weights (Step 2 Quality Reranking)
STUDY_TYPE_WEIGHTS = {
    "systematic review / meta-analysis": 5.0,
    "randomized controlled trial": 4.0,
    "cohort study": 3.0,
    "case-control study": 2.0,
    "cross-sectional study": 2.0,
    "qualitative study": 1.5,
    "mixed-methods study": 1.5,
    "case report / case series": 0.5,
    "guideline / consensus": 3.0,
    "unspecified": 0.0,
    "observational study": 1.5 # Fallback
}
