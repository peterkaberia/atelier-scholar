import os
from dotenv import load_dotenv

load_dotenv()

# Configuration variables loaded from the .env file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

REDIS_URL = os.getenv("REDIS_URL")

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
