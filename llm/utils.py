import logging
from typing import Callable, Optional, List

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY
)

logger = logging.getLogger(__name__)

class BufferedStreamingHandler(BaseCallbackHandler):
    """
    A custom callback handler for streaming LLM tokens.
    Buffers tokens until a newline or limit is reached, then flushes them to a UI callback.
    Perfect for making Streamlit chat interfaces feel responsive but not glitchy.
    """
    def __init__(self, buffer_limit: int = 60, ui_callback: Optional[Callable[[str], None]] = None):
        self.buffer = ""
        self.buffer_limit = buffer_limit
        self.ui_callback = ui_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer += token
        if "\n" in token or len(self.buffer) >= self.buffer_limit:
            logger.info(self.buffer)
            if self.ui_callback:
                self.ui_callback(self.buffer)
            self.buffer = ""

    def on_llm_end(self, response, **kwargs) -> None:
        logger.info(self.buffer)
        if self.buffer and self.ui_callback:
            self.ui_callback(self.buffer)
            self.buffer = ""

# --- Configuration Data ---
# Instantiate common dependencies once
_common_callbacks = [BufferedStreamingHandler()]

# Define common parameters for most LLMs
_common_llm_params = {"temperature": 0.0, "streaming": True, "callbacks": _common_callbacks}

# Map input model choices (lowercased) to their configuration
# Each config includes the class and any model-specific constructor parameters
_llm_config_map = {
    'gpt-4.1': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-4.1'} 
    },
    'gpt-5.2': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.2'} 
    },
    'gpt-5.1': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5.1'} 
    },
    'gpt-5-mini': {
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-mini'} 
    },
    'gpt-5-nano': { 
        'class': ChatOpenAI,
        'constructor_params': {'model_name': 'gpt-5-nano'} 
    },
    'claude-sonnet-4-5': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-5'}
    },
    'claude-sonnet-4-0': {
        'class': ChatAnthropic,
        'constructor_params': {'model': 'claude-sonnet-4-0'}
    },
    'gemini-2.5-flash': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash', 'google_api_key': GOOGLE_API_KEY }
    },
    'gemini-2.5-flash-lite': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-flash-lite', 'google_api_key': GOOGLE_API_KEY}
    },
    'gemini-2.5-pro': {
        'class': ChatGoogleGenerativeAI,
        'constructor_params': {'model': 'gemini-2.5-pro', 'google_api_key': GOOGLE_API_KEY}
    },
    'gpt-oss-120b-groq': {
        'class': ChatGroq,
        'constructor_params': {'model_name': 'openai/gpt-oss-120b' }
    },
    'llama-3.3-70b-versatile': {
        'class': ChatGroq,
        'constructor_params': {'model_name': 'llama-3.3-70b-versatile' }
    }
}

def _normalize_model_name(name: str) -> str:
    """Standardizes model names for reliable dictionary lookups."""
    return name.strip().lower()

def _is_set(v: Optional[str]) -> bool:
    """Checks if an environment variable or API key is actually populated."""
    return bool(v and str(v).strip() and "your_" not in str(v))

def get_model_choices() -> List[str]:
    """
    Returns a list of model strings that the user is actually authorized to use,
    based on which API keys are present in their .env file.
    """
    gated_base_models: List[str] = []

    anthropic_ok = _is_set(ANTHROPIC_API_KEY)
    google_ok = _is_set(GOOGLE_API_KEY)
    groq_ok = _is_set(GROQ_API_KEY)
    openai_ok = _is_set(OPENAI_API_KEY)

    for k, cfg in _llm_config_map.items():
        cls = cfg.get('class')

        # Anthropic
        if cls is ChatAnthropic and anthropic_ok:
            gated_base_models.append(k)
        elif cls is ChatGoogleGenerativeAI and google_ok:
            gated_base_models.append(k)
        elif cls is ChatGroq and groq_ok:
            gated_base_models.append(k)
        elif cls is ChatOpenAI and openai_ok:
            gated_base_models.append(k)

    return gated_base_models

def resolve_model_config(model_name: str):
    """
    Takes a model string from the UI and returns its underlying LangChain class 
    and constructor parameters.
    """
    normalized_choice_lower = _normalize_model_name(model_name)
    return _llm_config_map.get(normalized_choice_lower)