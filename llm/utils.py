import logging
from typing import Callable, Optional, List

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.config import resolve_key
from llm.model_catalog import list_provider_models

logger = logging.getLogger(__name__)

# Default endpoints for locally-hosted providers, overridable via Settings.
# Deliberately the 127.0.0.1 literal, not "localhost" - see model_catalog.py.
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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

# Cloud providers: (provider prefix, LangChain class, Settings/.env key name).
# No per-model entries here anymore - see resolve_model_config()/get_model_choices()
# below, which list each provider's actual current models live instead of a
# hardcoded, perpetually-stale table.
_CLOUD_PROVIDERS = [
    ("anthropic", ChatAnthropic, "ANTHROPIC_API_KEY"),
    ("google", ChatGoogleGenerativeAI, "GOOGLE_API_KEY"),
    ("groq", ChatGroq, "GROQ_API_KEY"),
    ("openai", ChatOpenAI, "OPENAI_API_KEY"),
]
_CLOUD_PROVIDER_CLASSES = {name: cls for name, cls, _ in _CLOUD_PROVIDERS}
_CLOUD_PROVIDER_KEYS = {name: env_key for name, _, env_key in _CLOUD_PROVIDERS}

def _normalize_model_name(name: str) -> str:
    """Standardizes model names for reliable dictionary lookups."""
    return name.strip().lower()

def _is_set(v: Optional[str]) -> bool:
    """Checks if an environment variable or API key is actually populated."""
    return bool(v and str(v).strip() and "your_" not in str(v))

def get_model_choices() -> List[str]:
    """
    Returns "provider:model" choices for every provider that's currently
    reachable - a live "list of set providers", not a hand-maintained one.

    - Cloud providers (Anthropic/Google/Groq/OpenAI) are gated by a
      configured key, then list THEIR OWN current models via that
      provider's models API - a new model release shows up automatically,
      no code change needed.
    - Ollama/LM Studio are probed directly (fast local timeout); whatever
      they report as loaded is listed, no manual model name needed.
    - OpenRouter needs a configured key; its full model catalog is public
      and listed the same way.

    Re-checked (with short-lived caching in model_catalog.py) on every call,
    so this reflects Settings changes without an app restart.
    """
    choices: List[str] = []

    for provider, _cls, env_key in _CLOUD_PROVIDERS:
        api_key = resolve_key(env_key)
        if _is_set(api_key):
            choices += [f"{provider}:{m}" for m in list_provider_models(provider, api_key)]

    # Local providers are opt-in, same as cloud ones: only probed once the
    # user has explicitly configured them in Settings (even just re-saving
    # the default URL) - never auto-detected/silently network-probed on
    # every render for users who haven't set them up at all.
    ollama_url = resolve_key("OLLAMA_BASE_URL")
    if _is_set(ollama_url):
        choices += [f"ollama:{m}" for m in list_provider_models("ollama", ollama_url)]

    lmstudio_url = resolve_key("LMSTUDIO_BASE_URL")
    if _is_set(lmstudio_url):
        choices += [f"lmstudio:{m}" for m in list_provider_models("lmstudio", lmstudio_url)]

    openrouter_key = resolve_key("OPENROUTER_API_KEY")
    if _is_set(openrouter_key):
        choices += [f"openrouter:{m}" for m in list_provider_models("openrouter", openrouter_key)]

    return choices

def resolve_model_config(model_name: str):
    """
    Takes a "provider:model" string from the UI and returns its underlying
    LangChain class and constructor parameters.

    Local/proxy providers (ollama/lmstudio/openrouter) get fully
    self-contained constructor_params (base_url + api_key already filled
    in), since their model name is user/server-supplied rather than one of
    a small fixed set. Cloud providers resolve to their LangChain class with
    just the model name - AtelierAIEngine injects the actual API key.
    """
    normalized = _normalize_model_name(model_name)
    if ":" not in normalized:
        return None  # every valid choice is "provider:model" - see get_model_choices()

    provider, _, _ = normalized.partition(":")
    model = model_name.split(":", 1)[1].strip()  # preserve original casing for the real API call

    if provider == "ollama":
        return {
            'class': ChatOllama,
            'constructor_params': {
                'model': model,
                'base_url': resolve_key("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            }
        }

    if provider == "lmstudio":
        return {
            'class': ChatOpenAI,
            'constructor_params': {
                'model': model,
                'base_url': resolve_key("LMSTUDIO_BASE_URL", DEFAULT_LMSTUDIO_BASE_URL),
                # LM Studio's local server doesn't validate this - it just
                # has to be a non-empty string for the OpenAI client.
                'api_key': 'lm-studio',
            }
        }

    if provider == "openrouter":
        return {
            'class': ChatOpenAI,
            'constructor_params': {
                'model': model,
                'base_url': OPENROUTER_BASE_URL,
                'api_key': resolve_key("OPENROUTER_API_KEY"),
            }
        }

    cls = _CLOUD_PROVIDER_CLASSES.get(provider)
    if not cls:
        return None
    return {'class': cls, 'constructor_params': {'model': model}}
