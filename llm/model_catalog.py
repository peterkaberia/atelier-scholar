"""
LLM Model Catalog Module

Lists each provider's currently available models by calling that provider's
OWN models-listing API, instead of maintaining a hand-curated table that goes
stale the moment a provider ships a new model. Results are cached briefly
in-memory (per provider+credential) since get_model_choices() can be called
on nearly every page render.
"""

import logging
import time
from typing import Callable, Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 600  # 10 minutes
_CACHE: Dict[str, Tuple[float, List[str]]] = {}

# Cloud providers' /models endpoints list every model they host, including
# non-chat ones (embeddings, TTS, image, moderation). Filtered out by name
# since none of the APIs expose a clean "is this a chat model" flag.
_NON_CHAT_HINTS = (
    "embedding", "whisper", "tts", "dall-e", "moderation",
    "davinci-002", "babbage-002", "audio", "image", "transcribe", "realtime",
)


def _looks_like_chat_model(model_id: str) -> bool:
    lower = model_id.lower()
    return not any(hint in lower for hint in _NON_CHAT_HINTS)


# Floor for a model to be usable here at all. The final synthesis prompt
# (llm/engine.py's generate_copilot_synthesis) concatenates up to 20 papers'
# structured extraction fields plus a full-text excerpt each
# (SYNTHESIS_PASSAGE_BUDGET=400 chars/paper - pipeline/nodes.py) on top of
# the batched-extraction and query-planning prompts elsewhere in the
# pipeline - worst case that's several thousand prompt tokens before the
# model has written a word of its own answer. Checked live against
# OpenRouter's actual catalog (2026-08): the wide majority of real chat
# models report 128K+, with only a handful of legacy models (old GPT-3.5-
# turbo variants, older 8B/13B checkpoints) below 16K - so this floor
# excludes exactly the models actually too small for Atelier's prompts,
# not a meaningful chunk of the real catalog.
MIN_CONTEXT_LENGTH = 16000


def _cached(cache_key: str, fetch_fn: Callable[[], List[str]]) -> List[str]:
    now = time.monotonic()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        models = fetch_fn()
        _CACHE[cache_key] = (now, models)
        return models
    except Exception as e:
        logger.warning(f"Model listing failed for '{cache_key}': {e}")
        # Prefer stale-but-known-good over empty, so a transient outage
        # doesn't suddenly hide every model the user was just using.
        return cached[1] if cached else []


# ==========================================
# PER-PROVIDER FETCHERS
# ==========================================

def _fetch_openai(api_key: str) -> List[str]:
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}, timeout=8,
    )
    resp.raise_for_status()
    ids = [m["id"] for m in resp.json().get("data", []) if _looks_like_chat_model(m["id"])]
    return sorted(ids)


def _fetch_anthropic(api_key: str) -> List[str]:
    resp = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}, timeout=8,
    )
    resp.raise_for_status()
    return sorted([m["id"] for m in resp.json().get("data", [])])


def _fetch_groq(api_key: str) -> List[str]:
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}, timeout=8,
    )
    resp.raise_for_status()
    ids = [m["id"] for m in resp.json().get("data", []) if _looks_like_chat_model(m["id"])]
    return sorted(ids)


def _fetch_google(api_key: str) -> List[str]:
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key}, timeout=8,
    )
    resp.raise_for_status()
    models = resp.json().get("models", [])
    # Names come back as "models/gemini-2.5-flash" - strip the prefix, only
    # keep ones that actually support chat generation, and (see
    # MIN_CONTEXT_LENGTH's docstring) drop anything too small for Atelier's
    # prompts - Google's list response reports inputTokenLimit per model,
    # unlike OpenAI/Anthropic/Groq's list endpoints which expose no capacity
    # metadata at all, so this check only applies here and to OpenRouter.
    ids = [
        m["name"].split("/", 1)[-1] for m in models
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and m.get("inputTokenLimit", MIN_CONTEXT_LENGTH) >= MIN_CONTEXT_LENGTH
    ]
    return sorted(ids)


def _fetch_ollama(base_url: str) -> List[str]:
    # Short timeout, and callers should default base_url to the 127.0.0.1
    # literal rather than "localhost" - on this stack, "localhost" triggers
    # a slow sequential IPv6-then-IPv4 resolution (~3s to fail) when nothing
    # is listening, which would otherwise tax every cold-cache page render
    # for users who aren't running Ollama at all.
    resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=0.8)
    resp.raise_for_status()
    names = [m["name"] for m in resp.json().get("models", []) if _looks_like_chat_model(m["name"])]
    return sorted(names)


def _fetch_lmstudio(base_url: str) -> List[str]:
    resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=0.8)
    resp.raise_for_status()
    # LM Studio's /v1/models mixes loaded chat and embedding models with no
    # type flag to distinguish them - filter by name like the cloud providers.
    ids = [m["id"] for m in resp.json().get("data", []) if _looks_like_chat_model(m["id"])]
    return sorted(ids)


def _fetch_openrouter(_credential: str) -> List[str]:
    # OpenRouter's catalog is public - listing needs no key, only actually
    # invoking a model does. `_credential` is accepted for signature
    # consistency with the other fetchers but unused.
    #
    # Unlike the other providers, OpenRouter's /models response carries real
    # capability metadata per model (architecture.output_modalities,
    # context_length) instead of just an id - so filtering here is exact,
    # not a name-substring guess like _looks_like_chat_model. Without this,
    # the raw catalog includes plenty of genuinely non-chat entries (Lyria
    # music generation, GPT Audio, Gemini image models...) that would
    # otherwise sit in the dropdown as if they were usable for Atelier's
    # text synthesis/extraction/chat calls and fail outright if picked.
    # output_modalities == ["text"] is the "can chat" check: anything that
    # ALSO emits audio/image isn't a plain text-chat model. context_length
    # is the "has the required context" check - see MIN_CONTEXT_LENGTH.
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=8)
    resp.raise_for_status()
    ids = [
        m["id"] for m in resp.json().get("data", [])
        if m.get("architecture", {}).get("output_modalities") == ["text"]
        and m.get("context_length", 0) >= MIN_CONTEXT_LENGTH
    ]
    return sorted(ids)


def _validate_openrouter(api_key: str) -> None:
    # Unlike _fetch_openrouter (listing), actually validating a key needs
    # OpenRouter's dedicated auth-check endpoint - the models list doesn't
    # require auth at all, so it can't tell a bad key from a good one.
    resp = requests.get(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"}, timeout=8,
    )
    resp.raise_for_status()


def _validate_openalex(api_key: str) -> None:
    # OpenAlex isn't an LLM provider (no chat models to list) - this exists
    # purely so Settings can validate the academic-search key before saving.
    resp = requests.get(
        "https://api.openalex.org/works",
        params={"per_page": 1, "api_key": api_key}, timeout=8,
    )
    resp.raise_for_status()


def _validate_semantic_scholar(api_key: str) -> None:
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": "test", "limit": 1},
        headers={"x-api-key": api_key}, timeout=8,
    )
    resp.raise_for_status()


_FETCHERS: Dict[str, Callable[[str], List[str]]] = {
    "openai": _fetch_openai,
    "anthropic": _fetch_anthropic,
    "groq": _fetch_groq,
    "google": _fetch_google,
    "ollama": _fetch_ollama,
    "lmstudio": _fetch_lmstudio,
    "openrouter": _fetch_openrouter,
}

# Providers whose real validation check differs from "list its models"
# (or that aren't LLM providers at all, so have no fetcher to fall back on -
# OpenAlex/Semantic Scholar). Everything else is validated by just calling
# its fetcher (_FETCHERS) and seeing if it raises.
_VALIDATORS: Dict[str, Callable[[str], None]] = {
    "openrouter": _validate_openrouter,
    "openalex": _validate_openalex,
    "semanticscholar": _validate_semantic_scholar,
}


def list_provider_models(provider: str, credential: str) -> List[str]:
    """
    Returns the live list of model ids a provider currently exposes.

    Args:
        provider: one of the keys in _FETCHERS ("openai", "anthropic", ...).
        credential: an API key for cloud providers, a base_url for local
            ones (Ollama/LM Studio), or unused for OpenRouter's public
            catalog endpoint.

    Returns:
        A sorted list of model ids, or [] if the provider is unknown, the
        request failed, and no prior cached result exists to fall back on.
    """
    fetch_fn = _FETCHERS.get(provider)
    if not fetch_fn:
        return []

    cache_key = f"{provider}:{credential}"
    return _cached(cache_key, lambda: fetch_fn(credential))


def validate_provider_credential(provider: str, credential: str) -> Tuple[bool, str]:
    """
    Live, UNCACHED check of a credential/URL before it's saved to Settings -
    so a typo or an unreachable server produces an immediate, specific error
    instead of a silently-broken saved value that only surfaces as "no
    models available" much later.

    Args:
        provider: one of the keys in _FETCHERS.
        credential: the exact value the user is about to save - an API key
            for cloud providers, a base_url for local ones.

    Returns:
        (True, "") if it works. (False, human-readable reason) if not.
    """
    check_fn = _VALIDATORS.get(provider) or _FETCHERS.get(provider)
    if not check_fn:
        return False, f"Unknown provider '{provider}'."

    try:
        check_fn(credential)
        return True, ""
    except requests.exceptions.Timeout:
        return False, "Timed out - check the value and make sure the server is reachable."
    except requests.exceptions.ConnectionError:
        return False, "Couldn't connect - check the URL and make sure the server is running."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (401, 403):
            return False, "Rejected - check that the key is correct and active."
        return False, f"Request failed (HTTP {status})."
    except Exception as e:
        return False, f"Could not validate: {e}"
