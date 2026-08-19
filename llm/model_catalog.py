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
from typing import Callable, Dict, List, Optional, Tuple

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
# (llm/engine.py's chat_with_literature) concatenates up to 20 papers'
# structured extraction fields plus a full-text excerpt each, on top of the
# batched-extraction and query-planning prompts elsewhere in the pipeline -
# worst case that's several thousand prompt tokens before the model has
# written a word of its own answer. Checked live against OpenRouter's
# actual catalog (2026-08): the wide majority of real chat models report
# 128K+, with only a handful of legacy models (old GPT-3.5-turbo variants,
# older 8B/13B checkpoints) below 16K - so this floor excludes exactly the
# models actually too small for Atelier's prompts, not a meaningful chunk
# of the real catalog.
MIN_CONTEXT_LENGTH = 16000

# Fallback context length (tokens) for any model whose provider's listing
# endpoint reports no capacity metadata at all - OpenAI/Anthropic/Groq/
# Ollama/LM Studio's /models endpoints expose only ids (see each fetcher's
# docstring below), unlike OpenRouter/Google's, which is the only reason
# get_model_context_length can be exact for those two and not the rest.
# Deliberately conservative (this module's whole premise, per its own
# docstring, is not hand-maintaining a per-model-name table that goes
# stale) - guessing too LOW only costs unused headroom in the budget
# functions below; guessing too HIGH risks an oversized-prompt API error
# the provider itself would reject.
DEFAULT_CONTEXT_LENGTH = 32000


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


# ==========================================
# CONTEXT LENGTH LOOKUP
# ==========================================
#
# Only OpenRouter and Google's /models responses carry real per-model
# capacity metadata (context_length / inputTokenLimit - see
# _fetch_openrouter/_fetch_google above); OpenAI/Anthropic/Groq/Ollama/LM
# Studio's list endpoints expose none at all. These two fetchers mirror
# _fetch_openrouter/_fetch_google exactly (same filtering) but return
# {model_id: context_length} instead of discarding the number after
# filtering - a second, parallel request rather than sharing _CACHE with
# the id-only fetchers, since the two caches can independently expire and
# there's no clean way to recover a discarded number from a cached List[str].

_CONTEXT_CACHE: Dict[str, Tuple[float, Dict[str, int]]] = {}


def _cached_context_lengths(cache_key: str, fetch_fn: Callable[[], Dict[str, int]]) -> Dict[str, int]:
    now = time.monotonic()
    cached = _CONTEXT_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        lengths = fetch_fn()
        _CONTEXT_CACHE[cache_key] = (now, lengths)
        return lengths
    except Exception as e:
        logger.warning(f"Context-length listing failed for '{cache_key}': {e}")
        return cached[1] if cached else {}


def _fetch_openrouter_context_lengths(_credential: str) -> Dict[str, int]:
    resp = requests.get("https://openrouter.ai/api/v1/models", timeout=8)
    resp.raise_for_status()
    return {
        m["id"]: m.get("context_length", 0)
        for m in resp.json().get("data", [])
        if m.get("architecture", {}).get("output_modalities") == ["text"]
        and m.get("context_length", 0) >= MIN_CONTEXT_LENGTH
    }


def _fetch_google_context_lengths(api_key: str) -> Dict[str, int]:
    resp = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key}, timeout=8,
    )
    resp.raise_for_status()
    out: Dict[str, int] = {}
    for m in resp.json().get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        limit = m.get("inputTokenLimit", 0)
        if limit < MIN_CONTEXT_LENGTH:
            continue
        out[m["name"].split("/", 1)[-1]] = limit
    return out


_CONTEXT_LENGTH_FETCHERS: Dict[str, Callable[[str], Dict[str, int]]] = {
    "openrouter": _fetch_openrouter_context_lengths,
    "google": _fetch_google_context_lengths,
}


def get_model_context_length(model_choice: Optional[str]) -> int:
    """
    Best-effort real context window (in tokens) for `model_choice`
    ("provider:model", the same string threaded through everywhere as
    AtelierAIEngine's model_choice). Falls back to DEFAULT_CONTEXT_LENGTH
    for any provider without capacity metadata, an unrecognized/malformed
    model_choice, or a failed lookup - see that constant's docstring for
    why the fallback is conservative rather than a guessed table.

    Used by the budget functions below to size how much paper content
    Atelier feeds this model per call, instead of a single hardcoded
    constant identical for an 8K local model and a 1M+-token model alike.
    """
    if not model_choice or ":" not in model_choice:
        return DEFAULT_CONTEXT_LENGTH

    provider, model_name = model_choice.split(":", 1)
    fetch_fn = _CONTEXT_LENGTH_FETCHERS.get(provider)
    if not fetch_fn:
        return DEFAULT_CONTEXT_LENGTH

    if provider == "google":
        from core.config import resolve_key
        credential = resolve_key("GOOGLE_API_KEY") or ""
    else:
        credential = ""  # OpenRouter's catalog is public - see _fetch_openrouter's docstring.

    lengths = _cached_context_lengths(f"{provider}:{credential}", lambda: fetch_fn(credential))
    return lengths.get(model_name, DEFAULT_CONTEXT_LENGTH)


# ==========================================
# CONTEXT-BUDGET SCALING
# ==========================================
#
# Every content budget the pipeline feeds an LLM (how much of a paper's
# text goes into extraction, how many full-text chunks get RAG-retrieved,
# how much of a paper's excerpt survives into the final synthesis prompt)
# used to be a single hardcoded constant, identical regardless of whether
# the selected model had an 8K or a 2M-token window - a huge-context model
# got exactly as much material as a small local one, wasting its real
# capacity for no reason. These derive every such budget from
# get_model_context_length instead, so a bigger-context model genuinely
# gets more material to work with, and a smaller one still gets a budget
# that fits.

CHARS_PER_TOKEN = 4  # rough heuristic for English text

# Reserved for system-prompt boilerplate + the model's own response,
# regardless of context size - without this floor, a tiny-context model's
# "available" budget could get divided down toward zero.
RESERVED_OVERHEAD_TOKENS = 3000

# Even a genuinely enormous window (Gemini's 1M-2M-token tier) doesn't get
# UNLIMITED content - a single call stuffed with hundreds of thousands of
# tokens of paper text would be extremely slow and expensive for a
# marginal quality gain over "a lot more than before." This caps the POOL
# every budget function below draws from, not any one of them individually.
MAX_AVAILABLE_CONTENT_CHARS = 800_000  # ~200K tokens


def _available_content_chars(model_choice: Optional[str]) -> int:
    """Total characters of PAPER CONTENT `model_choice`'s context window can reasonably hold in one prompt, after reserving room for instructions and the model's own response."""
    usable_tokens = max(get_model_context_length(model_choice) - RESERVED_OVERHEAD_TOKENS, 1000)
    return min(usable_tokens * CHARS_PER_TOKEN, MAX_AVAILABLE_CONTENT_CHARS)


def batch_extraction_content_budget(model_choice: Optional[str], batch_size: int) -> int:
    """
    Per-paper character budget when `batch_size` papers share one batched
    extraction prompt (pipeline/nodes.py's extract_records) - the total
    across the whole batch stays within _available_content_chars either
    way, this just decides how that pool is split.
    """
    per_paper = _available_content_chars(model_choice) // max(batch_size, 1)
    return max(min(per_paper, 200_000), 800)


def synthesis_passage_budget(model_choice: Optional[str], paper_count: int) -> int:
    """
    Character budget for the short full-text excerpt (Record.top_passage)
    carried into the FINAL synthesis/chat prompt per paper, when up to
    `paper_count` papers' excerpts all share that ONE prompt together
    (llm.engine._format_synthesis_entry, called once per paper but
    concatenated into a single call).
    """
    per_paper = _available_content_chars(model_choice) // max(paper_count, 1)
    return max(min(per_paper, 4000), 200)


def full_text_tool_budget(model_choice: Optional[str]) -> int:
    """
    Character budget for a single tool observation
    (pipeline/agent_tools.py's get_full_text) - that paper's text is alone
    in its turn's context (not sharing a prompt with other papers the way
    batched extraction does), so it gets a generous share of the pool.
    """
    return min(_available_content_chars(model_choice), 100_000)


def chat_history_turn_count(model_choice: Optional[str]) -> int:
    """
    How many of the most recent chat_history messages get included in a
    synthesis/chat/investigation prompt (llm.engine.chat_with_literature,
    pipeline.agent.run_investigation) - each message can be a full prior
    synthesis (thousands of characters), so a fixed "last 6" wastes a
    big-context model's real capacity to actually remember the
    conversation, the same way the other fixed content budgets did.

    Deliberately NOT scaled off _available_content_chars/full_text_tool_
    budget the way those are: chat history SHARES its one prompt with the
    paper content those functions already size independently (see
    synthesis_passage_budget), so greedily maxing this out too would
    double-book the same context window instead of leaving it real room.
    Tiered and conservative instead - a bigger-context model still
    remembers meaningfully more than the old fixed 6, just with headroom
    left over for the paper content sharing that same prompt.
    """
    context_length = get_model_context_length(model_choice)
    if context_length >= 200_000:
        return 20
    if context_length >= 100_000:
        return 12
    if context_length >= 32_000:
        return 6
    return 4


def rag_chunk_count(model_choice: Optional[str]) -> int:
    """
    How many top-scored full-text chunks (~1200 chars each - see
    core.utils.chunk_text's default chunk_size) a single TARGETED RAG
    query retrieves (pipeline/agent_tools.py's rag_search_chunks tool).

    Scales modestly with context window, but with its own tight ceiling
    rather than reusing full_text_tool_budget's - unlike a raw content
    budget, a "targeted" retrieval should stay small even for a
    huge-context model; past roughly 10 chunks it stops being a targeted
    answer and starts being "read the whole paper," which get_full_text
    already covers.
    """
    return max(3, min(10, get_model_context_length(model_choice) // 20_000))


def session_paper_list_count(model_choice: Optional[str]) -> int:
    """
    How many of a session's already-processed papers
    (pipeline/agent_tools.py's list_session_papers tool) get listed in one
    call. Scales with context window so a large-context model
    investigating a session with many accumulated papers (repeated
    searches/investigations over time) can see more of them at once,
    rather than the same fixed slice a small model would - each line is
    compact (title + one-line takeaway), so even a generous count costs
    relatively little of the budget.
    """
    return max(30, min(200, get_model_context_length(model_choice) // 1000))


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
