"""
Settings Layout Module

Route: '/settings' - two independent provider registries (LLM, Academic
Search) each drive their own dropdown -> one field -> configured list. Local
LLM providers (Ollama/LM Studio) are configured the same way as everything
else here - saving a value (even the default URL) is what "enables" them;
see llm/utils.py's get_model_choices(), which only probes a local provider
once it has an explicitly-saved value, never automatically.

Values are never echoed back once saved - the dynamic field re-renders
blank after a save, and the configured lists only ever show a provider's
name, never its value.
"""

from dash import html, dcc
from database import AtelierRepository

# (id, display label, Settings/.env key, field label, input type, icon, hint)
# Icon is a Material Symbols Outlined ligature name (same icon set used
# throughout the app - see index_string's font import in main.py).
LLM_PROVIDERS = [
    {"id": "anthropic", "label": "Anthropic", "env_key": "ANTHROPIC_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "psychology", "hint": "Powers Claude models."},
    {"id": "google", "label": "Google", "env_key": "GOOGLE_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "cloud", "hint": "Powers Gemini models."},
    {"id": "groq", "label": "Groq", "env_key": "GROQ_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "bolt", "hint": "Powers Llama / GPT-OSS models."},
    {"id": "openai", "label": "OpenAI", "env_key": "OPENAI_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "auto_awesome", "hint": "Powers GPT models."},
    {"id": "openrouter", "label": "OpenRouter", "env_key": "OPENROUTER_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "alt_route", "hint": "Proxies hundreds of models from many labs - full catalog listed automatically once set."},
    {"id": "ollama", "label": "Ollama (local)", "env_key": "OLLAMA_BASE_URL", "field_label": "Base URL", "input_type": "text", "icon": "dns", "hint": "No API key needed. Save a URL to enable it - even the default, http://127.0.0.1:11434 - then its loaded models are detected automatically."},
    {"id": "lmstudio", "label": "LM Studio (local)", "env_key": "LMSTUDIO_BASE_URL", "field_label": "Base URL", "input_type": "text", "icon": "computer", "hint": "No API key needed. Save a URL to enable it - even the default, http://127.0.0.1:1234/v1 - then its loaded models are detected automatically."},
]

SEARCH_PROVIDERS = [
    {"id": "openalex", "label": "OpenAlex", "env_key": "OPENALEX_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "menu_book", "hint": "Optional - raises your OpenAlex rate limit."},
    {"id": "semanticscholar", "label": "Semantic Scholar", "env_key": "SEMANTIC_SCHOLAR_API_KEY", "field_label": "API Key", "input_type": "password", "icon": "science", "hint": "Optional - raises your Semantic Scholar rate limit."},
]

PROVIDER_BY_ID = {p["id"]: p for p in LLM_PROVIDERS + SEARCH_PROVIDERS}


def status_message(text: str, ok: bool = True):
    """A dismissible-looking inline banner for save/remove feedback."""
    tone = "emerald" if ok else "amber"
    icon = "check_circle" if ok else "error"
    return html.Div(
        className=f"flex items-center gap-2 px-4 py-3 bg-{tone}-50 border border-{tone}-100 rounded-xl text-sm font-semibold text-{tone}-700",
        children=[
            html.Span(icon, className="material-symbols-outlined text-lg"),
            html.Span(text),
        ],
    )


def _dropdown_option(p: dict):
    """A leading-icon row: icon, bold name, muted field-type hint below it."""
    requires = "an API key" if p["input_type"] == "password" else "a base URL"
    return {
        "label": html.Div(className="dash-dropdown-option-rich flex items-center gap-3 py-1", children=[
            html.Span(p["icon"], className="material-symbols-outlined text-xl text-primary/70 flex-shrink-0"),
            html.Div(children=[
                html.Div(p["label"], className="text-sm font-bold text-slate-800"),
                html.Div(f"Requires {requires}", className="text-[11px] text-slate-400"),
            ]),
        ]),
        "value": p["id"],
        # label is a rich component, not plain text - "search" tells the
        # dropdown's own typeahead what text to filter against, so typing
        # in the dropdown IS the search (no separate search box needed).
        "search": p["label"],
    }


def render_provider_field(provider_id: str, field_input_id: str, registry: list, configured_keys: set):
    """The single dynamic field shown once a provider is picked from its dropdown."""
    p = PROVIDER_BY_ID.get(provider_id)
    if not p or p not in registry:
        return html.Div(children=[
            html.P("Choose a provider above to configure it.", className="text-sm text-slate-400 italic py-2"),
            # A hidden placeholder with the same id: the save callback's
            # State(field_input_id, 'value') must always resolve to SOME
            # component, even before a provider is picked and the real
            # dcc.Input doesn't exist yet - otherwise clicking Save with
            # nothing selected crashes the callback instead of showing the
            # graceful "choose a provider first" message.
            dcc.Input(id=field_input_id, style={"display": "none"}),
        ])

    is_configured = p["env_key"] in configured_keys
    is_secret = p["input_type"] == "password"
    placeholder = (
        ("•••••••••• (leave blank to keep current)" if is_configured else "Enter API key")
        if is_secret
        else ("Leave blank to keep current" if is_configured else "Enter value")
    )

    return html.Div(children=[
        html.Div(className="flex items-center justify-between mb-1.5", children=[
            html.Label(f"{p['label']} – {p['field_label']}", className="text-sm font-bold text-slate-700"),
            html.Span(
                "● Configured" if is_configured else "Not set",
                className=(
                    "text-[10px] font-bold uppercase tracking-widest "
                    + ("text-emerald-600" if is_configured else "text-slate-400")
                ),
            ),
        ]),
        html.P(p["hint"], className="text-xs text-slate-400 mb-2"),
        dcc.Input(
            id=field_input_id,
            type=p["input_type"],
            autoComplete="off",
            placeholder=placeholder,
            className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-800 focus:border-primary transition-colors",
        ),
    ])


def render_configured_list(registry: list, configured_keys: set):
    """The list of providers (within one registry) that currently have a value set, each removable."""
    configured_providers = [p for p in registry if p["env_key"] in configured_keys]

    if not configured_providers:
        return html.P("None configured yet - pick one above to get started.", className="text-sm text-slate-400 italic")

    return html.Div(className="space-y-2", children=[
        html.Div(className="flex items-center justify-between px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl", children=[
            html.Div(className="flex items-center gap-2.5", children=[
                html.Span("check_circle", className="material-symbols-outlined text-emerald-600 text-lg"),
                html.Span(p["label"], className="text-sm font-bold text-slate-700"),
            ]),
            html.Button(
                html.Span("close", className="material-symbols-outlined text-lg"),
                id={"type": "settings-remove-btn", "provider": p["id"]},
                n_clicks=0,
                className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors",
            ),
        ]) for p in configured_providers
    ])


def _provider_dialog(*, backdrop_id: str, close_btn_id: str, dialog_status_id: str, title: str, dropdown_id: str, field_container_id: str, field_input_id: str, save_btn_id: str, registry: list, configured_keys: set):
    """
    A centered modal (hidden by default). Its open/closed state lives in a
    dcc.Store (see ui/callbacks/settings.py), rendered to this element's
    `style.display` by exactly one callback - every trigger (Add button,
    close button, a successful save) writes to that store, never directly
    to this element. Mixing an imperative clientside DOM toggle with a
    declarative Dash Output on the SAME element's className was the
    original design here, and it desyncs React's virtual DOM from the real
    DOM: whichever mechanism writes second can silently no-op because React
    doesn't know the other one moved the element out from under it. `style`
    is also more robust than a className-based hidden/flex swap regardless -
    inline display always wins, no Tailwind cascade-order ambiguity.

    Keeps the dropdown+field+save flow out of the page's normal layout, so
    the dropdown's trigger-button-plus-separate-search-box (a Radix
    Combobox, not something Atelier controls the internals of) reads as one
    focused control in a dialog rather than sitting inline on the page.
    Failure messages render inside the dialog (dialog_status_id) - the
    page-level status message sits behind the modal overlay and wouldn't be
    visible while it's open.
    """
    return html.Div(
        id=backdrop_id,
        className="fixed inset-0 bg-black/40 z-[300] items-center justify-center p-4",
        style={"display": "none"},
        children=[
        html.Div(className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6", children=[
            html.Div(className="flex items-center justify-between mb-5", children=[
                html.H3(f"Add {title}", className="text-lg font-extrabold text-slate-900"),
                html.Button(
                    html.Span("close", className="material-symbols-outlined text-xl"),
                    id=close_btn_id, n_clicks=0,
                    className="p-1.5 text-slate-400 hover:text-slate-900 hover:bg-slate-50 rounded-lg transition-colors",
                ),
            ]),
            html.Div(id=dialog_status_id, className="mb-4 empty:hidden"),
            dcc.Dropdown(
                id=dropdown_id,
                options=[_dropdown_option(p) for p in registry],
                placeholder="Choose a provider...",
                clearable=False,
                searchable=True,
                optionHeight=52,
                className="mb-4",
            ),
            html.Div(id=field_container_id, children=render_provider_field(None, field_input_id, registry, configured_keys)),
            html.Button(
                "Save", id=save_btn_id, n_clicks=0,
                className="w-full scholar-gradient text-white px-8 py-2.5 rounded-xl font-bold text-sm uppercase tracking-widest hover:opacity-90 active:scale-95 transition-all shadow-lg shadow-primary/20 mt-4",
            ),
        ]),
    ])


def _provider_section(*, title: str, description: str, add_btn_id: str, dialog_open_store_id: str, backdrop_id: str, close_btn_id: str, dialog_status_id: str, dropdown_id: str, field_container_id: str, field_input_id: str, save_btn_id: str, registry: list, configured_list_id: str, configured_keys: set):
    return html.Div(className="mb-12", children=[
        dcc.Store(id=dialog_open_store_id, data=False),

        html.Div(className="flex items-center justify-between mb-1", children=[
            html.H2(title, className="text-lg font-extrabold text-slate-900"),
            html.Button(
                [html.Span("add", className="material-symbols-outlined text-lg"), html.Span("Add Provider", className="text-xs font-bold uppercase tracking-widest")],
                id=add_btn_id, n_clicks=0,
                className="flex items-center gap-1.5 px-4 py-2 bg-primary/10 text-primary hover:bg-primary/20 rounded-xl transition-colors flex-shrink-0",
            ),
        ]),
        html.P(description, className="text-sm text-slate-500 mb-5"),

        _provider_dialog(
            backdrop_id=backdrop_id, close_btn_id=close_btn_id, dialog_status_id=dialog_status_id, title=title,
            dropdown_id=dropdown_id, field_container_id=field_container_id, field_input_id=field_input_id,
            save_btn_id=save_btn_id, registry=registry, configured_keys=configured_keys,
        ),

        html.H3("Configured", className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3"),
        html.Div(id=configured_list_id, children=render_configured_list(registry, configured_keys)),
    ])


def layout_settings():
    """Route: '/settings' - Encrypted, UI-configurable API key / endpoint management."""
    configured_keys = set(AtelierRepository.get_configured_setting_keys())

    return html.Div(className="flex-1 overflow-y-auto px-8 py-12 bg-[#FDFDFD]", children=[
        html.Div(className="max-w-2xl mx-auto", children=[
            html.H1("Settings", className="text-3xl font-extrabold tracking-tighter text-slate-900 mb-2"),
            html.P(
                "API keys and endpoints entered here are encrypted and stored locally on this machine. "
                "They take priority over your .env file and apply immediately - no restart needed.",
                className="text-sm text-slate-500 mb-8 max-w-lg",
            ),

            html.Div(id="settings-status-message", className="mb-8 empty:hidden"),

            _provider_section(
                title="LLM Providers",
                description="Powers query planning, screening, and synthesis. Select at least one to use Atelier.",
                add_btn_id="settings-llm-add-btn",
                dialog_open_store_id="settings-llm-dialog-open",
                backdrop_id="settings-llm-dialog-backdrop",
                close_btn_id="settings-llm-dialog-close-btn",
                dialog_status_id="settings-llm-dialog-status",
                dropdown_id="settings-llm-provider-select",
                field_container_id="settings-llm-field-container",
                field_input_id="settings-llm-dynamic-input",
                save_btn_id="settings-llm-save-btn",
                registry=LLM_PROVIDERS,
                configured_list_id="settings-llm-configured-list",
                configured_keys=configured_keys,
            ),

            _provider_section(
                title="Academic Search Providers",
                description="PubMed and Europe PMC always work with no setup. These two are optional - only raise your rate limit.",
                add_btn_id="settings-search-add-btn",
                dialog_open_store_id="settings-search-dialog-open",
                backdrop_id="settings-search-dialog-backdrop",
                close_btn_id="settings-search-dialog-close-btn",
                dialog_status_id="settings-search-dialog-status",
                dropdown_id="settings-search-provider-select",
                field_container_id="settings-search-field-container",
                field_input_id="settings-search-dynamic-input",
                save_btn_id="settings-search-save-btn",
                registry=SEARCH_PROVIDERS,
                configured_list_id="settings-search-configured-list",
                configured_keys=configured_keys,
            ),
        ]),
    ])
