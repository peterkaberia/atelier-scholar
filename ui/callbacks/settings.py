"""
Settings Callbacks Module

Drives the two independent provider sections on '/settings' (LLM, Academic
Search). Each dialog's open/closed state lives in ONE dcc.Store, written by
whichever of three triggers fires (Add button, close button, a successful
save) and rendered to the dialog's `style.display` by exactly one callback -
deliberately never touched by a clientside DOM toggle. An earlier version
mixed an imperative clientside classList toggle with a declarative Dash
Output on the same element's className; that desyncs React's virtual DOM
from the real DOM (whichever mechanism writes second can silently no-op),
which is why the dialogs sometimes opened on page load or failed to close
on save. Routing everything through one Store with one rendering callback
removes that class of bug entirely.

Values are never echoed back once saved - each dynamic field re-renders
blank after every save.
"""

from dash import ALL, ctx, Input, Output, State, callback, no_update

from database import AtelierRepository
from llm.model_catalog import validate_provider_credential
from ui.layouts.settings import (
    LLM_PROVIDERS,
    SEARCH_PROVIDERS,
    PROVIDER_BY_ID,
    render_configured_list,
    render_provider_field,
    status_message,
)


_DIALOG_OPEN_STYLE = {"display": "flex"}
_DIALOG_CLOSED_STYLE = {"display": "none"}


def _register_section_callbacks(*, add_btn_id, close_btn_id, dialog_open_store_id, backdrop_id, dropdown_id, field_container_id, field_input_id, save_btn_id, dialog_status_id, registry, configured_list_id):
    """
    Wires one section's open/close, dropdown->field, and save->persist+
    refresh flow. Called once per section below with each section's own
    component ids, so the LLM and Academic Search sections stay fully
    independent.
    """

    @callback(
        Output(dialog_open_store_id, 'data'),
        Input(add_btn_id, 'n_clicks'),
        Input(close_btn_id, 'n_clicks'),
        prevent_initial_call=True,
    )
    def toggle_dialog(_open_clicks, _close_clicks):
        return ctx.triggered_id == add_btn_id

    @callback(
        Output(backdrop_id, 'style'),
        Input(dialog_open_store_id, 'data'),
    )
    def render_dialog_visibility(is_open):
        return _DIALOG_OPEN_STYLE if is_open else _DIALOG_CLOSED_STYLE

    @callback(
        Output(field_container_id, 'children'),
        Output(dialog_status_id, 'children'),
        Input(dropdown_id, 'value'),
    )
    def show_field(provider_id):
        # Clears any previous error the moment the user picks a different
        # provider, rather than leaving a stale failure message showing.
        configured_keys = set(AtelierRepository.get_configured_setting_keys())
        return render_provider_field(provider_id, field_input_id, registry, configured_keys), ""

    @callback(
        Output('settings-status-message', 'children', allow_duplicate=True),
        Output(configured_list_id, 'children'),
        Output(field_container_id, 'children', allow_duplicate=True),
        Output(dialog_open_store_id, 'data', allow_duplicate=True),
        Output(dialog_status_id, 'children', allow_duplicate=True),
        Input(save_btn_id, 'n_clicks'),
        State(dropdown_id, 'value'),
        State(field_input_id, 'value'),
        running=[
            (Output(save_btn_id, 'children'), 'Checking...', 'Save'),
            (Output(save_btn_id, 'disabled'), True, False),
        ],
        prevent_initial_call=True,
    )
    def save(n_clicks, provider_id, value):
        provider = PROVIDER_BY_ID.get(provider_id)

        # Failures render INSIDE the dialog (dialog_status_id) and keep it
        # open (no_update on the store) - the page-level status message
        # sits behind the modal overlay and wouldn't be visible while the
        # dialog is up.
        if not provider:
            return no_update, no_update, no_update, no_update, status_message("Choose a provider first.", ok=False)

        if not value or not value.strip():
            return no_update, no_update, no_update, no_update, status_message("Enter a value before saving.", ok=False)

        value = value.strip()

        # Live-check before persisting: a bad key or dead URL should fail
        # loudly here, not get saved and only surface later as "no models".
        is_valid, reason = validate_provider_credential(provider_id, value)
        if not is_valid:
            return no_update, no_update, no_update, no_update, status_message(f"{provider['label']}: {reason}", ok=False)

        AtelierRepository.set_setting(provider['env_key'], value)

        configured_keys = set(AtelierRepository.get_configured_setting_keys())
        return (
            status_message(f"Saved {provider['label']}. Changes apply immediately."),
            render_configured_list(registry, configured_keys),
            render_provider_field(provider_id, field_input_id, registry, configured_keys),  # badge flips, input clears
            False,  # close the dialog on success
            "",  # clear any prior in-dialog error
        )


_register_section_callbacks(
    add_btn_id='settings-llm-add-btn',
    close_btn_id='settings-llm-dialog-close-btn',
    dialog_open_store_id='settings-llm-dialog-open',
    backdrop_id='settings-llm-dialog-backdrop',
    dropdown_id='settings-llm-provider-select',
    field_container_id='settings-llm-field-container',
    field_input_id='settings-llm-dynamic-input',
    save_btn_id='settings-llm-save-btn',
    dialog_status_id='settings-llm-dialog-status',
    registry=LLM_PROVIDERS,
    configured_list_id='settings-llm-configured-list',
)

_register_section_callbacks(
    add_btn_id='settings-search-add-btn',
    close_btn_id='settings-search-dialog-close-btn',
    dialog_open_store_id='settings-search-dialog-open',
    backdrop_id='settings-search-dialog-backdrop',
    dropdown_id='settings-search-provider-select',
    field_container_id='settings-search-field-container',
    field_input_id='settings-search-dynamic-input',
    save_btn_id='settings-search-save-btn',
    dialog_status_id='settings-search-dialog-status',
    registry=SEARCH_PROVIDERS,
    configured_list_id='settings-search-configured-list',
)


@callback(
    Output('settings-llm-configured-list', 'children', allow_duplicate=True),
    Output('settings-search-configured-list', 'children', allow_duplicate=True),
    Output('settings-status-message', 'children', allow_duplicate=True),
    Input({'type': 'settings-remove-btn', 'provider': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def remove_provider(all_clicks):
    """Deletes whichever configured provider's remove (x) button was clicked, in either section."""
    if not any(all_clicks):
        return no_update, no_update, no_update

    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update, no_update

    provider = PROVIDER_BY_ID.get(triggered.get('provider'))
    if not provider:
        return no_update, no_update, no_update

    AtelierRepository.delete_setting(provider['env_key'])

    configured_keys = set(AtelierRepository.get_configured_setting_keys())
    return (
        render_configured_list(LLM_PROVIDERS, configured_keys),
        render_configured_list(SEARCH_PROVIDERS, configured_keys),
        status_message(f"Removed {provider['label']}."),
    )
