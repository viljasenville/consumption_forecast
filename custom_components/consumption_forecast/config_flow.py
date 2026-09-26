"""Config flow: selecting input entities, the model and parameters from the UI."""
from __future__ import annotations

import logging
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.helpers import selector

from .addon import async_available_models
from .const import (
    CONF_BASE_TEMP,
    CONF_ENERGY,
    CONF_EXPORT,
    CONF_FORECAST,
    CONF_HISTORY_DAYS,
    CONF_INDOOR,
    CONF_MAX_HOURLY_KWH,
    CONF_MODEL,
    CONF_OUTDOOR,
    CONF_PEAK_HOURS,
    CONF_PRODUCTION,
    CONF_THERMOSTAT,
    CONF_TIMEZONE,
    DEFAULT_BASE_TEMP,
    DEFAULT_MAX_HOURLY_KWH,
    DEFAULT_MODEL,
    DEFAULT_PEAK_HOURS,
    DEFAULT_TIMEZONE,
    DOMAIN,
    MODEL_PROFILE,
)

_LOGGER = logging.getLogger(__name__)

# Label for the built-in model, which is always offered first.
_PROFILE_LABEL = "Built-in profile model (no add-on needed)"


def _is_valid_timezone(value: str) -> bool:
    """Return True if value is a valid IANA timezone name.

    Used for manual validation after submit — NOT as a schema validator, because
    a bare function cannot be serialized to JSON for the config-flow UI.
    """
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


async def _model_options(hass, model_id: str) -> list[selector.SelectOptionDict]:
    """The models to offer, discovered from the add-on at form time.

    The built-in profile model is always first and always present — it needs no
    add-on. Everything after it comes from the add-on's ``GET /models``, so an
    add-on that gains a new backend shows up here without a release of this
    integration. With no add-on installed the list is just the one entry.
    """
    options = [
        selector.SelectOptionDict(value=MODEL_PROFILE, label=_PROFILE_LABEL)
    ]
    for model in await async_available_models(hass, model_id):
        options.append(
            selector.SelectOptionDict(
                value=model["id"], label=f"{model.get('name') or model['id']} (add-on)"
            )
        )
    return options


def _model_default(defaults: dict, options: list[selector.SelectOptionDict]) -> str:
    """The selection to pre-fill, falling back to the built-in model.

    A stored choice whose add-on is no longer installed is not offered, so the
    form would otherwise show an invalid value. Showing the built-in model
    instead makes the fallback the integration is already using visible.
    """
    stored = defaults.get(CONF_MODEL) or DEFAULT_MODEL
    if any(o["value"] == stored for o in options):
        return stored
    return MODEL_PROFILE


def _base_schema(
    defaults: dict, model_options: list[selector.SelectOptionDict]
) -> vol.Schema:
    """Shared schema for the config and options flow."""
    return vol.Schema(
        {
            vol.Required(
                CONF_ENERGY, default=defaults.get(CONF_ENERGY)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy")
            ),
            vol.Optional(
                CONF_PRODUCTION, default=defaults.get(CONF_PRODUCTION)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy")
            ),
            vol.Optional(
                CONF_EXPORT, default=defaults.get(CONF_EXPORT)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy")
            ),
            vol.Required(
                CONF_OUTDOOR, default=defaults.get(CONF_OUTDOOR)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor", device_class="temperature"
                )
            ),
            vol.Optional(
                CONF_INDOOR, default=defaults.get(CONF_INDOOR)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor", device_class="temperature"
                )
            ),
            vol.Required(
                CONF_FORECAST, default=defaults.get(CONF_FORECAST)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="weather")
            ),
            vol.Optional(
                CONF_THERMOSTAT, default=defaults.get(CONF_THERMOSTAT)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="climate")
            ),
            # Which model produces the forecast. The built-in profile model runs
            # in Home Assistant itself; every other choice is trained and served
            # by the model provider add-on, and falls back to the built-in model
            # whenever the add-on cannot answer.
            vol.Optional(
                CONF_MODEL,
                default=_model_default(defaults, model_options),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=model_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_BASE_TEMP,
                default=defaults.get(CONF_BASE_TEMP, DEFAULT_BASE_TEMP),
            ): vol.Coerce(float),
            # Optional training window. Leave empty to use ALL available
            # history (bounded by the recorder's retention); a number narrows
            # it to that many days (min 14). A NumberSelector in box mode renders
            # a field that can be left blank -- unlike vol.Any(), which the
            # config-flow frontend cannot render into an input at all.
            vol.Optional(
                CONF_HISTORY_DAYS,
                description={"suggested_value": defaults.get(CONF_HISTORY_DAYS)},
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=14, max=3650, step=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_TIMEZONE,
                default=defaults.get(CONF_TIMEZONE, DEFAULT_TIMEZONE),
            ): selector.TextSelector(),
            vol.Optional(
                CONF_MAX_HOURLY_KWH,
                default=defaults.get(CONF_MAX_HOURLY_KWH, DEFAULT_MAX_HOURLY_KWH),
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1)),
            # How many highest/lowest hours per day the peak binary sensors flag.
            # 0 disables them.
            vol.Optional(
                CONF_PEAK_HOURS,
                default=defaults.get(CONF_PEAK_HOURS, DEFAULT_PEAK_HOURS),
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=12)),
        }
    )


def _clean(user_input: dict) -> dict:
    """Drop empty optional entities so they are not stored as None.

    Used for the initial config flow, where there is no prior value to
    override -- an unset optional field simply should not be stored.
    """
    return {k: v for k, v in user_input.items() if v not in (None, "")}


# Optional entity inputs that the user may later want to clear. When cleared in
# the options flow they must be stored as explicit "" so they override the
# original entry.data value the coordinator merges underneath options.
_OPTIONAL_ENTITY_KEYS = (
    CONF_INDOOR,
    CONF_THERMOSTAT,
    CONF_PRODUCTION,
    CONF_EXPORT,
)


def _normalize_options(user_input: dict) -> dict:
    """Prepare options-flow input for storage.

    Non-empty values are kept as-is. Optional entity fields the user cleared are
    recorded as "" (not dropped) so they override entry.data; other empty fields
    are dropped so their defaults apply.

    IMPORTANT: a Home Assistant options form OMITS a cleared optional field from
    user_input entirely (rather than passing it as "" or None). So clearing
    cannot be detected by iterating user_input alone -- we must also check which
    "clearable" keys are ABSENT and record the cleared marker for them, or the
    old entry.data value keeps winning and the clear appears to do nothing.
    """
    out: dict = {}
    for key, value in user_input.items():
        if value not in (None, ""):
            out[key] = value
        elif key in _OPTIONAL_ENTITY_KEYS:
            out[key] = ""  # explicit "cleared" marker that overrides entry.data
        elif key == CONF_HISTORY_DAYS:
            out[key] = None  # cleared training window -> use all history
        # other empty fields: drop, so the default is used

    # Handle keys the form dropped because they were cleared (absent from
    # user_input): record the same cleared marker so they override entry.data.
    for key in _OPTIONAL_ENTITY_KEYS:
        if key not in user_input:
            out[key] = ""
    if CONF_HISTORY_DAYS not in user_input:
        out[CONF_HISTORY_DAYS] = None

    return out


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup config flow."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            if not _is_valid_timezone(user_input.get(CONF_TIMEZONE, DEFAULT_TIMEZONE)):
                errors[CONF_TIMEZONE] = "invalid_timezone"
            if not errors:
                return self.async_create_entry(
                    title="Consumption Forecast", data=_clean(user_input)
                )

        # No config entry exists yet, so there is no entry id to use as the
        # add-on's model_id; listing models does not need one.
        model_options = await _model_options(self.hass, DOMAIN)
        return self.async_show_form(
            step_id="user",
            data_schema=_base_schema(user_input or {}, model_options),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(entry):
        return OptionsFlow(entry)


class OptionsFlow(config_entries.OptionsFlow):
    """Editing settings later (all the same fields)."""

    def __init__(self, entry):
        self.entry = entry

    async def async_step_init(self, user_input=None):
        errors = {}
        if user_input is not None:
            if not _is_valid_timezone(user_input.get(CONF_TIMEZONE, DEFAULT_TIMEZONE)):
                errors[CONF_TIMEZONE] = "invalid_timezone"
            if not errors:
                # Store the full form. Optional entities the user cleared arrive
                # as None/"" and are recorded as explicit empty strings so they
                # OVERRIDE any value in the original entry.data (which the
                # coordinator merges underneath options). Without this, clearing
                # an optional entity in the UI has no effect because the old
                # entry.data value keeps winning.
                cleaned = _normalize_options(user_input)
                return self.async_create_entry(title="", data=cleaned)

        defaults = {**self.entry.data, **self.entry.options}
        # a cleared entity is stored as "" -> show the field empty, not the old value
        defaults = {k: v for k, v in defaults.items() if v not in (None, "")}
        # Re-read the add-on's model list every time the form is opened, so a
        # newly installed add-on appears without restarting Home Assistant.
        model_options = await _model_options(self.hass, self.entry.entry_id)
        return self.async_show_form(
            step_id="init",
            data_schema=_base_schema(defaults, model_options),
            errors=errors,
        )
