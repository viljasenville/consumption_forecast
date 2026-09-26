"""Services for manual model training, forecast refresh and ad-hoc forecasts.

Services registered under the integration domain:

  consumption_forecast.train_model       - force an immediate retrain
  consumption_forecast.refresh_forecast  - recompute the forecast with the current model
  consumption_forecast.forecast_period   - forecast consumption over an arbitrary
                                           period from a start, end and a single
                                           outdoor temperature (returns response data)

train_model and refresh_forecast accept an optional ``entry_id`` to target a
specific config entry; without it they apply to every configured entry.
"""
from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

SERVICE_TRAIN_MODEL = "train_model"
SERVICE_REFRESH_FORECAST = "refresh_forecast"
SERVICE_FORECAST_PERIOD = "forecast_period"

ATTR_ENTRY_ID = "entry_id"
ATTR_START = "start"
ATTR_END = "end"
ATTR_OUTDOOR_TEMP = "outdoor_temp"
ATTR_TARGET_TEMP = "target_temp"

_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTRY_ID): cv.string,
    }
)

_FORECAST_PERIOD_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_START): cv.datetime,
        vol.Required(ATTR_END): cv.datetime,
        vol.Required(ATTR_OUTDOOR_TEMP): vol.Coerce(float),
        vol.Optional(ATTR_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(ATTR_ENTRY_ID): cv.string,
    }
)


def _target_coordinators(hass: HomeAssistant, call: ServiceCall):
    """Resolve the coordinators the service call applies to."""
    store = hass.data.get(DOMAIN, {})
    if not store:
        raise HomeAssistantError("Consumption Forecast is not set up.")

    entry_id = call.data.get(ATTR_ENTRY_ID)
    if entry_id is not None:
        coordinator = store.get(entry_id)
        if coordinator is None:
            raise HomeAssistantError(f"Unknown entry_id: {entry_id}")
        return [coordinator]
    return list(store.values())


def _single_coordinator(hass: HomeAssistant, call: ServiceCall):
    """Resolve exactly one coordinator for a response-returning service.

    entry_id may be omitted only when a single entry is configured.
    """
    store = hass.data.get(DOMAIN, {})
    if not store:
        raise HomeAssistantError("Consumption Forecast is not set up.")

    entry_id = call.data.get(ATTR_ENTRY_ID)
    if entry_id is not None:
        coordinator = store.get(entry_id)
        if coordinator is None:
            raise HomeAssistantError(f"Unknown entry_id: {entry_id}")
        return coordinator
    if len(store) > 1:
        raise HomeAssistantError(
            "Multiple Consumption Forecast entries configured; specify entry_id."
        )
    return next(iter(store.values()))


async def _async_handle_train(hass: HomeAssistant, call: ServiceCall) -> None:
    for coordinator in _target_coordinators(hass, call):
        try:
            trained = await coordinator.async_train()
        except Exception as err:  # surface training failures to the caller
            raise HomeAssistantError(f"Training failed: {err}") from err
        if trained:
            _LOGGER.info(
                "Manual training complete, active model: %s",
                coordinator.active_model_name,
            )
        else:
            _LOGGER.warning("Manual training skipped: not enough data yet.")


async def _async_handle_refresh(hass: HomeAssistant, call: ServiceCall) -> None:
    for coordinator in _target_coordinators(hass, call):
        # recompute the forecast with the current model and push new state
        # to the sensors.
        await coordinator.async_request_refresh()
        _LOGGER.info("Manual forecast refresh complete.")


async def _async_handle_forecast_period(
    hass: HomeAssistant, call: ServiceCall
) -> ServiceResponse:
    coordinator = _single_coordinator(hass, call)
    start = call.data[ATTR_START]
    end = call.data[ATTR_END]
    outdoor_temp = call.data[ATTR_OUTDOOR_TEMP]
    target_temp = call.data.get(ATTR_TARGET_TEMP)
    try:
        result = await coordinator.async_forecast_period(
            start, end, outdoor_temp, target_temp
        )
    except HomeAssistantError:
        raise
    except Exception as err:  # surface unexpected failures to the caller
        raise HomeAssistantError(f"Forecast failed: {err}") from err
    return result


def async_register_services(hass: HomeAssistant) -> None:
    """Register the integration services once (idempotent)."""
    if hass.services.has_service(DOMAIN, SERVICE_TRAIN_MODEL):
        return

    async def train_model(call: ServiceCall) -> None:
        await _async_handle_train(hass, call)

    async def refresh_forecast(call: ServiceCall) -> None:
        await _async_handle_refresh(hass, call)

    async def forecast_period(call: ServiceCall) -> ServiceResponse:
        return await _async_handle_forecast_period(hass, call)

    hass.services.async_register(
        DOMAIN, SERVICE_TRAIN_MODEL, train_model, schema=_SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN, SERVICE_REFRESH_FORECAST, refresh_forecast, schema=_SERVICE_SCHEMA
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_FORECAST_PERIOD,
        forecast_period,
        schema=_FORECAST_PERIOD_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def async_unregister_services(hass: HomeAssistant) -> None:
    """Remove services when the last config entry is unloaded."""
    hass.services.async_remove(DOMAIN, SERVICE_TRAIN_MODEL)
    hass.services.async_remove(DOMAIN, SERVICE_REFRESH_FORECAST)
    hass.services.async_remove(DOMAIN, SERVICE_FORECAST_PERIOD)
