"""Consumption Forecast integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .addon import ModelServiceError, async_get_client
from .const import DOMAIN
from .coordinator import ForecastCoordinator
from .services import async_register_services, async_unregister_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR, Platform.BINARY_SENSOR, Platform.BUTTON]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coordinator = ForecastCoordinator(hass, entry)
    await coordinator.async_setup()
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload))

    # Register services once; safe to call on every entry setup (idempotent).
    async_register_services(hass)
    return True


async def _async_reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id, None)
        if coordinator:
            await coordinator.async_shutdown()
        # Remove services when the last entry is gone.
        if not hass.data.get(DOMAIN):
            async_unregister_services(hass)
    return unloaded


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the models the add-on trained for this entry.

    Models the add-on holds are stored under this config entry's id, outside
    Home Assistant, so removing the entry has to tell the add-on to drop them or
    they would linger on disk forever. Every backend is asked (the call is
    idempotent) so models left behind by an earlier selection go too. Purely
    best-effort: a missing or stopped add-on is not a reason to fail removal.
    """
    client = await async_get_client(hass, entry.entry_id)
    if client is None:
        return
    try:
        models = await client.async_list_models()
    except ModelServiceError as err:
        _LOGGER.debug("Could not list add-on models while removing entry: %s", err)
        return
    for model in models:
        try:
            await client.async_delete_instance(model["id"])
        except ModelServiceError as err:
            _LOGGER.debug("Could not delete add-on model %s: %s", model["id"], err)
