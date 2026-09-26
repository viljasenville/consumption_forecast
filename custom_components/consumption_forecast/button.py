"""Buttons for on-demand actions.

- Train model: retrains the model (same as the train_model service).
- Refresh forecast: recomputes the forecast with the current model, fetching a
  fresh weather forecast (same as the refresh_forecast service). Light and fast,
  no retraining.
"""
from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import ForecastCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: ForecastCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [TrainModelButton(coordinator), RefreshForecastButton(coordinator)]
    )


class TrainModelButton(ButtonEntity):
    _attr_has_entity_name = True
    _attr_icon = "mdi:brain"

    def __init__(self, coordinator: ForecastCoordinator):
        self._coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_train_model"
        self._attr_translation_key = "train_model"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="Consumption Forecast",
            manufacturer="Consumption Forecast",
            model="Forecast",
        )

    async def async_press(self) -> None:
        """Retrain now. Surfaces a clear error if there is not enough data."""
        try:
            trained = await self._coordinator.async_train()
        except Exception as err:  # surface training failures to the UI
            raise HomeAssistantError(f"Training failed: {err}") from err
        if not trained:
            raise HomeAssistantError(
                "Not enough recorded history to train yet."
            )


class RefreshForecastButton(ButtonEntity):
    _attr_has_entity_name = True
    _attr_icon = "mdi:refresh"

    def __init__(self, coordinator: ForecastCoordinator):
        self._coordinator = coordinator
        self._attr_unique_id = f"{coordinator.entry.entry_id}_refresh_forecast"
        self._attr_translation_key = "refresh_forecast"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="Consumption Forecast",
            manufacturer="Consumption Forecast",
            model="Forecast",
        )

    async def async_press(self) -> None:
        """Recompute the forecast now with the current model (no retraining)."""
        await self._coordinator.async_request_refresh()
