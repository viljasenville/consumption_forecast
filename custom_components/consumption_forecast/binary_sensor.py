"""Binary sensors flagging high / low consumption hours.

The coordinator marks, for each forecast day, the N highest and N lowest hours
(N = CONF_PEAK_HOURS) that also sit on the correct side of that day's mean. These
sensors are ON while the current clock hour is one of those flagged hours, so
automations can shift load toward low hours or away from high ones.
"""
from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import CONF_PEAK_HOURS, DEFAULT_PEAK_HOURS, DOMAIN
from .coordinator import ForecastCoordinator, _parse_iso


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: ForecastCoordinator = hass.data[DOMAIN][entry.entry_id]

    # peak_hours = 0 disables the feature -> add no binary sensors.
    if int(coordinator.cfg.get(CONF_PEAK_HOURS, DEFAULT_PEAK_HOURS)) <= 0:
        return

    async_add_entities(
        [
            PeakHourBinarySensor(
                coordinator, key="peak_high",
                name="High consumption hour", data_key="peak_high",
            ),
            PeakHourBinarySensor(
                coordinator, key="peak_low",
                name="Low consumption hour", data_key="peak_low",
            ),
        ]
    )


class PeakHourBinarySensor(
    CoordinatorEntity[ForecastCoordinator], BinarySensorEntity
):
    _attr_has_entity_name = True

    def __init__(self, coordinator, key: str, name: str, data_key: str):
        super().__init__(coordinator)
        self._data_key = data_key
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{key}"
        # translated via entity.binary_sensor.<key>.name
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="Consumption Forecast",
            manufacturer="Consumption Forecast",
            model="Forecast",
        )

    async def async_added_to_hass(self) -> None:
        """Re-evaluate at the top of every hour.

        `is_on` depends on the current clock hour, but the coordinator refreshes
        the forecast only about once an hour and not aligned to the wall clock,
        so without this the sensor would flip minutes late (whenever the next
        coordinator update happens to land). Ticking at :00 makes it turn on/off
        exactly on the hour, independently of the refresh cadence.
        """
        await super().async_added_to_hass()
        self.async_on_remove(
            async_track_time_change(
                self.hass, self._handle_hour_tick, minute=0, second=0
            )
        )

    @callback
    def _handle_hour_tick(self, now) -> None:
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool | None:
        data = self.coordinator.data or {}
        flagged = data.get(self._data_key)
        if flagged is None:
            return None
        # Compare by INSTANT, not by ISO string. The flagged timestamps come from
        # the weather forecast and may carry a different tz offset spelling than
        # dt_util.now(), so a text compare ("... in flagged") can miss even when
        # the hour is the same. Parse both sides to the local hour and compare.
        now_hour = dt_util.now(self.coordinator.tz).replace(
            minute=0, second=0, microsecond=0
        )
        for iso in flagged:
            dt = _parse_iso(iso)
            if dt is None:
                continue
            if dt.astimezone(self.coordinator.tz).replace(
                minute=0, second=0, microsecond=0
            ) == now_hour:
                return True
        return False

    @property
    def extra_state_attributes(self):
        flagged = (self.coordinator.data or {}).get(self._data_key, [])
        return {"flagged_hours": flagged}
