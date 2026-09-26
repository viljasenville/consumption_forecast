"""Presentation layer: consumption forecast sensors."""
from __future__ import annotations

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import ForecastCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: ForecastCoordinator = hass.data[DOMAIN][entry.entry_id]

    # The integration forecasts a single consumption quantity (total house
    # consumption). Production/export inputs, when configured, only correct the
    # history the model trains on; there is no separate grid-purchase forecast.
    entities: list[SensorEntity] = [
        DailySensor(coordinator, 0, "today", "Today"),
        DailySensor(coordinator, 1, "tomorrow", "Tomorrow"),
        WeekSensor(coordinator),
        HourlySensor(coordinator),
        # Current-hour forecast as a plain recordable STATE (not an attribute),
        # so the recorder keeps its history and it can be charted against the
        # actual consumption to see how past forecasts landed.
        CurrentHourForecastSensor(
            coordinator, key="current_hour",
            name="Current hour", hourly_key="hourly",
        ),
        # forecast-vs-actual accuracy (state = deviation %, MAE in attributes)
        AccuracySensor(
            coordinator, key="accuracy_today",
            name="Accuracy today", window="today",
        ),
        AccuracySensor(
            coordinator, key="accuracy_yesterday",
            name="Accuracy yesterday", window="yesterday",
        ),
        AccuracySensor(
            coordinator, key="accuracy_week",
            name="Accuracy week", window="week",
        ),
        AccuracySensor(
            coordinator, key="accuracy_month",
            name="Accuracy month", window="month",
        ),
    ]
    async_add_entities(entities)


class _BaseEntity(CoordinatorEntity[ForecastCoordinator], SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: ForecastCoordinator, key: str, name: str = ""):
        super().__init__(coordinator)
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{key}"
        # Name comes from translations (entity.sensor.<key>.name), not a
        # hardcoded string, so it can be shown in the user's language. The
        # English translation matches the old name, keeping entity_ids stable.
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, coordinator.entry.entry_id)},
            name="Consumption Forecast",
            manufacturer="Consumption Forecast",
            model="Forecast",
        )


class _EnergyEntity(_BaseEntity):
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = "kWh"
    _attr_state_class = SensorStateClass.MEASUREMENT


class DailySensor(_EnergyEntity):
    """Single-day consumption forecast (index 0=today, 1=tomorrow)."""

    def __init__(self, coordinator, day_index: int, key: str, name: str,
                 daily_key: str = "daily"):
        super().__init__(coordinator, key, name)
        self._i = day_index
        self._daily_key = daily_key

    @property
    def native_value(self):
        daily = (self.coordinator.data or {}).get(self._daily_key, [])
        if len(daily) > self._i:
            return daily[self._i]["kwh"]
        return None

    @property
    def extra_state_attributes(self):
        daily = (self.coordinator.data or {}).get(self._daily_key, [])
        if len(daily) > self._i:
            return {"date": daily[self._i]["date"]}
        return {}


class WeekSensor(_EnergyEntity):
    """Total consumption over the NEXT 7 days (excluding today) + daily breakdown.

    Mirrors the accuracy sensors: the current-day forecast has its own sensor
    (partial day in progress), and this "week" window covers the next 7 whole
    days ahead — tomorrow through today+7 — so it is a clean forward window, not
    skewed by the hours already lived today.
    """

    # next 7 days: daily indices 1..7 (0 is today, which has its own sensor)
    _SLICE = slice(1, 8)

    def __init__(self, coordinator, key: str = "week", name: str = "Week"):
        super().__init__(coordinator, key, name)

    @property
    def native_value(self):
        daily = (self.coordinator.data or {}).get("daily", [])
        days = daily[self._SLICE]
        return round(sum(x["kwh"] for x in days), 1) if days else None

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data or {}
        coeffs = data.get("coeffs", {}) or {}
        attrs = {
            # only the next-7-day rows, so this breakdown sums to the state
            "days": data.get("daily", [])[self._SLICE],
            # which model produced this forecast, and whether it ran in Home
            # Assistant or in the model provider add-on -- so a number can
            # always be traced to where it came from
            "active_model": data.get("active_model"),
            "model_source": data.get("model_source"),
            # validation MAE of the model that produced the forecast (kWh/h);
            # falls back to the built-in model's figure on older data
            "val_mae": data.get("val_mae", coeffs.get("val_mae")),
            "base_load": coeffs.get("base_load"),
            "per_hdd": coeffs.get("per_hdd"),
            # add-on state and its last training result; None without an add-on
            "addon": data.get("addon"),
            # weather (temps + HDD) the forecast used, per day — for diagnosing
            # why the forecast is higher or lower than expected.
            "forecast_weather": data.get("forecast_weather", []),
            "generated": data.get("generated"),
        }
        # training diagnostics: when the model was last trained and over what
        # span of data (set on the coordinator, persisted across restarts).
        info = self.coordinator.training_info or {}
        attrs.update(
            {
                "trained_at": info.get("trained_at"),
                "training_start": info.get("training_start"),
                "training_end": info.get("training_end"),
                "training_span_days": info.get("training_span_days"),
                "training_hours": info.get("training_hours"),
                # which input supplied the heating-degree reference
                "hdd_reference": info.get("hdd_reference"),
            }
        )
        return attrs


class HourlySensor(_EnergyEntity):
    """Next-hour estimate; full hourly table in the attribute (ApexCharts)."""

    def __init__(self, coordinator, key: str = "hourly",
                 name: str = "Hourly",
                 hourly_key: str = "hourly"):
        super().__init__(coordinator, key, name)
        self._hourly_key = hourly_key

    @property
    def native_value(self):
        hourly = (self.coordinator.data or {}).get(self._hourly_key, [])
        return hourly[0]["kwh"] if hourly else None

    @property
    def extra_state_attributes(self):
        hourly = (self.coordinator.data or {}).get(self._hourly_key, [])
        return {
            "forecast": [
                {"datetime": x["ts"].isoformat(), "kwh": x["kwh"]} for x in hourly
            ]
        }


class CurrentHourForecastSensor(_EnergyEntity):
    """Forecast for the CURRENT clock hour, exposed as the sensor state.

    Unlike HourlySensor (whose full table lives in an attribute and is not kept
    by the recorder), this sensor's *state* is a single number: the forecast for
    the hour we are in right now. The recorder stores state history, so plotting
    this sensor's history next to the actual consumption shows how each past
    hour's forecast compared with what actually happened.
    """

    def __init__(self, coordinator, key: str, name: str, hourly_key: str = "hourly"):
        super().__init__(coordinator, key, name)
        self._hourly_key = hourly_key

    async def async_added_to_hass(self) -> None:
        """Re-evaluate at the top of every hour.

        The state is the forecast for the current clock hour, so it must change
        when the hour changes — not only when the coordinator next refreshes
        (which is roughly hourly but not aligned to :00).
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
    def native_value(self):
        hourly = (self.coordinator.data or {}).get(self._hourly_key, [])
        if not hourly:
            return None
        # match the row whose timestamp is the current local hour
        now_hour = dt_util.now(self.coordinator.tz).replace(
            minute=0, second=0, microsecond=0
        )
        for row in hourly:
            ts = row["ts"]
            if ts.replace(minute=0, second=0, microsecond=0) == now_hour:
                return row["kwh"]
        # no exact match (e.g. just after restart before a fresh forecast):
        # fall back to the first upcoming hour so the sensor is not blank.
        return hourly[0]["kwh"]

    @property
    def extra_state_attributes(self):
        return {"hour": dt_util.now(self.coordinator.tz).replace(
            minute=0, second=0, microsecond=0).isoformat()}



class AccuracySensor(_BaseEntity):
    """Forecast-vs-actual accuracy for a window ("yesterday" or "week").

    The state is the deviation in percent (how far the forecast total was from
    the actual total). The MAE (kWh/h) and the underlying totals are in the
    attributes. Unavailable until there are hours with both a recorded forecast
    and an actual reading.
    """

    _attr_native_unit_of_measurement = "%"
    _attr_icon = "mdi:target"

    def __init__(self, coordinator, key: str, name: str, window: str):
        super().__init__(coordinator, key, name)
        self._window = window

    def _block(self):
        acc = (self.coordinator.data or {}).get("accuracy") or {}
        return acc.get(self._window)

    @property
    def native_value(self):
        block = self._block()
        return block.get("deviation_pct") if block else None

    @property
    def extra_state_attributes(self):
        block = self._block()
        if not block:
            return {}
        return {
            "mae_kwh_per_h": block.get("mae_kwh_per_h"),
            "deviation_pct": block.get("deviation_pct"),
            "forecast_kwh": block.get("forecast_kwh"),
            "actual_kwh": block.get("actual_kwh"),
            "hours_compared": block.get("hours"),
        }
