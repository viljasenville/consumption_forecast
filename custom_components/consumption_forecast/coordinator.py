"""Coordinator: data reading, model training, forecasting and cost.

Two schedules:
  - training once a day (heavy): reads history, trains the built-in profile
    model and, when one is selected, the add-on model as well.
  - forecasting once an hour (light): fetches the weather forecast, runs the
    selected model, produces the daily and hourly sensors plus cost.

Where the models live:
  - The built-in profile model always runs here, in-process. It is the default,
    the fallback, and the only model used by the period-forecast service.
  - Heavier models (LightGBM) run in the optional model provider add-on, which
    the integration talks to over HTTP (see addon.py). The add-on is an
    accelerator, never a requirement: if it is absent, stopped, or a call to it
    fails, the profile model takes over for that refresh.
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .addon import (
    InsufficientData,
    ModelServiceError,
    async_get_client,
    hour_actual,
    hour_future,
)
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
    DEFAULT_HISTORY_DAYS,
    DEFAULT_MAX_HOURLY_KWH,
    DEFAULT_MODEL,
    DEFAULT_PEAK_HOURS,
    DEFAULT_TIMEZONE,
    DOMAIN,
    FORECAST_RETRY_MINUTES,
    INDOOR_REF_DAYS,
    LAG_TAIL_HOURS,
    MIN_TRAIN_HOURS,
    MODEL_PROFILE,
    TRAIN_HOUR,
)
from .grid import assemble_grid
from .history import fetch_series
from .model import ProfileModel

_LOGGER = logging.getLogger(__name__)


# How many days of per-hour forecasts to keep for accuracy scoring. Covers the
# monthly window (last 30 full days) with a couple of days of headroom.
_ACCURACY_DAYS = 33


def _parse_iso(value: str):
    """Parse an ISO timestamp string, or return None if it is malformed."""
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _fmt_mae(value) -> str:
    """Format a validation MAE for the log, tolerating None."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _series_for_field(series, field: str):
    """Return a shallow copy of the series with ``energy`` set from ``field``.

    The models always read the "energy" key, so to forecast a different
    quantity (e.g. total_energy) we hand them a series where that quantity has
    been copied into "energy". Other keys (out_temp, target, ts) are preserved.
    """
    if field == "energy":
        return series
    out = []
    for r in series:
        row = dict(r)
        row["energy"] = r.get(field, r["energy"])
        out.append(row)
    return out


class ForecastCoordinator(DataUpdateCoordinator):
    """Orchestrates data reading, training and forecasting."""

    def __init__(self, hass: HomeAssistant, entry):
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            # No relative polling timer: the hourly refresh is driven by a
            # wall-clock tick at :00 (see async_setup), so it lands exactly on
            # the hour. update_interval is only set to the short retry interval
            # while a forecast cannot yet be produced (see _retry_soon).
            update_interval=timedelta(minutes=FORECAST_RETRY_MINUTES),
        )
        self.entry = entry
        self.cfg = {**entry.data, **entry.options}
        self.tz = ZoneInfo(self.cfg.get(CONF_TIMEZONE, DEFAULT_TIMEZONE))
        self.base = float(self.cfg.get(CONF_BASE_TEMP, DEFAULT_BASE_TEMP))
        self.max_kwh = float(self.cfg.get(CONF_MAX_HOURLY_KWH, DEFAULT_MAX_HOURLY_KWH))
        # Mean of the last INDOOR_REF_DAYS days of the indoor sensor, refreshed
        # every forecast run. The heating-degree reference used for FUTURE hours
        # when an indoor sensor is configured (see _current_target); None until
        # the first successful read, and always None without an indoor sensor.
        self._indoor_ref: float | None = None
        # Production/export are optional inputs used only to reconstruct the true
        # total consumption from history (grid + production - export). The
        # integration forecasts that single total consumption; it does not
        # forecast grid purchase separately, because grid purchase depends on
        # future solar generation, which cannot be forecast from temperature.
        self.model = ProfileModel(base_temp=self.base)       # total consumption
        # Which model the user picked: MODEL_PROFILE (built-in) or an add-on
        # backend id such as "lightgbm". The add-on client is resolved in
        # async_setup and stays None whenever the built-in model is to be used.
        self.selected_model = self.cfg.get(CONF_MODEL) or DEFAULT_MODEL
        self.addon = None
        # Which model actually produced the latest forecast. Equals
        # selected_model while the add-on answers, MODEL_PROFILE otherwise.
        self.active_model_name = MODEL_PROFILE
        # last add-on training result (val_mae, trained_at, or why it failed)
        self.remote_info: dict | None = None
        # training diagnostics, exposed in the week sensor's attributes
        self.training_info: dict | None = None
        # peak/low flags cached per day so a day's flags stay stable as the
        # forecast horizon shrinks through the day: {date: {"high": [...], "low": [...]}}
        self._peak_cache: dict = {}
        # rolling record of {hour_iso: forecast_kwh} for the current hour at the
        # time it was forecast, used to score forecast vs actual. Kept ~8 days,
        # persisted across restarts. Self-contained (no dependency on a sensor's
        # entity_id, which the user can rename).
        self._forecast_log: dict[str, float] = {}
        self._store = hass.config.path(f".{DOMAIN}_{entry.entry_id}_model.pkl")
        self._unsub_train = None
        self._unsub_hourly = None

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #
    async def async_setup(self):
        """Load a saved model or train, and start the clock-aligned timers.

        Both timers fire on the hour (at :00): the forecast refresh every hour,
        and the retrain once a day at TRAIN_HOUR. This keeps every hourly sensor
        (peak-hour flags, current-hour forecast) in step with the wall clock
        instead of drifting to whatever minute the last refresh happened to run.
        """
        await self._async_resolve_addon()
        await self._load_or_train()
        self._unsub_train = async_track_time_change(
            self.hass, self._daily_train, hour=TRAIN_HOUR, minute=0, second=0
        )
        self._unsub_hourly = async_track_time_change(
            self.hass, self._hourly_tick, minute=0, second=0
        )

    async def _async_resolve_addon(self):
        """Look up the model provider add-on, if a model from it is selected.

        Nothing is contacted when the built-in profile model is chosen: that
        model needs no add-on, so there is no reason to probe for one. When an
        add-on model is selected but the add-on is not installed or not running,
        ``self.addon`` stays None and the profile model is used instead — the
        stored selection is left alone, so the add-on simply starts being used
        again once it is back.

        Called at setup and again on every training run, which is what makes
        that recovery automatic: an add-on installed or reconfigured while Home
        Assistant is running is found without a restart.
        """
        if self.selected_model == MODEL_PROFILE:
            self.addon = None
            return
        self.addon = await async_get_client(self.hass, self.entry.entry_id)
        if self.addon is None:
            _LOGGER.warning(
                "Model '%s' is selected but the model provider add-on is not "
                "available; using the built-in profile model.",
                self.selected_model,
            )
        else:
            _LOGGER.debug(
                "Using model provider add-on %s at %s",
                self.addon.info.slug,
                self.addon.info.url,
            )

    def _remote_selected(self) -> bool:
        """Whether an add-on model is selected and the add-on is reachable."""
        return self.addon is not None and self.selected_model != MODEL_PROFILE

    def _addon_diagnostics(self) -> dict | None:
        """Add-on state for the diagnostic sensor attributes, or None."""
        if self.addon is None:
            return None
        info = self.addon.info
        out = {
            "slug": info.slug,
            "version": info.version,
            "url": info.url,
            "selected_model": self.selected_model,
        }
        if self.remote_info:
            out.update(
                {
                    k: v
                    for k, v in self.remote_info.items()
                    if k in ("trained", "val_mae", "baseline_val_mae",
                             "n_hours", "trained_at", "reason", "error")
                }
            )
        return out

    @callback
    def _hourly_tick(self, now) -> None:
        """Refresh the forecast at the top of every hour."""
        self.hass.async_create_task(self.async_request_refresh())

    async def async_shutdown(self):
        if self._unsub_train:
            self._unsub_train()
        if self._unsub_hourly:
            self._unsub_hourly()
        await super().async_shutdown()

    async def _load_or_train(self):
        def _load():
            with open(self._store, "rb") as f:
                return pickle.load(f)

        try:
            payload = await self.hass.async_add_executor_job(_load)
            self.model = payload["model"]
            self.active_model_name = payload.get("active_model_name") or MODEL_PROFILE
            self.remote_info = payload.get("remote_info")
            self.training_info = payload.get("training_info")
            self._forecast_log = payload.get("forecast_log") or {}
            self._peak_cache = payload.get("peak_cache") or {}
            # Restore the last forecast so sensors show it immediately after a
            # restart instead of "unknown" while the weather entity and history
            # come back. It is refreshed to a current forecast within a minute.
            saved = payload.get("last_data")
            if saved:
                self.async_set_updated_data(saved)
            _LOGGER.info("Loaded saved model: %s", self.active_model_name)
        # Deliberately broad: a saved file can be missing, truncated, corrupt, or
        # written by a version whose model classes no longer exist, and pickle
        # reports those as anything from OSError to AttributeError to MemoryError.
        # Retraining is the right answer to every one of them, and refusing to
        # set up the integration is never the right answer to a bad cache file.
        except Exception as err:  # noqa: BLE001
            _LOGGER.info("No usable saved model (%s), training.", err)
            await self._daily_train()
            return

        await self._async_match_selected_model(payload.get("selected_model"))

    async def _async_match_selected_model(self, trained_for):
        """Make the loaded state agree with the model the user has selected.

        The profile model comes back from the pickle fully trained, but a model
        held by the add-on does not: only the add-on knows whether it still has
        an instance for this config entry. So after a restart, or after the user
        changes the selection, ask the add-on before trusting it — and retrain
        only when there really is nothing to use.
        """
        if not self._remote_selected():
            # Built-in model: it is loaded and trained, nothing else to check.
            self.active_model_name = MODEL_PROFILE
            return

        try:
            info = await self.addon.async_instance_info(self.selected_model)
        except ModelServiceError as err:
            _LOGGER.debug("No add-on instance info for %s: %s", self.selected_model, err)
            info = None

        if info and info.get("trained_at"):
            self.active_model_name = self.selected_model
            self.remote_info = {"trained": True, **info}
            _LOGGER.info(
                "Using add-on model %s (trained %s, val_mae %s)",
                self.selected_model,
                info.get("trained_at"),
                info.get("val_mae"),
            )
            return

        if trained_for == self.selected_model:
            # The add-on lost the instance (reinstalled, data wiped). Training
            # at the next daily run is enough; until then the profile model,
            # which is loaded and trained, carries the forecast.
            _LOGGER.warning(
                "The add-on has no trained '%s' instance; using the built-in "
                "profile model until the next training run.",
                self.selected_model,
            )
            self.active_model_name = MODEL_PROFILE
            # Drop the loaded training result: it described an instance that no
            # longer exists, and leaving it in place would have the diagnostic
            # attributes claim a trained add-on model while the profile model is
            # the one actually forecasting.
            self.remote_info = {"trained": False, "reason": "instance_missing"}
            return

        # The selection changed since the last training run, so the add-on has
        # never been asked to train this model. Do it now rather than leaving
        # the user on the profile model until 03:00.
        _LOGGER.info(
            "Model selection changed to '%s'; training it now.", self.selected_model
        )
        await self._daily_train()

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #
    async def _daily_train(self, _now=None) -> bool:
        # Re-resolve the add-on on every training run, so an add-on that was
        # installed, started or reconfigured after Home Assistant came up is
        # picked up without a restart -- including a changed API token, which
        # would otherwise keep failing every hourly forecast.
        if self.selected_model != MODEL_PROFILE:
            await self._async_resolve_addon()

        await self._async_refresh_indoor_ref()

        # None (or unset) -> fetch all available history; a number narrows it.
        history_days = self.cfg.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS)
        series = await self._build_hourly_series(history_days)
        if len(series) < MIN_TRAIN_HOURS:
            _LOGGER.warning(
                "Not enough data to train (%d hours, %d required).",
                len(series),
                MIN_TRAIN_HOURS,
            )
            return False

        # Train the single consumption model on "total_energy" (grid + prod -
        # export; equals grid purchase when no solar inputs are configured).
        total_series = _series_for_field(series, "total_energy")
        # The built-in profile model is always trained, even when an add-on model
        # is selected: it is the fallback for every refresh the add-on cannot
        # answer, and the only model the period-forecast service uses.
        profile = ProfileModel(base_temp=self.base)
        await self.hass.async_add_executor_job(profile.train, total_series)
        self.model = profile
        self.active_model_name = MODEL_PROFILE

        if self._remote_selected():
            self.remote_info = await self._async_train_remote(total_series)
            if self.remote_info.get("trained"):
                self.active_model_name = self.selected_model
        else:
            self.remote_info = None

        # Record training diagnostics: when it ran and the span of data used.
        # The series is chronological; first/last timestamps bound the period,
        # while len(series) is the number of usable hours (may be fewer than the
        # calendar span if there were gaps).
        first_ts = series[0]["ts"].astimezone(self.tz)
        last_ts = series[-1]["ts"].astimezone(self.tz)
        span_hours = (last_ts - first_ts).total_seconds() / 3600.0
        self.training_info = {
            "trained_at": dt_util.now(self.tz).isoformat(),
            "training_start": first_ts.isoformat(),
            "training_end": last_ts.isoformat(),
            "training_span_days": round(span_hours / 24.0, 1),
            "training_hours": len(series),
            # which heating-degree reference the model learned from -- the first
            # thing to check when a forecast is systematically too high or low
            "hdd_reference": self._hdd_reference_source(),
        }

        _LOGGER.info(
            "Selected model: %s (trained on %d h spanning %.1f days)",
            self.active_model_name,
            len(series),
            span_hours / 24.0,
        )
        await self.hass.async_add_executor_job(self._save_model)
        await self.async_request_refresh()
        return True

    async def _async_train_remote(self, series):
        """Have the add-on train the selected model on ``series``.

        Returns the add-on's answer as a diagnostics dict; ``trained`` is False
        for every outcome that leaves us without a usable remote model (too
        little data, add-on error), and the profile model then stays active.
        Training is synchronous in the add-on and can take minutes, which is why
        it runs only on the daily schedule.
        """
        payload = await self.hass.async_add_executor_job(
            lambda: [hour_actual(r) for r in series]
        )
        try:
            body = await self.addon.async_train(
                self.selected_model, payload, base_temp=self.base
            )
        except InsufficientData as err:
            _LOGGER.warning(
                "The add-on needs more history to train '%s' (%s); using the "
                "built-in profile model.",
                self.selected_model,
                err,
            )
            return {"trained": False, "reason": str(err)}
        except ModelServiceError as err:
            _LOGGER.warning(
                "Add-on training of '%s' failed (%s); using the built-in "
                "profile model.",
                self.selected_model,
                err,
            )
            return {"trained": False, "error": str(err)}

        if not body.get("trained"):
            _LOGGER.warning(
                "The add-on did not train '%s' (%s); using the built-in "
                "profile model.",
                self.selected_model,
                body.get("reason") or "no reason given",
            )
            return {"trained": False, **body}

        # Both MAE figures are time-ordered validation errors in kWh/h, so they
        # are directly comparable — logged together so a model that is not worth
        # its container is visible in the log.
        _LOGGER.info(
            "Validation MAE: profile=%s  %s=%s",
            _fmt_mae(self.model.coefficients().get("val_mae")),
            self.selected_model,
            _fmt_mae(body.get("val_mae")),
        )
        return {"trained": True, **body}

    async def async_train(self) -> bool:
        """Public trigger for manual (service-initiated) retraining.

        Returns True if a model was trained and saved, False if there was not
        enough data. Raised exceptions propagate to the caller.
        """
        return bool(await self._daily_train())

    def _save_model(self):
        with open(self._store, "wb") as f:
            pickle.dump(self._store_payload(self.data), f)

    def _store_payload(self, data) -> dict:
        """What is written to disk. Only the built-in model is stored here; a
        model trained in the add-on stays in the add-on, keyed by entry id, so
        the selection it was trained for is recorded instead."""
        return {
            "model": self.model,
            "active_model_name": self.active_model_name,
            # which model the add-on was last asked to train, so a changed
            # selection is noticed after a restart (see _async_match_selected_model)
            "selected_model": self.selected_model if self._remote_selected() else None,
            "remote_info": self.remote_info,
            "training_info": self.training_info,
            "forecast_log": self._forecast_log,
            "peak_cache": self._peak_cache,
            # last forecast, so it can be shown right after a restart
            "last_data": data,
        }

    # ------------------------------------------------------------------ #
    #  Data reading -> hourly grid                                        #
    # ------------------------------------------------------------------ #
    async def _build_hourly_series(self, days: int | None):
        # Meters are cumulative kWh -> statistics store the end-of-hour STATE.
        # Outdoor temperature is instantaneous -> statistics store the MEAN.
        energy_raw = await fetch_series(
            self.hass, self.cfg[CONF_ENERGY], days, cumulative=True
        )
        outdoor_raw = await fetch_series(
            self.hass, self.cfg[CONF_OUTDOOR], days, cumulative=False
        )
        # The heating-degree reference. Preference order:
        #   1. indoor temperature sensor  2. thermostat target  3. heating threshold
        target_raw = None
        target_instantaneous = False
        indoor = self.cfg.get(CONF_INDOOR)
        thermostat = self.cfg.get(CONF_THERMOSTAT)
        if indoor:
            # A temperature SENSOR has long-term statistics, so its real
            # per-hour history covers the whole training window -- no need for
            # the thermostat approximation below, and a seasonal change of the
            # setpoint is already recorded in the history itself.
            target_raw = await fetch_series(self.hass, indoor, days, cumulative=False)
            target_instantaneous = True
        elif thermostat:
            # A thermostat's target is an ATTRIBUTE, so its history reaches only
            # the recorder's raw window (~10 days) -- it is not in long-term
            # statistics. Training on a year of data would then fall back to the
            # heating threshold for most days, while the forecast uses the
            # CURRENT target -- so the HDD threshold (and thus the forecast)
            # would be inconsistent. To keep them identical, apply the current
            # target to the whole training history. This assumes a stable
            # setpoint; if you change it seasonally, retrain after each change.
            current = self._current_target()
            if energy_raw:
                target_raw = [
                    (energy_raw[0][0], current),
                    (energy_raw[-1][0], current),
                ]

        production_raw = None
        if self.cfg.get(CONF_PRODUCTION):
            production_raw = await fetch_series(
                self.hass, self.cfg[CONF_PRODUCTION], days, cumulative=True
            )
        export_raw = None
        if self.cfg.get(CONF_EXPORT):
            export_raw = await fetch_series(
                self.hass, self.cfg[CONF_EXPORT], days, cumulative=True
            )

        if len(energy_raw) < 48 or len(outdoor_raw) < 48:
            return []

        # positional args only: async_add_executor_job takes no keywords
        return await self.hass.async_add_executor_job(
            assemble_grid,
            energy_raw,
            outdoor_raw,
            target_raw,
            self.base,
            self.tz,
            self.max_kwh,
            production_raw,
            export_raw,
            target_instantaneous,
        )

    async def _recent_tail(self, hours: int = LAG_TAIL_HOURS):
        """Recent history for the add-on model's lag features.

        The add-on's LightGBM backend needs 168 h of actuals before the forecast
        origin; LAG_TAIL_HOURS leaves slack for recorder gaps and for using
        midnight as the origin (see _today_full_day_kwh).
        """
        days = max(2, (hours // 24) + 3)
        series = await self._build_hourly_series(days)
        return series[-hours:] if len(series) > hours else series

    # ------------------------------------------------------------------ #
    #  Forecast (run hourly)                                              #
    # ------------------------------------------------------------------ #
    def _retry_soon(self):
        """Shorten the update interval so a failed/empty update is retried
        quickly (used after a restart before weather/history is ready)."""
        retry = timedelta(minutes=FORECAST_RETRY_MINUTES)
        if self.update_interval != retry:
            self.update_interval = retry
            _LOGGER.debug(
                "Forecast not ready; retrying in %d min.", FORECAST_RETRY_MINUTES
            )
        return self.data or {}

    def _resume_hourly(self):
        """Stop the short retry timer after a successful update.

        Setting update_interval to None disables the coordinator's relative
        polling, so the next refresh comes only from the on-the-hour clock tick
        (see async_setup) — keeping the hourly update aligned to :00.
        """
        if self.update_interval is not None:
            self.update_interval = None

    async def _async_update_data(self):
        if not getattr(self.model, "trained", False):
            return self._retry_soon()

        weather_hours = await self._get_weather_forecast()
        if not weather_hours:
            return self._retry_soon()

        # Refresh the indoor reference before any target is read, so training
        # and prediction use the same heating-degree reference.
        await self._async_refresh_indoor_ref()

        future = self._build_future(weather_hours)
        if not future:
            return self._retry_soon()

        # Single consumption forecast. The history tail is read once per refresh
        # and shared by both add-on calls below (the rolling forecast and today's
        # whole-day figure), so a remote model costs one extra recorder read, not
        # two. It is not needed at all by the built-in model.
        tail = await self._recent_tail() if self._remote_active() else None

        hourly, source = await self._async_forecast_hours(future, tail)
        if not hourly:
            return self._retry_soon()
        daily = self._aggregate_daily(hourly)

        # "today" from the aggregate above covers only the hours still ahead, so
        # it would shrink through the day. Replace it with a stable whole-day
        # forecast (all 24 h). Tomorrow onward are already full days.
        today_kwh = await self._today_full_day_kwh(
            weather_hours, tail if source != MODEL_PROFILE else None
        )
        today_iso = dt_util.now(self.tz).date().isoformat()
        if today_kwh is not None and daily and daily[0]["date"] == today_iso:
            daily[0] = {"date": today_iso, "kwh": today_kwh}

        # record the forecast made for the CURRENT hour, to score later against
        # the actual consumption for that hour.
        self._record_forecast(hourly)

        result = {
            "daily": daily,
            "hourly": hourly,
            # coefficients of the built-in model; still useful (base load, per-HDD
            # slope) even when the add-on produced the forecast
            "coeffs": self.model.coefficients(),
            # which model actually produced THIS forecast, not merely which one
            # is selected — they differ whenever the add-on failed and the
            # profile model stood in
            "active_model": source,
            "model_source": "built-in" if source == MODEL_PROFILE else "add-on",
            "val_mae": self._active_val_mae(source),
            # add-on state and its last training result, or None when the
            # built-in model is in use
            "addon": self._addon_diagnostics(),
            # per-day weather the forecast actually used, so an unexpectedly high
            # forecast can be traced to the temperatures / HDD driving it.
            "forecast_weather": self._weather_diagnostics(future),
            # forecast-vs-actual accuracy for yesterday and the last 7 days
            "accuracy": await self._compute_accuracy(),
            "generated": dt_util.now(self.tz).isoformat(),
            # timestamps flagged as the day's highest / lowest consumption hours
            "peak_high": self._peak_hours(hourly, high=True),
            "peak_low": self._peak_hours(hourly, high=False),
        }

        # A real forecast was produced: return to the normal hourly cadence and
        # persist it so a restart can show it immediately instead of "unknown".
        self._resume_hourly()
        self._save_last_data(result)
        return result

    def _record_forecast(self, hourly):
        """Store the forecast for the current hour and prune the log."""
        now_hour = dt_util.now(self.tz).replace(minute=0, second=0, microsecond=0)
        for row in hourly:
            if row["ts"].replace(minute=0, second=0, microsecond=0) == now_hour:
                self._forecast_log[now_hour.isoformat()] = row["kwh"]
                break
        cutoff = now_hour - timedelta(days=_ACCURACY_DAYS)
        self._forecast_log = {
            k: v
            for k, v in self._forecast_log.items()
            if _parse_iso(k) is not None and _parse_iso(k) >= cutoff
        }

    async def _compute_accuracy(self):
        """Compare recorded forecasts with actual consumption.

        Returns {"today": {...}, "yesterday": {...}, "week": {...},
        "month": {...}} where each block has the forecast/actual totals, the
        hourly MAE (kWh/h) and the deviation (%), computed only over hours that
        have BOTH a recorded forecast and an actual reading. None until there is
        data to compare.

        Windows: "today" is the (partial, running) current day; "yesterday" is
        the last completed day; "week" and "month" are the last 7 and 30
        completed days, ending yesterday.
        """
        if not self._forecast_log:
            return None
        actual_series = await self._build_hourly_series(_ACCURACY_DAYS)
        if not actual_series:
            return None

        actual = {}
        for r in actual_series:
            key = r["ts"].astimezone(self.tz).replace(
                minute=0, second=0, microsecond=0
            )
            actual[key] = r["total_energy"]

        forecast = {}
        for iso, kwh in self._forecast_log.items():
            dt = _parse_iso(iso)
            if dt is not None:
                forecast[
                    dt.astimezone(self.tz).replace(minute=0, second=0, microsecond=0)
                ] = kwh

        def window(start_date, end_date):
            pairs = [
                (forecast[h], a)
                for h, a in actual.items()
                if start_date <= h.date() <= end_date and h in forecast
            ]
            if not pairs:
                return None
            f_sum = sum(f for f, _ in pairs)
            a_sum = sum(a for _, a in pairs)
            mae = sum(abs(f - a) for f, a in pairs) / len(pairs)
            dev = (abs(f_sum - a_sum) / a_sum * 100) if a_sum > 0 else None
            return {
                "forecast_kwh": round(f_sum, 2),
                "actual_kwh": round(a_sum, 2),
                "mae_kwh_per_h": round(mae, 3),
                "deviation_pct": round(dev, 1) if dev is not None else None,
                "hours": len(pairs),
            }

        # Rolling "previous N days" windows, all ending yesterday (the last
        # completed day). "today" is the only partial, running window.
        today = dt_util.now(self.tz).date()
        yesterday = today - timedelta(days=1)
        return {
            # "today" is a partial, running comparison: the window is today, but
            # only elapsed hours have an actual reading, so the "both present"
            # filter naturally limits it to hours lived so far. It fluctuates
            # through the day and is most meaningful late in the day.
            "today": window(today, today),
            # "yesterday" = the last completed day; "week"/"month" = the last 7
            # and 30 completed days (ending yesterday), so they are full days
            # only, not a partial current day.
            "yesterday": window(yesterday, yesterday),
            "week": window(today - timedelta(days=7), yesterday),
            "month": window(today - timedelta(days=30), yesterday),
        }

    def _save_last_data(self, data):
        """Persist only the latest forecast (best-effort; never blocks state)."""
        try:
            self.hass.async_add_executor_job(self._write_last_data, data)
        except Exception:  # pragma: no cover - persistence is best-effort
            pass

    def _write_last_data(self, data):
        try:
            with open(self._store, "wb") as f:
                pickle.dump(self._store_payload(data), f)
        except OSError:
            pass

    def _peak_hours(self, hourly, high: bool):
        """Return the set of hour timestamps (ISO) flagged high or low.

        For each calendar day, take the N highest (high=True) or N lowest
        (high=False) forecast hours, where N is CONF_PEAK_HOURS. An hour is only
        flagged if it is also on the correct side of that day's mean consumption,
        so on a flat day near-average hours are not mislabelled. N=0 disables.
        """
        n = int(self.cfg.get(CONF_PEAK_HOURS, DEFAULT_PEAK_HOURS))
        if n <= 0 or not hourly:
            return []

        side = "high" if high else "low"

        # group hours by local calendar day
        by_day: dict = {}
        for row in hourly:
            d = row["ts"].astimezone(self.tz).date()
            by_day.setdefault(d, []).append(row)

        today = dt_util.now(self.tz).date()
        # drop cache entries for days now in the past
        self._peak_cache = {
            d: v for d, v in self._peak_cache.items() if d >= today
        }

        flagged: list[str] = []
        for d, rows in by_day.items():
            cached = self._peak_cache.get(d, {})
            # Recompute a day's flags only when this refresh gives us MORE hours
            # for that day than we cached before. As the day progresses the
            # forecast horizon shrinks, so an already-computed day would only get
            # fewer hours -- keeping the cached (fuller) ranking avoids the flags
            # jittering or disappearing through the day.
            if cached.get("hours", 0) >= len(rows) and side in cached:
                flagged.extend(cached[side])
                continue

            mean = sum(r["kwh"] for r in rows) / len(rows)
            day_flags = {"high": [], "low": [], "hours": len(rows)}
            for want_high in (True, False):
                ordered = sorted(rows, key=lambda r: r["kwh"], reverse=want_high)
                key = "high" if want_high else "low"
                for r in ordered[:n]:
                    ok = r["kwh"] > mean if want_high else r["kwh"] < mean
                    if ok:
                        local_hour = (
                            r["ts"]
                            .astimezone(self.tz)
                            .replace(minute=0, second=0, microsecond=0)
                        )
                        day_flags[key].append(local_hour.isoformat())
            self._peak_cache[d] = day_flags
            flagged.extend(day_flags[side])

        return sorted(flagged)

    def _remote_active(self) -> bool:
        """Whether the add-on holds the trained model we should be forecasting
        with. False also when the add-on is reachable but its model has not been
        trained (yet), in which case the profile model is active."""
        return self._remote_selected() and self.active_model_name == self.selected_model

    def _active_val_mae(self, source: str):
        """Validation MAE of the model that produced the forecast (kWh/h)."""
        if source == MODEL_PROFILE:
            return self.model.coefficients().get("val_mae")
        return (self.remote_info or {}).get("val_mae")

    async def _async_forecast_hours(self, future, tail):
        """Hourly forecast [{ts, kwh}] and the name of the model that made it.

        Tries the add-on first when its model is active, and falls back to the
        built-in profile model for this refresh if the call fails — a forecast
        from the simpler model is better than no forecast at all. Only this
        refresh falls back: the selection is untouched, so the add-on is used
        again as soon as it answers.
        """
        if tail is not None:
            rows = await self._async_remote_forecast(future, tail)
            if rows is not None:
                return rows, self.selected_model

        rows = await self.hass.async_add_executor_job(
            self._profile_predict, self.model, future
        )
        return rows, MODEL_PROFILE

    async def _async_remote_forecast(self, future, tail):
        """Forecast ``future`` with the add-on, or None if it could not.

        The add-on takes the first future hour as the forecast origin and needs
        the history tail to cover its model's lag depth before that hour, so the
        tail is sent as actual total consumption — the quantity the model was
        trained on.
        """
        if not tail:
            _LOGGER.debug("No history tail available for the add-on forecast.")
            return None

        tail_rows = _series_for_field(tail, "total_energy")
        payload_tail, payload_future = await self.hass.async_add_executor_job(
            lambda: (
                [hour_actual(r) for r in tail_rows],
                [hour_future(r) for r in future],
            )
        )
        try:
            pairs = await self.addon.async_predict(
                self.selected_model, payload_future, payload_tail
            )
        except ModelServiceError as err:
            _LOGGER.warning(
                "Add-on forecast with '%s' failed (%s); falling back to the "
                "built-in profile model for this refresh.",
                self.selected_model,
                err,
            )
            return None

        # Normalize to the house timezone: downstream code (peak hours, the
        # forecast log) compares these timestamps with local hour boundaries.
        return [
            {"ts": ts.astimezone(self.tz), "kwh": round(max(0.0, kwh), 3)}
            for ts, kwh in pairs
        ]

    # ---- profile model forecast ---- #
    def _profile_predict(self, model, future):
        """Group future by day, predict daily consumption and split to hours."""
        by_day: dict = {}
        for fr in future:
            d = fr["ts"].date()
            by_day.setdefault(d, []).append(fr)

        hdd_median = getattr(model, "_hdd_median", 0.0)
        out = []
        for d, rows in sorted(by_day.items()):
            hdd = sum(max(0.0, r["target"] - r["out_temp"]) for r in rows)
            weekday = rows[0]["ts"].weekday()
            day_kwh = model.predict_day(hdd, weekday)
            is_cold = hdd >= hdd_median
            # predict_hours returns absolute kWh per hour (already floored at the
            # base load and summing to day_kwh). For a partial day we simply take
            # the values for the hours present -- no re-normalization, which would
            # otherwise reintroduce sub-base-load hours.
            profile = model.predict_hours(day_kwh, weekday, is_cold)
            for r in rows:
                out.append({"ts": r["ts"], "kwh": round(profile[r["ts"].hour], 3)})
        out.sort(key=lambda x: x["ts"])
        return out

    # ---- weather forecast ---- #
    async def _get_weather_forecast(self):
        entity = self.cfg[CONF_FORECAST]
        for ftype in ("hourly", "daily"):
            try:
                resp = await self.hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    {"entity_id": entity, "type": ftype},
                    blocking=True,
                    return_response=True,
                )
            except Exception:  # the entity may not support this type
                continue
            forecasts = (resp or {}).get(entity, {}).get("forecast") or []
            if not forecasts:
                continue
            if ftype == "hourly":
                return self._parse_hourly(forecasts)
            return self._parse_daily(forecasts)
        return []

    def _parse_hourly(self, forecasts):
        out = []
        for f in forecasts:
            ts = dt_util.parse_datetime(f.get("datetime", ""))
            if ts is None or f.get("temperature") is None:
                continue
            out.append((ts.astimezone(self.tz), float(f["temperature"])))
        out.sort(key=lambda x: x[0])
        return out

    def _parse_daily(self, forecasts):
        """Daily forecast -> hourly using a coarse diurnal curve.
        Fallback when an hourly forecast is not available."""
        out = []
        for f in forecasts:
            ts = dt_util.parse_datetime(f.get("datetime", ""))
            if ts is None or f.get("temperature") is None:
                continue
            day_start = ts.astimezone(self.tz).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            t_hi = float(f["temperature"])
            t_lo = float(f.get("templow", t_hi))
            for h in range(24):
                # minimum ~05:00, maximum ~15:00
                frac = 0.5 - 0.5 * np.cos(2 * np.pi * (h - 5) / 24)
                temp = t_lo + (t_hi - t_lo) * frac
                out.append((day_start + timedelta(hours=h), temp))
        out.sort(key=lambda x: x[0])
        return out

    def _weather_diagnostics(self, future):
        """Per-day summary of the temperatures and HDD the forecast used.

        Exposed on the week sensor so a high forecast can be traced to the
        weather driving it (min/max/mean °C, target, and heating degree hours).
        """
        by_day: dict = {}
        for fr in future:
            d = fr["ts"].astimezone(self.tz).date()
            by_day.setdefault(d, []).append(fr)
        out = []
        for d, rows in sorted(by_day.items()):
            temps = [r["out_temp"] for r in rows]
            hdd = sum(max(0.0, r["target"] - r["out_temp"]) for r in rows)
            out.append(
                {
                    "date": d.isoformat(),
                    "hours": len(rows),
                    "temp_min": round(min(temps), 1),
                    "temp_max": round(max(temps), 1),
                    "temp_mean": round(sum(temps) / len(temps), 1),
                    "target": round(rows[0]["target"], 1),
                    "hdd": round(hdd, 1),
                }
            )
        return out

    # ---- whole-day forecast for today ---- #
    async def _today_hi_lo(self):
        """(hi, lo) temperature for today from the daily weather forecast.

        Used to fill in the hours already past when building today's whole-day
        forecast. Returns None if a daily forecast is not available.
        """
        entity = self.cfg[CONF_FORECAST]
        try:
            resp = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": entity, "type": "daily"},
                blocking=True,
                return_response=True,
            )
        except Exception:  # entity may not support a daily forecast
            return None
        forecasts = (resp or {}).get(entity, {}).get("forecast") or []
        today = dt_util.now(self.tz).date()
        for f in forecasts:
            ts = dt_util.parse_datetime(f.get("datetime", ""))
            if ts is None or f.get("temperature") is None:
                continue
            if ts.astimezone(self.tz).date() == today:
                hi = float(f["temperature"])
                lo = float(f.get("templow", hi))
                return (hi, lo)
        return None

    async def _today_full_day_kwh(self, weather_hours, tail=None):
        """Stable whole-day forecast for today (all 24 h).

        The live hourly forecast starts at the current hour, so aggregating it
        gives today only its remaining hours — the today figure would shrink as
        the day goes on. Instead we forecast the complete calendar day from a
        24-hour temperature series: the weather forecast where it reaches, and
        today's daily hi/lo (via a diurnal curve) for the hours already past.
        The result changes only as the weather forecast for today is revised,
        not simply because time passed. Returns None when the model or weather
        cannot give a whole-day estimate, so the caller leaves today as-is.

        ``tail`` is the recent history, passed in when the add-on model is the
        one to use; the whole day is then forecast from midnight as the origin,
        the same way the add-on was trained. Without it the built-in model's
        day-level prediction is used.
        """
        today = dt_util.now(self.tz).date()
        target = self._current_target()

        temp_by_hour: dict[int, float] = {}
        for ts, temp in weather_hours:
            tsl = ts.astimezone(self.tz)
            if tsl.date() == today:
                temp_by_hour.setdefault(tsl.hour, temp)

        hi_lo = await self._today_hi_lo()
        if hi_lo is not None:
            t_hi, t_lo = hi_lo
        elif temp_by_hour:
            t_hi = max(temp_by_hour.values())
            t_lo = min(temp_by_hour.values())
        else:
            return None  # no weather information for today at all

        temps = []
        for h in range(24):
            if h in temp_by_hour:
                temps.append(temp_by_hour[h])
            else:
                # same diurnal curve as the daily-forecast fallback: min ~05:00,
                # max ~15:00
                frac = 0.5 - 0.5 * np.cos(2 * np.pi * (h - 5) / 24)
                temps.append(t_lo + (t_hi - t_lo) * frac)

        if tail is not None:
            # Origin = today 00:00, so every hour of the day is forecast from
            # yesterday's last observation — a full-day horizon, not a
            # part-day one.
            midnight = dt_util.now(self.tz).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            day_future = [
                {
                    "ts": midnight + timedelta(hours=h),
                    "out_temp": temps[h],
                    "target": target,
                }
                for h in range(24)
            ]
            rows = await self._async_remote_forecast(day_future, tail)
            if rows is not None:
                return round(sum(r["kwh"] for r in rows), 1)
            # add-on unavailable for this call: fall through to the profile model

        hdd = sum(max(0.0, target - t) for t in temps)
        day_kwh = self.model.predict_day(hdd, today.weekday())
        return round(float(day_kwh), 1)

    # ---- future rows ---- #
    def _build_future(self, weather_hours):
        target = self._current_target()
        now_local = dt_util.now(self.tz)
        # Start from the CURRENT hour (not the next one) so the forecast covers
        # the hour we are in. This lets the current-hour forecast sensor record
        # a value for each hour that can later be compared with its actual
        # consumption; starting at now+1h would leave the current hour blank and
        # put the recorded forecast one hour out of step with the actuals.
        start = now_local.replace(minute=0, second=0, microsecond=0)
        future = []
        for ts, temp in weather_hours:
            if ts < start:
                continue
            future.append({"ts": ts, "out_temp": temp, "target": target})
        return future

    def _hdd_reference_source(self) -> str:
        """Which input supplies the heating-degree reference (for diagnostics)."""
        if self.cfg.get(CONF_INDOOR):
            return "indoor_sensor"
        if self.cfg.get(CONF_THERMOSTAT):
            return "thermostat_target"
        return "heating_threshold"

    async def _async_refresh_indoor_ref(self):
        """Update the forecast-time indoor reference from recent history.

        Future indoor temperature is unknown, so forecast hours need a single
        representative value. The mean of the last INDOOR_REF_DAYS days is an
        unbiased estimate of the same per-hour quantity the model trained on,
        which is what keeps training and prediction on the same scale -- the
        absolute value matters far less than that consistency, because a constant
        offset is absorbed by the day model's fit.

        Leaves the previous value (or None) in place if the read yields nothing,
        so a momentary recorder gap cannot silently swing the whole forecast.
        """
        ent = self.cfg.get(CONF_INDOOR)
        if not ent:
            self._indoor_ref = None
            return
        try:
            raw = await fetch_series(self.hass, ent, INDOOR_REF_DAYS, cumulative=False)
        except Exception:  # noqa: BLE001 - never let this break a forecast
            _LOGGER.debug("Indoor reference read failed for %s.", ent, exc_info=True)
            return
        if not raw:
            _LOGGER.debug("No recent history for indoor sensor %s.", ent)
            return
        self._indoor_ref = sum(v for _, v in raw) / len(raw)

    def _current_target(self):
        """Heating-degree reference for FUTURE hours.

        Same preference order as the training history in _build_hourly_series
        (indoor sensor -> thermostat target -> heating threshold), so the model
        predicts against the reference it learned from.
        """
        if self.cfg.get(CONF_INDOOR):
            # An indoor sensor outranks the thermostat, and keeps outranking it
            # when it turns out to be unreadable: _build_hourly_series then
            # trains on the heating threshold, so predicting with the thermostat
            # target instead would put training and prediction on different
            # references -- the one error that scales the whole forecast.
            return self._indoor_ref if self._indoor_ref is not None else self.base
        ent = self.cfg.get(CONF_THERMOSTAT)
        if ent:
            st = self.hass.states.get(ent)
            if st and st.attributes.get("temperature") is not None:
                try:
                    return float(st.attributes["temperature"])
                except (ValueError, TypeError):
                    pass
        return self.base

    async def async_forecast_period(self, start, end, outdoor_temp, target_temp=None):
        """Forecast consumption over an arbitrary period [start, end).

        The caller supplies a single outdoor temperature applied to every hour
        in the period (and optionally the thermostat target), so this works for
        any past, present or future range without a weather forecast. Uses the
        profile model, which is a pure function of temperature and time.

        Returns hourly values, daily sums, and the period total -- for total
        consumption and, when solar is configured, for grid purchase, plus cost.
        """
        if not getattr(self.model, "trained", False):
            raise HomeAssistantError(
                "No trained model yet. Let it train first, or call the "
                "train_model service."
            )

        # normalize both ends to timezone-aware local hour boundaries
        start = self._to_local_hour(start)
        end = self._to_local_hour(end)
        if end <= start:
            raise HomeAssistantError("`end` must be after `start`.")

        if target_temp is None and self.cfg.get(CONF_INDOOR) and self._indoor_ref is None:
            # service called before the first hourly refresh (e.g. just after a
            # restart): read the reference now rather than silently using another
            # one than the model was trained on
            await self._async_refresh_indoor_ref()
        target = target_temp if target_temp is not None else self._current_target()
        target = float(target)
        outdoor_temp = float(outdoor_temp)

        # build the list of hours in [start, end)
        hours: list[datetime] = []
        t = start
        while t < end:
            hours.append(t)
            t += timedelta(hours=1)
        # guard against absurd ranges
        if len(hours) > 24 * 400:
            raise HomeAssistantError("Period too long (max ~400 days).")

        def _profile_hours(model):
            # predict each hour independently from the single supplied temperature
            hdd = max(0.0, target - outdoor_temp)
            out = []
            for h in hours:
                weekday = h.weekday()
                hdd_median = getattr(model, "_hdd_median", 0.0)
                is_cold = (hdd * 24.0) >= hdd_median
                day_kwh = model.predict_day(hdd * 24.0, weekday)
                shape = model.predict_hours(day_kwh, weekday, is_cold)
                out.append((h, round(shape[h.hour], 3)))
            return out

        total_pairs = await self.hass.async_add_executor_job(
            _profile_hours, self.model
        )
        total_hourly = [{"ts": ts, "kwh": kwh} for ts, kwh in total_pairs]

        result = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "outdoor_temp": outdoor_temp,
            "target_temp": target,
            "model": ProfileModel.name,
            "total_consumption": self._period_breakdown(total_hourly),
        }

        return result

    def _to_local_hour(self, dt):
        """Normalize a datetime to a timezone-aware local hour boundary."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz).replace(minute=0, second=0, microsecond=0)

    def _period_breakdown(self, hourly):
        """Build {hourly, daily, total_kwh} for a list of {ts, kwh}."""
        daily = self._aggregate_daily(hourly)
        total = round(sum(x["kwh"] for x in hourly), 1)
        return {
            "total_kwh": total,
            "daily": daily,
            "hourly": [
                {"datetime": x["ts"].isoformat(), "kwh": x["kwh"]} for x in hourly
            ],
        }

    # ---- daily aggregation ---- #
    def _aggregate_daily(self, hourly):
        days: dict = {}
        for row in hourly:
            d = row["ts"].astimezone(self.tz).date()
            days[d] = days.get(d, 0.0) + row["kwh"]
        return [
            {"date": d.isoformat(), "kwh": round(v, 1)}
            for d, v in sorted(days.items())
        ]
