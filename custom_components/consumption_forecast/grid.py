"""Assembling the hourly grid from three unevenly sampled time series.

Pure module with no Home Assistant dependency -> unit-testable.

The three series are handled with different rules:
  - energy (cumulative kWh): difference between consecutive hourly readings;
    meter resets / anomalies are filtered out.
  - outdoor temperature (instantaneous): mean of samples falling within the
    hour, forward-filled when needed.
  - target, the heating-degree reference (indoor temperature or thermostat
    setpoint): a measured indoor sensor is instantaneous and is averaged within
    the hour like the outdoor temperature; a thermostat setpoint is stepwise and
    takes the last known value. Both are forward-filled over gaps.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np


def _floor_hour(ts: datetime, tz: ZoneInfo) -> datetime:
    """Convert to local time and floor to the full hour.

    Local time matters: hourly features (hour_sin/cos, profiles) must learn
    the house's real daily rhythm, not UTC.
    """
    return ts.astimezone(tz).replace(minute=0, second=0, microsecond=0)


def _sample_last_known(raw, hours):
    """For each hour: the latest value whose timestamp <= the hour boundary.

    Used for cumulative energy and the stepwise target.
    """
    result: dict[datetime, float | None] = {}
    idx = 0
    n = len(raw)
    last_val = None
    for h in hours:
        while idx < n and raw[idx][0] <= h:
            last_val = raw[idx][1]
            idx += 1
        result[h] = last_val
    return result


def _sample_mean_in_hour(raw, hours, forward_fill=False):
    """For each hour [h, h+1): mean of samples falling inside it.

    Used for the instantaneous outdoor temperature.
    """
    result: dict[datetime, float | None] = {}
    idx = 0
    n = len(raw)
    last_val = None
    for h in hours:
        h_next = h + timedelta(hours=1)
        vals: list[float] = []
        while idx < n and raw[idx][0] < h_next:
            if raw[idx][0] >= h:
                vals.append(raw[idx][1])
            last_val = raw[idx][1]
            idx += 1
        if vals:
            result[h] = float(np.mean(vals))
        elif forward_fill and last_val is not None:
            result[h] = last_val
        else:
            result[h] = None
    return result


def _forward_fill_inplace(mapping, hours, fallback):
    """Fill None values with the previous known value; before the first -> fallback."""
    last = None
    for h in hours:
        v = mapping.get(h)
        if v is None:
            mapping[h] = last if last is not None else fallback
        else:
            last = v


def _cumulative_to_hourly(raw, hours, max_hourly_kwh):
    """Convert a cumulative kWh series into per-hour consumption.

    The reading sampled at hour ``h`` is the meter value at the *start* of that
    hour. Consumption *during* hour ``h`` is therefore
    ``reading(h+1) - reading(h)`` and must be recorded against ``h`` -- the hour
    in which the energy was actually used -- not against ``h+1``. Getting this
    wrong shifts the whole daily profile one hour late (a 12-13 peak would show
    up at 13-14).

    Returns {hour -> kWh or None}. None marks a gap or a filtered anomaly
    (negative diff from a meter reset, or a spike above the cap). Used
    identically for grid purchase, solar production and grid export series.
    """
    reading_at = _sample_last_known(raw, hours)
    per_hour: dict[datetime, float | None] = {}
    for i, h in enumerate(hours):
        cur = reading_at.get(h)
        # the reading at the START of the next hour closes this hour's interval
        nxt = reading_at.get(hours[i + 1]) if i + 1 < len(hours) else None
        if cur is None or nxt is None:
            # last hour has no closing reading; gaps leave None
            per_hour[h] = None
        else:
            diff = nxt - cur
            per_hour[h] = diff if 0 <= diff < max_hourly_kwh else None
    return per_hour


def assemble_grid(
    energy_raw,
    outdoor_raw,
    target_raw,
    base_temp: float,
    tz: ZoneInfo,
    max_hourly_kwh: float,
    production_raw=None,
    export_raw=None,
    target_instantaneous=False,
):
    """Assemble a unified hourly grid.

    Returns a chronologically ordered list of dicts:
        {ts, energy, total_energy, out_temp, target}
    where ``energy`` is grid purchase (the billed quantity) and
    ``total_energy`` is real house consumption = grid + production - export.
    When production/export are not supplied, ``total_energy`` equals ``energy``.
    Timestamps (ts) are in local, timezone-aware time.

    ``target_instantaneous`` selects how ``target_raw`` is sampled: False for a
    stepwise thermostat setpoint (last known value), True for a measured indoor
    temperature sensor (mean within the hour, as for the outdoor temperature).
    """
    if not energy_raw or not outdoor_raw:
        return []

    # --- grid bounds in local time ---
    start = _floor_hour(max(energy_raw[0][0], outdoor_raw[0][0]), tz)
    end = _floor_hour(min(energy_raw[-1][0], outdoor_raw[-1][0]), tz)
    if end <= start:
        return []

    hours: list[datetime] = []
    t = start
    while t <= end:
        hours.append(t)
        t += timedelta(hours=1)

    # --- ENERGY: cumulative -> hourly, for each supplied series ---
    grid_at = _cumulative_to_hourly(energy_raw, hours, max_hourly_kwh)
    production_at = (
        _cumulative_to_hourly(production_raw, hours, max_hourly_kwh)
        if production_raw
        else None
    )
    export_at = (
        _cumulative_to_hourly(export_raw, hours, max_hourly_kwh)
        if export_raw
        else None
    )

    # --- OUTDOOR TEMPERATURE: instantaneous -> representative hourly value ---
    outdoor_at = _sample_mean_in_hour(outdoor_raw, hours, forward_fill=True)

    # --- TARGET (heating-degree reference) -> forward-fill ---
    if target_raw:
        if target_instantaneous:
            # measured indoor temperature: same sampling rule as outdoor
            target_at = _sample_mean_in_hour(target_raw, hours, forward_fill=True)
        else:
            target_at = _sample_last_known(target_raw, hours)
        # Hours BEFORE the first reading (an indoor sensor added later than the
        # energy meter) are back-filled with that first reading rather than with
        # base_temp: the whole training history must sit on ONE reference scale,
        # and a configured threshold mixed into a house the sensor shows sitting
        # somewhere else would distort the fitted slope far more than the unknown
        # early hours themselves do.
        _forward_fill_inplace(target_at, hours, fallback=float(target_raw[0][1]))
    else:
        target_at = {h: base_temp for h in hours}

    # --- build rows; drop hours missing grid energy or temperature ---
    series: list[dict] = []
    for h in hours:
        grid_e = grid_at.get(h)
        o = outdoor_at.get(h)
        if grid_e is None or o is None:
            continue  # gap: do not interpolate consumption, leave it out

        # total consumption = grid + production - export.
        # If production/export are configured but missing/anomalous this hour,
        # fall back to grid alone rather than producing a wrong total.
        total_e = grid_e
        if production_at is not None or export_at is not None:
            prod = production_at.get(h) if production_at is not None else 0.0
            exp = export_at.get(h) if export_at is not None else 0.0
            if prod is not None and exp is not None:
                total_e = max(0.0, grid_e + prod - exp)

        series.append(
            {
                "ts": h,
                "energy": float(grid_e),
                "total_energy": float(total_e),
                "out_temp": float(o),
                "target": float(target_at.get(h, base_temp)),
            }
        )
    return series
