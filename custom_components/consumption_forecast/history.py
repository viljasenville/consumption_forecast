"""Reading history data from the recorder.

Home Assistant keeps two kinds of history:

  * RAW state changes (the `states` table) — exact, but purged after
    `purge_keep_days` (default 10 days).
  * LONG-TERM statistics (the `statistics` table) — hourly aggregates, kept
    indefinitely.

For training we want as much history as possible, so for long windows we read
the long-term statistics (which reach back months or years) and top them up with
the raw states for the most recent days that statistics may not cover yet.

All recorder queries run in an executor thread so the Home Assistant event loop
is not blocked by heavy database queries.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from homeassistant.components.recorder import get_instance, history, statistics
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

# When no explicit window is set, we look back this far. The recorder / statistics
# return only what they still hold, so this effectively means "all data".
_ALL_HISTORY_DAYS = 365 * 20


async def fetch_series(
    hass: HomeAssistant,
    entity_id: str,
    days: int | None,
    attribute: str | None = None,
    cumulative: bool = False,
) -> list[tuple[datetime, float]]:
    """Return [(timestamp, value), ...] in chronological order.

    days=None  -> read ALL history (long-term statistics + recent raw states).
    days=<int> -> read only the last that many days.

    attribute=None -> read the entity state.
    attribute given -> read that attribute (e.g. climate "temperature").
        Attributes are NOT in long-term statistics, so an attribute read always
        uses raw states only (bounded by purge_keep_days).

    cumulative=False -> instantaneous quantity (e.g. temperature). Statistics
        provide the hourly MEAN.
    cumulative=True  -> cumulative meter (kWh). Statistics provide the hourly
        end-of-hour STATE (the meter reading), which the caller differences into
        per-hour consumption exactly as it does with raw readings.

    unknown/unavailable states and non-numeric values are skipped.
    """
    now = dt_util.now().astimezone()
    look_back = _ALL_HISTORY_DAYS if days is None else days
    start = now - timedelta(days=look_back)

    def _load_raw(raw_start: datetime) -> list[tuple[datetime, float]]:
        raw = history.state_changes_during_period(
            hass,
            raw_start,
            entity_id=entity_id,
            no_attributes=(attribute is None),
        )
        out: list[tuple[datetime, float]] = []
        for st in raw.get(entity_id, []):
            if st.state in ("unknown", "unavailable", None, ""):
                if attribute is None:
                    continue
            value = st.attributes.get(attribute) if attribute else st.state
            if value is None:
                continue
            try:
                out.append((st.last_changed, float(value)))
            except (ValueError, TypeError):
                continue
        out.sort(key=lambda x: x[0])
        return out

    def _load_stats(stat_start: datetime, stat_end: datetime):
        # Long-term statistics are hourly. Read MEAN for instantaneous values,
        # STATE (end-of-hour reading) for cumulative meters.
        stat_type = "state" if cumulative else "mean"
        result = statistics.statistics_during_period(
            hass,
            stat_start,
            stat_end,
            {entity_id},
            "hour",
            None,
            {stat_type},
        )
        rows = result.get(entity_id, [])
        out: list[tuple[datetime, float]] = []
        for row in rows:
            val = row.get(stat_type)
            ts = row.get("start")
            if val is None or ts is None:
                continue
            # "start" is a POSIX timestamp (float) in recent cores
            when = (
                datetime.fromtimestamp(ts, tz=dt_util.UTC)
                if isinstance(ts, (int, float))
                else ts
            )
            try:
                out.append((when, float(val)))
            except (ValueError, TypeError):
                continue
        out.sort(key=lambda x: x[0])
        return out

    def _load() -> list[tuple[datetime, float]]:
        # Attributes are not in statistics -> raw only.
        if attribute is not None:
            return _load_raw(start)

        # Recent raw states (exact, last purge_keep_days).
        raw = _load_raw(start)

        # If we want more than the raw window covers, prepend long-term stats.
        # Determine where raw data actually begins; fetch stats before that.
        raw_begin = raw[0][0] if raw else now
        if start < raw_begin - timedelta(hours=1):
            stats = _load_stats(start, raw_begin)
            # merge: stats strictly before the first raw sample, then raw
            merged = [s for s in stats if s[0] < raw_begin] + raw
            merged.sort(key=lambda x: x[0])
            return merged
        return raw

    return await get_instance(hass).async_add_executor_job(_load)
