# Consumption Forecast (Home Assistant integration)

Forecasts a home's electricity consumption for the coming days from historical
data and a weather forecast. All input entities are selected from the UI — no
YAML required.

## What it does

- **Daily consumption estimate** — `sensor.consumption_forecast_today` (the
  **whole current day**: a forecast for all 24 hours, so it stays stable
  through the day rather than shrinking as hours pass),
  `sensor.consumption_forecast_tomorrow` (the next day), and
  `sensor.consumption_forecast_week` (the **next 7 days**, excluding today; the
  per-day breakdown that sums to it is in the `days` attribute). Except for the
  current-day sensor these are forward "next N days" windows, mirroring the
  accuracy sensors' "last N days" windows.
- **Hourly consumption estimate** (`sensor.consumption_forecast_hourly`, with the
  full hourly table in the `forecast` attribute)
- **Current-hour forecast** (`sensor.consumption_forecast_current_hour`) — the
  forecast for the hour in progress, published as a plain state so the recorder
  keeps its history; chart it against actual consumption to see how past
  forecasts landed.
- **High / low consumption hour** binary sensors
  (`binary_sensor.consumption_forecast_high_consumption_hour`,
  `binary_sensor.consumption_forecast_low_consumption_hour`) — ON while the
  current hour is one of the day's highest or lowest forecast hours. Useful for
  load-shifting automations. The number of hours flagged per day is set by the
  **Peak hours** option (default 4; 0 disables the sensors).
- **Train model** button (`button.consumption_forecast_train_model`) — retrains
  the model on demand, the same as the `train_model` service. Errors (e.g. not
  enough recorded history) surface in the UI.
- **Refresh forecast** button (`button.consumption_forecast_refresh_forecast`) —
  recomputes the forecast with the current model, fetching a fresh weather
  forecast. Light and fast; no retraining. Same as the `refresh_forecast`
  service.
- **Forecast accuracy** sensors
  (`sensor.consumption_forecast_accuracy_today`,
  `sensor.consumption_forecast_accuracy_yesterday`,
  `sensor.consumption_forecast_accuracy_week`,
  `sensor.consumption_forecast_accuracy_month`) — how well the forecast matched
  actual consumption over **today so far**, **yesterday**, the **last 7 days**,
  and the **last 30 days**. The state is the deviation in percent (forecast
  total vs actual total); the attributes add the hourly `mae_kwh_per_h`, the
  `forecast_kwh`/`actual_kwh` totals, and how many hours were compared. The
  integration keeps its own log of each hour's forecast (persisted across
  restarts, ~33 days) and scores it against the actual consumption, so these
  fill in over time. **Today** is a partial, running figure — it covers only
  the hours lived so far and fluctuates through the day (its `hours_compared`
  attribute shows how many hours are in it), so it is most meaningful late in
  the day. **Yesterday**, **week** and **month** cover only completed days
  (ending yesterday), so they are not skewed by a partial current day.

The integration forecasts a single quantity: total house consumption. It does
not forecast grid purchase or cost separately — grid purchase depends on future
solar generation, which cannot be predicted from temperature.

## How it works

Consumption is modeled as the sum of heating (weather-dependent) and a base load.
The heating driver is heating degree hours, computed from the difference between
the thermostat target temperature (or a fixed heating threshold) and the outdoor
temperature.

Which model does the forecasting is your choice, made in the integration's
settings:

1. **Built-in profile model** (default) — ridge regression at day level (numpy
   only, no extra dependencies) + a robust (median) historical hourly shape with
   a base-load floor, so no hour is ever forecast below the load that is
   essentially always present (fridge, standby, HVAC idle). Robust with little
   data, does not overfit, and shrugs off occasional meter gaps or resets. It
   runs inside Home Assistant on every architecture and needs nothing installed.
2. **Add-on models** (optional) — heavier models such as LightGBM, trained and
   served by the [Consumption Forecast Model Provider][addon] add-on. They are
   more accurate at the hourly level once you have a few months of data. The
   integration still owns the data: it reads the recorder, assembles the hourly
   series and sends it to the add-on, which only fits models and returns
   forecasts. See "Add-on models" below.

[addon]: https://github.com/viljasenville/home-assistant-apps/tree/main/cfmp

The profile model always runs in the integration's own code, even when the
add-on is installed, and it is the fallback for everything: if the add-on is
missing, stopped, still short of data, or simply fails to answer a request, the
forecast comes from the profile model instead. A forecast is never lost because
of the add-on.

The forecast is refreshed once an hour, **exactly on the hour** (at :00), so
the hourly sensors — the peak-hour flags and the current-hour forecast — change
in step with the wall clock rather than drifting to whatever minute the last
refresh ran. The model is retrained once a day, at 03:00 local time. (The only
exception to on-the-hour timing is a brief one-minute retry right after a
restart, until the weather entity and recorder history are ready.) The
`sensor.consumption_forecast_week` attributes expose diagnostics: the model that
actually produced the forecast (`active_model`) and where it ran
(`model_source`: `built-in` or `add-on`), its validation MAE (`val_mae`), the
add-on's state and last training result (`addon`), and when the model was
last trained and over what span of data — `trained_at`, `training_start`,
`training_end`, `training_span_days` (calendar span of the data), and
`training_hours` (usable hours, which can be fewer than the span if there were
gaps).

After a Home Assistant restart the weather entity and recorder history may not be
ready immediately. To avoid the sensors showing "unknown" in the meantime, the
integration restores the **last forecast from disk** on startup, so the previous
values appear right away. In parallel it retries every minute until a fresh
forecast is produced, then returns to the hourly cadence.

## Installation

1. Copy `custom_components/consumption_forecast` into Home Assistant's
   `config/custom_components/` directory (or install via HACS as a custom
   repository).
2. Restart Home Assistant.
3. **Settings → Devices & Services → Add Integration → "Consumption Forecast"**.
4. Select the input entities and parameters.

The integration installs with no build step and no required third-party
packages — the profile model runs on numpy alone, which Home Assistant already
ships. See "Add-on models" below to enable the more accurate hour-level models.

## Configurable inputs

| Field | Required | Description |
|-------|:--------:|-------------|
| Energy meter | yes | Cumulative kWh consumption meter (device_class `energy`) |
| Solar production | no | Cumulative kWh produced by panels. Used only to correct history |
| Grid export | no | Cumulative kWh sold to the grid. Used only to correct history |
| Outdoor temperature | yes | °C |
| Weather | yes | `weather` entity the forecast is fetched from |
| Indoor temperature | no | °C |
| Thermostat | no | `climate` entity; its **current** target is applied to the whole training history (see note below) |
| Forecast model | no | Which model forecasts. Default is the built-in profile model; other entries appear only when the model provider add-on is installed |
| Heating threshold | no | Used if no thermostat. Default 17 °C. Set this to the temperature your home actually heats to |
| Training window | no | Days of history to train on. Leave empty to use all available data (bounded by the recorder's own retention); set a number to narrow it |
| Timezone | no | IANA name. Default `Europe/Helsinki` |
| Max hourly kWh | no | Anomaly filter (meter resets). Default 100 |
| Peak hours | no | How many highest/lowest hours per day the high/low binary sensors flag. Default 4; 0 disables them |

All settings can be changed later from the integration's **Configure** button.
Optional entities (thermostat, indoor temperature, solar production and export)
can also be cleared there — emptying the field removes that input.

**Training window.** Left empty, the integration trains on all available
history. It reads Home Assistant's **long-term statistics** (hourly aggregates,
kept indefinitely) for the older history and tops them up with the raw recorder
states for the most recent days — so it reaches back months or years, not just
the ~10 days of raw state history the recorder keeps by default.

This requires your energy meter and temperature sensor to have a **state_class**
(most `device_class: energy` and `temperature` sensors do), because only those
entities get long-term statistics. If a sensor has no statistics, the
integration falls back to raw history alone (~10 days). If your consumption
habits changed a lot at some point (a new heat pump, a renovation), set the
window to a number of days that covers only the period that still represents
your home, rather than training on outdated years.

**Heating threshold / thermostat target.** Consumption is driven by heating
degree hours — how far the outdoor temperature sits below your indoor target.
For the forecast to be right, the model must learn and predict with the **same**
target. A thermostat's target is an attribute, so its history only reaches the
recorder's raw window (~10 days), not the long-term statistics; to keep the
threshold consistent across the whole (possibly year-long) training history, the
integration applies the thermostat's **current** target to all of it. This
assumes a stable setpoint — if you change it seasonally, retrain (the Train
button) after the change. If you have no thermostat, set the **heating
threshold** to the temperature your home actually heats to (e.g. 21 °C); a wrong
threshold here systematically scales the whole forecast up or down.

### Solar panels

If your energy meter measures only **grid purchase**, sunny hours read near
zero even though the house is still consuming — and the forecast would learn
those zeros. Add the **solar production** and **grid export** entities so the
integration can reconstruct real consumption from history:

**total consumption = grid + production − export**

The model then trains on that real consumption and forecasts it. Production and
export are used **only to correct the training history** — the integration does
not forecast solar generation or grid purchase, because grid purchase depends on
future sunshine, which temperature cannot predict. Adding just one of the two
entities also works; the other is treated as zero. Without either, the meter is
taken as-is.

For a true consumption meter that already measures the whole house (before solar
self-consumption), you don't need these — just point the energy meter at it.

## Services

Training runs automatically once a day and the forecast refreshes hourly. Two
services let you trigger either step by hand — from Developer Tools → Actions,
an automation, or a dashboard button:

- **`consumption_forecast.train_model`** — force an immediate retrain from the recorded
  history (trains both models and keeps the better one).
- **`consumption_forecast.refresh_forecast`** — recompute the forecast right away using
  the current model, without retraining.
- **`consumption_forecast.forecast_period`** — forecast consumption over an arbitrary
  period from a start, an end, and a single outdoor temperature. Works for any
  past, present, or future range and returns the result as response data.

`train_model` and `refresh_forecast` take an optional `entry_id` to target a
specific configuration; leave it empty to apply to all configured entries.

```yaml
# Example: retrain a specific entry
action: consumption_forecast.train_model
data:
  entry_id: 1a2b3c4d5e6f7g8h

# Example: refresh all forecasts
action: consumption_forecast.refresh_forecast

# Example: forecast a 3-day period (returns response data)
action: consumption_forecast.forecast_period
data:
  start: "2026-01-15 00:00:00"
  end: "2026-01-18 00:00:00"
  outdoor_temp: -12
  target_temp: 21        # optional; defaults to current thermostat target
response_variable: result
```

`train_model` raises an error if training fails, and logs a warning (without
error) when there is not yet enough data to train.

### forecast_period

Because you supply the outdoor temperature yourself, `forecast_period` needs no
weather forecast and works for any range — including dates far beyond the
weather forecast horizon, and past dates (useful for checking the model against
what actually happened). It always uses the built-in profile
model, which is a pure function of temperature and time — regardless of which
model is selected for the rolling forecast. An add-on model needs a history tail
ending right before the hours it forecasts, so it cannot answer for a detached,
arbitrary range.

The single `outdoor_temp` is applied to every hour in the period, so for long
periods it is an approximation. `start` is inclusive and `end` is exclusive
(e.g. midnight-to-midnight over three dates yields 72 hours). The response
contains, under `total_consumption`, the `total_kwh`, the `daily` sums, and the
`hourly` values. If more than one configuration is set up, pass `entry_id`.

## Requirements

- Home Assistant 2024.4.0 or newer
- `recorder` enabled (it is by default)
- At least ~2 weeks of history for the built-in profile model
- Nothing else. Add-on models are optional, and the add-on itself declares how
  much history each of them needs (LightGBM asks for ~3 weeks, and is at its
  best with several months)

## Add-on models

Heavier models do not run inside Home Assistant. Home Assistant Core runs on
Alpine/musl, where LightGBM has no wheel, so the integration would have to
compile it on the device. Instead those models live in the [Consumption Forecast
Model Provider][addon] add-on, which runs in its own Debian/glibc container and
serves them over HTTP. The integration therefore needs **no third-party package
and no build step at all**.

To use one:

1. Add `https://github.com/viljasenville/home-assistant-apps` as an add-on
   repository (**Settings → Add-ons → Add-on Store → ⋮ → Repositories**).
2. Install **Consumption Forecast Model Provider**, set an `api_token` in its
   configuration, and start it.
3. Open the integration's **Configure** dialog. The **Forecast model** dropdown
   now lists the add-on's models alongside the built-in one. Pick one and save.

Nothing else is needed: the integration finds the add-on through the Supervisor
API — so it works whether the add-on was installed from a repository or built
locally — and reads the API token from the add-on's own configuration. There is
no host, port or token to type twice, and no host port needs to be published.

Changing the selection retrains immediately rather than waiting for the nightly
run. Training happens once a day and can take minutes on slow hardware;
forecasting is a fast request made once an hour. Models the add-on trains are
stored under the config entry's id, so one add-on can serve several Home
Assistant configurations — and deleting the integration tells the add-on to drop
them.

The model list is read from the add-on at the moment the dialog is opened, not
hard-coded here, so an add-on version that adds a backend shows up without an
update to this integration. The add-on's own `profile` backend is deliberately
hidden from the list: the built-in profile model is the same idea in the
integration's own code, and offering both would be two names for one thing.

**LightGBM runs on amd64 and aarch64 only** — Intel/AMD hardware and 64-bit
Raspberry Pi 4/5. On 32-bit ARM (armv7/armhf) the add-on has no image, and the
built-in profile model remains the only option. That is not a downgrade to a
stub: the profile model forecasts correctly on its own, and the hourly accuracy
an add-on model adds only materialises once you have a few months of data.

Which model produced the current forecast, and whether it ran in Home Assistant
or in the add-on, is shown in the `active_model` and `model_source` attributes
of `sensor.consumption_forecast_week`; the `addon` attribute carries the add-on's
version and its last training result.

## Lovelace

See `lovelace-example.yaml`. The hourly forecast is drawn directly with the
`custom:apexcharts-card` card from the forecast sensor's `forecast` attribute.

## Notes

- Over a longer horizon (beyond ~2 days) the uncertainty comes mainly from the
  weather forecast, not the model.
- In the forecast, the thermostat's current target is assumed to hold for the
  whole horizon.
- The add-on's LightGBM model is trained for horizons up to 48 hours. When the
  weather entity offers only a daily forecast, the horizon stretches to several
  days and the furthest hours are extrapolated — accurate enough in shape, but
  do not read too much into day 5. The built-in profile model has no horizon
  limit of its own; there too the weather forecast is the binding constraint.
