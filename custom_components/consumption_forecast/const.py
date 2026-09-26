"""Constants and configuration keys."""

DOMAIN = "consumption_forecast"

# --- Configurable input entities ---
CONF_ENERGY = "energy_entity"          # cumulative kWh consumption meter (device_class energy)
CONF_OUTDOOR = "outdoor_temp_entity"   # outdoor temperature (C)
CONF_INDOOR = "indoor_temp_entity"     # indoor temperature (C, optional); preferred heating-degree reference
CONF_FORECAST = "weather_entity"       # weather entity (forecast source)
CONF_THERMOSTAT = "thermostat_entity"  # climate entity, target from attr "temperature"
CONF_PRODUCTION = "production_entity"   # cumulative kWh solar production (optional, history only)
CONF_EXPORT = "export_entity"          # cumulative kWh grid export/sale (optional, history only)

# --- Configurable parameters ---
CONF_BASE_TEMP = "base_temp"           # heating threshold when no indoor sensor or thermostat is set
CONF_HISTORY_DAYS = "history_days"     # training window (days); empty = all available
CONF_TIMEZONE = "timezone"             # house timezone
CONF_MAX_HOURLY_KWH = "max_hourly_kwh" # per-hour consumption cap (anomaly filter)
CONF_PEAK_HOURS = "peak_hours"         # how many high/low peak hours per day to flag
CONF_MODEL = "model"                   # forecast model: "profile" or an add-on backend id

# --- Defaults ---
DEFAULT_BASE_TEMP = 21.0
DEFAULT_PEAK_HOURS = 4                  # flag the 4 highest and 4 lowest hours/day
# History window default is None -> fetch ALL data the recorder still has
# (the recorder's own retention bounds it). A number narrows it to that many days.
DEFAULT_HISTORY_DAYS = None
DEFAULT_TIMEZONE = "Europe/Helsinki"
DEFAULT_MAX_HOURLY_KWH = 100.0
# Future indoor temperature is unknown, so the forecast uses the mean of the last
# INDOOR_REF_DAYS days of the indoor sensor as its heating-degree reference. An
# unbiased estimate of the same quantity the model trained on, and long enough to
# average out day/night swings without lagging a real setpoint change.
INDOOR_REF_DAYS = 7
# The built-in profile model is the default and the fallback: it needs no
# add-on, runs anywhere, and is always available.
MODEL_PROFILE = "profile"
DEFAULT_MODEL = MODEL_PROFILE

# --- Schedules ---
# The forecast is refreshed once an hour, exactly on the hour (at :00), driven
# by a wall-clock tick rather than a relative timer, so every hourly sensor
# (peak hours, current-hour forecast, …) updates on the hour. The model is
# retrained once a day, also on the hour, at TRAIN_HOUR local time.
TRAIN_HOUR = 3                         # retrain daily at 03:00 local time
# When an update produces no forecast (e.g. right after a restart, before the
# weather entity or recorder history is ready), retry after this short interval
# instead of waiting for the next hour, so sensors do not stay "unknown" for
# long. This is the only case where updates are not aligned to the hour.
FORECAST_RETRY_MINUTES = 1

# --- Model minimum requirements ---
MIN_TRAIN_HOURS = 24 * 14              # at least ~2 weeks of data to train
# History tail sent with a remote prediction, for the add-on model's lag
# features. It must cover the backend's lag depth (168 h for LightGBM) BEFORE
# the forecast origin; today's whole-day forecast uses midnight as the origin,
# so the tail needs 168 h + a day of slack for recorder gaps.
LAG_TAIL_HOURS = 240

# --- Model provider add-on ---
# The add-on (github.com/viljasenville/home-assistant-apps/tree/main/cfmp)
# hosts the heavy models (LightGBM) in its own glibc container, so the
# integration itself needs no build dependency. It is optional: without it the
# built-in profile model is used.
#
# Supervisor prefixes a store-installed add-on's slug with its repository hash
# ("a1b2c3d4_cfmp") and a locally installed one with "local_", so the slug is
# matched by suffix rather than compared outright.
ADDON_SLUG = "cfmp"
ADDON_PORT = 8099
SUPERVISOR_URL = "http://supervisor"

# Request timeouts (seconds). Training is synchronous in the add-on and can
# take minutes on slow hardware; everything else is quick.
ADDON_HEALTH_TIMEOUT = 5
ADDON_MODELS_TIMEOUT = 10
ADDON_PREDICT_TIMEOUT = 30
ADDON_TRAIN_TIMEOUT = 900
