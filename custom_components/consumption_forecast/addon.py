"""Optional model provider add-on: discovery and HTTP client.

The heavy models (LightGBM today, others later) do not run inside Home
Assistant. They live in the *Consumption Forecast Model Provider* add-on
(github.com/viljasenville/home-assistant-apps/tree/main/cfmp), which trains
and serves them over HTTP from its own glibc container. The integration owns
the data — it reads the recorder, assembles the hourly series and sends it —
and the add-on only fits models and returns forecasts.

Two rules shape this module:

* **The add-on is an optional accelerator.** Every call is bounded by a
  timeout, and every failure is reported to the caller so it can fall back to
  the built-in profile model. A missing, stopped or broken add-on must never
  break the integration.
* **Models are discovered, never hard-coded.** ``GET /models`` is the
  authoritative list, so an add-on that later gains another backend shows up
  in the integration's settings with no release here.

Discovery goes through the Supervisor API rather than a fixed hostname,
because Supervisor derives an add-on's hostname from its slug and prefixes
that slug per install source (``local_cfmp`` versus ``a1b2c3d4_cfmp``). The
same lookup also yields the add-on's own options, so the API token is picked
up automatically and never has to be typed twice.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    ADDON_HEALTH_TIMEOUT,
    ADDON_MODELS_TIMEOUT,
    ADDON_PORT,
    ADDON_PREDICT_TIMEOUT,
    ADDON_SLUG,
    ADDON_TRAIN_TIMEOUT,
    MODEL_PROFILE,
    SUPERVISOR_URL,
)

_LOGGER = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """The add-on was unreachable or answered with an error."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class InsufficientData(ModelServiceError):
    """The add-on had too little usable history to train (HTTP 422)."""


@dataclass(frozen=True)
class AddonInfo:
    """What Supervisor knows about the installed add-on."""

    slug: str
    name: str
    version: str | None
    state: str
    url: str
    token: str | None

    @property
    def running(self) -> bool:
        return self.state == "started"


# --------------------------------------------------------------------------- #
#  Discovery via the Supervisor API                                            #
# --------------------------------------------------------------------------- #
async def _supervisor_get(hass: HomeAssistant, path: str):
    """GET a Supervisor API path and return its ``data`` payload, or None.

    Returns None for every failure mode — not running under Supervisor, no
    token, network error, non-ok response — because the only question this
    module ever asks is "is the add-on usable?", and anything short of a clean
    answer means "no".
    """
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return None  # not a Supervisor installation (Core/Container)
    session = async_get_clientsession(hass)
    try:
        async with session.get(
            f"{SUPERVISOR_URL}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=aiohttp.ClientTimeout(total=ADDON_MODELS_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                _LOGGER.debug("Supervisor %s returned HTTP %s", path, resp.status)
                return None
            body = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
        _LOGGER.debug("Supervisor %s failed: %s", path, err)
        return None
    if not isinstance(body, dict) or body.get("result") != "ok":
        return None
    return body.get("data")


def _matches_slug(slug: str) -> bool:
    """Whether a Supervisor add-on slug is our add-on.

    Supervisor stores an add-on under a slug prefixed by its install source:
    ``local_cfmp`` for a local build, ``<repository hash>_cfmp`` for one
    installed from a repository. Matching the suffix covers both without
    knowing the repository.
    """
    return slug == ADDON_SLUG or slug.endswith(f"_{ADDON_SLUG}")


async def async_find_addon(hass: HomeAssistant) -> AddonInfo | None:
    """Locate the installed model provider add-on, or None if there is none.

    Only *installed* add-ons are listed by Supervisor, so a hit here means the
    user has the add-on; ``AddonInfo.running`` says whether it is also started.
    """
    data = await _supervisor_get(hass, "/addons")
    if not data:
        return None
    installed = [a for a in data.get("addons", []) if _matches_slug(a.get("slug", ""))]
    if not installed:
        return None

    slug = installed[0]["slug"]
    info = await _supervisor_get(hass, f"/addons/{slug}/info")
    if not info:
        return None

    # Prefer the Supervisor-assigned hostname (resolvable on the internal
    # network); fall back to the container IP if it is missing.
    host = info.get("hostname") or info.get("ip_address")
    if not host:
        return None

    options = info.get("options") or {}
    token = (options.get("api_token") or "").strip() or None
    return AddonInfo(
        slug=slug,
        name=info.get("name") or slug,
        version=info.get("version"),
        state=info.get("state") or "unknown",
        url=f"http://{host}:{ADDON_PORT}",
        token=token,
    )


# --------------------------------------------------------------------------- #
#  HTTP client                                                                 #
# --------------------------------------------------------------------------- #
class ModelServiceClient:
    """Thin aiohttp client for the add-on's HTTP API.

    ``model_id`` identifies *whose* trained model (the config entry id), while
    ``model`` in each call identifies *which backend* — two different things
    that both appear in most requests.
    """

    def __init__(self, hass: HomeAssistant, info: AddonInfo, model_id: str):
        self._session = async_get_clientsession(hass)
        self.info = info
        self._url = info.url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {info.token}"} if info.token else {}
        self._model_id = model_id

    async def _request(self, method: str, path: str, *, json=None, timeout: int):
        try:
            async with self._session.request(
                method,
                f"{self._url}{path}",
                json=json,
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 204:
                    return None
                try:
                    body = await resp.json(content_type=None)
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                if resp.status == 422:
                    # A valid request with a negative answer: not enough data.
                    raise InsufficientData(
                        str(body.get("reason") or "insufficient_data"),
                        code=body.get("reason"),
                        status=422,
                    )
                if resp.status >= 400:
                    raise ModelServiceError(
                        body.get("message")
                        or body.get("error")
                        or f"HTTP {resp.status}",
                        code=body.get("error"),
                        status=resp.status,
                    )
                return body
        except asyncio.TimeoutError as err:
            raise ModelServiceError(f"Timed out after {timeout}s on {path}") from err
        except aiohttp.ClientError as err:
            raise ModelServiceError(f"Connection error on {path}: {err}") from err

    async def async_health(self) -> dict | None:
        """The add-on's /health payload, or None if it is not answering."""
        try:
            return await self._request(
                "GET", "/health", timeout=ADDON_HEALTH_TIMEOUT
            )
        except ModelServiceError as err:
            _LOGGER.debug("Add-on health check failed: %s", err)
            return None

    async def async_list_models(self) -> list[dict]:
        """Backends this add-on can actually run, profile excluded.

        The built-in profile model always uses the integration's own code, so
        the add-on's equivalent backend is filtered out here — offering both
        would present the user with two names for the same thing.
        """
        body = await self._request("GET", "/models", timeout=ADDON_MODELS_TIMEOUT)
        models = (body or {}).get("models") or []
        return [
            m
            for m in models
            if m.get("available") and m.get("id") and m["id"] != MODEL_PROFILE
        ]

    async def async_train(
        self, model: str, series: list[dict], *, base_temp: float
    ) -> dict:
        """Train ``model`` on an hourly series and persist it in the add-on.

        ``series`` rows are the integration's assembled hourly rows
        (ts/energy/out_temp/target); see ``_hour_actual``.
        """
        payload = {
            "model": model,
            "model_id": self._model_id,
            "base_temp": base_temp,
            "series": series,
        }
        return await self._request(
            "POST", "/train", json=payload, timeout=ADDON_TRAIN_TIMEOUT
        ) or {}

    async def async_predict(
        self, model: str, future: list[dict], history_tail: list[dict] | None = None
    ) -> list[tuple[datetime, float]]:
        """Forecast the ``future`` hours, oldest first.

        The add-on takes the first future hour as the forecast origin and needs
        ``history_tail`` to reach at least the backend's lag depth before it.
        Returns [(ts, kwh), …] with timezone-aware timestamps.
        """
        payload: dict = {
            "model": model,
            "model_id": self._model_id,
            "future": future,
        }
        if history_tail:
            payload["history_tail"] = history_tail
        body = await self._request(
            "POST", "/predict", json=payload, timeout=ADDON_PREDICT_TIMEOUT
        )
        out = []
        for row in (body or {}).get("hourly") or []:
            try:
                out.append((datetime.fromisoformat(row["ts"]), float(row["kwh"])))
            except (KeyError, TypeError, ValueError) as err:
                raise ModelServiceError(f"Malformed forecast row {row!r}") from err
        if not out:
            raise ModelServiceError("Add-on returned an empty forecast.")
        return out

    async def async_instance_info(self, model: str) -> dict:
        """Diagnostics for an already trained instance (val_mae, importances)."""
        return await self._request(
            "GET",
            f"/models/{model}/instances/{self._model_id}",
            timeout=ADDON_MODELS_TIMEOUT,
        ) or {}

    async def async_delete_instance(self, model: str) -> None:
        """Drop a trained instance; called when the config entry is removed."""
        await self._request(
            "DELETE",
            f"/models/{model}/instances/{self._model_id}",
            timeout=ADDON_MODELS_TIMEOUT,
        )


async def async_get_client(
    hass: HomeAssistant, model_id: str
) -> ModelServiceClient | None:
    """Discover the add-on and return a client for it, or None if unavailable.

    None covers every "use the built-in model" case: no Supervisor, add-on not
    installed, or installed but not started.
    """
    info = await async_find_addon(hass)
    if info is None:
        return None
    if not info.running:
        _LOGGER.debug("Add-on %s is installed but %s.", info.slug, info.state)
        return None
    return ModelServiceClient(hass, info, model_id)


async def async_available_models(hass: HomeAssistant, model_id: str) -> list[dict]:
    """Backends the add-on offers right now; empty when there is no add-on.

    Used by the config flow to populate the model selector, so the choice
    always reflects what is actually installed.
    """
    client = await async_get_client(hass, model_id)
    if client is None:
        return []
    try:
        return await client.async_list_models()
    except ModelServiceError as err:
        _LOGGER.warning("Could not list add-on models: %s", err)
        return []


# --------------------------------------------------------------------------- #
#  Payload helpers                                                             #
# --------------------------------------------------------------------------- #
def hour_actual(row: dict) -> dict:
    """One assembled hourly row -> the add-on's HourActual JSON."""
    out = {
        "ts": row["ts"].isoformat(),
        "energy": float(row["energy"]),
        "out_temp": float(row["out_temp"]),
    }
    if row.get("target") is not None:
        out["target"] = float(row["target"])
    return out


def hour_future(row: dict) -> dict:
    """One future row -> the add-on's HourFuture JSON (no energy)."""
    out = {"ts": row["ts"].isoformat(), "out_temp": float(row["out_temp"])}
    if row.get("target") is not None:
        out["target"] = float(row["target"])
    return out
