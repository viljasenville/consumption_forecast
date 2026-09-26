"""The built-in forecast model.

ProfileModel is a day-level ridge regression (numpy only) plus an hourly
profile split. It is robust with little data, does not overfit, and needs no
third-party build dependency, so it runs on every architecture Home Assistant
supports. It is both the default model and the fallback whenever the optional
model provider add-on is absent or unreachable.

Heavier models (LightGBM and anything added later) live in that add-on, not
here — see addon.py. This module is Home Assistant-independent and
unit-testable.
"""
from __future__ import annotations

import numpy as np


class _Ridge:
    """Minimal ridge regression via the closed-form normal equation.

    A numpy-only drop-in for sklearn.linear_model.Ridge exposing the same
    ``fit`` / ``predict`` / ``coef_`` / ``intercept_`` interface. Used so the
    profile model has no heavy build-from-source dependency; scikit-learn is
    not required for the integration to work.

    Solves  min_w ||Xw - y||^2 + alpha ||w||^2  with an unpenalized intercept.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        # augment with a bias column so the intercept is fit jointly
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        # L2 penalty on all weights except the intercept (last column)
        penalty = self.alpha * np.eye(n_features + 1)
        penalty[-1, -1] = 0.0
        # (X^T X + penalty) w = X^T y  -> solve for w
        gram = Xb.T @ Xb + penalty
        rhs = Xb.T @ y
        w = np.linalg.solve(gram, rhs)
        self.coef_ = w[:-1]
        self.intercept_ = float(w[-1])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
#  Profile model (day level + hourly profile)                                  #
# --------------------------------------------------------------------------- #
class ProfileModel:
    """Day level: ridge  E_day = a*HDD + weekday coefficients + c.
    Hour level: robust (median) hourly shape split by (weekend, cold/warm),
    with a base-load floor so no hour is forecast below the load that is
    essentially always present.
    """

    name = "profile"

    def __init__(self, base_temp: float = 21.0):
        self.base_temp = base_temp
        self.daily = _Ridge(alpha=1.0)
        self.hour_profiles: dict[tuple[bool, bool], np.ndarray] = {}
        self._hdd_median = 0.0
        self._base_load = 0.0  # per-hour floor (kWh); consumption never drops below this
        self.trained = False
        self._val_mae: float | None = None

    def _hdd(self, target: float, out_temp: float) -> float:
        return max(0.0, target - out_temp)

    def _aggregate_days(self, series):
        """series (hourly rows) -> {date: {e, hdd, weekday}}."""
        days: dict = {}
        for r in series:
            d = r["ts"].date()
            day = days.setdefault(
                d, {"e": 0.0, "hdd": 0.0, "weekday": r["ts"].weekday()}
            )
            day["e"] += r["energy"]
            tgt = r.get("target", self.base_temp)
            day["hdd"] += self._hdd(tgt, r["out_temp"])
        return days

    def train(self, series):
        if len(series) < 24:
            return
        days = self._aggregate_days(series)

        # --- day model ---
        X, y = [], []
        for d, v in days.items():
            wd = np.zeros(7)
            wd[v["weekday"]] = 1
            X.append([v["hdd"], *wd])
            y.append(v["e"])
        X, y = np.array(X), np.array(y)
        self.daily.fit(X, y)
        self._hdd_median = float(np.median([v["hdd"] for v in days.values()]))

        # time-ordered val-MAE (last 20% of days), on the same kWh/h scale as
        # the add-on models report, so the two figures can be compared
        self._val_mae = self._compute_val_mae(days)

        # --- base load: robust floor from the quietest hours ---
        # The 5th percentile of all hourly consumption approximates the load
        # that is essentially always present (fridge, standby, HVAC idle).
        # Using a low percentile (not the min) ignores meter-artifact zero-hours.
        all_hourly = np.array([r["energy"] for r in series], dtype=float)
        self._base_load = float(np.percentile(all_hourly, 5))

        # --- hourly profiles (robust): median energy per hour, per key ---
        # Collect every hourly value per (weekend, cold, hour) then take the
        # median. Median ignores occasional near-zero hours caused by stale or
        # reset meter readings, which a mean-based profile would bake in.
        buckets: dict[tuple[bool, bool], list[list[float]]] = {}
        for r in series:
            d = r["ts"].date()
            is_cold = days[d]["hdd"] >= self._hdd_median
            is_weekend = r["ts"].weekday() >= 5
            key = (is_weekend, is_cold)
            if key not in buckets:
                buckets[key] = [[] for _ in range(24)]
            buckets[key][r["ts"].hour].append(r["energy"])

        self.hour_profiles = {}
        for key, hours in buckets.items():
            shape = np.array(
                [float(np.median(v)) if v else 0.0 for v in hours]
            )
            # never allow a profiled hour below the base-load floor
            shape = np.maximum(shape, self._base_load)
            self.hour_profiles[key] = shape
        self.trained = True

    def _compute_val_mae(self, days):
        """Train temporarily on a time-ordered 80/20 split and compute the
        day-level MAE normalized to hours (kWh/h), so the figure is comparable
        with the hourly val_mae an add-on model reports."""
        items = sorted(days.items())
        if len(items) < 10:
            return None
        split = int(len(items) * 0.8)
        train_items, val_items = items[:split], items[split:]
        if not val_items:
            return None
        Xtr, ytr = [], []
        for _, v in train_items:
            wd = np.zeros(7)
            wd[v["weekday"]] = 1
            Xtr.append([v["hdd"], *wd])
            ytr.append(v["e"])
        tmp = _Ridge(alpha=1.0).fit(np.array(Xtr), np.array(ytr))
        errs = []
        for _, v in val_items:
            wd = np.zeros(7)
            wd[v["weekday"]] = 1
            pred = float(tmp.predict([[v["hdd"], *wd]])[0])
            errs.append(abs(pred - v["e"]) / 24.0)  # day -> hour/h
        return float(np.mean(errs))

    def predict_day(self, hdd_day: float, weekday: int) -> float:
        wd = np.zeros(7)
        wd[weekday] = 1
        return float(self.daily.predict([[hdd_day, *wd]])[0])

    def predict_hours(self, day_kwh: float, weekday: int, is_cold: bool):
        """Distribute a day's kWh over 24 hours using the learned shape,
        never letting any hour fall below the base-load floor.

        The stored profile is the robust (median) kWh-per-hour shape. We scale
        it to match ``day_kwh`` in total, but reserve the base load on every
        hour first so a low-consumption hour cannot collapse toward zero.
        """
        key = (weekday >= 5, is_cold)
        shape = self.hour_profiles.get(key)
        if shape is None:
            # no matching profile: flat split with the base-load floor applied
            floor = max(self._base_load, day_kwh / 24.0 if day_kwh > 0 else 0.0)
            return [floor] * 24

        floor = self._base_load
        reserved = floor * 24.0
        # variable (above-floor) energy to distribute across the day
        variable_total = max(0.0, day_kwh - reserved)

        # variable shape = how much each hour exceeds the floor, historically
        variable_shape = np.maximum(shape - floor, 0.0)
        s = variable_shape.sum()
        if s > 0:
            weights = variable_shape / s
        else:
            weights = np.full(24, 1.0 / 24.0)

        hours = floor + weights * variable_total
        return hours.tolist()

    def coefficients(self):
        return {
            "model": self.name,
            "val_mae": self._val_mae,
            "per_hdd": float(self.daily.coef_[0]) if self.trained else None,
            "intercept": float(self.daily.intercept_) if self.trained else None,
            "hdd_median": self._hdd_median,
            "base_load": self._base_load,
        }
