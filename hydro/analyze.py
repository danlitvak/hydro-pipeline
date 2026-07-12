"""Statistics: rolling baselines, confidence intervals, anomaly flags.

All functions are pure (pandas/NumPy in, pandas/NumPy out). Statistical
reasoning, in the spirit of an intro mathematical-statistics course
(STATS 302):

**Rolling mean and standard deviation.** For each day *t* we treat the
trailing ``window`` days (default 7) as a small sample x_1..x_n from the
river's "current regime" and compute the sample mean x-bar_t and sample
standard deviation s_t (ddof=1, the unbiased-variance estimator).

**Confidence interval on the rolling mean.** With n small (<= 7) and the
population variance unknown, the pivotal quantity
``(x-bar - mu) / (s / sqrt(n))`` follows (approximately) a Student-t
distribution with n-1 degrees of freedom, so the 95 % CI is::

    x-bar_t  +/-  t_{0.975, n-1} * s_t / sqrt(n)

using the t quantile rather than z = 1.96 — with n = 7 the multiplier is
2.447, i.e. ~25 % wider, which matters at these sample sizes. Honest caveat:
consecutive daily observations of a river are positively autocorrelated, so
the effective sample size is below n and the stated interval is somewhat
narrower than the true one. It is presented as a descriptive "uncertainty of
the weekly running mean", not an inferential claim about independent draws.

**Z-score anomalies.** Each day's value is compared against the *previous*
window's baseline (mean and std shifted by one day)::

    z_t = (x_t - x-bar_{t-1}) / s_{t-1}

Excluding today from its own baseline matters: a large spike included in its
own window drags the mean toward itself and inflates the std, muting exactly
the signal we want to detect. Under an approximate-normal baseline,
P(|Z| > 2) ~ 4.6 %, so |z| > 2 flags roughly the most surprising ~5 % of
days — a deliberate "worth a look" threshold (river data is non-stationary
and autocorrelated, so this is a screening rule, not a hypothesis test with
a calibrated Type-I error rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

STATS_COLUMNS = [
    "station_id", "date", "parameter", "value", "roll_mean", "roll_std",
    "n_obs", "ci_lo", "ci_hi", "zscore", "is_anomaly",
]


def rolling_stats(
    values: pd.Series, window: int = 7, min_periods: int = 4
) -> pd.DataFrame:
    """Trailing-window mean, sample std (ddof=1) and observation count.

    ``min_periods`` guards the warm-up edge: with fewer than that many
    observations in the window the estimates are too noisy to report, so
    they are NaN. ``n_obs`` counts non-missing days actually in the window
    (missing days shrink n, and the CI widens accordingly via sqrt(n)).
    """
    roll = values.rolling(window=window, min_periods=min_periods)
    out = pd.DataFrame(
        {
            "roll_mean": roll.mean(),
            "roll_std": roll.std(ddof=1),
            "n_obs": values.rolling(window=window, min_periods=1).count(),
        }
    )
    out.loc[out["roll_mean"].isna(), "n_obs"] = np.nan
    return out


def t_multiplier(n_obs: np.ndarray | pd.Series, ci_level: float = 0.95) -> np.ndarray:
    """Two-sided Student-t quantile t_{(1+level)/2, n-1}; NaN where n < 2."""
    n = np.asarray(n_obs, dtype=float)
    dof = np.where(n >= 2, n - 1, np.nan)
    with np.errstate(invalid="ignore"):
        return scipy_stats.t.ppf((1.0 + ci_level) / 2.0, dof)


def ci_bounds(
    roll_mean: pd.Series,
    roll_std: pd.Series,
    n_obs: pd.Series,
    ci_level: float = 0.95,
) -> tuple[pd.Series, pd.Series]:
    """t-based confidence interval for the rolling mean (see module docs)."""
    n = n_obs.astype(float)
    mult = t_multiplier(n, ci_level)
    with np.errstate(invalid="ignore"):
        half_width = mult * roll_std / np.sqrt(n)
    return roll_mean - half_width, roll_mean + half_width


def zscores(
    values: pd.Series, roll_mean: pd.Series, roll_std: pd.Series
) -> pd.Series:
    """z of each value against the *previous* day's rolling baseline.

    The shift keeps today's observation out of its own baseline. A zero or
    missing baseline std yields NaN (a constant week gives no yardstick for
    surprise — flagging anything against sigma = 0 would be division noise).
    """
    base_mean = roll_mean.shift(1)
    base_std = roll_std.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (values - base_mean) / base_std
    return z.mask(~(base_std > 0))


def flag_anomalies(z: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Strict two-sided flag: |z| > threshold (NaN z is never an anomaly)."""
    return (z.abs() > threshold).fillna(False).astype(int)


def analyze_series(
    daily: pd.DataFrame,
    window: int = 7,
    min_periods: int = 4,
    z_threshold: float = 2.0,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Stats for one station/parameter clean_daily slice (single group).

    Expects columns ``date`` (YYYY-MM-DD strings) and ``value``. The series
    is re-indexed onto its full daily range first so that calendar gaps count
    as missing observations instead of silently compressing the window.
    """
    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    s = daily.set_index(pd.to_datetime(daily["date"]))["value"].reindex(idx)

    rs = rolling_stats(s, window=window, min_periods=min_periods)
    ci_lo, ci_hi = ci_bounds(rs["roll_mean"], rs["roll_std"], rs["n_obs"], ci_level)
    z = zscores(s, rs["roll_mean"], rs["roll_std"])
    flags = flag_anomalies(z, threshold=z_threshold)
    # A flag needs an actual observation for the day.
    flags = flags.where(s.notna(), 0).astype(int)

    return pd.DataFrame(
        {
            "date": idx.strftime("%Y-%m-%d"),
            "value": s.to_numpy(),
            "roll_mean": rs["roll_mean"].to_numpy(),
            "roll_std": rs["roll_std"].to_numpy(),
            "n_obs": rs["n_obs"].to_numpy(),
            "ci_lo": ci_lo.to_numpy(),
            "ci_hi": ci_hi.to_numpy(),
            "zscore": z.to_numpy(),
            "is_anomaly": flags.to_numpy(),
        }
    )


def analyze(
    clean_daily: pd.DataFrame,
    window: int = 7,
    min_periods: int = 4,
    z_threshold: float = 2.0,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Run :func:`analyze_series` for every (station, parameter) group."""
    if clean_daily.empty:
        return pd.DataFrame(columns=STATS_COLUMNS)

    pieces = []
    for (station_id, parameter), grp in clean_daily.groupby(["station_id", "parameter"]):
        stats = analyze_series(
            grp,
            window=window,
            min_periods=min_periods,
            z_threshold=z_threshold,
            ci_level=ci_level,
        )
        stats.insert(0, "station_id", station_id)
        stats.insert(2, "parameter", parameter)
        pieces.append(stats)

    return pd.concat(pieces, ignore_index=True)[STATS_COLUMNS]
