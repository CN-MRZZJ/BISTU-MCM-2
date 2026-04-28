from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


THRESHOLD = 37.0
SEASON_DAYS = 365.0


def filter_sort_key(filter_no: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in filter_no if not ch.isdigit())
    digits = "".join(ch for ch in filter_no if ch.isdigit())
    return prefix, int(digits) if digits else -1


@dataclass(frozen=True)
class Config:
    q1_panel_csv: Path
    q1_indicators_csv: Path
    q1_event_metrics_csv: Path
    output_dir: Path
    max_forecast_years: int = 40
    event_window: int = 7
    half_life_grid: tuple[int, ...] = (7, 14, 21, 30, 45, 60, 90, 120, 180)


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def coefficient_bounds(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(names), -np.inf)
    upper = np.full(len(names), np.inf)
    for idx, name in enumerate(names):
        if name.startswith("beta_t_"):
            upper[idx] = 0.0
        elif name in {"cum_mid_maintenance", "cum_big_maintenance"}:
            upper[idx] = 0.0
        elif name in {"decay_mid_maintenance", "decay_big_maintenance"}:
            lower[idx] = 0.0
    return lower, upper


def bounded_lstsq(
    y: np.ndarray,
    x: np.ndarray,
    names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y_fit = y[mask]
    x_fit = x[mask]
    lower, upper = coefficient_bounds(names)
    fit = lsq_linear(x_fit, y_fit, bounds=(lower, upper), method="trf", lsmr_tol="auto")
    if not fit.success:
        raise RuntimeError(f"Constrained least-squares failed: {fit.message}")
    beta = fit.x
    pred = x_fit @ beta
    resid = y_fit - pred
    n, p = x_fit.shape
    dof = max(n - p, 1)
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(x_fit.T @ x_fit)
    stderr = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    t_value = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    sst = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1.0 - float((resid @ resid) / sst) if sst > 0 else np.nan
    result = pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "std_error": stderr,
            "t_value": t_value,
            "n_obs": n,
            "r2": r2,
            "rss": float(resid @ resid),
            "lower_bound": lower,
            "upper_bound": upper,
            "at_lower_bound": np.isclose(beta, lower, rtol=0.0, atol=1e-8),
            "at_upper_bound": np.isclose(beta, upper, rtol=0.0, atol=1e-8),
        }
    )
    fitted = np.full(len(y), np.nan)
    fitted[mask] = x[mask] @ beta
    return result, fitted


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(cfg.q1_panel_csv, parse_dates=["date"])
    indicators = pd.read_csv(cfg.q1_indicators_csv)
    event_metrics = pd.read_csv(cfg.q1_event_metrics_csv, parse_dates=["date"])

    required = {
        "filter_no",
        "date",
        "permeability_clean",
        "maintenance_type",
        "cum_mid_maintenance",
        "cum_big_maintenance",
        "day_index",
        "season_sin",
        "season_cos",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Q1 panel is missing required columns: {missing}")
    return panel, indicators, event_metrics


def maintenance_events(panel: pd.DataFrame) -> pd.DataFrame:
    events = panel.loc[
        panel["maintenance_type"].notna(),
        ["filter_no", "date", "maintenance_type", "day_index"],
    ].copy()
    events["day_index"] = events["day_index"].astype(int)
    return events.sort_values(["filter_no", "date"]).reset_index(drop=True)


def add_decay_terms(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    lambda_mid: float,
    lambda_big: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for filter_no, group in frame.groupby("filter_no", sort=True):
        g = group.sort_values("date").copy()
        t = g["day_index"].to_numpy(float)
        mid = np.zeros(len(g), dtype=float)
        big = np.zeros(len(g), dtype=float)
        e = events[events["filter_no"] == filter_no]

        for row in e.itertuples(index=False):
            diff = t - float(row.day_index)
            active = diff > 0
            if row.maintenance_type == "中维护":
                mid[active] += np.exp(-lambda_mid * diff[active])
            elif row.maintenance_type == "大维护":
                big[active] += np.exp(-lambda_big * diff[active])

        g["decay_mid_maintenance"] = mid
        g["decay_big_maintenance"] = big
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    filters = sorted(frame["filter_no"].unique(), key=filter_sort_key)
    x_parts: list[np.ndarray] = []
    names: list[str] = []

    for filter_no in filters:
        mask = (frame["filter_no"] == filter_no).to_numpy(float)
        x_parts.append(mask.reshape(-1, 1))
        names.append(f"alpha_{filter_no}")

    for filter_no in filters:
        mask = (frame["filter_no"] == filter_no).to_numpy(float)
        x_parts.append((mask * frame["day_index"].to_numpy(float)).reshape(-1, 1))
        names.append(f"beta_t_{filter_no}")

    common_cols = [
        "season_sin",
        "season_cos",
        "decay_mid_maintenance",
        "decay_big_maintenance",
        "cum_mid_maintenance",
        "cum_big_maintenance",
    ]
    for col in common_cols:
        x_parts.append(frame[col].to_numpy(float).reshape(-1, 1))
        names.append(col)
    return np.hstack(x_parts), names


def fit_life_model(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    half_life_grid: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, int, int]:
    y = panel["permeability_clean"].to_numpy(float)
    candidates: list[dict[str, object]] = []

    for mid_half_life in half_life_grid:
        for big_half_life in half_life_grid:
            lambda_mid = np.log(2) / mid_half_life
            lambda_big = np.log(2) / big_half_life
            trial = add_decay_terms(panel, events, lambda_mid, lambda_big)
            x, names = design_matrix(trial)
            result, fitted = bounded_lstsq(y, x, names)
            candidates.append(
                {
                    "mid_half_life_days": mid_half_life,
                    "big_half_life_days": big_half_life,
                    "lambda_mid": lambda_mid,
                    "lambda_big": lambda_big,
                    "rss": float(result["rss"].iloc[0]),
                    "r2": float(result["r2"].iloc[0]),
                    "result": result,
                    "fitted": fitted,
                    "frame": trial,
                }
            )

    best = min(candidates, key=lambda item: item["rss"])
    fitted_panel = best["frame"].copy()
    fitted_panel["fitted_permeability"] = best["fitted"]
    search_table = pd.DataFrame(
        [
            {
                "mid_half_life_days": item["mid_half_life_days"],
                "big_half_life_days": item["big_half_life_days"],
                "lambda_mid": item["lambda_mid"],
                "lambda_big": item["lambda_big"],
                "rss": item["rss"],
                "r2": item["r2"],
            }
            for item in candidates
        ]
    ).sort_values("rss")
    return (
        best["result"],
        fitted_panel,
        float(best["lambda_mid"]),
        float(best["lambda_big"]),
        int(best["mid_half_life_days"]),
        int(best["big_half_life_days"]),
    ), search_table


def maintenance_intervals(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_means: dict[str, float] = {}
    for maintenance_type in ["中维护", "大维护"]:
        diffs = (
            events[events["maintenance_type"] == maintenance_type]
            .sort_values(["filter_no", "day_index"])
            .groupby("filter_no")["day_index"]
            .diff()
            .dropna()
        )
        global_means[maintenance_type] = float(diffs.mean()) if len(diffs) else 90.0

    for filter_no in sorted(events["filter_no"].unique(), key=filter_sort_key):
        for maintenance_type in ["中维护", "大维护"]:
            e = events[
                (events["filter_no"] == filter_no)
                & (events["maintenance_type"] == maintenance_type)
            ].sort_values("day_index")
            if len(e) >= 2:
                interval = float(e["day_index"].diff().dropna().mean())
                source = "filter_mean"
            else:
                interval = float(global_means[maintenance_type])
                source = "global_mean"
            last_day = int(e["day_index"].iloc[-1]) if len(e) else np.nan
            last_date = e["date"].iloc[-1] if len(e) else pd.NaT
            rows.append(
                {
                    "filter_no": filter_no,
                    "maintenance_type": maintenance_type,
                    "avg_interval_days": interval,
                    "interval_source": source,
                    "last_event_day_index": last_day,
                    "last_event_date": last_date,
                    "event_count": int(len(e)),
                }
            )
    return pd.DataFrame(rows)


def future_events_for_filter(
    filter_no: str,
    historical_events: pd.DataFrame,
    intervals: pd.DataFrame,
    last_day: int,
    horizon_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for maintenance_type in ["中维护", "大维护"]:
        rule = intervals[
            (intervals["filter_no"] == filter_no)
            & (intervals["maintenance_type"] == maintenance_type)
        ].iloc[0]
        interval = max(1, int(round(float(rule["avg_interval_days"]))))
        if pd.notna(rule["last_event_day_index"]):
            next_day = int(rule["last_event_day_index"]) + interval
        else:
            next_day = last_day + interval
        while next_day <= horizon_days:
            rows.append(
                {
                    "filter_no": filter_no,
                    "day_index": next_day,
                    "maintenance_type": maintenance_type,
                    "is_future": 1,
                }
            )
            next_day += interval
    future = pd.DataFrame(rows)
    hist = historical_events[historical_events["filter_no"] == filter_no][
        ["filter_no", "day_index", "maintenance_type"]
    ].copy()
    hist["is_future"] = 0
    return pd.concat([hist, future], ignore_index=True).sort_values("day_index")


def make_future_frame(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    intervals: pd.DataFrame,
    max_forecast_years: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    horizon_add = int(max_forecast_years * 365)
    for filter_no, group in panel.groupby("filter_no", sort=True):
        g = group.sort_values("date")
        start_date = g["date"].min()
        last_date = g["date"].max()
        last_day = int(g["day_index"].max())
        horizon_day = last_day + horizon_add
        all_events = future_events_for_filter(
            filter_no, events, intervals, last_day, horizon_day
        )
        future = pd.DataFrame(
            {
                "filter_no": filter_no,
                "day_index": np.arange(last_day + 1, horizon_day + 1),
            }
        )
        future["date"] = start_date + pd.to_timedelta(future["day_index"], unit="D")
        future["is_history"] = 0
        future = future.merge(
            all_events[all_events["is_future"] == 1][
                ["filter_no", "day_index", "maintenance_type"]
            ],
            on=["filter_no", "day_index"],
            how="left",
        )
        future["is_mid_maintenance"] = (future["maintenance_type"] == "中维护").astype(int)
        future["is_big_maintenance"] = (future["maintenance_type"] == "大维护").astype(int)
        future["cum_mid_maintenance"] = (
            int(g["cum_mid_maintenance"].iloc[-1]) + future["is_mid_maintenance"].cumsum()
        )
        future["cum_big_maintenance"] = (
            int(g["cum_big_maintenance"].iloc[-1]) + future["is_big_maintenance"].cumsum()
        )
        future["season_sin"] = np.sin(2 * np.pi * future["day_index"] / SEASON_DAYS)
        future["season_cos"] = np.cos(2 * np.pi * future["day_index"] / SEASON_DAYS)
        future["last_history_date"] = last_date
        rows.append(future)
    return pd.concat(rows, ignore_index=True)


def predict_with_model(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    model: pd.DataFrame,
    lambda_mid: float,
    lambda_big: float,
) -> pd.DataFrame:
    frame_with_decay = add_decay_terms(frame, events, lambda_mid, lambda_big)
    x, names = design_matrix(frame_with_decay)
    coef = model.set_index("term").loc[names, "coef"].to_numpy(float)
    frame_with_decay["predicted_permeability"] = x @ coef
    return frame_with_decay


def maintenance_effect_threshold(event_metrics: pd.DataFrame) -> float:
    positive = event_metrics.loc[event_metrics["effect_delta"] > 0, "effect_delta"].dropna()
    if len(positive):
        return float(positive.quantile(0.25))
    return 0.0


def predicted_event_effect(
    forecast: pd.DataFrame,
    event_day: int,
    event_window: int,
) -> float:
    before = forecast.loc[
        (forecast["day_index"] >= event_day - event_window)
        & (forecast["day_index"] <= event_day - 1),
        "predicted_permeability",
    ]
    after = forecast.loc[
        (forecast["day_index"] >= event_day + 1)
        & (forecast["day_index"] <= event_day + event_window),
        "predicted_permeability",
    ]
    if len(before) == 0 or len(after) == 0:
        return np.nan
    return float(after.mean() - before.mean())


def life_predictions(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    intervals: pd.DataFrame,
    indicators: pd.DataFrame,
    model: pd.DataFrame,
    delta: float,
    event_window: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indicator_by_filter = indicators.set_index("filter_no")
    coef_by_term = model.set_index("term")["coef"]

    for filter_no, f in forecast.groupby("filter_no", sort=True):
        f = f.sort_values("date").copy()
        h = history[history["filter_no"] == filter_no].sort_values("date").copy()
        h_tail = h[["filter_no", "date", "day_index", "permeability_clean"]].rename(
            columns={"permeability_clean": "predicted_permeability"}
        )
        combined = pd.concat(
            [
                h_tail[["filter_no", "date", "day_index", "predicted_permeability"]],
                f[["filter_no", "date", "day_index", "predicted_permeability"]],
            ],
            ignore_index=True,
        ).sort_values("day_index")
        combined["rolling_365_mean"] = (
            combined["predicted_permeability"].rolling(365, min_periods=365).mean()
        )
        combined["safety_margin"] = combined["rolling_365_mean"] - THRESHOLD
        future = combined[combined["day_index"] > h["day_index"].max()].copy()
        f = f.merge(
            future[["day_index", "rolling_365_mean", "safety_margin"]],
            on="day_index",
            how="left",
        )

        future_events = f[f["maintenance_type"].notna()][
            ["day_index", "date", "maintenance_type"]
        ].copy()
        future_events["predicted_effect_delta"] = future_events["day_index"].apply(
            lambda day: predicted_event_effect(f, int(day), event_window)
        )
        ineffective_events = future_events[
            future_events["predicted_effect_delta"].fillna(np.inf) < delta
        ]

        failure = pd.DataFrame()
        threshold_days = f[f["rolling_365_mean"] < THRESHOLD]
        if len(threshold_days) and len(ineffective_events):
            first_threshold_day = int(threshold_days["day_index"].iloc[0])
            candidate_events = ineffective_events[
                ineffective_events["day_index"] >= first_threshold_day
            ]
            if len(candidate_events):
                event_day = int(candidate_events["day_index"].iloc[0])
                failure = f[f["day_index"] >= event_day].head(1)

        last_date = h["date"].max()
        mid_interval = intervals[
            (intervals["filter_no"] == filter_no) & (intervals["maintenance_type"] == "中维护")
        ]["avg_interval_days"].iloc[0]
        big_interval = intervals[
            (intervals["filter_no"] == filter_no) & (intervals["maintenance_type"] == "大维护")
        ]["avg_interval_days"].iloc[0]
        ind = indicator_by_filter.loc[filter_no]
        constrained_annual_decline = float(coef_by_term[f"beta_t_{filter_no}"] * 365.0)

        if len(failure):
            failure_date = failure["date"].iloc[0]
            remaining_days = int((failure_date - last_date).days)
            failure_day_index = int(failure["day_index"].iloc[0])
            rolling_mean_at_failure = float(failure["rolling_365_mean"].iloc[0])
            safety_margin_at_failure = float(failure["safety_margin"].iloc[0])
            status = "failed_within_horizon"
        else:
            failure_date = pd.NaT
            remaining_days = np.nan
            failure_day_index = np.nan
            rolling_mean_at_failure = np.nan
            safety_margin_at_failure = np.nan
            status = "not_failed_within_horizon"

        rows.append(
            {
                "filter_no": filter_no,
                "annual_decline_rate": constrained_annual_decline,
                "q1_unconstrained_annual_decline_rate": float(ind["annual_decline_rate"]),
                "current_safety_margin_G": float(ind["safety_margin_G"]),
                "mid_interval_days": float(mid_interval),
                "big_interval_days": float(big_interval),
                "failure_date": failure_date.date().isoformat()
                if pd.notna(failure_date)
                else "",
                "failure_day_index": failure_day_index,
                "remaining_days": remaining_days,
                "remaining_years": remaining_days / 365.0
                if np.isfinite(remaining_days)
                else np.nan,
                "rolling_mean_at_failure": rolling_mean_at_failure,
                "safety_margin_at_failure": safety_margin_at_failure,
                "maintenance_effect_threshold_delta": delta,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.rcParams["axes.unicode_minus"] = False

    filters = sorted(history["filter_no"].unique(), key=filter_sort_key)
    nrows = int(np.ceil(len(filters) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(14, max(10, nrows * 2.4)), sharex=False)
    axes = np.ravel(axes)
    pred_by_filter = predictions.set_index("filter_no")

    for ax, filter_no in zip(axes, filters):
        h = history[history["filter_no"] == filter_no].sort_values("date")
        f = forecast[forecast["filter_no"] == filter_no].sort_values("date")
        below_threshold = f[f["predicted_permeability"] <= THRESHOLD]
        if len(below_threshold):
            cutoff_day = int(below_threshold["day_index"].iloc[0])
            f_plot = f[f["day_index"] <= cutoff_day].copy()
        else:
            f_plot = f.copy()
        ax.plot(h["date"], h["permeability_clean"], color="#4c78a8", lw=0.8, alpha=0.45)
        ax.plot(f_plot["date"], f_plot["predicted_permeability"], color="#f58518", lw=1.0)
        ax.axhline(THRESHOLD, color="#e45756", lw=0.9, ls="--")
        failure_date = pred_by_filter.loc[filter_no, "failure_date"]
        if isinstance(failure_date, str) and failure_date:
            ax.axvline(pd.to_datetime(failure_date), color="#e45756", lw=1.0)
        y_min = min(
            THRESHOLD - 3,
            float(h["permeability_clean"].min()),
            float(f_plot["predicted_permeability"].min()),
        )
        y_max = max(
            float(h["permeability_clean"].max()),
            float(f_plot["predicted_permeability"].max()),
        )
        ax.set_ylim(y_min, y_max + 5)
        ax.set_title(filter_no)
        ax.set_ylabel("Permeability")
        years = max(1, int(np.ceil((f_plot["date"].max() - h["date"].min()).days / 365.25)))
        year_interval = max(1, int(np.ceil(years / 4)))
        ax.xaxis.set_major_locator(mdates.YearLocator(base=year_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.2)

    for ax in axes[len(filters) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "q2_life_forecast_curves.png", dpi=200)
    plt.close(fig)

    ordered = predictions.sort_values("remaining_years", na_position="last")
    fig, ax = plt.subplots(figsize=(9, 5))
    values = ordered["remaining_years"].fillna(ordered["remaining_years"].max(skipna=True) + 1)
    ax.bar(ordered["filter_no"], values, color="#4c78a8")
    ax.set_ylabel("Remaining life / years")
    ax.set_title("Predicted remaining life")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "q2_remaining_life.png", dpi=200)
    plt.close(fig)


def write_summary(
    predictions: pd.DataFrame,
    model: pd.DataFrame,
    lambda_mid: float,
    lambda_big: float,
    mid_half_life: int,
    big_half_life: int,
    output_dir: Path,
) -> None:
    top_risk = predictions.sort_values("remaining_years", na_position="last").head(5)
    terms = model[
        model["term"].isin(
            [
                "season_sin",
                "season_cos",
                "decay_mid_maintenance",
                "decay_big_maintenance",
                "cum_mid_maintenance",
                "cum_big_maintenance",
            ]
        )
    ]
    lines = [
        "# Q2 Result Summary",
        "",
        "## Physical constraints",
        "- Filter trend coefficients are constrained by beta_i <= 0.",
        "- Cumulative maintenance coefficients are constrained by eta_M <= 0 and eta_B <= 0.",
        "- Exponential maintenance recovery coefficients are constrained by gamma_M >= 0 and gamma_B >= 0.",
        "",
        "## Selected maintenance decay",
        f"- Mid maintenance half-life: {mid_half_life} days, lambda = {lambda_mid:.6f}",
        f"- Big maintenance half-life: {big_half_life} days, lambda = {lambda_big:.6f}",
        "",
        "## Shortest predicted remaining life",
        top_risk[
            [
                "filter_no",
                "annual_decline_rate",
                "current_safety_margin_G",
                "failure_date",
                "remaining_years",
                "status",
            ]
        ].to_markdown(index=False),
        "",
        "## Key model terms",
        terms[["term", "coef", "std_error", "t_value", "r2"]].to_markdown(index=False),
        "",
    ]
    (output_dir / "q2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> None:
    table_dir, figure_dir = ensure_dirs(cfg.output_dir)
    panel, indicators, event_metrics = load_inputs(cfg)
    events = maintenance_events(panel)
    (model, fitted_panel, lambda_mid, lambda_big, mid_half_life, big_half_life), search = (
        fit_life_model(panel, events, cfg.half_life_grid)
    )
    intervals = maintenance_intervals(events)
    forecast_base = make_future_frame(panel, events, intervals, cfg.max_forecast_years)
    all_future_events = pd.concat(
        [
            future_events_for_filter(
                filter_no,
                events,
                intervals,
                int(panel.loc[panel["filter_no"] == filter_no, "day_index"].max()),
                int(panel.loc[panel["filter_no"] == filter_no, "day_index"].max())
                + cfg.max_forecast_years * 365,
            )
            for filter_no in sorted(panel["filter_no"].unique(), key=filter_sort_key)
        ],
        ignore_index=True,
    )
    forecast = predict_with_model(
        forecast_base,
        all_future_events,
        model,
        lambda_mid,
        lambda_big,
    )
    delta = maintenance_effect_threshold(event_metrics)
    predictions = life_predictions(
        panel,
        forecast,
        intervals,
        indicators,
        model,
        delta,
        cfg.event_window,
    )

    model.to_csv(table_dir / "q2_model_coefficients.csv", index=False, encoding="utf-8-sig")
    search.to_csv(table_dir / "q2_decay_grid_search.csv", index=False, encoding="utf-8-sig")
    fitted_panel.to_csv(table_dir / "q2_history_fitted.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(table_dir / "q2_maintenance_intervals.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(table_dir / "q2_future_forecast.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(table_dir / "q2_life_predictions.csv", index=False, encoding="utf-8-sig")

    save_plots(panel, forecast, predictions, figure_dir)
    write_summary(
        predictions,
        model,
        lambda_mid,
        lambda_big,
        mid_half_life,
        big_half_life,
        cfg.output_dir,
    )

    print(f"History rows: {len(panel)}")
    print(f"Forecast rows: {len(forecast)}")
    print(f"Selected mid half-life: {mid_half_life} days")
    print(f"Selected big half-life: {big_half_life} days")
    print(f"Maintenance effectiveness threshold delta: {delta:.4f}")
    print(f"Tables: {table_dir.resolve()}")
    print(f"Figures: {figure_dir.resolve()}")
    print(f"Summary: {(cfg.output_dir / 'q2_summary.md').resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Q2 filter life prediction model")
    parser.add_argument(
        "--q1-panel",
        type=Path,
        default=Path("output/q1/tables/q1_daily_panel.csv"),
        help="Q1 daily panel CSV.",
    )
    parser.add_argument(
        "--q1-indicators",
        type=Path,
        default=Path("output/q1/tables/q1_filter_indicators.csv"),
        help="Q1 filter indicators CSV.",
    )
    parser.add_argument(
        "--q1-event-metrics",
        type=Path,
        default=Path("output/q1/tables/q1_maintenance_event_metrics.csv"),
        help="Q1 maintenance event metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/q2"),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-forecast-years",
        type=int,
        default=40,
        help="Maximum forecast horizon in years.",
    )
    parser.add_argument(
        "--event-window",
        type=int,
        default=7,
        help="Days before and after maintenance for predicted effect calculation.",
    )
    args = parser.parse_args()
    return Config(
        q1_panel_csv=args.q1_panel,
        q1_indicators_csv=args.q1_indicators,
        q1_event_metrics_csv=args.q1_event_metrics,
        output_dir=args.output,
        max_forecast_years=args.max_forecast_years,
        event_window=args.event_window,
    )


if __name__ == "__main__":
    run(parse_args())
