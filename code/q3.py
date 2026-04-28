from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import q2


PURCHASE_COST = 300.0
MID_MAINTENANCE_COST = 3.0
BIG_MAINTENANCE_COST = 12.0


@dataclass(frozen=True)
class Config:
    q1_panel_csv: Path
    q1_indicators_csv: Path
    q1_event_metrics_csv: Path
    output_dir: Path
    max_forecast_years: int = 40
    event_window: int = 7
    mid_interval_candidates: tuple[int, ...] = (30, 45, 60, 75, 90, 105, 120, 150)
    big_interval_candidates: tuple[int, ...] = (90, 120, 150, 180, 240, 300, 365)
    purchase_cost: float = PURCHASE_COST
    mid_cost: float = MID_MAINTENANCE_COST
    big_cost: float = BIG_MAINTENANCE_COST


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def fit_q2_model(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float]:
    panel, indicators, event_metrics = q2.load_inputs(
        q2.Config(
            q1_panel_csv=cfg.q1_panel_csv,
            q1_indicators_csv=cfg.q1_indicators_csv,
            q1_event_metrics_csv=cfg.q1_event_metrics_csv,
            output_dir=Path("output/q2"),
            max_forecast_years=cfg.max_forecast_years,
            event_window=cfg.event_window,
        )
    )
    events = q2.maintenance_events(panel)
    (model, _, lambda_mid, lambda_big, _, _), _ = q2.fit_life_model(
        panel, events, (7, 14, 21, 30, 45, 60, 90, 120, 180)
    )
    delta = q2.maintenance_effect_threshold(event_metrics)
    return panel, indicators, events, model, lambda_mid, lambda_big, delta


def strategy_events(
    filter_no: str,
    historical_events: pd.DataFrame,
    last_day: int,
    horizon_day: int,
    mid_interval: int,
    big_interval: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hist = historical_events[historical_events["filter_no"] == filter_no][
        ["filter_no", "day_index", "maintenance_type"]
    ].copy()
    hist["is_future"] = 0

    for maintenance_type, interval in [("中维护", mid_interval), ("大维护", big_interval)]:
        type_events = hist[hist["maintenance_type"] == maintenance_type].sort_values("day_index")
        if len(type_events):
            next_day = int(type_events["day_index"].iloc[-1]) + interval
        else:
            next_day = last_day + interval
        while next_day <= horizon_day:
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
    return pd.concat([hist, future], ignore_index=True).sort_values("day_index")


def strategy_frame(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    filter_no: str,
    mid_interval: int,
    big_interval: int,
    horizon_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = panel[panel["filter_no"] == filter_no].sort_values("date")
    start_date = h["date"].min()
    last_day = int(h["day_index"].max())
    horizon_day = last_day + horizon_years * 365
    all_events = strategy_events(
        filter_no, events, last_day, horizon_day, mid_interval, big_interval
    )

    future = pd.DataFrame(
        {
            "filter_no": filter_no,
            "day_index": np.arange(last_day + 1, horizon_day + 1),
        }
    )
    future["date"] = start_date + pd.to_timedelta(future["day_index"], unit="D")
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
        int(h["cum_mid_maintenance"].iloc[-1]) + future["is_mid_maintenance"].cumsum()
    )
    future["cum_big_maintenance"] = (
        int(h["cum_big_maintenance"].iloc[-1]) + future["is_big_maintenance"].cumsum()
    )
    future["season_sin"] = np.sin(2 * np.pi * future["day_index"] / q2.SEASON_DAYS)
    future["season_cos"] = np.cos(2 * np.pi * future["day_index"] / q2.SEASON_DAYS)
    return future, all_events


def evaluate_strategy(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    model: pd.DataFrame,
    lambda_mid: float,
    lambda_big: float,
    delta: float,
    cfg: Config,
    filter_no: str,
    mid_interval: int,
    big_interval: int,
) -> dict[str, object]:
    future_base, all_events = strategy_frame(
        panel,
        events,
        filter_no,
        mid_interval,
        big_interval,
        cfg.max_forecast_years,
    )
    forecast = q2.predict_with_model(future_base, all_events, model, lambda_mid, lambda_big)
    h = panel[panel["filter_no"] == filter_no].sort_values("date")
    h_tail = h[["filter_no", "date", "day_index", "permeability_clean"]].rename(
        columns={"permeability_clean": "predicted_permeability"}
    )
    combined = pd.concat(
        [
            h_tail[["filter_no", "date", "day_index", "predicted_permeability"]],
            forecast[["filter_no", "date", "day_index", "predicted_permeability"]],
        ],
        ignore_index=True,
    ).sort_values("day_index")
    combined["rolling_365_mean"] = (
        combined["predicted_permeability"].rolling(365, min_periods=365).mean()
    )
    combined["safety_margin"] = combined["rolling_365_mean"] - q2.THRESHOLD
    forecast = forecast.merge(
        combined[["day_index", "rolling_365_mean", "safety_margin"]],
        on="day_index",
        how="left",
    )

    future_events = forecast[forecast["maintenance_type"].notna()][
        ["day_index", "date", "maintenance_type"]
    ].copy()
    future_events["predicted_effect_delta"] = future_events["day_index"].apply(
        lambda day: q2.predicted_event_effect(forecast, int(day), cfg.event_window)
    )
    ineffective = future_events[
        future_events["predicted_effect_delta"].fillna(np.inf) < delta
    ]
    threshold_days = forecast[forecast["rolling_365_mean"] < q2.THRESHOLD]

    failure = pd.DataFrame()
    if len(threshold_days) and len(ineffective):
        first_threshold_day = int(threshold_days["day_index"].iloc[0])
        candidate_events = ineffective[ineffective["day_index"] >= first_threshold_day]
        if len(candidate_events):
            failure = forecast[forecast["day_index"] >= int(candidate_events["day_index"].iloc[0])].head(1)

    last_date = h["date"].max()
    if len(failure):
        end_date = failure["date"].iloc[0]
        life_days = max(1, int((end_date - last_date).days))
        status = "failed_within_horizon"
    else:
        end_date = forecast["date"].max()
        life_days = max(1, int((end_date - last_date).days))
        status = "censored_at_horizon"

    used = future_events[future_events["date"] <= end_date]
    mid_count = int((used["maintenance_type"] == "中维护").sum())
    big_count = int((used["maintenance_type"] == "大维护").sum())
    life_years = life_days / 365.0
    total_cost = cfg.purchase_cost + mid_count * cfg.mid_cost + big_count * cfg.big_cost
    annual_cost = total_cost / life_years

    return {
        "filter_no": filter_no,
        "mid_interval_days": mid_interval,
        "big_interval_days": big_interval,
        "failure_date": end_date.date().isoformat(),
        "life_days": life_days,
        "life_years": life_years,
        "future_mid_count": mid_count,
        "future_big_count": big_count,
        "purchase_cost": cfg.purchase_cost,
        "maintenance_cost": mid_count * cfg.mid_cost + big_count * cfg.big_cost,
        "total_cycle_cost": total_cost,
        "annual_cost": annual_cost,
        "status": status,
    }


def search_optimal_strategies(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    model: pd.DataFrame,
    lambda_mid: float,
    lambda_big: float,
    delta: float,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    filters = sorted(panel["filter_no"].unique(), key=q2.filter_sort_key)
    for filter_no in filters:
        for mid_interval in cfg.mid_interval_candidates:
            for big_interval in cfg.big_interval_candidates:
                rows.append(
                    evaluate_strategy(
                        panel,
                        events,
                        model,
                        lambda_mid,
                        lambda_big,
                        delta,
                        cfg,
                        filter_no,
                        mid_interval,
                        big_interval,
                    )
                )
    strategy_table = pd.DataFrame(rows)
    best = (
        strategy_table.sort_values(["filter_no", "annual_cost", "life_years"], ascending=[True, True, False])
        .groupby("filter_no", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return strategy_table, best


def save_plots(strategy_table: pd.DataFrame, best: pd.DataFrame, figure_dir: Path) -> None:
    filters = sorted(best["filter_no"].unique(), key=q2.filter_sort_key)
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = best.set_index("filter_no").loc[filters].reset_index()
    ax.bar(ordered["filter_no"], ordered["annual_cost"], color="#4c78a8")
    ax.set_ylabel("Annual cost / 10k CNY")
    ax.set_title("Q3 optimal annual cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "q3_optimal_annual_cost.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ordered))
    width = 0.38
    ax.bar(x - width / 2, ordered["mid_interval_days"], width, label="mid")
    ax.bar(x + width / 2, ordered["big_interval_days"], width, label="big")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["filter_no"])
    ax.set_ylabel("Interval / days")
    ax.set_title("Q3 optimal maintenance intervals")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "q3_optimal_intervals.png", dpi=200)
    plt.close(fig)


def write_summary(best: pd.DataFrame, output_dir: Path) -> None:
    total_annual_cost = float(best["annual_cost"].sum())
    lines = [
        "# Q3 Result Summary",
        "",
        f"Total annual cost of 10 filters: {total_annual_cost:.4f} (10k CNY/year)",
        "",
        "## Optimal maintenance plan",
        best[
            [
                "filter_no",
                "mid_interval_days",
                "big_interval_days",
                "life_years",
                "future_mid_count",
                "future_big_count",
                "annual_cost",
                "status",
            ]
        ].to_markdown(index=False),
        "",
    ]
    (output_dir / "q3_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> None:
    table_dir, figure_dir = ensure_dirs(cfg.output_dir)
    panel, _, events, model, lambda_mid, lambda_big, delta = fit_q2_model(cfg)
    strategy_table, best = search_optimal_strategies(
        panel, events, model, lambda_mid, lambda_big, delta, cfg
    )
    strategy_table.to_csv(table_dir / "q3_strategy_search.csv", index=False, encoding="utf-8-sig")
    best.to_csv(table_dir / "q3_optimal_plan.csv", index=False, encoding="utf-8-sig")
    save_plots(strategy_table, best, figure_dir)
    write_summary(best, cfg.output_dir)
    print(f"Strategy rows: {len(strategy_table)}")
    print(f"Tables: {table_dir.resolve()}")
    print(f"Figures: {figure_dir.resolve()}")
    print(f"Summary: {(cfg.output_dir / 'q3_summary.md').resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Q3 optimal maintenance plan")
    parser.add_argument("--q1-panel", type=Path, default=Path("output/q1/tables/q1_daily_panel.csv"))
    parser.add_argument("--q1-indicators", type=Path, default=Path("output/q1/tables/q1_filter_indicators.csv"))
    parser.add_argument("--q1-event-metrics", type=Path, default=Path("output/q1/tables/q1_maintenance_event_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/q3"))
    parser.add_argument("--max-forecast-years", type=int, default=40)
    args = parser.parse_args()
    return Config(
        q1_panel_csv=args.q1_panel,
        q1_indicators_csv=args.q1_indicators,
        q1_event_metrics_csv=args.q1_event_metrics,
        output_dir=args.output,
        max_forecast_years=args.max_forecast_years,
    )


if __name__ == "__main__":
    run(parse_args())
