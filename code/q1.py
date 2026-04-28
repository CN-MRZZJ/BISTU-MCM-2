from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLD = 37.0
SEASON_DAYS = 365.0


@dataclass(frozen=True)
class Config:
    input_csv: Path
    output_dir: Path
    smooth_windows: tuple[int, ...] = (7, 15, 30)
    event_window: int = 7
    recent_maintenance_days: int = 7
    outlier_window: int = 15
    outlier_sigma: float = 4.0
    return_tolerance: float = 0.5


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "filiter_no": "filter_no",
        "date&time": "timestamp",
        "permeability": "permeability",
        "maintenance_type": "maintenance_type",
    }
    missing = [col for col in rename_map if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.rename(columns=rename_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["permeability"] = pd.to_numeric(df["permeability"], errors="coerce")
    df["maintenance_type"] = df["maintenance_type"].replace("", np.nan)
    df = df.dropna(subset=["filter_no", "timestamp"]).copy()
    df["filter_no"] = df["filter_no"].astype(str)
    df["date"] = df["timestamp"].dt.floor("D")
    return df


def build_daily_panel(raw: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = raw.dropna(subset=["permeability"])
    daily_perm = (
        obs.groupby(["filter_no", "date"], as_index=False)["permeability"]
        .mean()
        .rename(columns={"permeability": "permeability_raw"})
    )

    maintenance_events = (
        raw.dropna(subset=["maintenance_type"])
        .sort_values(["filter_no", "date", "timestamp"])
        .drop_duplicates(["filter_no", "date", "maintenance_type"])
        [["filter_no", "date", "maintenance_type"]]
        .reset_index(drop=True)
    )

    parts: list[pd.DataFrame] = []
    for filter_no, group in daily_perm.groupby("filter_no", sort=True):
        event_dates = maintenance_events.loc[
            maintenance_events["filter_no"] == filter_no, "date"
        ]
        date_min = min(group["date"].min(), event_dates.min()) if len(event_dates) else group["date"].min()
        date_max = max(group["date"].max(), event_dates.max()) if len(event_dates) else group["date"].max()
        frame = pd.DataFrame(
            {
                "filter_no": filter_no,
                "date": pd.date_range(date_min, date_max, freq="D"),
            }
        )
        parts.append(frame)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.merge(daily_perm, on=["filter_no", "date"], how="left")
    panel = panel.merge(maintenance_events, on=["filter_no", "date"], how="left")
    panel["is_mid_maintenance"] = (panel["maintenance_type"] == "中维护").astype(int)
    panel["is_big_maintenance"] = (panel["maintenance_type"] == "大维护").astype(int)

    corrected_parts: list[pd.DataFrame] = []
    for _, group in panel.groupby("filter_no", sort=True):
        g = group.sort_values("date").copy()
        g["missing_before_interpolation"] = g["permeability_raw"].isna().astype(int)
        g["permeability_interp"] = (
            g["permeability_raw"].interpolate(method="linear").ffill().bfill()
        )

        near_maintenance = np.zeros(len(g), dtype=bool)
        event_positions = np.flatnonzero(
            (g["is_mid_maintenance"].to_numpy() + g["is_big_maintenance"].to_numpy()) > 0
        )
        for pos in event_positions:
            lo = max(0, pos - 3)
            hi = min(len(g), pos + 4)
            near_maintenance[lo:hi] = True

        rolling_median = g["permeability_interp"].rolling(
            cfg.outlier_window, center=True, min_periods=max(3, cfg.outlier_window // 2)
        ).median()
        abs_dev = (g["permeability_interp"] - rolling_median).abs()
        rolling_mad = abs_dev.rolling(
            cfg.outlier_window, center=True, min_periods=max(3, cfg.outlier_window // 2)
        ).median()
        robust_sigma = 1.4826 * rolling_mad
        floor = max(1.0, g["permeability_interp"].std(skipna=True) * 0.05)
        outlier = (abs_dev > np.maximum(cfg.outlier_sigma * robust_sigma, floor)) & ~near_maintenance
        rolling_mean = g["permeability_interp"].rolling(7, center=True, min_periods=1).mean()
        g["is_outlier_corrected"] = outlier.fillna(False).astype(int)
        g["permeability_clean"] = g["permeability_interp"].where(~outlier, rolling_mean)

        for window in cfg.smooth_windows:
            g[f"permeability_ma{window}"] = (
                g["permeability_clean"].rolling(window, min_periods=1).mean()
            )
            g[f"local_decline_{window}d"] = (
                g["permeability_clean"] - g["permeability_clean"].shift(window)
            ) / window

        g["cum_mid_maintenance"] = g["is_mid_maintenance"].cumsum()
        g["cum_big_maintenance"] = g["is_big_maintenance"].cumsum()
        g["day_index"] = (g["date"] - g["date"].min()).dt.days.astype(float)
        corrected_parts.append(g)

    panel = pd.concat(corrected_parts, ignore_index=True)
    panel = add_recent_maintenance_flags(panel, cfg.recent_maintenance_days)
    panel["season_sin"] = np.sin(2 * np.pi * panel["day_index"] / SEASON_DAYS)
    panel["season_cos"] = np.cos(2 * np.pi * panel["day_index"] / SEASON_DAYS)
    return panel, maintenance_events


def add_recent_maintenance_flags(panel: pd.DataFrame, recent_days: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in panel.groupby("filter_no", sort=True):
        g = group.sort_values("date").copy()
        mid = np.zeros(len(g), dtype=int)
        big = np.zeros(len(g), dtype=int)
        for idx, row in enumerate(g.itertuples(index=False)):
            if row.is_mid_maintenance:
                mid[idx : min(len(g), idx + recent_days + 1)] = 1
            if row.is_big_maintenance:
                big[idx : min(len(g), idx + recent_days + 1)] = 1
        g["recent_mid_maintenance"] = mid
        g["recent_big_maintenance"] = big
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> pd.DataFrame:
    mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[mask]
    x = x[mask]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    resid = y - y_hat
    n, p = x.shape
    dof = max(n - p, 1)
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    t_value = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    return pd.DataFrame(
        {
            "term": names,
            "coef": beta,
            "std_error": stderr,
            "t_value": t_value,
            "n_obs": n,
            "r2": 1.0 - float((resid @ resid) / np.sum((y - y.mean()) ** 2)),
        }
    )


def simple_trend(series: pd.DataFrame) -> tuple[float, float, float]:
    y = series["permeability_clean"].to_numpy(float)
    t = series["day_index"].to_numpy(float)
    x = np.column_stack([np.ones(len(t)), t])
    result = ols(y, x, ["intercept", "daily_slope"])
    slope = float(result.loc[result["term"] == "daily_slope", "coef"].iloc[0])
    intercept = float(result.loc[result["term"] == "intercept", "coef"].iloc[0])
    r2 = float(result["r2"].iloc[0])
    return intercept, slope, r2


def seasonal_fit(series: pd.DataFrame) -> tuple[float, float, float]:
    y = series["permeability_clean"].to_numpy(float)
    x = series[["day_index", "season_sin", "season_cos"]].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    result = ols(y, x, ["intercept", "daily_slope", "season_sin", "season_cos"])
    a1 = float(result.loc[result["term"] == "season_sin", "coef"].iloc[0])
    b1 = float(result.loc[result["term"] == "season_cos", "coef"].iloc[0])
    amplitude = float(np.hypot(a1, b1))
    return a1, b1, amplitude


def maintenance_event_metrics(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for filter_no, group in panel.groupby("filter_no", sort=True):
        g = group.sort_values("date").reset_index(drop=True)
        for idx, row in g.loc[g["maintenance_type"].notna()].iterrows():
            before = g.loc[
                max(0, idx - cfg.event_window) : idx - 1, "permeability_clean"
            ]
            after = g.loc[
                idx + 1 : min(len(g) - 1, idx + cfg.event_window),
                "permeability_clean",
            ]
            before_mean = float(before.mean()) if len(before) else np.nan
            after_mean = float(after.mean()) if len(after) else np.nan
            effect = after_mean - before_mean if np.isfinite(before_mean) and np.isfinite(after_mean) else np.nan

            duration = np.nan
            if np.isfinite(before_mean):
                future = g.loc[idx + 1 :, ["date", "permeability_clean"]].copy()
                returned = future[
                    future["permeability_clean"] <= before_mean + cfg.return_tolerance
                ]
                if len(returned):
                    duration = int((returned["date"].iloc[0] - row["date"]).days)

            rows.append(
                {
                    "filter_no": filter_no,
                    "date": row["date"],
                    "maintenance_type": row["maintenance_type"],
                    "before_mean": before_mean,
                    "after_mean": after_mean,
                    "effect_delta": effect,
                    "duration_days": duration,
                }
            )
    return pd.DataFrame(rows)


def filter_indicators(panel: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for filter_no, group in panel.groupby("filter_no", sort=True):
        g = group.sort_values("date")
        intercept, daily_slope, trend_r2 = simple_trend(g)
        a1, b1, amplitude = seasonal_fit(g)
        last_365 = g.tail(min(365, len(g)))
        avg_last_365 = float(last_365["permeability_clean"].mean())
        safety_margin = avg_last_365 - THRESHOLD
        events = event_metrics[event_metrics["filter_no"] == filter_no]
        mid_events = events[events["maintenance_type"] == "中维护"]
        big_events = events[events["maintenance_type"] == "大维护"]

        row = {
            "filter_no": filter_no,
            "start_date": g["date"].min().date().isoformat(),
            "end_date": g["date"].max().date().isoformat(),
            "n_days": len(g),
            "intercept": intercept,
            "daily_slope": daily_slope,
            "annual_decline_rate": 365.0 * daily_slope,
            "trend_r2": trend_r2,
            "season_sin_coef": a1,
            "season_cos_coef": b1,
            "season_amplitude": amplitude,
            "avg_last_365d": avg_last_365,
            "safety_margin_G": safety_margin,
            "risk_level": risk_level(safety_margin),
            "mid_maintenance_count": int(g["is_mid_maintenance"].sum()),
            "big_maintenance_count": int(g["is_big_maintenance"].sum()),
            "mid_effect_mean": float(mid_events["effect_delta"].mean())
            if len(mid_events)
            else np.nan,
            "big_effect_mean": float(big_events["effect_delta"].mean())
            if len(big_events)
            else np.nan,
            "mid_duration_mean": float(mid_events["duration_days"].mean())
            if len(mid_events)
            else np.nan,
            "big_duration_mean": float(big_events["duration_days"].mean())
            if len(big_events)
            else np.nan,
            "missing_days": int(g["missing_before_interpolation"].sum()),
            "outlier_corrected_days": int(g["is_outlier_corrected"].sum()),
        }
        for window in (7, 15, 30):
            col = f"local_decline_{window}d"
            row[f"latest_{col}"] = float(g[col].dropna().iloc[-1]) if g[col].notna().any() else np.nan
            row[f"min_{col}"] = float(g[col].min(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def risk_level(safety_margin: float) -> str:
    if safety_margin < 0:
        return "high"
    if safety_margin < 3:
        return "medium"
    return "low"


def global_regression(panel: pd.DataFrame) -> pd.DataFrame:
    filters = sorted(panel["filter_no"].unique())
    x_parts: list[np.ndarray] = []
    names: list[str] = []

    for filter_no in filters:
        mask = (panel["filter_no"] == filter_no).to_numpy(float)
        x_parts.append(mask.reshape(-1, 1))
        names.append(f"alpha_{filter_no}")

    for filter_no in filters:
        mask = (panel["filter_no"] == filter_no).to_numpy(float)
        x_parts.append((mask * panel["day_index"].to_numpy(float)).reshape(-1, 1))
        names.append(f"beta_t_{filter_no}")

    common_cols = [
        "season_sin",
        "season_cos",
        "recent_mid_maintenance",
        "recent_big_maintenance",
        "cum_mid_maintenance",
        "cum_big_maintenance",
    ]
    for col in common_cols:
        x_parts.append(panel[col].to_numpy(float).reshape(-1, 1))
        names.append(col)

    x = np.hstack(x_parts)
    y = panel["permeability_clean"].to_numpy(float)
    result = ols(y, x, names)
    return result


def save_plots(panel: pd.DataFrame, event_metrics: pd.DataFrame, figure_dir: Path) -> None:
    plt.rcParams["axes.unicode_minus"] = False

    filters = sorted(panel["filter_no"].unique())
    nrows = int(np.ceil(len(filters) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(14, max(10, nrows * 2.3)), sharex=False)
    axes = np.ravel(axes)
    for ax, filter_no in zip(axes, filters):
        g = panel[panel["filter_no"] == filter_no].sort_values("date")
        ax.plot(g["date"], g["permeability_clean"], color="#4c78a8", lw=0.8, alpha=0.45)
        ax.plot(g["date"], g["permeability_ma30"], color="#f58518", lw=1.5)
        events = g[g["maintenance_type"].notna()]
        ax.scatter(
            events["date"],
            events["permeability_clean"],
            color="#54a24b",
            s=16,
            zorder=3,
        )
        ax.axhline(THRESHOLD, color="#e45756", lw=0.9, ls="--")
        ax.set_title(filter_no)
        ax.set_ylabel("Permeability")
        ax.grid(alpha=0.2)
    for ax in axes[len(filters) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "q1_permeability_trends.png", dpi=200)
    plt.close(fig)

    if len(event_metrics):
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [
            event_metrics.loc[event_metrics["maintenance_type"] == "中维护", "effect_delta"].dropna(),
            event_metrics.loc[event_metrics["maintenance_type"] == "大维护", "effect_delta"].dropna(),
        ]
        ax.boxplot(data, tick_labels=["mid", "big"], showmeans=True)
        ax.axhline(0, color="#e45756", lw=0.9, ls="--")
        ax.set_ylabel("After mean - before mean")
        ax.set_title("Maintenance effect by type")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "q1_maintenance_effects.png", dpi=200)
        plt.close(fig)


def write_markdown_summary(
    indicators: pd.DataFrame,
    event_metrics: pd.DataFrame,
    regression: pd.DataFrame,
    output_dir: Path,
) -> None:
    top_decline = indicators.sort_values("annual_decline_rate").head(3)
    top_risk = indicators.sort_values("safety_margin_G").head(3)
    mid_mean = event_metrics.loc[
        event_metrics["maintenance_type"] == "中维护", "effect_delta"
    ].mean()
    big_mean = event_metrics.loc[
        event_metrics["maintenance_type"] == "大维护", "effect_delta"
    ].mean()
    common_terms = regression[
        regression["term"].isin(
            [
                "season_sin",
                "season_cos",
                "recent_mid_maintenance",
                "recent_big_maintenance",
                "cum_mid_maintenance",
                "cum_big_maintenance",
            ]
        )
    ]

    lines = [
        "# Q1 Result Summary",
        "",
        "## Fastest annual decline",
        top_decline[
            ["filter_no", "annual_decline_rate", "avg_last_365d", "safety_margin_G"]
        ].to_markdown(index=False),
        "",
        "## Highest life risk",
        top_risk[["filter_no", "avg_last_365d", "safety_margin_G", "risk_level"]].to_markdown(
            index=False
        ),
        "",
        "## Mean maintenance effect",
        f"- Mid maintenance mean delta: {mid_mean:.4f}",
        f"- Big maintenance mean delta: {big_mean:.4f}",
        "",
        "## Common regression terms",
        common_terms[["term", "coef", "std_error", "t_value", "r2"]].to_markdown(
            index=False
        ),
        "",
    ]
    (output_dir / "q1_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> None:
    table_dir, figure_dir = ensure_dirs(cfg.output_dir)
    raw = load_raw_data(cfg.input_csv)
    panel, maintenance_events = build_daily_panel(raw, cfg)
    event_metrics = maintenance_event_metrics(panel, cfg)
    indicators = filter_indicators(panel, event_metrics)
    regression = global_regression(panel)

    panel.to_csv(table_dir / "q1_daily_panel.csv", index=False, encoding="utf-8-sig")
    maintenance_events.to_csv(
        table_dir / "q1_maintenance_records.csv", index=False, encoding="utf-8-sig"
    )
    event_metrics.to_csv(
        table_dir / "q1_maintenance_event_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    indicators.to_csv(
        table_dir / "q1_filter_indicators.csv", index=False, encoding="utf-8-sig"
    )
    regression.to_csv(
        table_dir / "q1_global_regression.csv", index=False, encoding="utf-8-sig"
    )

    save_plots(panel, event_metrics, figure_dir)
    write_markdown_summary(indicators, event_metrics, regression, cfg.output_dir)

    print(f"Input rows: {len(raw)}")
    print(f"Daily panel rows: {len(panel)}")
    print(f"Maintenance events: {len(event_metrics)}")
    print(f"Tables: {table_dir.resolve()}")
    print(f"Figures: {figure_dir.resolve()}")
    print(f"Summary: {(cfg.output_dir / 'q1_summary.md').resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Q1 permeability indicator model")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw_data.csv"),
        help="Raw merged CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/q1"),
        help="Output directory.",
    )
    parser.add_argument(
        "--event-window",
        type=int,
        default=7,
        help="Days before and after maintenance for effect calculation.",
    )
    args = parser.parse_args()
    return Config(input_csv=args.input, output_dir=args.output, event_window=args.event_window)


if __name__ == "__main__":
    run(parse_args())
