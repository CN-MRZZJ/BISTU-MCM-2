from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import q2


BASE_PURCHASE_COST = 300.0
BASE_MID_COST = 3.0
BASE_BIG_COST = 12.0


@dataclass(frozen=True)
class Config:
    q3_strategy_search_csv: Path
    q3_optimal_plan_csv: Path
    output_dir: Path
    purchase_cost: float = BASE_PURCHASE_COST
    mid_cost: float = BASE_MID_COST
    big_cost: float = BASE_BIG_COST
    factors: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = pd.read_csv(cfg.q3_strategy_search_csv)
    base_plan = pd.read_csv(cfg.q3_optimal_plan_csv)
    required = {
        "filter_no",
        "mid_interval_days",
        "big_interval_days",
        "life_years",
        "future_mid_count",
        "future_big_count",
    }
    missing = sorted(required - set(strategy.columns))
    if missing:
        raise ValueError(f"Q3 strategy table is missing required columns: {missing}")
    return strategy, base_plan


def annual_cost(
    table: pd.DataFrame,
    purchase_cost: float,
    mid_cost: float,
    big_cost: float,
) -> pd.Series:
    return (
        purchase_cost
        + table["future_mid_count"] * mid_cost
        + table["future_big_count"] * big_cost
    ) / table["life_years"]


def best_plan_under_costs(
    strategy: pd.DataFrame,
    purchase_cost: float,
    mid_cost: float,
    big_cost: float,
) -> pd.DataFrame:
    scored = strategy.copy()
    scored["scenario_annual_cost"] = annual_cost(scored, purchase_cost, mid_cost, big_cost)
    return (
        scored.sort_values(
            ["filter_no", "scenario_annual_cost", "life_years"],
            ascending=[True, True, False],
        )
        .groupby("filter_no", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def base_plan_cost_under_scenario(
    strategy: pd.DataFrame,
    base_plan: pd.DataFrame,
    purchase_cost: float,
    mid_cost: float,
    big_cost: float,
) -> pd.DataFrame:
    keys = ["filter_no", "mid_interval_days", "big_interval_days"]
    base_rows = strategy.merge(base_plan[keys], on=keys, how="inner")
    base_rows = base_rows.copy()
    base_rows["scenario_annual_cost"] = annual_cost(
        base_rows, purchase_cost, mid_cost, big_cost
    )
    return base_rows


def compare_scenario(
    strategy: pd.DataFrame,
    base_plan: pd.DataFrame,
    purchase_cost: float,
    mid_cost: float,
    big_cost: float,
    label: str,
) -> dict[str, object]:
    best = best_plan_under_costs(strategy, purchase_cost, mid_cost, big_cost)
    base = base_plan_cost_under_scenario(
        strategy, base_plan, purchase_cost, mid_cost, big_cost
    )
    keys = ["filter_no", "mid_interval_days", "big_interval_days"]
    comparison = base[keys].merge(
        best[keys],
        on=keys,
        how="inner",
    )
    same_filters = int(comparison["filter_no"].nunique())
    total_best = float(best["scenario_annual_cost"].sum())
    total_base = float(base["scenario_annual_cost"].sum())
    penalty = total_base - total_best
    return {
        "scenario": label,
        "purchase_cost": purchase_cost,
        "mid_cost": mid_cost,
        "big_cost": big_cost,
        "same_filter_count": same_filters,
        "all_filters_same": same_filters == base_plan["filter_no"].nunique(),
        "base_plan_total_annual_cost": total_base,
        "reoptimized_total_annual_cost": total_best,
        "annual_cost_penalty": penalty,
        "annual_cost_penalty_pct": penalty / total_best * 100.0 if total_best > 0 else np.nan,
    }


def one_way_sensitivity(strategy: pd.DataFrame, base_plan: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in cfg.factors:
        rows.append(
            compare_scenario(
                strategy,
                base_plan,
                cfg.purchase_cost * factor,
                cfg.mid_cost,
                cfg.big_cost,
                f"purchase_x{factor:.1f}",
            )
        )
        rows.append(
            compare_scenario(
                strategy,
                base_plan,
                cfg.purchase_cost,
                cfg.mid_cost * factor,
                cfg.big_cost,
                f"mid_x{factor:.1f}",
            )
        )
        rows.append(
            compare_scenario(
                strategy,
                base_plan,
                cfg.purchase_cost,
                cfg.mid_cost,
                cfg.big_cost * factor,
                f"big_x{factor:.1f}",
            )
        )
    return pd.DataFrame(rows)


def combined_sensitivity(strategy: pd.DataFrame, base_plan: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for purchase_factor in cfg.factors:
        for mid_factor in cfg.factors:
            for big_factor in cfg.factors:
                rows.append(
                    compare_scenario(
                        strategy,
                        base_plan,
                        cfg.purchase_cost * purchase_factor,
                        cfg.mid_cost * mid_factor,
                        cfg.big_cost * big_factor,
                        f"purchase_{purchase_factor:.1f}_mid_{mid_factor:.1f}_big_{big_factor:.1f}",
                    )
                    | {
                        "purchase_factor": purchase_factor,
                        "mid_factor": mid_factor,
                        "big_factor": big_factor,
                    }
                )
    return pd.DataFrame(rows)


def plan_switch_details(
    strategy: pd.DataFrame,
    base_plan: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    scenarios = [
        ("purchase_low", 0.7, 1.0, 1.0),
        ("purchase_high", 1.3, 1.0, 1.0),
        ("mid_low", 1.0, 0.7, 1.0),
        ("mid_high", 1.0, 1.3, 1.0),
        ("big_low", 1.0, 1.0, 0.7),
        ("big_high", 1.0, 1.0, 1.3),
        ("all_low", 0.7, 0.7, 0.7),
        ("all_high", 1.3, 1.3, 1.3),
    ]
    rows: list[pd.DataFrame] = []
    keys = ["filter_no", "mid_interval_days", "big_interval_days"]
    base_keys = base_plan[keys].rename(
        columns={
            "mid_interval_days": "base_mid_interval_days",
            "big_interval_days": "base_big_interval_days",
        }
    )
    for label, purchase_factor, mid_factor, big_factor in scenarios:
        best = best_plan_under_costs(
            strategy,
            cfg.purchase_cost * purchase_factor,
            cfg.mid_cost * mid_factor,
            cfg.big_cost * big_factor,
        )
        detail = best[
            ["filter_no", "mid_interval_days", "big_interval_days", "scenario_annual_cost"]
        ].merge(base_keys, on="filter_no", how="left")
        detail["scenario"] = label
        detail["changed"] = (
            (detail["mid_interval_days"] != detail["base_mid_interval_days"])
            | (detail["big_interval_days"] != detail["base_big_interval_days"])
        )
        rows.append(detail)
    return pd.concat(rows, ignore_index=True)


def save_plots(one_way: pd.DataFrame, combined: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for prefix, label in [
        ("purchase", "purchase"),
        ("mid", "mid maintenance"),
        ("big", "big maintenance"),
    ]:
        data = one_way[one_way["scenario"].str.startswith(prefix)].copy()
        data["factor"] = data["scenario"].str.extract(r"x([0-9.]+)").astype(float)
        ax.plot(data["factor"], data["annual_cost_penalty_pct"], marker="o", label=label)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Cost multiplier")
    ax.set_ylabel("Penalty of Q3 plan vs reoptimized / %")
    ax.set_title("Q4 one-way cost sensitivity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "q4_one_way_sensitivity.png", dpi=200)
    plt.close(fig)

    heat = (
        combined[combined["big_factor"] == 1.0]
        .pivot(index="purchase_factor", columns="mid_factor", values="same_filter_count")
        .sort_index(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(heat.to_numpy(), cmap="YlGnBu", vmin=0, vmax=10)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([f"{x:.1f}" for x in heat.index])
    ax.set_xlabel("Mid maintenance cost multiplier")
    ax.set_ylabel("Purchase cost multiplier")
    ax.set_title("Unchanged filters when big cost is fixed")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, int(heat.iloc[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="same filter count")
    fig.tight_layout()
    fig.savefig(figure_dir / "q4_robustness_heatmap.png", dpi=200)
    plt.close(fig)


def write_summary(one_way: pd.DataFrame, combined: pd.DataFrame, switches: pd.DataFrame, output_dir: Path) -> None:
    robust_rate = float(combined["all_filters_same"].mean())
    max_penalty = float(combined["annual_cost_penalty_pct"].max())
    worst = combined.sort_values("annual_cost_penalty_pct", ascending=False).head(5)
    one_way_view = one_way[
        [
            "scenario",
            "same_filter_count",
            "base_plan_total_annual_cost",
            "reoptimized_total_annual_cost",
            "annual_cost_penalty_pct",
        ]
    ]
    lines = [
        "# Q4 Result Summary",
        "",
        f"Share of tested combined scenarios where the Q3 plan remains exactly optimal: {robust_rate:.2%}",
        f"Maximum annual-cost penalty of keeping the Q3 plan: {max_penalty:.4f}%",
        "",
        "## One-way sensitivity",
        one_way_view.to_markdown(index=False),
        "",
        "## Worst combined scenarios",
        worst[
            [
                "purchase_factor",
                "mid_factor",
                "big_factor",
                "same_filter_count",
                "annual_cost_penalty_pct",
            ]
        ].to_markdown(index=False),
        "",
        "## Representative plan switches",
        switches[switches["changed"]]
        .head(30)[
            [
                "scenario",
                "filter_no",
                "base_mid_interval_days",
                "base_big_interval_days",
                "mid_interval_days",
                "big_interval_days",
            ]
        ]
        .to_markdown(index=False),
        "",
    ]
    (output_dir / "q4_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> None:
    table_dir, figure_dir = ensure_dirs(cfg.output_dir)
    strategy, base_plan = load_inputs(cfg)
    one_way = one_way_sensitivity(strategy, base_plan, cfg)
    combined = combined_sensitivity(strategy, base_plan, cfg)
    switches = plan_switch_details(strategy, base_plan, cfg)

    one_way.to_csv(table_dir / "q4_one_way_sensitivity.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(table_dir / "q4_combined_sensitivity.csv", index=False, encoding="utf-8-sig")
    switches.to_csv(table_dir / "q4_plan_switch_details.csv", index=False, encoding="utf-8-sig")
    save_plots(one_way, combined, figure_dir)
    write_summary(one_way, combined, switches, cfg.output_dir)

    print(f"One-way scenarios: {len(one_way)}")
    print(f"Combined scenarios: {len(combined)}")
    print(f"Tables: {table_dir.resolve()}")
    print(f"Figures: {figure_dir.resolve()}")
    print(f"Summary: {(cfg.output_dir / 'q4_summary.md').resolve()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Q4 maintenance plan cost sensitivity")
    parser.add_argument(
        "--q3-strategy-search",
        type=Path,
        default=Path("output/q3/tables/q3_strategy_search.csv"),
    )
    parser.add_argument(
        "--q3-optimal-plan",
        type=Path,
        default=Path("output/q3/tables/q3_optimal_plan.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/q4"))
    args = parser.parse_args()
    return Config(
        q3_strategy_search_csv=args.q3_strategy_search,
        q3_optimal_plan_csv=args.q3_optimal_plan,
        output_dir=args.output,
    )


if __name__ == "__main__":
    run(parse_args())
