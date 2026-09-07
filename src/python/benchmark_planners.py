"""Reproducible comparison of the three tracked trajectory-sampling modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from part2_control import (
    Environment,
    MotionPlanningSimulation,
    PlannerConfig,
    dependency_versions,
    save_scenario_plots,
)


MODES = ("gmm_heuristic", "heuristic", "uniform")
MODE_LABELS = {
    "gmm_heuristic": "GMM + heuristic",
    "heuristic": "Heuristic only",
    "uniform": "Uniform random",
}
# A full seed takes about one minute on the tested host with the conservative
# adaptive checker.  Twenty seeds keep the tracked run practical while meeting
# the requested minimum for a portfolio-quality benchmark.
DEFAULT_SEEDS = tuple(range(20))
BINARY_METRICS = (
    "success",
    "goal_reached",
    "reached_goal_tolerance",
    "completed_dwell",
    "safe_success",
    "collision_failure",
)
SCALAR_METRICS = (
    "final_angular_error_l2_rad",
    "final_prismatic_error_m",
    "normalized_configuration_error",
    "end_effector_error_m",
    "end_effector_path_length_m",
    "mean_selected_horizon_cost",
    "realized_trajectory_cost",
    "minimum_link_clearance_m",
    "planning_time_mean_s",
    "planning_time_median_s",
    "planning_time_p95_s",
    "planning_time_max_s",
    "planning_time_total_s",
    "planning_deadline_miss_fraction",
    "query_solver_wall_time_s",
    "fitting_plus_online_planning_time_s",
    "execution_duration_s",
    "control_steps",
    "fallback_steps",
    "no_motion_steps",
)


def flatten_run(run: dict) -> dict:
    row = {
        "mode": run["mode"],
        "scenario": run["scenario"],
        "seed": run["seed"],
        "control_period_s": run["control_period_s"],
    }
    for metric in BINARY_METRICS:
        row[metric] = run.get(metric)
    for metric in SCALAR_METRICS:
        row[metric] = run.get(metric)
    row["planning_times_s"] = json.dumps(run["planning_times_s"])
    row["training_fitted"] = run["training"]["fitted"]
    row["training_feasible"] = run["training"]["feasible"]
    row["training_wall_time_s"] = run["training"]["training_wall_time_s"]
    return row


def wilson_interval(successes: int, runs: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if runs == 0:
        return math.nan, math.nan
    proportion = successes / runs
    denominator = 1.0 + z**2 / runs
    center = (proportion + z**2 / (2.0 * runs)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / runs + z**2 / (4.0 * runs**2))
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def aggregate_group(mode: str, group: list[dict], scenario: str | None = None) -> dict:
    row = {"mode": mode}
    if scenario is not None:
        row["scenario"] = scenario
    row["runs"] = len(group)
    for metric in BINARY_METRICS:
        row[f"{metric}_rate"] = float(np.mean([run[metric] for run in group]))

    safe_successes = sum(bool(run["safe_success"]) for run in group)
    ci_lower, ci_upper = wilson_interval(safe_successes, len(group))
    row["safe_success_count"] = safe_successes
    row["safe_success_rate_ci95_lower"] = ci_lower
    row["safe_success_rate_ci95_upper"] = ci_upper

    for metric in SCALAR_METRICS:
        values = [run[metric] for run in group if run.get(metric) is not None]
        row[f"{metric}_mean"] = float(np.mean(values)) if values else None
        row[f"{metric}_std"] = float(np.std(values)) if values else None

    successful_clearances = [
        run["minimum_link_clearance_m"] for run in group if run["safe_success"]
    ]
    failed_clearances = [
        run["minimum_link_clearance_m"] for run in group if not run["safe_success"]
    ]
    row["minimum_clearance_successful_runs_m"] = (
        float(min(successful_clearances)) if successful_clearances else None
    )
    row["minimum_clearance_failed_runs_m"] = (
        float(min(failed_clearances)) if failed_clearances else None
    )
    row["worst_case_minimum_clearance_m"] = float(
        min(run["minimum_link_clearance_m"] for run in group)
    )

    planning_times = np.asarray(
        [value for run in group for value in run["planning_times_s"]], dtype=float
    )
    row["planning_step_time_mean_s"] = float(np.mean(planning_times))
    row["planning_step_time_median_s"] = float(np.median(planning_times))
    row["planning_step_time_p95_s"] = float(np.quantile(planning_times, 0.95))
    row["planning_step_time_max_s"] = float(np.max(planning_times))
    row["planning_step_deadline_miss_fraction"] = float(
        np.mean(planning_times > group[0]["control_period_s"])
    )
    fitting_times = np.asarray(
        [run["training"]["training_wall_time_s"] for run in group], dtype=float
    )
    row["offline_fitting_time_mean_s"] = float(np.mean(fitting_times))
    row["offline_fitting_time_std_s"] = float(np.std(fitting_times))
    return row


def aggregate_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["mode"], run["scenario"])].append(run)

    summary = []
    for (mode, scenario), group in sorted(grouped.items()):
        summary.append(aggregate_group(mode, group, scenario))
    return summary


def aggregate_overall(runs: list[dict]) -> list[dict]:
    overall = []
    for mode in MODES:
        group = [run for run in runs if run["mode"] == mode]
        overall.append(aggregate_group(mode, group))
    return overall


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_figure(summary: list[dict], output_path: Path) -> None:
    scenarios = [scenario.name for scenario in MotionPlanningSimulation.scenarios()]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = (
        ("safe_success_rate", "Safe-success rate", (0.0, 1.05)),
        ("end_effector_error_m_mean", "Mean end-effector error (m)", None),
        ("worst_case_minimum_clearance_m", "Worst-case swept clearance (m)", None),
        ("planning_step_time_p95_s", "95th-percentile planning time / step (s)", None),
    )
    x = np.arange(len(scenarios))
    width = 0.25
    for axis, (metric, title, limits) in zip(axes.flat, metrics):
        for mode_index, mode in enumerate(MODES):
            values = []
            lower_errors = []
            upper_errors = []
            for scenario in scenarios:
                matching = [
                    row for row in summary if row["mode"] == mode and row["scenario"] == scenario
                ]
                row = matching[0] if matching else None
                value = row[metric] if row else np.nan
                values.append(value)
                if metric == "safe_success_rate" and row:
                    lower_errors.append(value - row["safe_success_rate_ci95_lower"])
                    upper_errors.append(row["safe_success_rate_ci95_upper"] - value)
            error_bars = None
            if metric == "safe_success_rate":
                error_bars = np.asarray([lower_errors, upper_errors])
            axis.bar(
                x + (mode_index - 1) * width,
                values,
                width,
                label=MODE_LABELS[mode],
                yerr=error_bars,
                capsize=3 if error_bars is not None else 0,
            )
        axis.set_xticks(x, scenarios, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
        if metric == "worst_case_minimum_clearance_m":
            axis.set_yscale("log")
            axis.axhline(
                Environment().clearance_tolerance,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
            )
        if metric == "planning_step_time_p95_s":
            axis.axhline(
                PlannerConfig().dt,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
            )
    axes[0, 0].legend()
    figure.suptitle("Seeded sampled receding-horizon benchmark")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_benchmark(
    output_directory: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    duration_s: float = 6.0,
    candidates: int = 12,
    training_trajectories: int = 48,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    representative_written: set[str] = set()
    common_planner_config = PlannerConfig(
        n_candidates=candidates, n_training=training_trajectories
    ).to_jsonable()
    common_planner_config.pop("seed")

    for seed in seeds:
        for scenario in MotionPlanningSimulation.scenarios():
            for mode in MODES:
                config = PlannerConfig(
                    seed=seed,
                    n_candidates=candidates,
                    n_training=training_trajectories,
                )
                simulation = MotionPlanningSimulation(config=config, mode=mode)
                run = simulation.run_scenario(scenario, duration_s=duration_s)
                runs.append(run)
                print(
                    f"{mode:15s} | seed={seed:3d} | {scenario.name:20s} | "
                    f"safe_success={run['safe_success']} | collision={run['collision_failure']} | "
                    f"ee={run['end_effector_error_m']:.4f} m | "
                    f"clearance={run['minimum_link_clearance_m']:.4f} m"
                )
                if mode == "gmm_heuristic" and seed == seeds[-1] and scenario.name not in representative_written:
                    save_scenario_plots(
                        run, simulation.environment, output_directory / "representative-runs"
                    )
                    representative_written.add(scenario.name)

    summary = aggregate_runs(runs)
    overall = aggregate_overall(runs)
    payload = {
        "method": "GMM-guided trajectory sampling with MPC-style sampled receding-horizon selection",
        "comparison_modes": list(MODES),
        "seeds": list(seeds),
        "duration_s": duration_s,
        "candidate_budget_per_control_step": candidates,
        "training_trajectory_budget": training_trajectories,
        "planner_config": common_planner_config,
        "seed_policy": "per-run seed; SeedSequence(seed).spawn(2) for training and online RNG streams",
        "collision_checker": Environment().to_jsonable(),
        "environment": dependency_versions(),
        "notes": [
            "Python and MATLAB studies are computationally independent.",
            "Wall-clock values vary by host load even when sampled trajectories are seeded.",
            "Uniform mode samples joint-space targets uniformly and applies the same step limit, cost, and safety checks.",
            "The comparison matches online candidate count, not candidate-generation compute cost.",
            "GMM fitting is scenario-specific offline work and is excluded from online planning time; both fitting and fitting-plus-online totals are reported separately.",
            "Training and online sampling use separate deterministic RNG streams derived from each run seed.",
            "Safe success requires completed goal dwell and no collision failure.",
            "Reported signed clearance subtracts obstacle radius and the configured link safety radius; feasibility also requires the positive clearance tolerance.",
            "No superiority claim is implied; interpret the measured table directly.",
        ],
        "summary": summary,
        "overall": overall,
        "runs": runs,
    }
    (output_directory / "benchmark-results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(output_directory / "benchmark-runs.csv", [flatten_run(run) for run in runs])
    write_csv(output_directory / "benchmark-summary.csv", summary)
    write_csv(output_directory / "benchmark-overall.csv", overall)
    save_summary_figure(summary, output_directory / "benchmark-summary.png")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results") / "corrected" / "benchmark"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--training-trajectories", type=int, default=48)
    arguments = parser.parse_args()
    run_benchmark(
        arguments.output,
        seeds=tuple(arguments.seeds),
        duration_s=arguments.duration,
        candidates=arguments.candidates,
        training_trajectories=arguments.training_trajectories,
    )


if __name__ == "__main__":
    main()
