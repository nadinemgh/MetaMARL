from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import plotly.graph_objects as go
import wandb

from core.utils import sanitize_key, to_float
from core.world.context import Context, EnvStepContext, MechanismStatus
from core.reporting.utils.env_step_context import _extract_info_series


_ENV_REDUCED_ITER_TABLES: dict[tuple[int, str], wandb.Table] = {}

EPS = 1e-8


@dataclass(frozen=True)
class ReductionSpec:
    name: str
    fn: Optional[Callable[[list[Context]], float]] = None


def build_default_fishery_reduction_specs() -> list[ReductionSpec]:
    return [ReductionSpec(name="auto_env_metrics")]
 

def _global_step(outer_iter: int, train_step: int) -> int:
    return int(outer_iter) * 1_000_000 + int(train_step)


def _status_to_phase(status: Any, *, fallback: str = "train") -> str:
    if status is None:
        return fallback

    value = str(getattr(status, "value", status))

    if value == MechanismStatus.eval.value:
        return "eval"
    if value == MechanismStatus.train.value:
        return "train"

    return value


def _mean_agent_values(values_by_agent: dict[str, float]) -> float | None:
    vals = []

    for value in values_by_agent.values():
        fv = to_float(value)
        if fv is not None:
            vals.append(float(fv))

    if not vals:
        return None

    return float(np.mean(vals))


def _ctx_to_metric_rows(ctx: Context) -> list[dict[str, Any]]:
    # KEEP_METRICS = {
    #     "info/reservoir_stage_m",
    #     "info/streamflow_m3s",
    #     "info/total_usage_m3s",
    #     "info/requested_m3s",
    #     "info/allowed_m3s",
    #     "info/quota_penalty",
    #     "info/underuse_penalty",
    #     "info/total_penalty",
    # }
    KEEP_METRICS = {
        "info/reservoir_stage",
        "info/reservoir_level_norm",
        "info/streamflow_m3s",
        "info/precip_mm_day",
        "info/outflow_m3s",
        "info/temp_c",
        "info/crop_satisfaction",
        "info/delivered_m3_day",
        "info/total_usage_m3s",
        "info/full_required_m3_day",
        "info/deficit_mm_day",
        "info/requested_m3_day",
        "info/allowed_m3_day",
        "info/quota_violation_m3",
        "info/quota_penalty",
        "info/violation_signal",
        "info/release_pressure",
        "info/residence_time_days",
        "info/02GA041_streamflow_m3s",
        "info/02GA041_streamflow_m3s_observed",
        "info/02GA014_streamflow_m3s",
        "info/02GA014_streamflow_m3s_observed",
        "info/West_Montrose_streamflow_m3s",
        "info/West_Montrose_streamflow_m3s_observed",
        "info/base_allowed_m3_day",
        "info/graded_floor_m3_day",
        "info/quota_source",
        "info/farm_area_m2",
        "info/farm_area_m2_std",
        "info/farm_area_m2_min",
        "info/farm_area_m2_p05",
        "info/farm_area_m2_p25",
        "info/farm_area_m2_p50",
        "info/farm_area_m2_p75",
        "info/farm_area_m2_p90",
        "info/farm_area_m2_p95",
        "info/farm_area_m2_p99",
        "info/farm_area_m2_max",

        "info/farm_area_ha",
        "info/farm_area_ha_std",
        "info/farm_area_ha_min",
        "info/farm_area_ha_p05",
        "info/farm_area_ha_p25",
        "info/farm_area_ha_p50",
        "info/farm_area_ha_p75",
        "info/farm_area_ha_p90",
        "info/farm_area_ha_p95",
        "info/farm_area_ha_p99",
        "info/farm_area_ha_max",

        "info/fish",
        "info/fish_next",
        "info/fish_norm",
        "info/fish_norm_next",
        "info/growth",
        "info/growth_noise",
        "info/H_attempted",
        "info/H_realized",
        "info/total_usage_norm",
        "info/full_required_harvest",
        "info/requested_harvest",
        "info/allowed_harvest",
        "info/delivered_harvest",
        "info/quota_violation",
        "info/risk_penalty",
        "info/quota_stress",
        "info/min_demand_frac",
        "info/max_demand_frac",
        "info/B_msy",
        "info/MSY",
        "info/F_msy",
        "info/intrinsic_utility",
        "info/n_taggers",
        "info/n_non_taggers",
        "info/n_fixed_non_taggers",
        "info/n_taggers_episode_end",
        "info/n_non_taggers_episode_end",
    }
    # KEEP_METRICS = {
    #     "info/fish",
    #     "info/fish_next",
    #     "info/fish_norm",
    #     "info/growth",
    #     "info/growth_noise",
    #     "info/harvest",
    #     "info/desired_harvest",
    #     "info/intrinsic_utility",
    #     "info/violation_signal",
    #     "info/H_attempted",
    #     "info/H_realized",
    #     "info/harvest_scale",
    #     "info/below_target_zone",
    #     "info/target_shortfall",
    #     "info/B_msy",
    #     "info/MSY",
    #     "info/F_msy",
    #     "info/quota_violation",
    #     "info/quota_penalty",
    #     "info/stock_penalty",
    # }

    if ctx is None or not isinstance(ctx.payload, EnvStepContext):
        return []

    payload = ctx.payload

    phase = _status_to_phase(getattr(payload, "status", None))
    mechanism = getattr(payload, "mechanism", None)
    seed = getattr(payload, "seed", None)
    env_id = getattr(payload, "env_id", None)
    env_step = int(ctx.step)

    rows: list[dict[str, Any]] = []

    def add_metric(metric_name: str, value: float | None) -> None:
        if metric_name not in KEEP_METRICS:
            return

        fv = to_float(value)
        if fv is None:
            return

        rows.append(
            {
                "env_step": env_step,
                "phase": phase,
                "mechanism": str(mechanism) if mechanism is not None else "unknown",
                "seed": str(seed) if seed is not None else "unknown",
                "env_id": str(env_id) if env_id is not None else "unknown",
                "metric": sanitize_key(metric_name),
                "value": float(fv),
            }
        )

    info_by_key = _extract_info_series(payload.info)

    for info_key, values_by_agent in info_by_key.items():
        vals = []

        for value in values_by_agent.values():
            fv = to_float(value)
            if fv is not None:
                vals.append(float(fv))

        if not vals:
            continue

        arr = np.asarray(vals, dtype=np.float64)

        # Always log mean
        add_metric(f"info/{info_key}", float(arr.mean()))

        # Extra distribution stats for heterogeneous farm sizes
        if info_key in {"farm_area_m2", "farm_area_ha"}:
            add_metric(f"info/{info_key}_std", float(arr.std()))
            add_metric(f"info/{info_key}_min", float(arr.min()))
            add_metric(f"info/{info_key}_p05", float(np.quantile(arr, 0.05)))
            add_metric(f"info/{info_key}_p25", float(np.quantile(arr, 0.25)))
            add_metric(f"info/{info_key}_p50", float(np.quantile(arr, 0.50)))
            add_metric(f"info/{info_key}_p75", float(np.quantile(arr, 0.75)))
            add_metric(f"info/{info_key}_p90", float(np.quantile(arr, 0.90)))
            add_metric(f"info/{info_key}_p95", float(np.quantile(arr, 0.95)))
            add_metric(f"info/{info_key}_p99", float(np.quantile(arr, 0.99)))
            add_metric(f"info/{info_key}_max", float(arr.max()))
    return rows


def _build_metric_tables(rows: list[dict[str, Any]]) -> dict[str, wandb.Table]:
    tables: dict[str, wandb.Table] = {}

    for row in rows:
        metric = sanitize_key(row["metric"])

        if metric not in tables:
            tables[metric] = wandb.Table(
                columns=[
                    "env_step",
                    "phase",
                    "mechanism",
                    "seed",
                    "env_id",
                    "value",
                ]
            )

        tables[metric].add_data(
            int(row["env_step"]),
            str(row["phase"]),
            str(row["mechanism"]),
            str(row["seed"]),
            str(row["env_id"]),
            float(row["value"]),
        )

    return tables


def _table_to_phase_mechanism_curves(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str = "value",
    phase_col: str = "phase",
    mechanism_col: str = "mechanism",
    seed_col: str = "seed",
) -> dict[tuple[str, str], dict[str, list[float]]]:
    if table is None or table.data is None:
        return {}

    cols = list(table.columns)
    ix = cols.index(x_col)
    iy = cols.index(y_col)
    iphase = cols.index(phase_col)
    imech = cols.index(mechanism_col)
    iseed = cols.index(seed_col)

    grouped: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = {}

    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])

        if x is None or y is None:
            continue

        phase = str(row[iphase])
        mechanism = str(row[imech])
        seed = str(row[iseed])
        step = int(x)

        grouped.setdefault((phase, mechanism), {}).setdefault(step, {}).setdefault(
            seed, []
        ).append(float(y))

    curves: dict[tuple[str, str], dict[str, list[float]]] = {}

    for key, step_to_seed_values in grouped.items():
        xs, means, stds, uppers, lowers = [], [], [], [], []

        for step in sorted(step_to_seed_values.keys()):
            seed_means = [
                float(np.mean(vals))
                for vals in step_to_seed_values[step].values()
                if vals
            ]

            if not seed_means:
                continue

            arr = np.asarray(seed_means, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())

            xs.append(float(step))
            means.append(mean)
            stds.append(std)
            uppers.append(mean + std)
            lowers.append(mean - std)

        curves[key] = {
            "x": xs,
            "mean": means,
            "std": stds,
            "upper": uppers,
            "lower": lowers,
        }

    return curves


def _log_train_eval_shaded_plot(
    *,
    wandb_run,
    prefix: str,
    metric_name: str,
    table: wandb.Table,
    step: int,
) -> None:
    curves = _table_to_phase_mechanism_curves(
        table,
        x_col="env_step",
    )

    if not curves:
        return

    fig = _make_train_eval_figure(
        curves=curves,
        title=f"{metric_name.replace('/', ' / ')}: train vs eval over horizon",
        xaxis_title="env_step",
    )

    wandb_run.log(
        {f"{prefix}/plots/{metric_name}/train_vs_eval": fig},
        step=step,
        commit=False,
    )


def _log_iteration_reduced_shaded_plot(
    *,
    wandb_run,
    prefix: str,
    metric_name: str,
    table: wandb.Table,
    step: int,
) -> None:
    curves = _table_to_phase_mechanism_curves(
        table,
        x_col="train_step",
    )

    if not curves:
        return

    fig = _make_train_eval_figure(
        curves=curves,
        title=f"{metric_name}: train vs eval over training",
        xaxis_title="train_step",
    )

    wandb_run.log(
        {f"{prefix}/plots_over_training/{metric_name}/train_vs_eval": fig},
        step=step,
        commit=False,
    )


def _make_train_eval_figure(
    *,
    curves: dict[tuple[str, str], dict[str, list[float]]],
    title: str,
    xaxis_title: str,
) -> go.Figure:
    fig = go.Figure()

    phase_colors = {
        "train": "rgba(31, 119, 180, 1.0)",
        "eval": "rgba(255, 127, 14, 1.0)",
    }

    phase_fill_colors = {
        "train": "rgba(31, 119, 180, 0.25)",
        "eval": "rgba(255, 127, 14, 0.25)",
    }

    dash_styles = ["solid", "dash", "dot", "dashdot"]

    mechanisms = sorted({mechanism for _, mechanism in curves.keys()})
    mechanism_dash = {
        mechanism: dash_styles[i % len(dash_styles)]
        for i, mechanism in enumerate(mechanisms)
    }

    ordered_keys = sorted(
        curves.keys(),
        key=lambda k: (0 if k[0] == "train" else 1, k[1]),
    )

    for phase, mechanism in ordered_keys:
        curve = curves[(phase, mechanism)]

        xs = curve["x"]
        means = curve["mean"]
        upper = curve["upper"]
        lower = curve["lower"]

        if not xs:
            continue

        color = phase_colors.get(phase, "rgba(0, 0, 0, 1.0)")
        fill_color = phase_fill_colors.get(phase, "rgba(0, 0, 0, 0.20)")
        dash = mechanism_dash.get(mechanism, "solid")

        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=upper + lower[::-1],
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=fill_color, width=0),
                name=f"{phase} m{mechanism} ±1 std",
                hoverinfo="skip",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line=dict(width=3, color=color, dash=dash),
                marker=dict(size=4, color=color),
                name=f"{phase} m{mechanism} mean",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="value",
        hovermode="x unified",
        template="plotly_white",
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def _rows_to_iteration_metric_rows(
    rows: list[dict[str, Any]],
    *,
    train_step: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}

    for row in rows:
        key = (row["phase"], row["mechanism"], row["seed"])
        metric = row["metric"]
        grouped.setdefault(key, {}).setdefault(metric, []).append(float(row["value"]))

    out: list[dict[str, Any]] = []

    def add(
        phase: str,
        mechanism: str,
        seed: str,
        metric: str,
        value: float | None,
    ) -> None:
        if value is None or not np.isfinite(value):
            return

        out.append(
            {
                "train_step": int(train_step),
                "phase": phase,
                "mechanism": mechanism,
                "seed": seed,
                "metric": metric,
                "value": float(value),
            }
        )

    for (phase, mechanism, seed), metric_values in grouped.items():
        for metric_name, values in metric_values.items():
            if not values:
                continue

            clean_name = metric_name.replace("info_", "")
            # clean_name = metric_name.split("/")[-1]
            arr = np.asarray(values, dtype=np.float64)

            # if clean_name == "total_usage_m3s":
            #     # Total extraction across the rollout.
            #     value = float(np.sum(arr))
            #     out_name = "total_usage_m3s_sum"

            # else:
            #     # Typical state / penalty / flow level over the rollout.
            #     value = float(np.mean(arr))
            #     out_name = f"{clean_name}_mean"

            # # if clean_name in {"harvest", "desired_harvest", "H_attempted", "H_realized"}:
            # #     value = float(np.sum(arr))
            # #     out_name = f"{clean_name}_sum"
            # # else:
            # #     value = float(np.mean(arr))
            # #     out_name = f"{clean_name}_mean"

            # add(
            #     phase,
            #     mechanism,
            #     seed,
            #     out_name,
            #     value,
            # )

            if clean_name in {
                "total_usage_m3s",
                "H_realized",
                "H_attempted",
                "requested_harvest",
                "allowed_harvest",
                "delivered_harvest",
                "requested_m3_day",
                "allowed_m3_day",
                "delivered_m3_day",
            }:
                add(phase, mechanism, seed, f"{clean_name}_sum", float(np.sum(arr)))
                add(phase, mechanism, seed, f"{clean_name}_mean", float(np.mean(arr)))
                add(phase, mechanism, seed, f"{clean_name}_max", float(np.max(arr)))

            elif clean_name in {
                # water state variables
                "reservoir_stage",
                "reservoir_level_norm",
                "streamflow_m3s",
                "outflow_m3s",
                "02GA041_streamflow_m3s",
                "02GA041_streamflow_m3s_observed",
                "02GA014_streamflow_m3s",
                "02GA014_streamflow_m3s_observed",
                "West_Montrose_streamflow_m3s",
                "West_Montrose_streamflow_m3s_observed",
                "release_pressure",
                "residence_time_days",

                # fishery state variables
                "fish",
                "fish_next",
                "fish_norm",
                "fish_norm_next",
                "growth",
                "total_usage_norm",
            }:
                add(phase, mechanism, seed, f"{clean_name}_mean", float(np.mean(arr)))
                add(phase, mechanism, seed, f"{clean_name}_last", float(arr[-1]))
                add(phase, mechanism, seed, f"{clean_name}_min", float(np.min(arr)))
                add(phase, mechanism, seed, f"{clean_name}_max", float(np.max(arr)))

            else:
                add(phase, mechanism, seed, f"{clean_name}_mean", float(np.mean(arr)))
    return out

def _log_02GA041_observed_vs_simulated(
    *,
    wandb_run,
    prefix: str,
    metric_tables: dict[str, wandb.Table],
    step: int,
) -> None:
    sim_key = sanitize_key("info/02GA041_streamflow_m3s")
    obs_key = sanitize_key("info/02GA041_streamflow_m3s_observed")

    if sim_key not in metric_tables or obs_key not in metric_tables:
        return

    sim_table = metric_tables[sim_key]
    obs_table = metric_tables[obs_key]

    fig = go.Figure()

    def add_trace(table: wandb.Table, name: str) -> None:
        cols = list(table.columns)
        ix = cols.index("env_step")
        iy = cols.index("value")

        xs = []
        ys = []

        for row in table.data:
            x = to_float(row[ix])
            y = to_float(row[iy])
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
            )
        )

    add_trace(sim_table, "02GA041 simulated")
    add_trace(obs_table, "02GA041 observed")

    fig.update_layout(
        title="02GA041 simulated vs observed streamflow",
        xaxis_title="env_step",
        yaxis_title="streamflow [m3/s]",
        hovermode="x unified",
        template="plotly_white",
        height=650,
    )

    wandb_run.log(
        {f"{prefix}/plots/02GA041_simulated_vs_observed": fig},
        step=step,
        commit=False,
    )

def _log_02GA014_observed_vs_simulated(
    *,
    wandb_run,
    prefix: str,
    metric_tables: dict[str, wandb.Table],
    step: int,
) -> None:
    sim_key = sanitize_key("info/02GA014_streamflow_m3s")
    obs_key = sanitize_key("info/02GA014_streamflow_m3s_observed")

    if sim_key not in metric_tables or obs_key not in metric_tables:
        return

    sim_table = metric_tables[sim_key]
    obs_table = metric_tables[obs_key]

    fig = go.Figure()

    def add_trace(table: wandb.Table, name: str) -> None:
        cols = list(table.columns)
        ix = cols.index("env_step")
        iy = cols.index("value")

        xs = []
        ys = []

        for row in table.data:
            x = to_float(row[ix])
            y = to_float(row[iy])
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
            )
        )

    add_trace(sim_table, "02GA014 simulated")
    add_trace(obs_table, "02GA014 observed")

    fig.update_layout(
        title="02GA014 simulated vs observed streamflow",
        xaxis_title="env_step",
        yaxis_title="streamflow [m3/s]",
        hovermode="x unified",
        template="plotly_white",
        height=650,
    )

    wandb_run.log(
        {f"{prefix}/plots/02GA014_simulated_vs_observed": fig},
        step=step,
        commit=False,
    )

def _log_West_Montrose_observed_vs_simulated(
    *,
    wandb_run,
    prefix: str,
    metric_tables: dict[str, wandb.Table],
    step: int,
) -> None:
    sim_key = sanitize_key("info/West_Montrose_streamflow_m3s")
    obs_key = sanitize_key("info/West_Montrose_streamflow_m3s_observed")

    if sim_key not in metric_tables or obs_key not in metric_tables:
        return

    sim_table = metric_tables[sim_key]
    obs_table = metric_tables[obs_key]

    fig = go.Figure()

    def add_trace(table: wandb.Table, name: str) -> None:
        cols = list(table.columns)
        ix = cols.index("env_step")
        iy = cols.index("value")

        xs = []
        ys = []

        for row in table.data:
            x = to_float(row[ix])
            y = to_float(row[iy])
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=name,
            )
        )

    add_trace(sim_table, "West_Montrose simulated")
    add_trace(obs_table, "West_Montrose observed")

    fig.update_layout(
        title="West_Montrose simulated vs observed streamflow",
        xaxis_title="env_step",
        yaxis_title="streamflow [m3/s]",
        hovermode="x unified",
        template="plotly_white",
        height=650,
    )

    wandb_run.log(
        {f"{prefix}/plots/West_Montrose_simulated_vs_observed": fig},
        step=step,
        commit=False,
    )

def plot_env_reduced(
    *,
    wandb_run,
    ctxs: list[Context],
    outer_iter: int,
    training_episode: int,
    reducers: list[ReductionSpec] | None = None,
    prefix: str = "env_reduced",
    agent_groups: dict[str, str] | None = None,  # optional fixed/learned experimental roles
) -> None:
    if wandb_run is None or not ctxs:
        return

    gs = _global_step(outer_iter, training_episode)

    # Added: exact per-agent rollout reward sums and observed tagging behavior.
    # Leaves existing aggregated environmental plots and tables unchanged.
    from core.reporting.utils.schelling_archive import log_agent_rollout_returns
    log_agent_rollout_returns(
        wandb_run=wandb_run,
        ctxs=ctxs,
        outer_iter=outer_iter,
        train_step=training_episode,
        prefix=prefix,
        expected_horizon=100,
        agent_groups=agent_groups,
    )

    rows: list[dict[str, Any]] = []

    for ctx in ctxs:
        rows.extend(_ctx_to_metric_rows(ctx))

    if not rows:
        return
    
    # ============================================================
    # RAW TABLES FOR POST-HOC ANALYSIS
    # ============================================================

    import pandas as pd

    # ------------------------------------------------------------
    # 1. Long-form timestep table
    # ------------------------------------------------------------
    raw_env_df = pd.DataFrame(rows)

    wandb_run.log(
        {
            f"{prefix}/tables/raw_env_steps": wandb.Table(
                dataframe=raw_env_df
            )
        },
        step=gs,
        commit=False,
    )

    # ------------------------------------------------------------
    # 2. Wide timestep table
    # ------------------------------------------------------------
    wide_rows: dict = {}

    for row in rows:
        key = (
            row["phase"],
            row["mechanism"],
            row["seed"],
            row["env_step"],
        )

        wide_rows.setdefault(
            key,
            {
                "phase": row["phase"],
                "mechanism": row["mechanism"],
                "seed": row["seed"],
                "env_step": row["env_step"],
            },
        )

        wide_rows[key][row["metric"]] = row["value"]

    wide_df = pd.DataFrame(
        list(wide_rows.values())
    )

    wandb_run.log(
        {
            f"{prefix}/tables/raw_env_steps_wide": wandb.Table(
                dataframe=wide_df
            )
        },
        step=gs,
        commit=False,
    )

    # ------------------------------------------------------------
    # 3. Useful derived quantities
    # ------------------------------------------------------------

    if (
        "info_requested_m3_day" in wide_df.columns
        and "info_allowed_m3_day" in wide_df.columns
    ):
        wide_df["requested_over_allowed"] = (
            wide_df["info_requested_m3_day"]
            / np.maximum(
                EPS,
                wide_df["info_allowed_m3_day"],
            )
        )

    if (
        "info_streamflow_m3s" in wide_df.columns
        and "info_total_usage_m3s" in wide_df.columns
    ):
        wide_df["usage_over_streamflow"] = (
            wide_df["info_total_usage_m3s"]
            / np.maximum(
                EPS,
                wide_df["info_streamflow_m3s"],
            )
        )

    wandb_run.log(
        {
            f"{prefix}/tables/raw_env_steps_wide_derived":
                wandb.Table(dataframe=wide_df)
        },
        step=gs,
        commit=False,
    )

    # ------------------------------------------------------------
    # 4. Correlation matrix
    # ------------------------------------------------------------

    exclude_cols = {
        "phase",
        "mechanism",
        "seed",
        "env_step",
    }

    numeric_cols = [
        c
        for c in wide_df.columns
        if c not in exclude_cols
    ]

    corr_df = (
        wide_df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .corr()
    )

    wandb_run.log(
        {
            f"{prefix}/tables/correlation_matrix":
                wandb.Table(
                    dataframe=corr_df.reset_index()
                )
        },
        step=gs,
        commit=False,
    )

    # ------------------------------------------------------------
    # 5. Distribution summary table
    # ------------------------------------------------------------

    summary_rows = []

    for col in numeric_cols:
        series = pd.to_numeric(
            wide_df[col],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        summary_rows.append(
            {
                "metric": col,
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p05": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    wandb_run.log(
        {
            f"{prefix}/tables/distribution_summary":
                wandb.Table(
                    dataframe=summary_df
                )
        },
        step=gs,
        commit=False,
    )

    # ============================================================
    # END RAW ANALYSIS TABLES
    # ============================================================

    metric_tables = _build_metric_tables(rows)

    _log_02GA041_observed_vs_simulated(
        wandb_run=wandb_run,
        prefix=prefix,
        metric_tables=metric_tables,
        step=gs,
    )

    _log_02GA014_observed_vs_simulated(
        wandb_run=wandb_run,
        prefix=prefix,
        metric_tables=metric_tables,
        step=gs,
    )

    _log_West_Montrose_observed_vs_simulated(
        wandb_run=wandb_run,
        prefix=prefix,
        metric_tables=metric_tables,
        step=gs,
    )

    # 1. Raw rollout plots over horizon/env_step.
    for metric_name, table in metric_tables.items():
        _log_train_eval_shaded_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            metric_name=metric_name,
            table=table,
            step=gs,
        )

    # 2. Reduced plots over training iterations.
    iter_rows = _rows_to_iteration_metric_rows(
        rows,
        train_step=gs,
    )

    # Tagging composition versus training iteration (W&B time-series).
    # Average across mechanism/seed groups; episode-end metrics include only
    # completed episodes, whereas plain counts use all sampled timesteps.
    count_metrics = {
        "n_taggers_mean": "taggers",
        "n_non_taggers_mean": "non_taggers",
        "n_fixed_non_taggers_mean": "fixed_non_taggers",
        "n_taggers_episode_end_mean": "taggers_episode_end",
        "n_non_taggers_episode_end_mean": "non_taggers_episode_end",
    }
    count_log = {}
    for metric_name, label in count_metrics.items():
        values = [
            row["value"] for row in iter_rows
            if row["phase"] == "train" and row["metric"] == metric_name
        ]
        if values:
            count_log[f"{prefix}/tagging/{label}"] = float(np.mean(values))
    if count_log:
        wandb_run.log(count_log, step=gs, commit=False)

    iter_df = pd.DataFrame(iter_rows)

    wandb_run.log(
        {
            f"{prefix}/tables/training_metrics":
                wandb.Table(dataframe=iter_df)
        },
        step=gs,
        commit=False,
    )

    for row in iter_rows:
        metric_name = sanitize_key(row["metric"])
        cache_key = (id(wandb_run), metric_name)

        table = _ENV_REDUCED_ITER_TABLES.get(cache_key)

        if table is None:
            table = wandb.Table(
                columns=[
                    "train_step",
                    "phase",
                    "mechanism",
                    "seed",
                    "value",
                ]
            )
            _ENV_REDUCED_ITER_TABLES[cache_key] = table

        table.add_data(
            int(row["train_step"]),
            str(row["phase"]),
            str(row["mechanism"]),
            str(row["seed"]),
            float(row["value"]),
        )

        _log_iteration_reduced_shaded_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            metric_name=metric_name,
            table=table,
            step=gs,
        )

    wandb_run.log({}, step=gs, commit=True)