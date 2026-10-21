from __future__ import annotations

import logging
import re  # MODIFIED: needed to extract mechanism ids like m0/m1 from policy names
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

# MODIFIED: Plotly is used because wandb.plot.line_series does not support shaded error regions.
# W&B can log Plotly figures directly.
import plotly.graph_objects as go

from core.utils import safe_ratio, to_float, finite, sanitize_key

logger = logging.getLogger(__name__)


def _cap_table(table: wandb.Table, max_rows: int) -> wandb.Table:
    if table is None:
        return table
    data = table.data
    if data is None or len(data) <= max_rows:
        return table
    tail = data[-max_rows:]
    new_t = wandb.Table(columns=table.columns)
    for r in tail:
        new_t.add_data(*r)
    return new_t


def _summarize_dict_of_scalars(d: Dict[str, Any]) -> Dict[str, float]:
    vals: list[float] = []
    for v in (d or {}).values():
        fv = finite(v)
        if fv is not None:
            vals.append(fv)
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "n": float(len(arr)),
    }


def _global_step(outer_iter: int, train_step: int) -> int:
    # monotonic across outer iters
    # TODO find a way to reduce outer iterations
    return int(outer_iter) * 1_000_000 + int(train_step)

# MODIFIED: helper to extract mechanism identity from seeded policy ids.
# Example:
#   fisher_policy_m0_s2669555309 -> m0
#   fisher_policy_m1_s3444837047 -> m1
_MECHANISM_RE = re.compile(r"(?:^|_)m(?P<mechanism>\d+)(?:_|$)")


# MODIFIED: new helper.
def _extract_mechanism_id(series_name: str) -> str:
    """
    Extracts mechanism identity from a policy/series name.

    Expected seeded policy format:
      fisher_policy_m0_s2669555309
      fisher_policy_m1_s3444837047

    Returns:
      m0, m1, ...

    Fallback:
      If no mechanism id is found, returns the original series name so behavior
      remains safe and non-breaking.
    """
    s = str(series_name)
    match = _MECHANISM_RE.search(s)
    if match is None:
        return s
    return f"m{match.group('mechanism')}"


# MODIFIED: new helper.
def _table_to_mechanism_mean_std_arrays(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Converts a table with per-seed/per-policy lines into mechanism-level
    mean/std curves.

    Input table columns contain:
      x_col      -> step
      y_col      -> value
      series_col -> seeded policy/series id, e.g. fisher_policy_m0_s2669555309

    Output:
      {
        "m0": {
            "x": [...],
            "mean": [...],
            "upper": [...],
            "lower": [...],
            "std": [...],
            "n": [...]
        },
        "m1": ...
      }
    """
    if table is None or table.data is None:
        return {}

    cols = list(table.columns)
    try:
        ix = cols.index(x_col)
        iy = cols.index(y_col)
        iser = cols.index(series_col)
    except ValueError as e:
        raise ValueError(
            f"Table missing required columns. Have={cols}, need={[x_col, series_col, y_col]}"
        ) from e

    # mechanism -> step -> values across seeded policies
    grouped: dict[str, dict[float, list[float]]] = {}

    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])
        series_name = row[iser]

        if x is None or y is None:
            continue

        mechanism_id = _extract_mechanism_id(str(series_name))

        grouped.setdefault(mechanism_id, {}).setdefault(float(x), []).append(float(y))

    out: Dict[str, Dict[str, List[float]]] = {}

    for mechanism_id in sorted(grouped.keys()):
        step_to_values = grouped[mechanism_id]
        xs = sorted(step_to_values.keys())

        means: list[float] = []
        stds: list[float] = []
        uppers: list[float] = []
        lowers: list[float] = []
        ns: list[float] = []

        for x in xs:
            vals = np.asarray(step_to_values[x], dtype=np.float64)
            mean = float(vals.mean())
            std = float(vals.std())  # population std; stable even when n=1

            means.append(mean)
            stds.append(std)
            uppers.append(mean + std)
            lowers.append(mean - std)
            ns.append(float(len(vals)))

        out[mechanism_id] = {
            "x": [float(x) for x in xs],
            "mean": means,
            "std": stds,
            "upper": uppers,
            "lower": lowers,
            "n": ns,
        }

    return out


# MODIFIED: new helper.
def _log_mechanism_shaded_error_plot(
    *,
    wandb_run: Run,
    prefix: str,
    plot_name: str,
    table: wandb.Table,
    x_col: str,
    y_col: str,
    series_col: str,
    step: int,
    max_rows: int,
    title: str,
) -> None:
    """
    Logs a Plotly figure to W&B where each mechanism has:
      - one mean line
      - one shaded ±1 std region

    This leaves the existing wandb.plot.line_series plots untouched.
    """
    table = _cap_table(table, max_rows)

    mechanism_curves = _table_to_mechanism_mean_std_arrays(
        table,
        x_col=x_col,
        y_col=y_col,
        series_col=series_col,
    )

    if not mechanism_curves:
        return

    fig = go.Figure()

    # MODIFIED: explicit color palette because W&B's Plotly renderer can make
    # default fill colors almost invisible.
    colors = [
        "rgba(31, 119, 180, 1.0)",
        "rgba(255, 127, 14, 1.0)",
        "rgba(44, 160, 44, 1.0)",
        "rgba(214, 39, 40, 1.0)",
        "rgba(148, 103, 189, 1.0)",
        "rgba(140, 86, 75, 1.0)",
    ]

    fill_colors = [
        "rgba(31, 119, 180, 0.35)",
        "rgba(255, 127, 14, 0.35)",
        "rgba(44, 160, 44, 0.35)",
        "rgba(214, 39, 40, 0.35)",
        "rgba(148, 103, 189, 0.35)",
        "rgba(140, 86, 75, 0.35)",
    ]

    for i, (mechanism_id, curve) in enumerate(mechanism_curves.items()):
        xs = curve["x"]
        means = curve["mean"]
        upper = curve["upper"]
        lower = curve["lower"]

        if not xs:
            continue

        color = colors[i % len(colors)]
        fill_color = fill_colors[i % len(fill_colors)]

        # MODIFIED:
        # Instead of using fill="tonexty", create a closed polygon:
        #
        #   upper line left -> right
        #   lower line right -> left
        #
        # This is much more robust in W&B than relying on Plotly's previous-trace
        # fill behavior.
        band_x = xs + xs[::-1]
        band_y = upper + lower[::-1]

        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=fill_color, width=0),
                name=f"{mechanism_id} ±1 std",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        # MODIFIED:
        # Draw the mean after the shaded region so the mean line stays on top.
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line=dict(width=3, color=color),
                marker=dict(size=5, color=color),
                name=f"{mechanism_id} mean",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="x unified",
        template="plotly_white",
    )

    # MODIFIED: remove range slider from W&B panel.
    fig.update_xaxes(rangeslider_visible=False)

    wandb_run.log(
        {f"{prefix}/plots/{plot_name}": fig},
        step=step,
        commit=False,
    )

def _log_train_eval_return_by_mechanism_plot(
    *,
    wandb_run: Run,
    base_prefix: str,
    table: wandb.Table,
    x_col: str,
    y_col: str,
    series_col: str,
    phase_col: str,
    step: int,
    max_rows: int,
) -> None:
    table = _cap_table(table, max_rows)

    cols = list(table.columns)
    ix = cols.index(x_col)
    iy = cols.index(y_col)
    iser = cols.index(series_col)
    iphase = cols.index(phase_col)

    grouped: dict[str, dict[float, list[float]]] = {}

    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])
        series = row[iser]
        phase = str(row[iphase])

        if x is None or y is None:
            continue

        mechanism_id = _extract_mechanism_id(str(series))
        key = f"{phase}/{mechanism_id}"

        grouped.setdefault(key, {}).setdefault(float(x), []).append(float(y))

    if not grouped:
        return

    fig = go.Figure()

    colors = {
        "train": "rgba(31, 119, 180, 1.0)",
        "eval": "rgba(255, 127, 14, 1.0)",
    }
    fill_colors = {
        "train": "rgba(31, 119, 180, 0.25)",
        "eval": "rgba(255, 127, 14, 0.25)",
    }

    for key, step_to_values in sorted(grouped.items()):
        phase, mechanism_id = key.split("/", 1)

        xs = sorted(step_to_values.keys())
        means, uppers, lowers = [], [], []

        for x in xs:
            vals = np.asarray(step_to_values[x], dtype=np.float64)
            mean = float(vals.mean())
            std = float(vals.std())

            means.append(mean)
            uppers.append(mean + std)
            lowers.append(mean - std)

        band_x = xs + xs[::-1]
        band_y = uppers + lowers[::-1]

        color = colors.get(phase, "rgba(0, 0, 0, 1.0)")
        fill_color = fill_colors.get(phase, "rgba(0, 0, 0, 0.20)")

        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=fill_color, width=0),
                name=f"{phase} {mechanism_id} ±1 std",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line=dict(width=3, color=color),
                marker=dict(size=5, color=color),
                name=f"{phase} {mechanism_id} mean",
            )
        )

    fig.update_layout(
        title="Train vs eval return mean by mechanism ±1 std across seeds",
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(rangeslider_visible=False)

    wandb_run.log(
        {f"{base_prefix}/plots/returns/train_vs_eval_by_mechanism_mean_std": fig},
        step=step,
        commit=False,
    )

def extract_episode_metrics_newstack(
    results: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
        RLlib raw episode metric.

        This is useful as a sanity/debug metric, but NOT as the main mechanism
        comparison metric.

        In MARL, episode_return_mean is:

            mean over completed episodes of:
               sum over timesteps:
                   sum over agents:
                       reward_t_agent
        
        Therefore it scales with:
          - number of agents
          - episode length
          - reward scale
        
        It can also mix mechanisms/seeds if multiple envs are sampled together.
        
        Main reporting metric should come from extract_series_returns_newstack(),
        which returns reward_per_agent_per_step by module/policy.
    """
    
    env = results.get("env_runners", {}) or {}

    # new-stack first, fallback to old-stack names if present
    # N.B. this is the total return over episode (num_agents * sum(rollout_fragement_length return mean))/ num_env_runners !
    return_mean = finite(env.get("episode_return_mean"))
    if return_mean is None:
        return_mean = finite(env.get("episode_reward_mean"))

    return {
        "episode_return_mean": return_mean,
        "episode_return_min": finite(env.get("episode_return_min")),
        "episode_return_max": finite(env.get("episode_return_max")),
        "episode_len_mean": finite(env.get("episode_len_mean")),
        "episode_len_min": finite(env.get("episode_len_min")),
        "episode_len_max": finite(env.get("episode_len_max")),
        "num_episodes": finite(env.get("num_episodes")),
        "num_episodes_lifetime": finite(env.get("num_episodes_lifetime")),
    }


def extract_series_returns_newstack(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns reward_per_agent_per_step instead of raw episode return.

    Future-proof version:
    - fisher:* agents are counted for fisher_policy_*
    - merchant:* agents are counted for merchant_policy_*
    - etc.

    Formula:

        module_env_steps =
            num_module_steps_sampled[module_id] / num_agents_in_that_module

        reward_per_agent_per_step =
            module_episode_return_mean / module_env_steps
    """

    env = results.get("env_runners", {}) or {}

    module_steps = env.get("num_module_steps_sampled")
    module_rewards = env.get("module_episode_returns_mean")
    agent_steps = env.get("num_agent_steps_sampled")

    def _agent_type_from_agent_id(agent_id: str) -> str:
        # Example:
        #   fisher:0 -> fisher
        #   merchant:2 -> merchant
        return str(agent_id).split(":", 1)[0]

    def _agent_type_from_module_id(module_id: str) -> Optional[str]:
        # Example:
        #   fisher_policy_m0_s123 -> fisher
        #   merchant_policy_m0_s123 -> merchant
        module_id = str(module_id)

        if "_policy" in module_id:
            return module_id.split("_policy", 1)[0]

        return None

    def _count_agents_for_module(module_id: str) -> Optional[int]:
        if not isinstance(agent_steps, dict) or not agent_steps:
            return None

        module_agent_type = _agent_type_from_module_id(module_id)

        if module_agent_type is None:
            return None

        count = 0

        for agent_id in agent_steps.keys():
            agent_type = _agent_type_from_agent_id(str(agent_id))

            if agent_type == module_agent_type:
                count += 1

        return count if count > 0 else None

    if (
        isinstance(module_rewards, dict)
        and module_rewards
        and isinstance(module_steps, dict)
        and module_steps
    ):
        normalized = {}

        for module_id, episode_reward_mean in module_rewards.items():
            episode_reward_mean = finite(episode_reward_mean)
            steps_for_module = finite(module_steps.get(module_id))
            num_agents_in_module = _count_agents_for_module(str(module_id))

            if (
                episode_reward_mean is None
                or steps_for_module is None
                or num_agents_in_module is None
                or num_agents_in_module <= 0
            ):
                continue

            module_env_steps = steps_for_module / num_agents_in_module

            if module_env_steps <= 0:
                continue

            # MODIFIED:
            # Normalize cumulative module return into mean reward per agent
            # per environment step.
            reward_per_agent_per_step = (
                episode_reward_mean / module_env_steps
            )

            normalized[module_id] = reward_per_agent_per_step

        return normalized

    agent_rewards = env.get("agent_episode_returns_mean")

    if (
        isinstance(agent_rewards, dict)
        and agent_rewards
        and isinstance(agent_steps, dict)
        and agent_steps
    ):
        normalized = {}

        for agent_id, episode_return in agent_rewards.items():
            episode_return = finite(episode_return)
            steps_for_agent = finite(agent_steps.get(agent_id))

            if episode_return is None or steps_for_agent is None:
                continue

            if steps_for_agent <= 0:
                continue

            normalized[agent_id] = episode_return / steps_for_agent

        return normalized

    return {}


def extract_perf_newstack(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    env = results.get("env_runners", {}) or {}
    timers = results.get("timers", {}) or {}

    thr = env.get("num_env_steps_sampled_lifetime_throughput")
    throughput = None
    if isinstance(thr, dict):
        throughput = finite(thr.get("throughput_since_last_reduce")) or finite(
            thr.get("throughput_since_last_restore")
        )

    agent_steps = env.get("num_agent_steps_sampled")
    agent_steps_lt = env.get("num_agent_steps_sampled_lifetime")

    agent_steps_sum = None
    agent_steps_lt_sum = None
    if isinstance(agent_steps, dict):
        agent_steps_sum = finite(
            sum((to_float(v) or 0.0) for v in agent_steps.values())
        )
    if isinstance(agent_steps_lt, dict):
        agent_steps_lt_sum = finite(
            sum((to_float(v) or 0.0) for v in agent_steps_lt.values())
        )

    return {
        "env_steps_this_iter": finite(env.get("num_env_steps_sampled")),
        "env_steps_lifetime": finite(env.get("num_env_steps_sampled_lifetime")),
        "agent_steps_this_iter_sum": agent_steps_sum,
        "agent_steps_lifetime_sum": agent_steps_lt_sum,
        "env_steps_throughput": throughput,
        "training_iteration_s": finite(timers.get("training_iteration")),
        "training_step_s": finite(timers.get("training_step")),
        "sample_s": finite(timers.get("sample")),
        "learner_update_s": finite(timers.get("learner_update_timer")),
        "weights_seq_no": finite(env.get("weights_seq_no")),
    }


def extract_learner_metrics_newstack(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Returns:
      { learner_id: {metric_name: float_or_none, ...}, ... }
    where learner_id is something like 'fisher_policy_0', '__all_modules__', etc.
    """
    learners = results.get("learners", {}) or {}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not isinstance(learners, dict):
        return out

    learner_group = results.get("learner_group", {}) or {}
    mean_training_calls_since_sync = finite(
        results.get("mean_num_training_step_calls_since_last_synch_worker_weights")
    )
    outstanding_async_reqs = finite(
        learner_group.get("actor_manager_num_outstanding_async_reqs")
    )

    # Useful global proxy source from __all_modules__
    all_modules_stats = learners.get("__all_modules__", {}) or {}
    learner_queue_wait = finite(
        all_modules_stats.get("learner_thread_in_queue_wait_timer")
    )

    for learner_id, stats in learners.items():
        if not isinstance(stats, dict):
            continue

        m: Dict[str, Optional[float]] = {}

        # Log all scalar-ish keys automatically (scales as RLlib changes).
        # Skip nested dicts/lists, and skip NaN/inf.
        for k, v in stats.items():
            if isinstance(v, (dict, list, tuple)):
                continue
            fv = finite(v)
            if fv is None:
                continue
            m[str(k)] = fv

        # Optionally unpack throughput dict if present (nice to have)
        thr = stats.get("num_module_steps_trained_lifetime_throughput")
        if isinstance(thr, dict):
            m["module_steps_throughput_since_last_reduce"] = finite(
                thr.get("throughput_since_last_reduce")
            )
            m["module_steps_throughput_since_last_restore"] = finite(
                thr.get("throughput_since_last_restore")
            )

        # Derived metrics
        # policy_relative_entropy = entropy / entropy coeff
        entropy = m.get("entropy")
        entropy_coeff = m.get("curr_entropy_coeff")

        m["policy_relative_entropy"] = safe_ratio(entropy, entropy_coeff)

        # optional but often useful
        if finite(entropy) is not None and finite(entropy_coeff) is not None:
            m["entropy_pressure"] = float(entropy) * float(entropy_coeff)

        # sample staleness proxy
        lag1 = m.get("diff_num_grad_updates_vs_sampler_policy")
        lag2 = mean_training_calls_since_sync
        lag3 = outstanding_async_reqs
        lag4 = learner_queue_wait

        parts = [x for x in (lag1, lag2, lag3, lag4) if x is not None]
        m["sample_staleness"] = float(sum(parts)) if parts else None

        out[str(learner_id)] = m

    return out


# --------------------------
# W&B: plot caches
# --------------------------

# Returns plot table: one per run
_RETURNS_TABLES: dict[tuple[int, str], wandb.Table] = {}

_TRAIN_EVAL_RETURN_TABLES: dict[tuple[int, str], wandb.Table] = {}

# Learner metric plot tables: per run -> per metric -> table
# table columns: ["step", "outer_iter", "train_step", "policy", "value"]
_LEARNER_METRIC_TABLES: dict[tuple[int, str], dict[str, wandb.Table]] = {}

# Keys we never want to plot (even if whitelisted by substring)
_DEFAULT_SKIP_PLOT_KEYS = {
    "num_module_steps_trained",
    "num_module_steps_trained_lifetime",
    "num_trainable_parameters",
    "num_non_trainable_parameters",
    "weights_seq_no",
    "diff_num_grad_updates_vs_sampler_policy",
    "module_train_batch_size_mean",
    "module_steps_throughput_since_last_reduce",
    "module_steps_throughput_since_last_restore",
}

# Only make plots for these (substring match on lower-cased key).
# Keep small to avoid W&B UI spam.
_DEFAULT_LEARNER_PLOT_WHITELIST = {
    "total_loss",
    "policy_loss",
    "vf_loss",
    "entropy",
    # "curr_entropy_coeff",
    # "policy_relative_entropy",
    # "entropy_pressure",
    # "kl",
    # "curr_kl_coeff",
    # "vf_explained_var",
    # "grad_gnorm",
    "sample_staleness",
    # "lr",
    # "learning_rate",
    # "default_optimizer_learning_rate",
}


def _should_plot_metric(metric_key: str, whitelist: set[str]) -> bool:
    return str(metric_key).lower() in whitelist


def _table_to_line_series_arrays(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
) -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """
    Convert a wandb.Table into (xs, ys, keys) for wandb.plot.line_series(xs=..., ys=..., keys=...).

    The table can contain extra columns; we only use x_col/y_col/series_col.
    """
    if table is None or table.data is None:
        return [], [], []

    cols = list(table.columns)
    try:
        ix = cols.index(x_col)
        iy = cols.index(y_col)
        iser = cols.index(series_col)
    except ValueError as e:
        raise ValueError(
            f"Table missing required columns. Have={cols}, need={[x_col, series_col, y_col]}"
        ) from e

    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])
        s = row[iser]
        if x is None or y is None:
            continue
        key = str(s)
        grouped.setdefault(key, []).append((float(x), float(y)))

    keys = sorted(grouped.keys())
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    for k in keys:
        pts = sorted(grouped[k], key=lambda t: t[0])
        xs.append([p[0] for p in pts])
        ys.append([p[1] for p in pts])

    return xs, ys, keys


def _log_multiline_table_as_plot(
    *,
    wandb_run: Run,
    prefix: str,
    plot_name: str,
    table: wandb.Table,
    x_col: str,
    y_col: str,
    series_col: str,
    step: int,
    max_rows: int,
    title: str,
) -> None:
    table = _cap_table(table, max_rows)

    xs, ys, keys = _table_to_line_series_arrays(
        table,
        x_col=x_col,
        y_col=y_col,
        series_col=series_col,
    )

    payload: Dict[str, Any] = {}

    # Only create plot if we have data.
    if keys and xs and ys:
        payload[f"{prefix}/plots/{plot_name}"] = wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=title,
            xname=x_col,
        )

    wandb_run.log(payload, step=step, commit=False)


# --------------------------
# main logger
# --------------------------


def plot_training_results_new_stack(
    wandb_run: Run,
    *,
    outer_iter: int,
    training_episode: int,
    results: Dict[str, Any],
    prefix: str = "rllib",
    # plotting controls
    max_lines_returns: int = 64,
    max_rows_returns: int = 50_000,
    max_rows_per_learner_metric: int = 50_000,
    include_all_modules_in_learner_plots: bool = False,  # usually False
    skip_learner_plot_keys: Optional[set[str]] = None,
    learner_plot_whitelist: Optional[set[str]] = None,
    log_per_policy_learner_scalars: bool = False,
    learner_scalar_whitelist: Optional[set[str]] = None,
    log_per_series_return_scalars: bool = False,
    log_return_multiline_plot: bool = False,
    log_learner_multiline_plots: bool = False,
    log_mechanism_shaded_plots: bool = False,
    log_raw_rllib_episode_metrics: bool = False,
) -> None:
    """
    Logs:
      - scalar summaries (episode/perf)
      - per-series return scalars (optional)
      - OPTIONAL per-policy learner scalars (off by default to reduce UI noise)
      - Multi-line plots:
          * returns: one plot with lines=policies/agents
          * learner metrics: ONE plot per metric with lines=policies (WHITELISTED)

    MODIFIED:
      Also logs mechanism-level shaded error plots:
          * returns/by_mechanism_mean_std
          * learner/<metric_name>_by_mechanism_mean_std

      These aggregate seeded policies by mechanism id:
          fisher_policy_m0_s2669555309 -> m0
          fisher_policy_m0_s3444837047 -> m0
          fisher_policy_m1_s2669555309 -> m1
    """
    if wandb_run is None or results is None:
        return

    gs = _global_step(outer_iter, training_episode)
    eps = extract_episode_metrics_newstack(results)
    perf = extract_perf_newstack(results)
    series_means = extract_series_returns_newstack(results)
    learner_by_policy = extract_learner_metrics_newstack(results)

    metrics: Dict[str, Any] = {
        f"{prefix}/outer_iter": outer_iter,
        f"{prefix}/train_step": training_episode,
        f"{prefix}/perf/env_steps_this_iter": perf["env_steps_this_iter"],
        f"{prefix}/perf/env_steps_lifetime": perf["env_steps_lifetime"],
        f"{prefix}/perf/agent_steps_this_iter_sum": perf["agent_steps_this_iter_sum"],
        f"{prefix}/perf/agent_steps_lifetime_sum": perf["agent_steps_lifetime_sum"],
        f"{prefix}/perf/env_steps_throughput": perf["env_steps_throughput"],
        f"{prefix}/perf/training_iteration_s": perf["training_iteration_s"],
        f"{prefix}/perf/training_step_s": perf["training_step_s"],
        f"{prefix}/perf/sample_s": perf["sample_s"],
        f"{prefix}/perf/learner_update_s": perf["learner_update_s"],
        f"{prefix}/perf/weights_seq_no": perf["weights_seq_no"],
    }
    custom = (
        results.get("env_runners", {})
        .get("custom_metrics", {})
    )

    water_metrics = [
        "reservoir_stage_m",
        "reservoir_depth_m",
        "reservoir_level_norm",
        "max_depth_m",
        "total_usage_m3s",
    ]

    for metric in water_metrics:
        for suffix in ["mean", "min", "max"]:
            val = finite(custom.get(f"{metric}_{suffix}"))
            if val is not None:
                metrics[f"{prefix}/water/{metric}_{suffix}"] = val

    if log_raw_rllib_episode_metrics:
        metrics.update(
            {
                f"{prefix}/rllib_raw/episode_return_mean": eps["episode_return_mean"],
                f"{prefix}/rllib_raw/episode_return_min": eps["episode_return_min"],
                f"{prefix}/rllib_raw/episode_return_max": eps["episode_return_max"],
                f"{prefix}/rllib_raw/episode_len_mean": eps["episode_len_mean"],
                f"{prefix}/rllib_raw/episode_len_min": eps["episode_len_min"],
                f"{prefix}/rllib_raw/episode_len_max": eps["episode_len_max"],
                f"{prefix}/rllib_raw/num_episodes": eps["num_episodes"],
                f"{prefix}/rllib_raw/num_episodes_lifetime": eps["num_episodes_lifetime"],
            }
        )

    # MODIFIED: keep the aggregate summary, but stop logging one scalar chart per seeded policy
    if isinstance(series_means, dict) and series_means:
        series_ids = list(series_means.keys())[:max_lines_returns]

        if log_per_series_return_scalars:
            for sid in series_ids:
                fv = finite(series_means.get(sid))
                if fv is None:
                    continue
                sid_clean = sanitize_key(sid)
                metrics[f"{prefix}/series/reward_per_agent_per_step/{sid_clean}"] = fv

        summary = _summarize_dict_of_scalars(
            {sid: series_means.get(sid) for sid in series_ids}
        )
        for k, v in summary.items():
            metrics[f"{prefix}/series/reward_per_agent_per_step_{k}"] = v

    # OPTIONAL: per-policy learner scalars (OFF by default)
    if log_per_policy_learner_scalars and isinstance(learner_by_policy, dict):
        scalar_wl = learner_scalar_whitelist or _DEFAULT_LEARNER_PLOT_WHITELIST
        for learner_id, lm in (learner_by_policy or {}).items():
            learner_clean = sanitize_key(learner_id)
            for k, v in (lm or {}).items():
                if v is None:
                    continue
                if not _should_plot_metric(k, scalar_wl):
                    continue
                k_clean = sanitize_key(k)
                metrics[f"{prefix}/learner/{learner_clean}/{k_clean}"] = v

    # drop Nones
    metrics = {k: v for k, v in metrics.items() if v is not None}
    wandb_run.log(metrics, step=gs, commit=False)

    # Added: lossless per-policy EPISODE RETURN MEAN table for Schelling analysis.
    # This does not change any existing metrics or plots.
    from core.reporting.utils.schelling_archive import log_module_episode_returns
    log_module_episode_returns(
        wandb_run=wandb_run,
        results=results,
        outer_iter=outer_iter,
        train_step=training_episode,
        prefix=prefix,
    )

    # --------------------------
    # 2) MULTI-LINE PLOT: returns (ONE plot, lines=series)
    # --------------------------
    if isinstance(series_means, dict) and series_means:
        run_key = (id(wandb_run), prefix)
        t = _RETURNS_TABLES.get(run_key)
        if t is None:
            t = wandb.Table(
                columns=["step", "outer_iter", "train_step", "series", "value"]
            )
            _RETURNS_TABLES[run_key] = t

        series_ids = list(series_means.keys())[:max_lines_returns]
        for sid in series_ids:
            fv = finite(series_means.get(sid))
            if fv is None:
                continue
            t.add_data(
                int(gs), int(outer_iter), int(training_episode), str(sid), float(fv)
            )

        if prefix.endswith("/train") or prefix.endswith("/eval"):
            base_prefix = prefix.rsplit("/", 1)[0]
            phase = "eval" if prefix.endswith("/eval") else "train"

            combo_key = (id(wandb_run), base_prefix)
            combo_t = _TRAIN_EVAL_RETURN_TABLES.get(combo_key)

            if combo_t is None:
                combo_t = wandb.Table(
                    columns=[
                        "step",
                        "outer_iter",
                        "train_step",
                        "phase",
                        "series",
                        "value",
                    ]
                )
                _TRAIN_EVAL_RETURN_TABLES[combo_key] = combo_t

            for sid in series_ids:
                fv = finite(series_means.get(sid))
                if fv is None:
                    continue

                combo_t.add_data(
                    int(gs),
                    int(outer_iter),
                    int(training_episode),
                    phase,
                    str(sid),
                    float(fv),
                )

            if phase == "eval":
                _log_train_eval_return_by_mechanism_plot(
                    wandb_run=wandb_run,
                    base_prefix=base_prefix,
                    table=combo_t,
                    x_col="step",
                    y_col="value",
                    series_col="series",
                    phase_col="phase",
                    step=gs,
                    max_rows=max_rows_returns,
                )

        # MODIFIED: disabled by default to avoid W&B UI spam
        if log_return_multiline_plot:
            _log_multiline_table_as_plot(
                wandb_run=wandb_run,
                prefix=prefix,
                plot_name="returns/all_series_return_mean",
                table=t,
                x_col="step",
                y_col="value",
                series_col="series",
                step=gs,
                max_rows=max_rows_returns,
                title="Return mean (all policies/series)",
            )

        # MODIFIED: disabled by default
        if log_mechanism_shaded_plots and not (
            prefix.endswith("/train") or prefix.endswith("/eval")
        ):
            _log_mechanism_shaded_error_plot(
                wandb_run=wandb_run,
                prefix=prefix,
                plot_name="returns/by_mechanism_mean_std",
                table=t,
                x_col="step",
                y_col="value",
                series_col="series",
                step=gs,
                max_rows=max_rows_returns,
                title="Return mean by mechanism ±1 std across seeds",
            )

    # --------------------------
    # 3) MULTI-LINE PLOTS: learner metrics (ONE plot per metric; lines=policies)
    #     IMPORTANT: whitelist metrics to avoid UI spam
    # --------------------------
    skip_keys = set(skip_learner_plot_keys or set()) | _DEFAULT_SKIP_PLOT_KEYS
    plot_wl = set(learner_plot_whitelist or _DEFAULT_LEARNER_PLOT_WHITELIST)

    if isinstance(learner_by_policy, dict) and learner_by_policy:
        run_key = (id(wandb_run), prefix)
        per_metric = _LEARNER_METRIC_TABLES.setdefault(run_key, {})
        touched: set[str] = set()

        for learner_id, lm in learner_by_policy.items():
            if (
                not include_all_modules_in_learner_plots
            ) and learner_id == "__all_modules__":
                continue

            series_name = str(learner_id)
            for k, v in (lm or {}).items():
                if v is None:
                    continue
                if k in skip_keys:
                    continue
                if not _should_plot_metric(k, plot_wl):
                    continue  # <- key: do NOT plot every learner scalar

                metric_name = sanitize_key(k)
                t = per_metric.get(metric_name)
                if t is None:
                    t = wandb.Table(
                        columns=["step", "outer_iter", "train_step", "policy", "value"]
                    )
                    per_metric[metric_name] = t

                t.add_data(
                    int(gs),
                    int(outer_iter),
                    int(training_episode),
                    series_name,
                    float(v),
                )
                touched.add(metric_name)

        for metric_name in touched:
            t = per_metric[metric_name]

            # MODIFIED: disabled by default to avoid W&B UI spam
            if log_learner_multiline_plots:
                _log_multiline_table_as_plot(
                    wandb_run=wandb_run,
                    prefix=prefix,
                    plot_name=f"learner/{metric_name}",
                    table=t,
                    x_col="step",
                    y_col="value",
                    series_col="policy",
                    step=gs,
                    max_rows=max_rows_per_learner_metric,
                    title=f"{metric_name} (all policies)",
                )

            # MODIFIED: disabled by default
            if log_mechanism_shaded_plots:
                _log_mechanism_shaded_error_plot(
                    wandb_run=wandb_run,
                    prefix=prefix,
                    plot_name=f"learner/{metric_name}_by_mechanism_mean_std",
                    table=t,
                    x_col="step",
                    y_col="value",
                    series_col="policy",
                    step=gs,
                    max_rows=max_rows_per_learner_metric,
                    title=f"{metric_name} by mechanism ±1 std across seeds",
                )

    # finalize
    wandb_run.log({}, step=gs, commit=False)