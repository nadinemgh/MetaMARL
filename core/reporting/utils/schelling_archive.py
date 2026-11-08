"""Per-step Schelling archives for W&B and durable compressed CSV files.

The old logger reused a mutable W&B Table while outer_iter stayed at 1.
W&B froze the first JSON but kept increasing its nrows metadata.

Every call here creates a *new*, immutable W&B Table containing only the
current training step. In addition, the same rows are written to independent
CSV.gz files under <wandb_run.dir>/analysis_archives/.

Two independent measurements:
  * RLlib module_episode_returns_mean: RLlib's reported module-level episode
    return mean (it is not necessarily an individual episode sample).
  * EnvStepContext: exact reward sum over the supplied contexts, along with
    assigned policy group and observed tagging behavior.

No changes to policy training, rewards, environment evolution, or evaluation.
"""
from __future__ import annotations

import csv
import gzip
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import wandb

from core.world.context import Context, EnvStepContext

_MODULE_ID = re.compile(r"(?:^|_)a(?P<agent>\d+)_m(?P<mech>\d+)_s(?P<policy_seed>\d+)(?:_|$)")
_MODULE_MECH = re.compile(r"(?:^|_)m(?P<mech>\d+)(?:_|$)")

_MODULE_COLUMNS = [
    "outer_iter", "train_step", "phase", "policy_id", "agent_id",
    "mechanism", "policy_seed", "module_episode_return_mean",
    "module_steps_sampled",
]
_ROLLOUT_COLUMNS = [
    "outer_iter", "train_step", "phase", "mechanism", "seed",
    "policy_seed", "env_id", "agent_id", "assigned_group",
    "observed_group", "tag_steps", "tag_observations",
    "num_fixed_non_taggers", "rollout_steps", "complete_horizon",
    "rollout_return", "rollout_reward_mean",
]


def _global_step(outer_iter: int, train_step: int) -> int:
    return int(outer_iter) * 1_000_000 + int(train_step)


def _valid_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _mechanism_id(value: Any) -> str:
    if value is None:
        return "unknown"
    s = str(value)
    return s if s.startswith("m") else f"m{s}"


def _tag_choice(action: Any) -> bool | None:
    """Match WaterRegulatedEdHsEnv._wants_tag: Box(2) uses action[1] > 0."""
    if isinstance(action, dict):
        if "tag" not in action:
            return None
        try:
            return int(np.asarray(action["tag"]).reshape(-1)[0]) == 1
        except (ValueError, TypeError, IndexError):
            return None
    if action is None:
        return None
    try:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        return bool(arr[1] > 0) if arr.size == 2 else None
    except (ValueError, TypeError):
        return None


def _info_value(info: Any, agent_id: str, key: str) -> Any:
    if not isinstance(info, dict):
        return None
    agent_info = info.get(agent_id)
    if agent_info is None:
        agent_info = next((v for k, v in info.items() if str(k) == agent_id), None)
    return agent_info.get(key) if isinstance(agent_info, dict) else None


def archive_csv_rows(
    *,
    wandb_run: Any,
    prefix: str,
    dataset: str,
    outer_iter: int,
    train_step: int,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> Path | None:
    """Write an independently recoverable, per-step CSV.gz, atomically.

    W&B's `run.dir` normally points to `offline-run-.../files`. The resulting
    files are intentionally *additional* to the W&B Table JSON media. Include
    `files/analysis_archives/` when copying the run off the cluster.
    """
    run_dir = getattr(wandb_run, "dir", None)
    if not run_dir:
        return None  # Some unit-test stubs / alternate W&B adapters lack .dir.

    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix))
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset))
    dest_dir = Path(run_dir) / "analysis_archives" / safe_prefix / safe_dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / f"outer_{int(outer_iter):04d}_train_{int(train_step):06d}.csv.gz"
    temp_path = destination.with_name(destination.name + f".{os.getpid()}.tmp")

    try:
        with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def _log_per_step_table(
    *,
    wandb_run: Any,
    prefix: str,
    name: str,
    columns: list[str],
    rows: list[list[Any]],
    outer_iter: int,
    train_step: int,
) -> None:
    if not rows:
        return

    archive_csv_rows(
        wandb_run=wandb_run,
        prefix=prefix,
        dataset=name,
        outer_iter=outer_iter,
        train_step=train_step,
        columns=columns,
        rows=rows,
    )
    # New object on every call: never reuse, cache, or mutate a logged Table.
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*row)

    wandb_run.log(
        {f"{prefix}/tables/{name}": table},
        step=_global_step(outer_iter, train_step),
        commit=False,
    )


def log_module_episode_returns(
    *,
    wandb_run: Any,
    results: Mapping[str, Any],
    outer_iter: int,
    train_step: int,
    prefix: str = "rllib",
) -> None:
    """Archive RLlib's unmodified per-module episode-return means for this step."""
    if wandb_run is None or not results:
        return
    er = results.get("env_runners", {}) or {}
    returns = er.get("module_episode_returns_mean") or {}
    if not isinstance(returns, dict) or not returns:
        return

    steps = er.get("num_module_steps_sampled") or {}
    phase = "eval" if prefix.endswith("/eval") else "train"
    rows: list[list[Any]] = []
    for module_id, value in sorted(returns.items()):
        episode_return = _valid_float(value)
        if episode_return is None:
            continue
        module_id = str(module_id)
        match = _MODULE_ID.search(module_id)
        mech_match = _MODULE_MECH.search(module_id)
        agent_id = f"a{match.group('agent')}" if match else "unknown"
        mechanism = f"m{match.group('mech')}" if match else (
            f"m{mech_match.group('mech')}" if mech_match else "unknown"
        )
        policy_seed = match.group("policy_seed") if match else "unknown"
        module_steps = _valid_float(steps.get(module_id)) if isinstance(steps, dict) else None
        rows.append([
            int(outer_iter), int(train_step), phase, module_id,
            agent_id, mechanism, policy_seed, episode_return, module_steps,
        ])

    _log_per_step_table(
        wandb_run=wandb_run,
        prefix=prefix,
        name="module_episode_returns",
        columns=_MODULE_COLUMNS,
        rows=rows,
        outer_iter=outer_iter,
        train_step=train_step,
    )


def log_agent_rollout_returns(
    *,
    wandb_run: Any,
    ctxs: Iterable[Context],
    outer_iter: int,
    train_step: int,
    prefix: str = "env_reduced",
    expected_horizon: int | None = 100,
    agent_groups: Mapping[str, str] | None = None,
) -> None:
    """Archive exact reward sums over supplied contexts and observed tagging.

    These are complete episode returns only if the supplied contexts represent
    a complete, single episode for each agent (check your sampling setup).
    """
    if wandb_run is None or ctxs is None:
        return

    records: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for ctx in ctxs:
        payload = getattr(ctx, "payload", None)
        if not isinstance(payload, EnvStepContext):
            continue
        rewards = getattr(payload, "reward", None)
        if not isinstance(rewards, dict):
            continue

        raw_status = getattr(payload, "status", None)
        phase = str(getattr(raw_status, "value", raw_status or "train"))
        mechanism = _mechanism_id(getattr(payload, "mechanism", None))
        seed = str(getattr(payload, "seed", None))
        policy_seed = str(getattr(payload, "policy_seed", None))
        env_id = str(getattr(payload, "env_id", None))
        actions = getattr(payload, "action", None)
        actions = actions if isinstance(actions, dict) else {}
        infos = getattr(payload, "info", None)

        for agent, reward in rewards.items():
            reward = _valid_float(reward)
            if reward is None:
                continue
            agent_id = str(agent)
            key = (phase, mechanism, seed, policy_seed, env_id, agent_id)
            row = records.setdefault(key, {
                "return_sum": 0.0, "steps": 0, "tag_steps": 0,
                "tag_observations": 0, "fixed_non_taggers": None,
            })
            row["return_sum"] += reward
            row["steps"] += 1

            action = actions.get(agent)
            if action is None:
                action = next((v for k, v in actions.items() if str(k) == agent_id), None)
            tag = _tag_choice(action)
            if tag is not None:
                row["tag_observations"] += 1
                row["tag_steps"] += int(tag)

            fixed = _valid_float(_info_value(infos, agent_id, "n_fixed_non_taggers"))
            if fixed is not None:
                row["fixed_non_taggers"] = int(fixed)

    if not records:
        return

    rows: list[list[Any]] = []
    for (phase, mechanism, seed, policy_seed, env_id, agent_id), v in sorted(records.items()):
        tag_obs = v["tag_observations"]
        observed_group = (
            "tagged" if v["tag_steps"] > 0 else "never_tagged"
        ) if tag_obs else "unknown"
        assigned_group = (
            (agent_groups or {}).get(agent_id)
            or (
                "fixed_non_tagger"
                if agent_id.startswith("utilizer_non_tagging:")
                else "tagging_capable"
                if agent_id.startswith("utilizer:")
                else "unknown"
            )
        )
        n = v["steps"]
        rows.append([
            int(outer_iter), int(train_step), phase, mechanism, seed,
            policy_seed, env_id, agent_id, assigned_group, observed_group,
            int(v["tag_steps"]), int(tag_obs), v["fixed_non_taggers"],
            int(n), (n == expected_horizon) if expected_horizon is not None else None,
            float(v["return_sum"]), float(v["return_sum"] / n),
        ])

    _log_per_step_table(
        wandb_run=wandb_run,
        prefix=prefix,
        name="per_agent_rollout_returns",
        columns=_ROLLOUT_COLUMNS,
        rows=rows,
        outer_iter=outer_iter,
        train_step=train_step,
    )
