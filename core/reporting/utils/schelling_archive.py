"""Minimal, non-invasive W&B archives for per-agent Schelling experiments.

Two independent input sources:
  - RLlib results: module_episode_returns_mean, already an EPISODE RETURN MEAN.
  - EnvStepContext batch: exact reward sum of the provided rollout, plus tag decisions.

No changes to RLlib training, evaluation, rewards, or existing plots.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Iterable, Mapping

import numpy as np
import wandb

from core.world.context import EnvStepContext, Context

_MODULE_ID = re.compile(r"(?:^|_)a(?P<agent>\d+)_m(?P<mech>\d+)_s(?P<policy_seed>\d+)(?:_|$)")
_MODULE_MECH = re.compile(r"(?:^|_)m(?P<mech>\d+)(?:_|$)")

# Separate buffers per W&B run / prefix / outer iteration. This avoids a very
# large single W&B Table when many outer iterations are run.
_MODULE_TABLES: dict[tuple[str, str, int], wandb.Table] = {}
_ROLLOUT_TABLES: dict[tuple[str, str, int], wandb.Table] = {}


def _run_key(wandb_run: Any) -> str:
    return str(getattr(wandb_run, "id", None) or id(wandb_run))


def _global_step(outer_iter: int, train_step: int) -> int:
    return int(outer_iter) * 1_000_000 + int(train_step)


def _valid_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mechanism_id(value: Any) -> str:
    if value is None:
        return "unknown"
    s = str(value)
    return s if s.startswith("m") else f"m{s}"


def _tag_choice(action: Any) -> bool | None:
    """Match WaterRegulatedEdHsEnv._wants_tag: Box(2) uses action[1] > 0.

    A Box(1) or other unrecognized action has no tag signal, not False.
    """
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
        # Agent IDs may be non-string keys in the original dict.
        agent_info = next((v for k, v in info.items() if str(k) == agent_id), None)
    if not isinstance(agent_info, dict):
        return None
    return agent_info.get(key)


def log_module_episode_returns(
    *,
    wandb_run: Any,
    results: Mapping[str, Any],
    outer_iter: int,
    train_step: int,
    prefix: str = "rllib",
) -> None:
    """Archive *unmodified* module_episode_returns_mean per policy.

    These are RLlib rolling/completed-episode means, NOT a list of individual
    episode samples. In the one-episode-per-policy training setup they can equal
    the episode's return; consult rollout archive for exact reward sums.
    """
    if wandb_run is None or not results:
        return
    er = results.get("env_runners", {}) or {}
    returns = er.get("module_episode_returns_mean") or {}
    if not isinstance(returns, dict) or not returns:
        return

    key = (_run_key(wandb_run), prefix, int(outer_iter))
    table = _MODULE_TABLES.get(key)
    if table is None:
        table = wandb.Table(columns=[
            "outer_iter", "train_step", "phase", "policy_id", "agent_id",
            "mechanism", "policy_seed", "module_episode_return_mean",
            "module_steps_sampled",
        ])
        _MODULE_TABLES[key] = table

    steps = er.get("num_module_steps_sampled") or {}
    phase = "eval" if prefix.endswith("/eval") else "train"
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
        table.add_data(
            int(outer_iter), int(train_step), phase, module_id,
            agent_id, mechanism, policy_seed, episode_return,
            module_steps,
        )

    wandb_run.log(
        {f"{prefix}/tables/module_episode_returns_outer_{outer_iter}": table},
        step=_global_step(outer_iter, train_step), commit=False,
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
    """Archive sums of ACTUAL per-agent rewards from EnvStepContext.

    One row per agent, mechanism, env, phase, and reporting iteration.
    `rollout_return` is an exact per-agent sum over the supplied contexts.
    If rollout_steps == expected_horizon and sampling is one episode per
    iteration, it is a complete episode return. If not, it remains a segment.

    `observed_group` is behavioral (tagged at least once vs never tagged).
    `assigned_group` is an optional experiment-design label; *only* use it to
    compare learned taggers vs fixed non-taggers. Do not equate never-tagged
    behavior with being assigned to a fixed non-tagger policy.
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

    key = (_run_key(wandb_run), prefix, int(outer_iter))
    table = _ROLLOUT_TABLES.get(key)
    if table is None:
        table = wandb.Table(columns=[
            "outer_iter", "train_step", "phase", "mechanism", "seed",
            "policy_seed", "env_id", "agent_id", "assigned_group",
            "observed_group", "tag_steps", "tag_observations",
            "num_fixed_non_taggers", "rollout_steps", "complete_horizon",
            "rollout_return", "rollout_reward_mean",
        ])
        _ROLLOUT_TABLES[key] = table

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
        table.add_data(
            int(outer_iter), int(train_step), phase, mechanism, seed,
            policy_seed, env_id, agent_id, assigned_group, observed_group,
            int(v["tag_steps"]), int(tag_obs), v["fixed_non_taggers"],
            int(n), (n == expected_horizon) if expected_horizon is not None else None,
            float(v["return_sum"]), float(v["return_sum"] / n),
        )

    wandb_run.log(
        {f"{prefix}/tables/per_agent_rollout_returns_outer_{outer_iter}": table},
        step=_global_step(outer_iter, train_step), commit=False,
    )
