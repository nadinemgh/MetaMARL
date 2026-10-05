"""Helpers translating RLlib result dictionaries into typed metric schemas.

The metric getters accept both the new API stack layout (``env_runners/...``)
and the classic one (top-level ``episode_reward_mean``, ``timesteps_total``,
``info/learner/...``), returning a neutral default when a key is absent.
``hash_weights`` fingerprints a weights structure, ``parse_learner_id`` splits
an RLModule name into policy, mechanism and seed, and the ``build_*``
functions turn a full result dict into the ``RolloutSchema``,
``PerformanceSchema`` and ``LearnerSchema`` consumed by the reporting layer.
``RayOptimizer`` is their only caller: it feeds each ``Algorithm.train()`` or
``Algorithm.evaluate()`` result through them before logging it.
"""

import hashlib
from collections.abc import Iterable
from typing import Optional, get_args

import numpy as np
import torch
from ray.rllib.utils.typing import ResultDict

from core.adaptors.ray.schema import (
    LearnerSchema,
    MechanismLearnerSchema,
    MechanismRolloutSchema,
    PerformanceSchema,
    PolicyLearnerSchema,
    RolloutSchema,
    SeedLearnerSchema,
    SeedRolloutSchema,
)
from core.envs.schema import EpisodeRolloutSchema
from core.types import MechanismID, PolicyID, SeedID
from core.utils import finite, safe_ratio, to_float


def _get_env(result: dict) -> dict:
    """Return the ``env_runners`` sub-dict of a result, or ``{}``."""

    return result.get("env_runners", {}) or {}


def _first_present(*values: Optional[float]) -> Optional[float]:
    """Return the first value that is not ``None``, keeping a ``0.0`` as is.

    ``a or b`` would skip a legitimate ``0.0``; the result getters need to
    tell a measured zero from a missing key.
    """

    return next((value for value in values if value is not None), None)


def get_episode_return_mean(result: dict) -> float:
    """Extract the mean episode return from an RLlib result.

    Looks at ``env_runners/episode_return_mean`` first (new API stack), then
    at the legacy top-level ``episode_reward_mean`` and finally at
    ``env_runners/episode_reward_mean``. The first key that holds a number
    wins, a value of ``0.0`` included.

    Parameters
    ----------
    result : dict
        Result of ``Algorithm.train()``.

    Returns
    -------
    float
        The mean episode return in reward units, or ``0.0`` if no key is
        present or the value cannot be converted.

    When to use: to log a single training-progress number per inner
    iteration, whichever RLlib API stack produced the result.

    Examples
    --------
    >>> get_episode_return_mean({"env_runners": {"episode_return_mean": 1.5}})
    1.5
    >>> get_episode_return_mean({"episode_reward_mean": 2.0})
    2.0
    >>> get_episode_return_mean({})
    0.0
    """

    env = _get_env(result)
    v = to_float(env.get("episode_return_mean"))

    if v is not None:
        return v

    v = _first_present(
        to_float(result.get("episode_reward_mean")),
        to_float(env.get("episode_reward_mean")),
    )

    return v if v is not None else 0.0


def get_env_steps(result: dict) -> tuple[int, int]:
    """Extract environment step counters from an RLlib result.

    Parameters
    ----------
    result : dict
        Result of ``Algorithm.train()``.

    Returns
    -------
    tuple[int, int]
        ``(steps_this_iteration, steps_lifetime)`` in environment steps, read
        from ``env_runners/num_env_steps_sampled[_lifetime]`` with the legacy
        ``timesteps_this_iter`` / ``timesteps_total`` as fallback; ``0`` when
        missing.

    When to use: to report how much experience an inner iteration consumed
    and how much the algorithm has consumed in total.

    Examples
    --------
    >>> result = {
    ...     "env_runners": {
    ...         "num_env_steps_sampled": 400,
    ...         "num_env_steps_sampled_lifetime": 1200,
    ...     }
    ... }
    >>> get_env_steps(result)
    (400, 1200)
    >>> get_env_steps({"timesteps_this_iter": 8, "timesteps_total": 80})
    (8, 80)
    >>> get_env_steps({})
    (0, 0)
    """

    env = _get_env(result)
    steps_iter = _first_present(
        to_float(env.get("num_env_steps_sampled")),
        to_float(result.get("timesteps_this_iter")),
    )
    steps_life = _first_present(
        to_float(env.get("num_env_steps_sampled_lifetime")),
        to_float(result.get("timesteps_total")),
    )

    return int(steps_iter or 0), int(steps_life or 0)


def get_policy_loss_if_present(result: dict) -> float:
    """Average ``policy_loss`` across policies, if the result exposes it.

    Reads the new API stack layout ``learners/<module>/policy_loss`` (the
    ``__all_modules__`` entry carries no loss), and falls back to the classic
    ``info/learner/<policy>/learner_stats/policy_loss`` when the new layout
    yields no finite loss. On the new layout, non-finite losses are skipped:
    RLlib's IMPALA/APPO learner reduces its stats only once every 20 gradient
    updates (``IMPALALearner.update`` in Ray 2.53), and the module entries of
    the iterations in between hold NaN. Those iterations return NaN and the
    training log prints ``policy_loss=NA``. The classic layout is filtered the
    same way, so a NaN loss of one policy does not poison the mean.

    Parameters
    ----------
    result : dict
        Result of ``Algorithm.train()``.

    Returns
    -------
    float
        Mean of the per-policy losses (dimensionless loss units), or ``nan``
        when none is found.

    When to use: to track one training-loss number per inner iteration
    without caring how many policies the society trains.

    Examples
    --------
    >>> result = {
    ...     "learners": {
    ...         "__all_modules__": {},
    ...         "fisher_m0_s1": {"policy_loss": -1.0},
    ...         "fisher_m1_s1": {"policy_loss": -2.0},
    ...     }
    ... }
    >>> get_policy_loss_if_present(result)
    -1.5
    >>> get_policy_loss_if_present({})
    nan
    """

    learners = result.get("learners") or {}
    losses = [
        v
        for module_id, stats in learners.items()
        if module_id != "__all_modules__" and isinstance(stats, dict)
        if (v := finite(stats.get("policy_loss"))) is not None
    ]

    if losses:
        return float(np.mean(losses))

    learner_info = (result.get("info") or {}).get("learner") or {}

    if isinstance(learner_info, dict):
        for _, policy_stats in learner_info.items():
            ls = (policy_stats or {}).get("learner_stats") or {}
            v = finite(ls.get("policy_loss"))

            if v is not None:
                losses.append(v)

    return float(np.mean(losses)) if losses else float("nan")


def hash_weights(weights: dict) -> str:
    """Compute a SHA-256 fingerprint of a (nested) weights structure.

    Used to check that ``PolicyActor.reset`` restores identical parameters
    across outer iterations and across runs. Shapes and dtypes are not
    hashed separately: they only influence the hash through the raw bytes.

    Parameters
    ----------
    weights : dict or torch.Tensor or numpy.ndarray or Any
        Typically the nested dict returned by ``Algorithm.get_weights()``.
        Dict keys are visited in sorted order so the hash is independent of
        insertion order; each leaf is hashed together with its ``/``-joined
        key path.

    Returns
    -------
    str
        Hex digest (64 characters). Tensors are moved to CPU and hashed by
        raw bytes, arrays likewise, and any other leaf by its ``repr``.

    When to use: to compare two sets of network parameters for exact
    equality, for example before and after a policy reset, without keeping
    both sets in memory.

    Examples
    --------
    >>> import numpy as np
    >>> a = {"layer": {"w": np.arange(3, dtype=np.float32)}}
    >>> b = {"layer": {"w": np.arange(3, dtype=np.float32)}}
    >>> hash_weights(a) == hash_weights(b)
    True
    >>> len(hash_weights(a))
    64
    >>> hash_weights(a) == hash_weights({"layer": {"w": np.ones(3, np.float32)}})
    False
    """

    h = hashlib.sha256()

    def update(obj, prefix=""):
        """Recursively feed ``obj`` into the running hash under ``prefix``."""

        if isinstance(obj, dict):
            for key in sorted(obj):
                update(obj[key], f"{prefix}/{key}")
        elif isinstance(obj, torch.Tensor):
            array = obj.detach().cpu().contiguous().numpy()

            h.update(prefix.encode())
            h.update(array.tobytes())
        elif isinstance(obj, np.ndarray):
            h.update(prefix.encode())
            h.update(np.ascontiguousarray(obj).tobytes())
        else:
            h.update(prefix.encode())
            h.update(repr(obj).encode())

    update(weights)

    return h.hexdigest()


def parse_learner_id(learner_id: str) -> tuple[PolicyID, MechanismID, SeedID]:
    """Split an RLModule name into its policy, mechanism and seed.

    ``RayOptimizerConfig`` names every RLModule
    ``<policy>_m<mechanism_idx>_s<policy_seed>``. The split starts from the
    right, so the policy part may itself contain ``_m`` or ``_s``.

    Parameters
    ----------
    learner_id : str
        Module name, for example ``"fisher_policy_m0_s123"``.

    Returns
    -------
    tuple[str, str, str]
        ``(policy_id, mechanism_id, policy_seed)``, all strings (the
        mechanism index and the seed are not converted to integers).

    Raises
    ------
    ValueError
        If ``learner_id`` does not have the ``<policy>_m<mechanism>_s<seed>``
        form.

    When to use: to file the learner statistics of a module under its
    mechanism and seed, as ``build_learner`` does.

    Examples
    --------
    >>> parse_learner_id("fisher_policy_m0_s123")
    ('fisher_policy', '0', '123')
    >>> parse_learner_id("fisher")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: Expected learner ID of the form '<policy>_m<mechanism>_s<seed>', ...
    """

    try:
        policy_and_mechanism, policy_seed = learner_id.rsplit("_s", 1)
        policy_id, mechanism_id = policy_and_mechanism.rsplit("_m", 1)
    except ValueError:
        raise ValueError(
            "Expected learner ID of the form '<policy>_m<mechanism>_s<seed>', "
            + f"got {learner_id!r}."
        ) from None

    return (policy_id, mechanism_id, policy_seed)


def build_episode_aggregate(results: ResultDict) -> EpisodeRolloutSchema:
    """Aggregate episode return and length statistics from ``env_runners``.

    Fields that RLlib does not provide at the aggregate level (total and
    terminal rewards, terminal values) are left ``None``, and so is any value
    that is missing or not finite.

    Parameters
    ----------
    results : ResultDict
        Result of ``Algorithm.train()`` or ``Algorithm.evaluate()``; only the
        ``env_runners`` block is read.

    Returns
    -------
    EpisodeRolloutSchema
        Mean, minimum and maximum episode return (reward units), episode
        length statistics (steps) and episode counts.

    When to use: to build the ``aggregate`` entry of a ``RolloutSchema``; use
    ``build_rollout`` when the per-episode breakdown is needed too.

    Examples
    --------
    >>> result = {"env_runners": {"episode_return_mean": 1.5, "num_episodes": 3}}
    >>> aggregate = build_episode_aggregate(result)
    >>> aggregate.reward_mean, aggregate.num_episodes, aggregate.reward_min
    (1.5, 3, None)
    """

    env = results.get("env_runners", {}) or {}

    return EpisodeRolloutSchema(
        reward_total=None,
        reward_mean=finite(env.get("episode_return_mean")),
        reward_min=finite(env.get("episode_return_min")),
        reward_max=finite(env.get("episode_return_max")),
        reward_terminal=None,
        value_terminal=None,
        value_penultimate=None,
        episode_len_mean=finite(env.get("episode_len_mean")),
        episode_len_min=finite(env.get("episode_len_min")),
        episode_len_max=finite(env.get("episode_len_max")),
        num_episodes=finite(env.get("num_episodes")),
        num_episodes_lifetime=finite(env.get("num_episodes_lifetime")),
    )


def build_performance(results: ResultDict) -> PerformanceSchema:
    """Collect step counters, throughput and timers into a ``PerformanceSchema``.

    Agent step counts, reported per agent by RLlib, are summed; throughput is
    read from the ``since_last_reduce`` window and falls back to
    ``since_last_restore``. A missing, non-finite or non-numeric value leaves
    its field ``None``.

    Parameters
    ----------
    results : ResultDict
        Result of ``Algorithm.train()`` or ``Algorithm.evaluate()``; the
        ``env_runners`` and ``timers`` blocks are read.

    Returns
    -------
    PerformanceSchema
        Step counters (steps), throughput (steps per second) and timers
        (seconds).

    When to use: to track the cost of an inner iteration alongside its
    learning metrics.

    Examples
    --------
    >>> result = {
    ...     "env_runners": {
    ...         "num_env_steps_sampled": 100,
    ...         "num_agent_steps_sampled": {"fisher:0": 60, "fisher:1": 40},
    ...     },
    ...     "timers": {"sample": 0.5},
    ... }
    >>> performance = build_performance(result)
    >>> performance.env_steps_this_iter, performance.agent_steps_this_iter_sum
    (100.0, 100.0)
    >>> performance.sample_s
    0.5
    """

    env = results.get("env_runners", {}) or {}
    timers = results.get("timers", {}) or {}
    throughput_data = env.get("num_env_steps_sampled_lifetime_throughput")
    throughput = None

    if isinstance(throughput_data, dict):
        throughput = _first_present(
            finite(throughput_data.get("throughput_since_last_reduce")),
            finite(throughput_data.get("throughput_since_last_restore")),
        )

    agent_steps = env.get("num_agent_steps_sampled")
    agent_steps_lifetime = env.get("num_agent_steps_sampled_lifetime")
    agent_steps_sum = None
    agent_steps_lifetime_sum = None

    if isinstance(agent_steps, dict):
        agent_steps_sum = finite(
            sum((to_float(value) or 0.0) for value in agent_steps.values())
        )

    if isinstance(agent_steps_lifetime, dict):
        agent_steps_lifetime_sum = finite(
            sum((to_float(value) or 0.0) for value in agent_steps_lifetime.values())
        )

    return PerformanceSchema(
        env_steps_this_iter=finite(env.get("num_env_steps_sampled")),
        env_steps_lifetime=finite(env.get("num_env_steps_sampled_lifetime")),
        agent_steps_this_iter_sum=agent_steps_sum,
        agent_steps_lifetime_sum=agent_steps_lifetime_sum,
        env_steps_throughput=throughput,
        training_iteration_s=finite(timers.get("training_iteration")),
        training_step_s=finite(timers.get("training_step")),
        sample_s=finite(timers.get("sample")),
        learner_update_s=finite(timers.get("learner_update_timer")),
        weights_seq_no=finite(env.get("weights_seq_no")),
    )


def build_rollout(results: ResultDict) -> RolloutSchema:
    """Group the ``env_runners/by_episode`` entries by mechanism and seed.

    Each ``by_episode`` entry is the list of ``EpisodeRolloutSchema`` objects
    that ``log_and_report_episode_metrics`` appended under one episode prefix
    during the iteration (empty when that sub-environment ended no episode).
    RLlib's ``MetricsLogger.compile`` (Ray 2.53) unwraps a list of one item,
    so a lone ``EpisodeRolloutSchema`` stands for a list holding it. The
    episode at position ``n`` of the list for prefix ``<prefix>`` is filed
    under ``by_mechanism[<mechanism_id>].by_seed[<seed>]
    .by_episode["<prefix>|n=<n>"]``, alongside the aggregate from
    ``build_episode_aggregate``. The mechanism ID and the seed are converted to
    strings.

    Parameters
    ----------
    results : ResultDict
        Result of ``Algorithm.train()`` or ``Algorithm.evaluate()``; the
        ``env_runners`` block is read.

    Returns
    -------
    RolloutSchema
        The aggregate statistics and the per-mechanism, per-seed,
        per-episode breakdown (empty when the result has no ``by_episode``).

    Raises
    ------
    TypeError
        If a ``by_episode`` entry is neither a list nor an
        ``EpisodeRolloutSchema``.

    When to use: to turn an RLlib result into the structure from which the
    regulator computes one fitness per candidate mechanism.

    Examples
    --------
    >>> first = EpisodeRolloutSchema(mechanism_id=0, seed=11, reward_mean=2.0)
    >>> second = EpisodeRolloutSchema(mechanism_id=0, seed=11, reward_mean=4.0)
    >>> prefix = "env=0|m=0|ps=11|ss=11"
    >>> result = {"env_runners": {"by_episode": {prefix: [first, second]}}}
    >>> rollout = build_rollout(result)
    >>> by_episode = rollout.by_mechanism["0"].by_seed["11"].by_episode
    >>> {key: episode.reward_mean for key, episode in by_episode.items()}
    {'env=0|m=0|ps=11|ss=11|n=0': 2.0, 'env=0|m=0|ps=11|ss=11|n=1': 4.0}

    A prefix that ended a single episode arrives unwrapped:

    >>> result = {"env_runners": {"by_episode": {prefix: first}}}
    >>> by_episode = build_rollout(result).by_mechanism["0"].by_seed["11"].by_episode
    >>> list(by_episode)
    ['env=0|m=0|ps=11|ss=11|n=0']
    """

    env = results.get("env_runners", {}) or {}
    episodes = env.get("by_episode", {}) or {}

    by_mechanism: dict[MechanismID, MechanismRolloutSchema] = {}

    for prefix, prefix_episodes in episodes.items():
        if isinstance(prefix_episodes, EpisodeRolloutSchema):
            prefix_episodes = [prefix_episodes]
        elif not isinstance(prefix_episodes, list):
            raise TypeError(
                f"by_episode entry {prefix!r} is a {type(prefix_episodes).__name__}, "
                + "expected a list of EpisodeRolloutSchema."
            )

        for index, episode in enumerate(prefix_episodes):
            mechanism_id = str(episode.mechanism_id)
            seed = str(episode.seed)
            mechanism = by_mechanism.setdefault(mechanism_id, MechanismRolloutSchema())
            seed_rollout = mechanism.by_seed.setdefault(seed, SeedRolloutSchema())
            seed_rollout.by_episode[f"{prefix}|n={index}"] = episode

    return RolloutSchema(
        aggregate=build_episode_aggregate(results), by_mechanism=by_mechanism
    )


def _missing_as_nan(metrics: PolicyLearnerSchema) -> PolicyLearnerSchema:
    """Return a copy of ``metrics`` whose unset float statistics are NaN."""

    missing = {
        name: float("nan")
        for name, field in type(metrics).model_fields.items()
        if getattr(metrics, name) is None and float in get_args(field.annotation)
    }

    return metrics.model_copy(update=missing)


def build_learner(results: ResultDict, module_ids: Iterable[str] = ()) -> LearnerSchema:
    """Build one ``PolicyLearnerSchema`` per module in ``results["learners"]``.

    Besides copying the finite scalar stats, it derives
    ``policy_relative_entropy`` (entropy / entropy coefficient),
    ``entropy_pressure`` (entropy * entropy coefficient) and
    ``sample_staleness`` (sum of the available lag indicators: gradient-update
    lag, training calls since the last weight sync, outstanding async requests
    and learner queue wait).

    Every module named in ``module_ids`` gets an entry even when the result
    has none for it, and every float statistic it lacks is NaN instead of
    unset. APPO reduces its learner statistics only once every 20 gradient
    updates (``IMPALALearner.update`` in Ray 2.53): on the other iterations
    ``learners`` is empty, or holds the modules with NaN values. The metric
    logger skips unset values, so without this a learner series would be
    shorter than the iteration axis and could not be plotted against it;
    with it, every series keeps one value per iteration and shows a gap
    where nothing was measured. Modules absent from ``module_ids`` keep
    their missing statistics unset.

    Parameters
    ----------
    results : ResultDict
        Result of ``Algorithm.train()``; the ``learners`` and ``learner_group``
        blocks and the legacy lag counter are read. Module names must follow
        ``<policy>_m<mechanism>_s<seed>`` (see ``parse_learner_id``).
    module_ids : iterable of str, optional
        Learner IDs that must appear in the output even when the result has
        no entry for them. Defaults to none.

    Returns
    -------
    LearnerSchema
        Statistics grouped as ``by_mechanism -> by_seed -> by_policy``.

    Raises
    ------
    ValueError
        If a module name in the result or in ``module_ids`` does not have the
        ``<policy>_m<mechanism>_s<seed>`` form.

    When to use: to type the learner block of a result before logging it,
    passing the modules declared in the RLlib config so that every series has
    one value per iteration.

    Examples
    --------
    >>> import math
    >>> result = {
    ...     "learners": {
    ...         "__all_modules__": {},
    ...         "fisher_m0_s1": {
    ...             "policy_loss": 0.5,
    ...             "entropy": 2.0,
    ...             "curr_entropy_coeff": 0.01,
    ...         },
    ...     }
    ... }
    >>> learner = build_learner(result)
    >>> stats = learner.by_mechanism["0"].by_seed["1"].by_policy["fisher"]
    >>> stats.policy_loss, stats.policy_relative_entropy
    (0.5, 200.0)
    >>> learner = build_learner({}, module_ids=["fisher_m0_s1"])
    >>> stats = learner.by_mechanism["0"].by_seed["1"].by_policy["fisher"]
    >>> math.isnan(stats.policy_loss)
    True
    """

    learners = results.get("learners", {}) or {}
    learner_group = results.get("learner_group", {}) or {}
    mean_training_calls_since_sync = finite(
        results.get("mean_num_training_step_calls_since_last_synch_worker_weights")
    )
    outstanding_async_reqs = finite(
        learner_group.get("actor_manager_num_outstanding_async_reqs")
    )
    all_modules_stats = learners.get("__all_modules__", {}) or {}
    learner_queue_wait = finite(
        all_modules_stats.get("learner_thread_in_queue_wait_timer")
    )

    by_mechanism: dict[MechanismID, MechanismLearnerSchema] = {}
    module_stats = {
        learner_id: stats
        for learner_id, stats in learners.items()
        if learner_id != "__all_modules__"
    }
    expected = set(module_ids)

    for module_id in module_ids:
        module_stats.setdefault(module_id, {})

    for learner_id, stats in module_stats.items():
        policy_id, mechanism_id, policy_seed = parse_learner_id(learner_id)

        m: dict[str, Optional[float]] = {}

        for key, value in stats.items():
            value = finite(value)

            if value is None:
                continue

            m[str(key)] = value

        # Legacy optionally unpacked this throughput dict.
        throughput = stats.get("num_module_steps_trained_lifetime_throughput")

        if isinstance(throughput, dict):
            m["module_steps_throughput_since_last_reduce"] = finite(
                throughput.get("throughput_since_last_reduce")
            )
            m["module_steps_throughput_since_last_restore"] = finite(
                throughput.get("throughput_since_last_restore")
            )

        entropy = m.get("entropy")
        entropy_coeff = m.get("curr_entropy_coeff")
        m["policy_relative_entropy"] = safe_ratio(entropy, entropy_coeff)

        if entropy is not None and entropy_coeff is not None:
            m["entropy_pressure"] = float(entropy) * float(entropy_coeff)

        lag1 = m.get("diff_num_grad_updates_vs_sampler_policy")
        lag2 = mean_training_calls_since_sync
        lag3 = outstanding_async_reqs
        lag4 = learner_queue_wait
        parts = [value for value in (lag1, lag2, lag3, lag4) if value is not None]
        m["sample_staleness"] = float(sum(parts)) if parts else None
        policy_metrics = PolicyLearnerSchema(
            batch_size=m.get("module_train_batch_size_mean"),
            total_loss=m.get("total_loss"),
            residual_variance=None,
            sample_staleness=m.get("sample_staleness"),
            policy_loss=m.get("policy_loss"),
            policy_entropy=m.get("entropy"),
            policy_entropy_coeff=m.get("curr_entropy_coeff"),
            policy_relative_entropy=m.get("policy_relative_entropy"),
            entropy_pressure=m.get("entropy_pressure"),
            policy_kl=m.get("kl"),
            policy_kl_coeff=m.get("curr_kl_coeff"),
            value_loss=m.get("vf_loss"),
            value_mean=m.get("value_mean"),
            value_target=m.get("value_target"),
            gradient_norm=_first_present(
                m.get("gradients_default_optimizer_global_norm"), m.get("grad_gnorm")
            ),
            gradient_noise=m.get("gradient_noise"),
        )

        if learner_id in expected:
            policy_metrics = _missing_as_nan(policy_metrics)

        mechanism = by_mechanism.setdefault(mechanism_id, MechanismLearnerSchema())
        seed = mechanism.by_seed.setdefault(policy_seed, SeedLearnerSchema())
        seed.by_policy[policy_id] = policy_metrics

    return LearnerSchema(by_mechanism=by_mechanism)
