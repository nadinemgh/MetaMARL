"""Bilevel fresh-water experiment: an ES regulator over a water policy, farms inside.

The script builds the experiment in Python and runs it. The inner level trains
a crowd of corn farms with APPO on ``WaterRegulatedEnv``: every day each farm
requests a share of its crop water deficit from one reservoir. The outer level
is an Evolution Strategies optimizer over the eight rules of the
``water_policy`` mechanism (a quota that tightens as the lake falls, fines for
requests above it and a penalty on large requests when the river depends on the
release, see :mod:`examples.fresh_water.mechanism`). The fitness of a candidate
mixes the farms' reward with how little the withdrawals change the river (see
:mod:`examples.fresh_water.regulator_env`). Every level logs a typed metric
schema and renders the queries of :mod:`examples.fresh_water.queries` through
the configured reporter.

The lake is the built-in surrogate by default. ``--hydrology raven`` runs the
external Raven model instead; its directory and executable are not part of the
repository, so they are given with ``--raven-cwd`` and ``--raven-cmd``.

Nothing runs at import. :func:`main` parses the options, calls
``ensure_hash_seed()`` (which may restart the process), builds the experiment
and trains it, so the module can be imported, and its doctests run, without side
effects.

Options (``--help`` prints the same list):

``--hydrology`` : ``{surrogate, raven}``, default ``surrogate``
    Lake model.
``--raven-cwd``, ``--raven-cmd`` : path
    Raven model directory and executable, required with ``--hydrology raven``.
``--raven-work-dir`` : path, optional
    Where the Raven run directories are created.
``--outer-iters`` : int, default 100
    ES generations of the outer level.
``--train-iters`` : int, default 200
    Inner training iterations for each generation.
``--horizon`` : int, default 150
    Episode length, in days.
``--num-agents`` : int, default 500
    Number of farms.
``--population`` : int, default 1
    Candidates per generation (1, or an even number).
``--reporter`` : ``{wandb, csv}``, default ``wandb``
    Where the queries are rendered. The CSV files are written under
    ``results/`` in the working directory.

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.fresh_water.debug \\
        --outer-iters 2 --train-iters 2 --horizon 10 --num-agents 4 \\
        --reporter csv
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.config.hash_seed import ensure_hash_seed
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.reporting.config import ReporterConfig
from core.reporting.csv import CSVConfig
from core.reporting.wandb import WandbConfig
from examples.fresh_water.mechanism import (
    DEFAULT_RULES,
    IRRIGATE_ID,
    OBSERVATION_SIZE,
    RULES_SPACE,
    WATER_POLICY_ID,
    WaterPolicyConfig,
    encode_rules,
)
from examples.fresh_water.metric_schema import WaterMetricSchema
from examples.fresh_water.queries import ES_QUERIES, INNER_QUERIES
from examples.fresh_water.regulated_env import (
    DEFAULT_ECOLOGY,
    IRRIGATE_SPACE,
    IrrigateConfig,
    UtilizerConfig,
    WaterRegulatedEnv,
)
from examples.fresh_water.regulator_env import WaterRegulatorEnv

PROJECT = "fresh_water"
"""Project name the reporters write under."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the options of the experiment; the defaults are the full run.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        The options listed in the module docstring.

    Raises
    ------
    SystemExit
        On an unknown option, or when ``--hydrology raven`` is given without
        ``--raven-cwd`` and ``--raven-cmd`` (the parser reports the error).

    When to use: from :func:`main`, or in a test to check the defaults.

    Examples
    --------
    >>> args = parse_args([])
    >>> args.hydrology, args.outer_iters, args.train_iters, args.horizon
    ('surrogate', 100, 200, 150)
    >>> args.num_agents, args.population, args.reporter
    (500, 1, 'wandb')
    >>> raven = ["--hydrology", "raven", "--raven-cwd", "m", "--raven-cmd", "x"]
    >>> parse_args(raven).raven_cwd
    PosixPath('m')
    """
    parser = argparse.ArgumentParser(description="Bilevel fresh-water experiment")
    parser.add_argument(
        "--hydrology", choices=("surrogate", "raven"), default="surrogate"
    )
    parser.add_argument("--raven-cwd", type=Path, default=None, help="Raven model dir")
    parser.add_argument("--raven-cmd", type=Path, default=None, help="Raven executable")
    parser.add_argument(
        "--raven-work-dir", type=Path, default=None, help="Raven run directories"
    )
    parser.add_argument(
        "--outer-iters", type=int, default=100, help="ES generations (outer level)"
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=200,
        help="inner training iterations per generation",
    )
    parser.add_argument("--horizon", type=int, default=150, help="episode days")
    parser.add_argument("--num-agents", type=int, default=500, help="number of farms")
    parser.add_argument(
        "--population", type=int, default=1, help="candidates per generation"
    )
    parser.add_argument(
        "--reporter",
        choices=("wandb", "csv"),
        default="wandb",
        help="where the queries are rendered",
    )
    args = parser.parse_args(argv)
    if args.hydrology == "raven" and (args.raven_cwd is None or args.raven_cmd is None):
        parser.error("--hydrology raven needs --raven-cwd and --raven-cmd")
    return args


def reporter_config(name: str) -> ReporterConfig:
    """Build the reporter configuration named on the command line.

    Parameters
    ----------
    name : str
        ``"csv"`` or ``"wandb"``.

    Returns
    -------
    ReporterConfig
        A ``CSVConfig`` writing under ``results/fresh_water`` of the working
        directory, or a ``WandbConfig`` for the project ``fresh_water``.

    When to use: from :func:`build_config`.

    Examples
    --------
    >>> type(reporter_config("csv")).__name__
    'CSVConfig'
    """
    if name == "csv":
        return CSVConfig(project=PROJECT)
    return WandbConfig(
        project=PROJECT,
        x_disable_stats=True,
        x_disable_meta=True,
        quiet=True,
        max_end_of_run_summary_metrics=0,
        max_end_of_run_history_metrics=0,
    )


def inner_env_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the ``env_config`` of the inner environment from the options.

    Parameters
    ----------
    args : argparse.Namespace
        Options returned by :func:`parse_args`.

    Returns
    -------
    dict
        Keyword arguments of :class:`examples.fresh_water.regulated_env.
        WaterRegulatedEnv`: the ecology constants of the first version of the
        example, the lake model and, for Raven, its paths as strings.

    When to use: from :func:`build_config`, or to check what the inner
    environment receives.

    Examples
    --------
    >>> config = inner_env_config(parse_args([]))
    >>> config["hydrology"], config["ecology_cfg"]["max_farm_area_m2"]
    ('surrogate', 1000000.0)
    >>> "raven_cwd" in config
    False
    """
    config: dict[str, Any] = {
        "ecology_cfg": dict(DEFAULT_ECOLOGY),
        "hydrology": args.hydrology,
        "seed": 0,
    }
    if args.hydrology == "raven":
        config["raven_cwd"] = str(args.raven_cwd)
        config["raven_cmd"] = str(args.raven_cmd)
        if args.raven_work_dir is not None:
            config["raven_work_dir"] = str(args.raven_work_dir)
    return config


def _inner_config(args: argparse.Namespace) -> APPOptimizerConfig:
    """Build the inner (society) optimizer configuration."""
    return (
        APPOptimizerConfig()
        .resources(num_cpus_for_main_process=1)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .environment(
            env=WaterRegulatedEnv,
            env_config=inner_env_config(args),
            horizon=args.horizon,
            disable_env_checking=False,
            schema=WaterMetricSchema,
            queries=(),
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            # One environment per candidate and seed.
            num_envs_per_env_runner=args.population,
            rollout_fragment_length=args.horizon,
            batch_mode="truncate_episodes",
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .callbacks(
            on_episode_created=tag_episode_with_env_idx,
            on_episode_end=log_and_report_episode_metrics,
        )
        .training(
            episodes=args.train_iters,
            vtrace=True,
            circular_buffer_num_batches=4,
            circular_buffer_iterations_per_batch=1,
            broadcast_interval=1,
            timeout_s_sampler_manager=10,
            timeout_s_aggregator_manager=10,
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=args.horizon,
            minibatch_size=args.horizon,
            num_epochs=1,
            entropy_coeff=0.001,
            grad_clip=40.0,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=0,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,  # greedy evaluation actions
                "rollout_fragment_length": args.horizon,
                "batch_mode": "complete_episodes",
            },
            base_seed=42,
            num_seeds=1,
        )
        .agents(
            UtilizerConfig(
                id="utilizer",
                policy_id="utilizer_policy",
                shared_policy=True,
                mechanisms=(
                    IrrigateConfig(id=IRRIGATE_ID, action_space=IRRIGATE_SPACE),
                ),
                count=args.num_agents,
                observation_space=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(OBSERVATION_SIZE,),
                    dtype=np.float32,
                ),
            )
        )
        .fault_tolerance(restart_failed_env_runners=False)
        .debugging(seed=42, num_seeds=1)
        .reporting(
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=0,
            min_train_timesteps_per_iteration=0,
            queries=INNER_QUERIES,
        )
    )


def build_config(args: argparse.Namespace) -> BilevelConfig:
    """Build the bilevel configuration of the fresh-water experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Options returned by :func:`parse_args`.

    Returns
    -------
    BilevelConfig
        The configuration; ``build_optimizer()`` starts Ray and builds the
        optimizers.

    When to use: from :func:`main`, or to inspect the configuration without
    running it.

    Examples
    --------
    >>> config = build_config(parse_args(["--reporter", "csv"]))
    >>> type(config).__name__
    'BilevelConfig'
    """
    regulator = AgentConfig(
        id="water_regulator",
        policy_id="water_policy_net",
        mechanisms=(
            WaterPolicyConfig(
                id=WATER_POLICY_ID,
                action_space=RULES_SPACE,
                acts_on=("utilizer", IRRIGATE_ID),
                default=encode_rules(DEFAULT_RULES).astype(np.float32),
            ),
        ),
    )
    return (
        BilevelConfig()
        .world(world_name="water_world")
        .reporter(config=reporter_config(args.reporter))
        .ray(
            device="cpu",
            num_cpus=4,
            omp_threads=1,
            logging_level="ERROR",
            runtime_env={
                "excludes": [
                    "wandb/",
                    ".git/",
                    ".venv/",
                    "__pycache__/",
                    "ray_results/",
                    "runs/",
                ]
            },
        )
        .regulator(
            ESConfig()
            .training(
                sigma=0.5,
                mean_lr=0.2,
                sigma_lr=0.05,
                min_sigma=0.01,
                max_sigma=0.6,
                episodes=args.outer_iters,
            )
            .agents(regulator)
            .environment(
                horizon=1,
                env=WaterRegulatorEnv,
                env_config={
                    "ecology_cfg": {
                        "economic_weight": 1.0,
                        "sustainability_weight": 1.0,
                        # The inflow is upstream of the reservoir: the farms
                        # cannot change it, so the deviation is read on the
                        # outflow, the river below the dam.
                        "deviation_series": "outflow",
                        "aggregation_status": "train",
                    }
                },
            )
            .debugging(seed=42, num_seeds=1)
            .reporting(queries=ES_QUERIES)
        )
        .society(_inner_config(args))
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the options, build the experiment and train it.

    ``ensure_hash_seed()`` runs first: when ``PYTHONHASHSEED`` is unset it
    restarts the process with the same command line. The root logger is then
    set to ``INFO`` so that the progress of the optimizers is printed.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line without the program name; ``None`` reads ``sys.argv``.

    When to use: as the body of the script. It starts Ray and runs the full
    bilevel loop, so it has no doctest.
    """
    ensure_hash_seed()
    # The library no longer configures the root logger, so the script does it to
    # show the progress lines of the optimizers.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args(argv)

    ray.shutdown()
    bilevel_opt = build_config(args).build_optimizer()
    bilevel_opt.train()
    ray.shutdown()


if __name__ == "__main__":
    main()
