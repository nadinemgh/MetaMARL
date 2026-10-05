"""Bilevel fishery experiment: an ES regulator over a quota, APPO fishers inside.

This script builds the experiment in Python and runs it; it is the programmatic
counterpart of ``config.yaml``. The outer Evolution Strategies optimizer
searches one parameter, the ``quota`` mechanism of ``core.mechanism.algorithms``
(a value in ``[0, 1]`` acting on the fishers' ``harvest``, started from 0.56224),
with a constant search spread of 0.15. The inner APPO optimizer trains ten
fishers by default against each candidate, on ``FisheryRegulatedEnv`` with a
carrying capacity of 5000. Every level logs a typed ``MetricSchema`` and
renders the queries of :mod:`examples.bilevel_fishery.queries` through the
configured reporter (Weights & Biases by default, CSV with ``--reporter csv``).

Nothing runs at import. :func:`main` calls ``ensure_hash_seed()`` (which may
restart the process, before Ray starts), configures the logging, parses the
options, builds the experiment and trains it, so the module can be imported,
and its doctests run, without side effects. Run it as a script.

Options (``--help`` prints the same list):

``--outer-iters`` : int, default 1000
    ES generations of the outer level.
``--train-iters`` : int, default 50
    APPO iterations the inner level runs for each generation.
``--num-agents`` : int, default 10
    Number of fishers.
``--horizon`` : int, default 100
    Episode length of the fishery, in steps.
``--reporter`` : ``{wandb, csv}``, default ``wandb``
    Where the queries are rendered.

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \\
        --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20

Full configuration: the defaults.
"""

import argparse
import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import ray
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.config.hash_seed import ensure_hash_seed
from core.mechanism.algorithms.quota import Quota
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.reporting.config import ReporterConfig
from core.reporting.csv import CSVConfig
from core.reporting.wandb import WandbConfig
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from examples.bilevel_fishery.queries import (
    ES_QUERIES,
    # FISHERY_ENV_QUERIES,
    INNER_QUERIES,
)
from examples.bilevel_fishery.regulated_env import FishermanConfig as Fisherman
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulated_env import FishingConfig as Fishing
from examples.bilevel_fishery.regulated_env import RestoreConfig as Restore
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the size options of the experiment; the defaults are the full run.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        The options ``outer_iters``, ``train_iters``, ``num_agents``,
        ``horizon`` and ``reporter``.

    When to use: from :func:`main`, or in a test to check the defaults.

    Examples
    --------
    >>> args = parse_args(["--reporter", "csv", "--horizon", "20"])
    >>> args.outer_iters, args.train_iters, args.num_agents, args.horizon
    (1000, 50, 10, 20)
    >>> args.reporter
    'csv'
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--outer-iters", type=int, default=1000, help="ES generations (outer level)"
    )
    parser.add_argument(
        "--train-iters", type=int, default=50, help="APPO iterations per generation"
    )
    parser.add_argument(
        "--num-agents", type=int, default=10, help="number of fishermen"
    )
    parser.add_argument(
        "--horizon", type=int, default=100, help="episode length of the fishery"
    )
    parser.add_argument(
        "--reporter",
        choices=("wandb", "csv"),
        default="wandb",
        help="where the queries are rendered",
    )
    return parser.parse_args(argv)


def reporter_config(name: str) -> ReporterConfig:
    """Build the reporter configuration named on the command line.

    Parameters
    ----------
    name : {"wandb", "csv"}
        ``"csv"`` writes the queries as CSV files; any other value builds the
        Weights & Biases reporter.

    Returns
    -------
    ReporterConfig
        The configuration of the reporter of every level.

    When to use: from :func:`build_config`.

    Examples
    --------
    >>> type(reporter_config("csv")).__name__
    'CSVConfig'
    """
    if name == "csv":
        return CSVConfig(project="bilevel")
    return WandbConfig(
        project="bilevel",
        x_disable_stats=True,
        x_disable_meta=True,
        quiet=True,
        max_end_of_run_summary_metrics=0,
        max_end_of_run_history_metrics=0,
    )


def build_config(args: argparse.Namespace) -> BilevelConfig:
    """Build the bilevel configuration of the fishery experiment.

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
    return (
        BilevelConfig()
        .world(world_name="fishery_world")
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
                sigma=0.15,
                mean_lr=0.10,
                sigma_decay=1.0,
                sigma_lr=0.00,
                min_sigma=0.15,
                max_sigma=0.15,
                episodes=args.outer_iters,
            )
            .agents(
                AgentConfig(
                    id="fisheries_regulator",
                    policy_id="quota_policy",
                    mechanisms=(
                        Quota(
                            id="quota",
                            action_space=spaces.Box(
                                low=0, high=1.0, shape=(1,), dtype=np.float32
                            ),
                            acts_on=("fisherman", "harvest"),
                            obs_map={"resource_level": "fish"},
                            default=np.asarray(0.56224),
                        ),
                    ),
                )
            )
            .environment(
                horizon=1,
                env=FisheryRegulatorEnv,
                env_config={
                    "ecology_cfg": {
                        "sustainability_weight": 2,  # assert between 0 and 5
                        "sustainability_threshold": 0.20,
                    }
                },
            )
            .debugging(seed=42, num_seeds=1)
            .reporting(queries=ES_QUERIES)
        )
        .society(
            APPOptimizerConfig()
            .resources(num_cpus_for_main_process=1)
            .framework(framework="torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(
                env=FisheryRegulatedEnv,
                env_config={
                    "ecology_cfg": {
                        # Pella-Tomlinson / Schaefer single-stock dynamics
                        "r": 0.3,
                        "K": 5_000,
                        "p": 1.0,
                        "B0": 4_000,
                        "fish_init": 4_000,
                        # Env stochasticity
                        "sigma": 0.02,
                        "initial_stock_log_sigma": 0.05,
                        "unregulated_f_multiplier": 2.0,
                    },
                    "seed": 0,
                },
                horizon=args.horizon,
                disable_env_checking=False,
                schema=FisheryMetricSchema,
                queries=(),
            )
            .env_runners(
                num_env_runners=0,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
                num_envs_per_env_runner=4,
                rollout_fragment_length=100,
                batch_mode="truncate_episodes",
                max_requests_in_flight_per_env_runner=1,
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
                train_batch_size_per_learner=100,
                minibatch_size=100,
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
                    "explore": False,
                    "rollout_fragment_length": 100,
                    "batch_mode": "complete_episodes",
                    "max_requests_in_flight_per_env_runner": 1,
                },
                base_seed=42,
                num_seeds=3,
            )
            .agents(
                Fisherman(
                    id="fisherman",
                    policy_id="fisher_policy",
                    shared_policy=True,
                    mechanisms=(
                        Fishing(
                            id="harvest",
                            action_space=spaces.Box(
                                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                            ),
                        ),
                        Restore(
                            id="restore",
                            action_space=spaces.Box(
                                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                            ),
                        ),
                    ),
                    count=args.num_agents,
                    observation_space=spaces.Box(
                        low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
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
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the options, build the experiment and train it.

    ``ensure_hash_seed()`` runs first: when ``PYTHONHASHSEED`` is unset it
    restarts the process with the same command line, before Ray starts. The
    root logger is then set to ``INFO`` so that the progress of the optimizers
    is printed.

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
