"""Bilevel cart-pole experiment: an ES regulator over an inert dial, RL agent inside.

The script builds the experiment in Python and runs it. The inner level trains
one agent on ``CartpoleRegulatedEnv`` (Gymnasium's ``CartPole-v1``) with PPO or
APPO. The outer level is an Evolution Strategies optimizer over the single
parameter of the inert ``dial`` mechanism, so the experiment checks a whole
bilevel run on a task with a known solution: the fitness of a candidate is the
mean per-step reward of the agent, which is the constant ``1.0`` because every
step of ``CartPole-v1`` pays ``1.0``. The run is a pipeline check in which the
ES gets no signal to follow. Every level logs a typed metric schema and
renders the queries of :mod:`examples.cartpole.queries` through the configured
reporter.

Nothing runs at import. :func:`main` parses the options, calls
``ensure_hash_seed()`` (which may restart the process), builds the experiment
and trains it, so the module can be imported, and its doctests run, without side
effects. ``examples.cartpole.main_ppo`` and ``examples.cartpole.main_appo`` are
the entry points of the two algorithms; both call :func:`main`.

Options (``--help`` prints the same list):

``--algo`` : ``{ppo, appo}``, default ``appo``
    Inner algorithm.
``--outer-iters`` : int, default 100
    ES generations of the outer level.
``--train-iters`` : int, default 100 for PPO and 200 for APPO
    Inner training iterations for each generation.
``--horizon`` : int, default 1000
    Episode length limit, in steps (Gymnasium's own limit is 500).
``--reporter`` : ``{wandb, csv}``, default ``wandb``
    Where the queries are rendered. The CSV files are written under
    ``results/`` in the working directory.

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.cartpole.debug \\
        --algo appo --outer-iters 2 --train-iters 2 --horizon 20 --reporter csv

Full configurations: the defaults, through ``main_ppo`` and ``main_appo``.
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
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.config import ReporterConfig
from core.reporting.csv import CSVConfig
from core.reporting.wandb import WandbConfig
from examples.cartpole.metric_schema import CartpoleMetricSchema
from examples.cartpole.queries import ES_QUERIES, INNER_QUERIES
from examples.cartpole.regulated_env import (
    DIAL_ID,
    DIAL_SPACE,
    OBSERVATION_SIZE,
    PUSH_ID,
    PUSH_SPACE,
    CartpoleAgentConfig,
    CartpoleRegulatedEnv,
    DialConfig,
    PushConfig,
)
from examples.cartpole.regulator_env import CartpoleRegulatorEnv

PROJECT = "cartpole"
"""Project name the reporters write under."""

DEFAULT_TRAIN_ITERS = {"ppo": 100, "appo": 200}
"""Inner training iterations per generation of the two full configurations."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the size options of the experiment; the defaults are the full run.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        The options, with ``train_iters`` resolved to the default of the chosen
        algorithm when it was not given.

    When to use: from :func:`main`, or in a test to check the defaults.

    Examples
    --------
    >>> args = parse_args(["--algo", "ppo"])
    >>> args.algo, args.outer_iters, args.train_iters, args.horizon, args.reporter
    ('ppo', 100, 100, 1000, 'wandb')
    >>> parse_args([]).train_iters
    200
    """
    parser = argparse.ArgumentParser(description="Bilevel cart-pole experiment")
    parser.add_argument("--algo", choices=("ppo", "appo"), default="appo")
    parser.add_argument(
        "--outer-iters", type=int, default=100, help="ES generations (outer level)"
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=None,
        help="inner iterations per generation (default 100 for PPO, 200 for APPO)",
    )
    parser.add_argument(
        "--horizon", type=int, default=1000, help="episode length limit, in steps"
    )
    parser.add_argument(
        "--reporter",
        choices=("wandb", "csv"),
        default="wandb",
        help="where the queries are rendered",
    )
    args = parser.parse_args(argv)
    if args.train_iters is None:
        args.train_iters = DEFAULT_TRAIN_ITERS[args.algo]
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
        A ``CSVConfig`` writing under ``results/cartpole`` of the working
        directory, or a ``WandbConfig`` for the project ``cartpole``.

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


def _inner_config(args: argparse.Namespace):
    """Build the inner (society) optimizer configuration of the chosen algorithm."""
    if args.algo == "ppo":
        inner = PPOptimizerConfig()
        training = dict(
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=4000,
            minibatch_size=512,
            entropy_coeff=0.01,
            grad_clip=40.0,
        )
        env_runners = dict(rollout_fragment_length=1000, batch_mode="complete_episodes")
        env_config: dict = {}
    else:
        inner = APPOptimizerConfig()
        training = dict(
            vtrace=True,
            circular_buffer_num_batches=2,
            circular_buffer_iterations_per_batch=1,
            broadcast_interval=5,
            timeout_s_sampler_manager=300,
            timeout_s_aggregator_manager=300,
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=200,
            minibatch_size=200,
            entropy_coeff=0.01,
            grad_clip=40.0,
        )
        env_runners = dict(rollout_fragment_length=200, batch_mode="truncate_episodes")
        env_config = {"seed": 0}

    return (
        inner.resources(num_cpus_for_main_process=1)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .environment(
            env=CartpoleRegulatedEnv,
            env_config=env_config,
            horizon=args.horizon,
            disable_env_checking=False,
            schema=CartpoleMetricSchema,
            queries=(),
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            # One environment: the dial population is a single candidate.
            num_envs_per_env_runner=1,
            **env_runners,
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .callbacks(
            on_episode_created=tag_episode_with_env_idx,
            on_episode_end=log_and_report_episode_metrics,
        )
        .training(episodes=args.train_iters, **training)
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
            CartpoleAgentConfig(
                id="agent_0",
                policy_id="cartpole_policy",
                mechanisms=(PushConfig(id=PUSH_ID, action_space=PUSH_SPACE),),
                count=1,
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
    """Build the bilevel configuration of the cart-pole experiment.

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
    >>> config = build_config(parse_args(["--algo", "ppo", "--reporter", "csv"]))
    >>> type(config).__name__
    'BilevelConfig'
    """
    regulator = AgentConfig(
        id="cartpole_regulator",
        policy_id="dial_policy",
        mechanisms=(DialConfig(id=DIAL_ID, action_space=DIAL_SPACE),),
    )
    return (
        BilevelConfig()
        .world(world_name="cartpole_world")
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
                env=CartpoleRegulatorEnv,
                # The fitness reads the inner training episodes.
                env_config={"aggregation_status": "train"},
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

    When to use: as the body of a script; the entry points ``main_ppo`` and
    ``main_appo`` call it with the algorithm fixed. It starts Ray and runs the
    full bilevel loop, so it has no doctest.
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
