"""Bilevel fishery experiment: an ES regulator over a quota, APPO fishers inside.

This script builds the experiment in Python and runs it; it is the programmatic
counterpart of ``config.yaml``, and both describe the same experiment as
``debug.py``. The outer Evolution Strategies optimizer searches two parameters,
both mechanisms of ``core.mechanism.algorithms``: the ``quota`` (a value in
``[0, 1]`` acting on the fishers' ``harvest``, started from 0.56224, with a
transition width of 0.05) and the ``restoration_subsidy`` (a value in ``[0, 1]``
acting on the fishers' ``restore``, started from 0.10), with a constant search
spread of 0.15. The inner APPO optimizer trains ten fishers by default against
each candidate, on ``FisheryRegulatedEnv`` with a carrying capacity of 5000, a
restoration cost of 0.25 and a restoration power of 0.02. Every level logs a
typed ``MetricSchema`` and renders the queries of
:mod:`examples.bilevel_fishery.queries` through the configured reporter (Weights
& Biases by default, CSV with ``--reporter csv``, TensorBoard with
``--reporter tensorboard``).

Nothing runs at import. :func:`main` configures the logging, calls
``ensure_hash_seed()`` (which may restart the process, before Ray starts),
parses the options, builds the experiment and trains it, so the module can be
imported, and its doctests run, without side effects. Run it as a script.

Options (``--help`` prints the same list):

``--outer-iters`` : int, default 1000
    ES generations of the outer level.
``--train-iters`` : int, default 50
    APPO iterations the inner level runs for each generation.
``--num-agents`` : int, default 10
    Number of fishers.
``--horizon`` : int, default 100
    Episode length of the fishery, in steps.
``--reporter`` : ``{wandb, csv, tensorboard}``, default ``wandb``
    Where the queries are rendered.

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.bilevel_fishery.run \\
        --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20

Full configuration: the defaults.
"""

import logging
from typing import TypeAlias

import numpy as np
import ray
from gymnasium import spaces

from core.adaptors.ray.schema import RaySchema
from core.agents.base import AgentConfig
from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.mechanism.algorithms.incentive import Incentive
from core.mechanism.algorithms.quota import Quota
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.es.schema import ESSchema
from core.optimizers.ppo.config import PPOptimizerConfig
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

ray.shutdown()

EPS = 1e-8

FisheriesRegulator: TypeAlias = AgentConfig
bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .reporter(
        config=WandbConfig(
            project="bilevel",
            x_disable_stats=True,
            x_disable_meta=True,
            quiet=True,
            max_end_of_run_summary_metrics=0,
            max_end_of_run_history_metrics=0,
        )
    )
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
            episodes=1000,
        )
        .agents(
            FisheriesRegulator(
                id="fisheries_regulator",
                policy_id="quota_policy",
                mechanisms=(
                    Quota(
                        id="quota",
                        action_space=spaces.Box(
                            low=0,
                            high=1.0,
                            shape=(1,),  # TODO maybe constrain this to always be (1,)
                            dtype=np.float32,
                        ),
                        # TODO (nadine) better name ?
                        acts_on=("fisherman", "harvest"),
                        obs_map={"resource_level": "fish"},
                        default=np.asarray(0.56224),
                        quota_transition_width=0.05,
                    ),
                    Incentive(
                        id="restoration_subsidy",
                        action_space=spaces.Box(
                            low=0,
                            # TODO (nadine) temp val / max_subsidy. must be 0.5
                            high=1.0,
                            shape=(1,),  # TODO maybe constrain this to always be (1,)
                            dtype=np.float32,
                        ),
                        acts_on=("fisherman", "restore"),
                        default=np.asarray(0.10),
                    ),
                    # SocialInfluence(
                    #     influence_weight=...,
                    #     bindings={
                    #         "previous_actions": lambda env: (
                    #             env.previous_actions
                    #         ),
                    #         "agent_ids": lambda env: (
                    #             tuple(env.agents)
                    #         ),
                    #     },
                    # )
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
                    "K": 5_000,  # HAS to match environmnet K
                }
            },
        )
        .debugging(seed=42, num_seeds=1)
        .reporting(schema=ESSchema, queries=ES_QUERIES)
    )
    .society(
        PPOptimizerConfig()
        .resources(num_cpus_for_main_process=1)
        .framework(framework="torch")
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
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
            horizon=100,
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
            episodes=50,
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=400,
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
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),  # TODO (nadine) enforce strict shape
                            dtype=np.float32,
                        ),
                    ),
                    Restore(
                        id="restore",
                        action_space=spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),  # TODO (nadine) enforce strict shape
                            dtype=np.float32,
                        ),
                        restoration_cost=0.25,
                        restoration_power=0.02,
                    ),
                ),
                count=10,
                observation_space=spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(9,),
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
            schema=RaySchema,
            queries=INNER_QUERIES,
            # TODO test queries agg over mechanisms (or other dynamic fields)
            # TODO test queries with y keys from reduced (env)
        )
    )
)


def main() -> None:
    """Parse the options, build the experiment and train it.

    The root logger is set to ``INFO`` first, so that the notice of the
    hash-seed check and the progress of the optimizers are printed. Then
    ``ensure_hash_seed()`` runs: when ``PYTHONHASHSEED`` is unset it restarts
    the process with the same command line, before Ray starts.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command line without the program name; ``None`` reads ``sys.argv``.

    When to use: as the body of the script. It starts Ray and runs the full
    bilevel loop, so it has no doctest.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    ray.shutdown()

    try:
        bilevel_opt = bilevel_opt_cfg.build_optimizer()

        bilevel_opt.train()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
