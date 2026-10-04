"""Bilevel fishery experiment with the typed metrics/reporting stack.

The outer Evolution Strategies optimizer searches ``fixed_quota`` and
``restoration_subsidy``; the inner APPO optimizer trains the fishers against
each candidate. Every level logs a typed ``MetricSchema`` and renders the
queries of :mod:`examples.bilevel_fishery.queries` through the configured
reporter (Weights & Biases by default, CSV with ``--reporter csv``).

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \\
        --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20

Full configuration: the defaults.
"""

import argparse
from typing import TypeAlias

import numpy as np
import ray
from gymnasium import spaces

from core.adaptors.ray.schema import RaySchema
from core.agents.base import AgentConfig
from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.config.hash_seed import ensure_hash_seed
from core.mechanism.algorithms.quota import Quota
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.es.schema import ESSchema
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

# Before any work: the call restarts the process when the seed is unset.
ensure_hash_seed()


def _parse_args() -> argparse.Namespace:
    """Parse the size options of the experiment; the defaults are the full run."""
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
    return parser.parse_args()


ARGS = _parse_args()

REPORTER_CONFIG: ReporterConfig = (
    CSVConfig(project="bilevel")
    if ARGS.reporter == "csv"
    else WandbConfig(
        project="bilevel",
        x_disable_stats=True,
        x_disable_meta=True,
        quiet=True,
        max_end_of_run_summary_metrics=0,
        max_end_of_run_history_metrics=0,
    )
)

ray.shutdown()

EPS = 1e-8

FisheriesRegulator: TypeAlias = AgentConfig

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .reporter(config=REPORTER_CONFIG)
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
            episodes=ARGS.outer_iters,
        )
        .agents(
            FisheriesRegulator(
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
                    "K": 5_000,  # HAS to match environmnet K
                }
            },
        )
        .debugging(seed=42, num_seeds=1)
        .reporting(schema=ESSchema, queries=ES_QUERIES)
    )
    .society(
        APPOptimizerConfig()
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
            horizon=ARGS.horizon,
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
            episodes=ARGS.train_iters,
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
                count=ARGS.num_agents,
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
            schema=RaySchema,
            queries=INNER_QUERIES,
        )
    )
)

bilevel_opt = bilevel_opt_cfg.build_optimizer()

bilevel_opt.train()
ray.shutdown()
