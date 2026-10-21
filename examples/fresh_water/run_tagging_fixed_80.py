"""Narval experiment: tagging_fixed_80.

Tagging treatment: 20 tag-capable, 80 fixed non-taggers; all learn extraction.

Execute: python -m examples.fresh_water.run_tagging_fixed_80
"""

import os

import numpy as np
import ray
from gymnasium import spaces

from core.callbacks import tag_episode_with_env_idx
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from examples.fresh_water.mechanism import WaterMechanismSpace
from examples.fresh_water.regulated_env_ed_hs_v4_banning import WaterRegulatedEdHsEnv
from examples.fresh_water.regulator_env_raven import WaterRegulatorRavenEnv


def build_config() -> BilevelConfig:
    raven_cwd = os.environ["RAVEN_CWD"]
    raven_cmd = os.environ["RAVEN_CMD"]

    return (
        BilevelConfig()
        .world(world_name="water_world")
        .reporting(
            reporter="wandb",
            project_name="MetaMARL",
            settings_dict={
                "x_disable_stats": True,
                "x_disable_meta": True,
                "quiet": True,
                "max_end_of_run_summary_metrics": 0,
                "max_end_of_run_history_metrics": 0,
            },
        )
        .mechanism(
            space=WaterMechanismSpace(
                optimize_params=["fixed_quota"],
                default_fixed_quota=0.56224,
                default_max_demand_frac=1.0,
                default_fine_amount=0.20,
                default_risk_penalty_scale=0.0,
                default_risk_penalty_power=1.0,
            ),
        )
        .training(outer_iters=1)
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
        .outer(
            ESConfig()
            .training(
                sigma=0.5,
                mean_lr=0.2,
                sigma_lr=0.05,
                min_sigma=0.01,
                max_sigma=0.6,
            )
            .environment(
                env=WaterRegulatorRavenEnv,
                env_config={
                    "ecology_cfg": {
                        "sustainability_weight": 1.0,
                        "sustainability_threshold": 0.1,
                        "fitness_tail_steps": 100,
                        "max_water": 100.0,
                    },
                },
                horizon=100,
                train_iters=500,
            )
            .debugging(seed=42, num_seeds=1)
        )
        .inner(
            PPOptimizerConfig()
            .resources(num_cpus_for_main_process=1)
            .framework(framework="torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(
                env=WaterRegulatedEdHsEnv,
                env_config={
                    "ecology_cfg": {
                        "max_farm_area_m2": 5_000_000.0,
                        "full_stage_m": 420.41,
                        "max_depth_m": 11.0,
                        "lake_area_m2": 5_756_935.89615,
                        "quota_transition_width": 0.05,
                        "irrigation_transition_width": 0.005,
                        "violation_transition_width": 0.03,
                    },
                    "use_raven": True,
                    "raven_cwd": raven_cwd,
                    "raven_cmd": raven_cmd,
                    "raven_freq": 1,
                    "seed": 0,
                    "num_fixed_non_taggers": 80,
                },
                horizon=100,
                disable_env_checking=False,
            )
            .env_runners(
                num_env_runners=0,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
                num_envs_per_env_runner=4,
                rollout_fragment_length=100,
                batch_mode="truncate_episodes",
            )
            .learners(num_learners=0, num_gpus_per_learner=0)
            .callbacks(on_episode_created=tag_episode_with_env_idx)
            .training(
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
                num_seeds=1,
            )
            .agents({
                "utilizer": {
                    "count": 20,
                    "policy": "utilizer_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(5 + WaterMechanismSpace().full_dimension,),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Dict({
                        "extraction": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "tag": spaces.Discrete(2),
                    }),
                },
                "utilizer_non_tagging": {
                    "count": 80,
                    "policy": "utilizer_non_tagging_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(5 + WaterMechanismSpace().full_dimension,),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Dict({
                        "extraction": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                        "tag": spaces.Discrete(1),  # Only valid value is 0.
                    }),
                },
            })
            .fault_tolerance(restart_failed_env_runners=False)
            .debugging(seed=42, num_seeds=1)
            .reporting(
                min_time_s_per_iteration=0,
                min_sample_timesteps_per_iteration=0,
                min_train_timesteps_per_iteration=0,
            )
        )
    )


def main() -> None:
    ray.shutdown()
    try:
        config = build_config()
        optimizer = config.build_optimizer()
        optimizer.run()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
