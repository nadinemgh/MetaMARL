"""``RayOptimizerConfig.build_optimizer``: missing inputs fail at build time.

Every environment fetches its mechanism from the shared ``World`` and builds
its followers from the declared agents, so a config without either cannot
produce a working optimizer. It must say so when it is built, not when RLlib
first creates an environment.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.optimizers.appo.config import APPOptimizerConfig
from examples.bilevel_fishery.regulated_env import FishermanConfig as Fisherman
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulated_env import FishingConfig as Fishing


@pytest.mark.unit
def test_build_without_a_world_is_a_clear_error():
    with pytest.raises(ValueError, match="needs the shared World actor"):
        APPOptimizerConfig().build_optimizer(world=None)


@pytest.mark.unit
def test_build_without_agents_is_a_clear_error():
    with pytest.raises(ValueError, match=r"no agents: call \.agents\("):
        APPOptimizerConfig().build_optimizer(world=object(), world_name="w")


def fisher_society(num_mechanisms: int, num_train_seeds: int) -> APPOptimizerConfig:
    """Fishery society config with the layout knobs of ``debug.py``."""
    box = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    return (
        APPOptimizerConfig()
        .environment(env=FisheryRegulatedEnv, env_config={}, horizon=5)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=num_mechanisms)
        .evaluation(
            evaluation_config={"rollout_fragment_length": 5}, base_seed=42, num_seeds=3
        )
        .agents(
            Fisherman(
                id="fisherman",
                policy_id="fisher_policy",
                mechanisms=(Fishing(id="harvest", action_space=box),),
                count=2,
                observation_space=spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
                ),
            )
        )
        .debugging(seed=42, num_seeds=num_train_seeds)
    )


@pytest.mark.unit
def test_evaluation_runners_host_one_environment_per_mechanism():
    # Training runners host one environment per (mechanism, training seed);
    # each evaluation runner tests a single training seed, so it must host one
    # environment per mechanism, which is what ``evaluation_duration`` counts.
    cfg = fisher_society(num_mechanisms=4, num_train_seeds=2)

    # The build stops at the missing world name, after the RLlib config is replayed.
    with pytest.raises(ValueError, match="world_name must be provided"):
        cfg.build_optimizer(world=object(), world_name=None)

    assert cfg.rllib_cfg.num_envs_per_env_runner == 8
    assert cfg.rllib_cfg.evaluation_config["num_envs_per_env_runner"] == 4
    assert cfg.rllib_cfg.evaluation_num_env_runners == 3 * 2
    assert cfg.rllib_cfg.evaluation_duration == 3 * 2 * 4


@pytest.mark.unit
@pytest.mark.parametrize("disable", [True, False])
def test_disable_env_checking_reaches_rllib(disable):
    cfg = fisher_society(num_mechanisms=1, num_train_seeds=1)
    cfg.environment(
        env=FisheryRegulatedEnv, env_config={}, horizon=5, disable_env_checking=disable
    )

    # The build stops at the missing world name, after the RLlib config is replayed.
    with pytest.raises(ValueError, match="world_name must be provided"):
        cfg.build_optimizer(world=object(), world_name=None)

    assert cfg.rllib_cfg.disable_env_checking is disable
