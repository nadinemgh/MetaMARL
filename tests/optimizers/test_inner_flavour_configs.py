"""PPO and APPO flavours of the inner (society) optimizer config.

Both subclasses only choose the RLlib algorithm whose default configuration the
recorded builder calls are replayed on; building one needs no Ray runtime.
"""

import pytest
from ray.rllib.algorithms.appo import APPO
from ray.rllib.algorithms.ppo import PPO

from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.optimizer_config import RayOptimizerConfig
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.ppo.config import PPOptimizerConfig


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_class, algorithm",
    [(PPOptimizerConfig, PPO), (APPOptimizerConfig, APPO)],
    ids=["ppo", "appo"],
)
def test_the_flavour_selects_its_rllib_algorithm(config_class, algorithm):
    assert issubclass(config_class, RayOptimizerConfig)
    assert config_class.algo_class is algorithm

    cfg = config_class()

    assert cfg.opt_class is RayOptimizer
    assert cfg.algo_class is algorithm


@pytest.mark.unit
def test_the_two_flavours_do_not_share_an_algorithm():
    assert PPOptimizerConfig.algo_class is not APPOptimizerConfig.algo_class
    assert RayOptimizerConfig.algo_class is None


@pytest.mark.unit
def test_the_base_config_without_an_algorithm_cannot_be_built():
    with pytest.raises(ValueError, match="must define `algo_class`"):
        RayOptimizerConfig()
