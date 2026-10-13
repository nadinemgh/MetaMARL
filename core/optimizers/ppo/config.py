"""PPO flavour of the RLlib inner optimizer config.

This module holds ``PPOptimizerConfig``, the inner-level configuration that
selects RLlib's synchronous PPO as the policy learner. It adds nothing to
``RayOptimizerConfig`` except the algorithm class, so every builder (``training``,
``environment``, ``debugging``, ...) comes from the parent.
"""

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class PPOptimizerConfig(RayOptimizerConfig):
    """``RayOptimizerConfig`` whose default RLlib config is ``PPO``'s.

    When to use: a synchronous inner learner, simpler to reason about than
    ``APPOptimizerConfig`` when debugging sample flow at the cost of idle
    environments during the learner update.

    Attributes
    ----------
    algo_class : type[Algorithm]
        RLlib algorithm class whose ``get_default_config()`` is the base of the
        recorded builder calls: ``ray.rllib.algorithms.ppo.PPO``.

    Examples
    --------
    >>> PPOptimizerConfig.algo_class.__name__
    'PPO'
    >>> PPOptimizerConfig().opt_class.__name__
    'RayOptimizer'
    """

    algo_class: Algorithm = PPO
