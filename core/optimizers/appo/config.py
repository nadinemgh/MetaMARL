"""APPO (asynchronous PPO) flavour of the RLlib inner optimizer config.

This module holds ``APPOptimizerConfig``, the inner-level configuration that
selects RLlib's asynchronous PPO as the policy learner. It adds nothing to
``RayOptimizerConfig`` except the algorithm class, so every builder
(``training``, ``environment``, ``debugging``, ...) comes from the parent.
"""

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.appo import APPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class APPOptimizerConfig(RayOptimizerConfig):
    """``RayOptimizerConfig`` whose default RLlib config is ``APPO``'s.

    When to use: the default inner learner of the bilevel fishery example;
    its asynchronous sampling keeps every regulated environment busy while
    the learner updates. Use ``PPOptimizerConfig`` for synchronous PPO.

    Attributes
    ----------
    algo_class : type[Algorithm]
        RLlib algorithm class whose ``get_default_config()`` is the base of the
        recorded builder calls: ``ray.rllib.algorithms.appo.APPO``.

    Examples
    --------
    >>> APPOptimizerConfig.algo_class.__name__
    'APPO'
    >>> APPOptimizerConfig().opt_class.__name__
    'RayOptimizer'
    """

    algo_class: Algorithm = APPO
