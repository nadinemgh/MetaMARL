"""Hyperparameters of the Evolution Strategies outer optimizer.

This module holds ``ESConfig``, the fluent configuration of the outer level of
a bilevel run. It stores the ES hyperparameters (search standard deviation,
learning rates, sigma bounds, symmetry and initial mean) on top of the shared
``OptimizerConfig`` builders and builds an ``ESOptimizer``.
"""

from typing import Any, Optional, Self

from core.annotations import override
from core.optimizers.config import OptimizerConfig
from core.optimizers.es.optimizer import ESOptimizer


class ESConfig(OptimizerConfig):
    """Configuration of :class:`~core.optimizers.es.optimizer.ESOptimizer`.

    The search dimension and the population size are not set here. The
    optimizer takes its dimension from the flattened action space of the
    mechanisms of the regulator agent given to ``agents``, and
    ``BilevelConfig`` sets the population size from the inner optimizer's batch
    capacity. The ``dimension`` attribute below is never read.

    Parameters
    ----------
    opt_class : type[Optimizer], optional
        Optimizer class to build (default ``ESOptimizer``).
    **kw : Any
        Accepted and ignored.

    Attributes
    ----------
    dimension : int or None
        Placeholder, ``None``; not used by ``ESOptimizer``.
    sigma : float
        Initial standard deviation of the search distribution in logit space
        (default 0.15, dimensionless).
    mean_lr : float
        Step size applied to the estimated gradient when moving the mean
        (default 0.1).
    sigma_lr : float
        Strength of the sigma adaptation; ``0`` disables it (default 0.05).
    sigma_decay : float
        Base multiplicative sigma factor, in ``(0, 1]`` (default 0.99; ``1.0``
        disables the adaptation).
    min_sigma, max_sigma : float
        Bounds on sigma (defaults 1e-3 and 0.5).
    break_symmetry : bool
        Replace one mirrored sample by an independent one (default ``False``).
    convergence_eps : float
        Displacement threshold of the convergence stop, in the normalized
        ``[0, 1]^dimension`` space (default 1e-4); ``0`` disables the stop.
    convergence_patience : int
        Number of consecutive generations whose displacement of the search mean
        stays below ``convergence_eps`` before the run stops (default 10).
    initial_mean : list of float or None
        Starting point in ``[0, 1]^dimension``; ``None`` means ``0.5`` in every
        coordinate.

    When to use: to configure the outer mechanism search of a bilevel run,
    typically passed to ``BilevelConfig.regulator``. Call ``agents`` with the
    regulator agent whose mechanisms define the search space, ``training`` for
    the hyperparameters and the number of generations, and ``debugging`` for
    the seed.

    Examples
    --------
    >>> cfg = ESConfig().training(episodes=100, sigma=0.2, mean_lr=0.05)
    >>> (cfg.episodes, cfg.sigma, cfg.mean_lr, cfg.sigma_decay)
    (100, 0.2, 0.05, 0.99)
    >>> cfg.opt_class.__name__
    'ESOptimizer'
    """

    def __init__(self, opt_class: Optional[type[ESOptimizer]] = None, **kw: Any):
        super().__init__(opt_class=opt_class or ESOptimizer)

        # Add default or from default
        # ES training hyperparameters
        self.dimension: int = None
        self.sigma: float = 0.15
        self.mean_lr: float = 0.1
        self.sigma_lr: float = 0.05
        self.sigma_decay: float = 0.99
        self.min_sigma: float = 1e-3
        self.max_sigma: float = 0.5
        self.break_symmetry: bool = False
        self.convergence_eps: float = 1e-4
        self.convergence_patience: int = 10
        self.initial_mean: Optional[list[float]] = None

    @override(OptimizerConfig)
    def training(
        self,
        *,
        episodes: Optional[int] = None,
        sigma: Optional[float] = None,
        mean_lr: Optional[float] = None,
        sigma_lr: Optional[float] = None,
        sigma_decay: Optional[float] = None,
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
        generation: Optional[int] = None,
        break_symmetry: Optional[bool] = None,
        convergence_eps: Optional[float] = None,
        convergence_patience: Optional[int] = None,
        initial_mean: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> Self:
        """Set the ES search hyperparameters. Unset arguments keep their current value.

        The values are stored without validation; ``ESOptimizer`` checks them
        when it is built.

        Parameters
        ----------
        episodes : int, optional
            Number of ES generations to run.
        sigma : float, optional
            Initial standard deviation of the search distribution in logit
            space.
        mean_lr : float, optional
            Step size applied to the estimated gradient when moving the mean.
        sigma_lr : float, optional
            Strength of the sigma adaptation (``0`` disables it).
        sigma_decay : float, optional
            Base multiplicative factor of the sigma adaptation, in ``(0, 1]``.
        min_sigma, max_sigma : float, optional
            Bounds of sigma: a floor keeps exploring, a ceiling avoids
            destabilizing jumps.
        generation : int, optional
            Stored as ``self.generation``; nothing reads it.
        break_symmetry : bool, optional
            Replace one mirrored sample by an independent one so the population
            is not strictly antithetic (allows odd population sizes).
        convergence_eps, convergence_patience : float, int, optional
            Convergence stop of ``ESOptimizer``: the run ends when the
            Euclidean norm of the change of the search mean between two
            consecutive generations, in the normalized ``[0, 1]^dimension``
            space, stays below ``convergence_eps`` for ``convergence_patience``
            consecutive generations (``convergence_eps=0`` disables it). This
            is a heuristic stopping rule, not taken from a paper.
        initial_mean : list[float], optional
            Starting point in ``[0, 1]^dimension`` (default ``0.5`` everywhere).
        **kwargs : Any
            Accepted and ignored.

        Returns
        -------
        ESConfig
            ``self``, for chaining.
        """
        super().training(episodes=episodes)

        if sigma is not None:
            self.sigma = sigma

        if mean_lr is not None:
            self.mean_lr = mean_lr

        if sigma_lr is not None:
            self.sigma_lr = sigma_lr

        if sigma_decay is not None:
            self.sigma_decay = sigma_decay

        if min_sigma is not None:
            self.min_sigma = min_sigma

        if max_sigma is not None:
            self.max_sigma = max_sigma

        if generation is not None:
            self.generation = generation

        if break_symmetry is not None:
            self.break_symmetry = break_symmetry

        if convergence_eps is not None:
            self.convergence_eps = convergence_eps

        if convergence_patience is not None:
            self.convergence_patience = convergence_patience

        if initial_mean is not None:
            self.initial_mean = initial_mean

        return self
