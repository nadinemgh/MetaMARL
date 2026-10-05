"""Evolution Strategies over mechanism parameters in ``[0, 1]^d``.

This module holds ``ESOptimizer``, the outer level of a bilevel run: it searches
the normalized mechanism vector that the regulator environment applies to the
inner learner. The optimizer keeps a search distribution over that vector: a
mean in ``(0, 1)^d`` handled in logit space (so candidates never leave the unit
cube) and a scalar standard deviation ``sigma``. Each generation samples an
antithetic population, asks the regulator environment for one fitness per
candidate and moves the mean along the fitness-weighted noise directions,
using the estimator of Salimans et al., 2017 (https://arxiv.org/abs/1703.03864)
with standardized fitness values in place of their rank transformation.
In population mode ``sigma`` expands after a worse generation and contracts
after a better one. The run stops early when the search mean stops moving (see
the convergence rule of :class:`ESOptimizer`).

Three regimes share the same ``train()``:

- population mode (``batch_capacity >= 2``): antithetic ES update;
- single-candidate mode (``batch_capacity == 1``): sequential (1+1)-ES, where
  ``sigma`` expands after an accepted candidate and contracts after a rejected
  one;
- fixed mode (``dimension == 0``): no parameters, the fixed mechanism is
  evaluated and reported. Every generation evaluates the same empty candidates;
  the logged payload has no parameter entries, only the fitness values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import flatdim, flatten_space, unflatten

from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.base import Optimizer
from core.optimizers.es.schema import ESCandidateSchema, ESParameterSchema, ESSchema

if TYPE_CHECKING:
    from core.optimizers.es.config import ESConfig
    from core.types import AgentID, MechanismID

logger = logging.getLogger(__name__)
EPS = 1e-8

# Ignore changes smaller than this relative scale when deciding whether
# population-level performance improved or deteriorated.
SIGMA_PERFORMANCE_REL_TOL = 1e-3


class ESOptimizer(Optimizer):
    """Outer optimizer searching the normalized mechanism vector by ES.

    ES stands for Evolution Strategies, a family of gradient-free methods that
    estimate an ascent direction from the fitness of randomly perturbed
    candidates.

    The state is a mean in ``(0, 1)^dimension`` (moved in logit space) and a
    scalar ``sigma``. One call to :meth:`train` runs ``config.episodes``
    generations. A generation samples the population, steps the regulator
    environment with it (which trains the inner learner against every candidate
    and returns one fitness each), updates the mean and ``sigma``, and logs an
    ``ESSchema`` payload. The population size comes from :attr:`batch_capacity`,
    which ``BilevelConfig`` sets to the inner optimizer's capacity. The
    dimension is the size of the flattened action space of the mechanisms of
    every regulator agent. The regulator environment hands the same candidate
    to every leader and each leader applies the mechanisms it owns, so the
    search space is the union of the leaders' mechanisms and a mechanism id may
    be declared by one leader only. Gymnasium orders the mechanisms by id, so
    the vector and ``parameter_names`` follow the alphabetical order of the
    mechanism ids, whichever leader owns them.

    The run stops before ``config.episodes`` generations when the search mean
    has converged: the Euclidean norm of the change of the mean between two
    consecutive generations, measured in the normalized ``[0, 1]^dimension``
    space, stays below ``convergence_eps`` for ``convergence_patience``
    consecutive generations, and :meth:`train` then reports ``converged=True``.
    In single-candidate mode the mean only moves when a candidate is accepted,
    so a streak of rejected candidates counts as a streak of zero displacement.
    The rule does not apply in fixed mode, where there is nothing to search and
    every generation is run. ``convergence_eps=0``, the default of ``ESConfig``,
    disables the rule. This is a heuristic stopping rule of this
    implementation, not taken from a paper:
    a flat fitness (no directional information) or a mean pushed against a
    bound of ``[0, 1]`` also satisfy it, whether or not the optimum was found,
    and in single-candidate mode ``convergence_patience`` rejected candidates
    in a row stop the run even far from the optimum.

    Parameters
    ----------
    config : ESConfig
        Hyperparameters (``sigma``, ``mean_lr``, sigma adaptation and bounds,
        ``break_symmetry``, ``initial_mean``), the number of generations
        (``episodes``), the convergence rule (``convergence_eps``,
        ``convergence_patience``) and the regulator agents (``agents_cfgs``);
        ``config.base_seed`` seeds the random generator.
    **kwargs : Any
        Forwarded to :class:`~core.optimizers.base.Optimizer` (``world``,
        ``reporting``).

    Attributes
    ----------
    dimension : int
        Number of searched parameters (``0`` selects the fixed mode).
    parameter_names : list of str
        One name per coordinate: the mechanism id, or ``"id[k]"`` when a
        mechanism has several values.
    mean : numpy.ndarray
        Mean of the search distribution, shape ``(dimension,)``, float32,
        values in ``[0, 1]``.
    sigma : float
        Current standard deviation in logit space (dimensionless), kept within
        ``[min_sigma, max_sigma]``.
    best_fitness : float
        Best fitness seen so far (objective units defined by the example);
        ``-inf`` before the first generation.
    best_candidate : numpy.ndarray
        Candidate that reached ``best_fitness``, shape ``(dimension,)``, values
        in ``[0, 1]``.
    convergence_eps : float
        Displacement threshold of the convergence rule, in the normalized
        ``[0, 1]^dimension`` space (``0`` disables the rule).
    convergence_patience : int
        Number of consecutive generations below ``convergence_eps`` that
        stops the run.
    stalled_generations : int
        Length of the current streak of generations whose mean displacement is
        below ``convergence_eps``; reset when :meth:`train` starts.
    population_history : list of tuple
        One ``(population, fitness)`` pair per generation, shapes
        ``(batch_capacity, dimension)`` and ``(batch_capacity,)``.

    Raises
    ------
    TypeError
        If a mechanism of a regulator agent has a non-``Box`` action space.
    ValueError
        If no regulator agent was set on the config (``agents``), if two
        regulator agents declare the same mechanism id, on action bounds other
        than ``[0, 1]``, a negative dimension, a non-positive ``mean_lr``, a
        negative ``sigma_lr``, a ``sigma_decay`` outside ``(0, 1]``, a
        non-positive ``min_sigma``, ``max_sigma`` below ``min_sigma``, a
        negative ``convergence_eps``, a ``convergence_patience`` below 1, or an
        ``initial_mean`` of the wrong shape, not finite or outside ``[0, 1]``.

    When to use: the mechanism has a handful of continuous parameters and the
    fitness is a noisy black box (an inner RL run), which is exactly where
    gradient-free ES is at home. See the module docstring for the three
    regimes (population, single-candidate, fixed). The sigma adaptation, the
    gradient clipping at norm 5 and the logit parametrization are design
    choices of this implementation, not part of the cited method.

    Examples
    --------
    A regulator with one mechanism, ``quota``, searched on a toy fitness that
    peaks at ``0.8``; the environment is a minimal stand-in for the regulator
    environment:

    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.agents.base import AgentConfig
    >>> from core.mechanism.config import MechanismConfig
    >>> from core.optimizers.es.config import ESConfig
    >>> regulator = AgentConfig(
    ...     id="regulator",
    ...     policy_id="regulator_policy",
    ...     mechanisms=(
    ...         MechanismConfig(
    ...             id="quota", action_space=spaces.Box(0.0, 1.0, (1,), np.float32)
    ...         ),
    ...     ),
    ... )
    >>> cfg = ESConfig().training(episodes=20, sigma=0.15, mean_lr=0.1)
    >>> cfg = cfg.agents(regulator).debugging(seed=0)
    >>> opt = ESOptimizer(cfg)
    >>> opt.batch_capacity = 4
    >>> opt.mean.tolist(), opt.parameter_names
    ([0.5], ['quota'])
    >>> class ToyRegulatorEnv:
    ...     def reset(self):
    ...         return None, {}
    ...     def step(self, actions):
    ...         fitness = [-((a["quota"][0] - 0.8) ** 2) for a in actions]
    ...         return None, fitness, True, False, {}
    >>> opt.env = ToyRegulatorEnv()
    >>> summary = opt.train()
    >>> summary["episodes"], summary["converged"]
    (20, False)
    >>> bool(abs(summary["best_mechanism"][0] - 0.8) < 0.05)
    True

    References
    ----------
    .. [1] Salimans, T., Ho, J., Chen, X., Sidor, S., Sutskever, I. (2017).
       Evolution Strategies as a Scalable Alternative to Reinforcement
       Learning. arXiv:1703.03864. https://arxiv.org/abs/1703.03864
    """

    def __init__(self, config: ESConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)

        if not config.agents_cfgs:
            raise ValueError(
                "ESOptimizer needs at least one regulator agent; declare it "
                + "with ESConfig.agents(...)."
            )

        # The regulator environment hands the same candidate dictionary to
        # every leader, and each leader applies the entries of the mechanisms
        # it owns, so the search space is the union of every leader's
        # mechanisms. A mechanism id must therefore name one mechanism only.
        owners: dict[MechanismID, AgentID] = {}
        mechanism_spaces: dict[MechanismID, spaces.Space] = {}
        for agent_id, agent_cfg in config.agents_cfgs.items():
            for mechanism in agent_cfg.mechanisms:
                if mechanism.id in owners:
                    raise ValueError(
                        f"Mechanism id {mechanism.id!r} is declared by both "
                        + f"{owners[mechanism.id]!r} and {agent_id!r}; the ES "
                        + "searches one coordinate block per mechanism id, so "
                        + "ids must be unique across regulator agents."
                    )
                owners[mechanism.id] = agent_id
                mechanism_spaces[mechanism.id] = mechanism.action_space
        self.action_space = spaces.Dict(mechanism_spaces)
        for mechanism_id, action_space in self.action_space.spaces.items():
            if not isinstance(action_space, spaces.Box):
                raise TypeError(
                    "ESOptimizer requires continuous Box action spaces; "
                    + f"{mechanism_id!r} has {type(action_space).__name__}."
                )
        self.search_space = flatten_space(self.action_space)
        self.dimension = flatdim(self.action_space)

        if not (
            np.allclose(self.search_space.low, 0.0)
            and np.allclose(self.search_space.high, 1.0)
        ):
            raise ValueError(
                "ESOptimizer currently requires action bounds in [0, 1]. "
                + f"Got low={self.search_space.low}, high={self.search_space.high}."
            )

        self.parameter_names: list[str] = []

        for mechanism_id, action_space in self.action_space.spaces.items():
            dimension = int(flatdim(action_space))

            if dimension == 1:
                self.parameter_names.append(str(mechanism_id))
            else:
                self.parameter_names.extend(
                    f"{mechanism_id}[{index}]" for index in range(dimension)
                )

        if len(self.parameter_names) != self.dimension:
            raise RuntimeError(
                "Action-space parameter count does not match ES dimension: "
                + f"{len(self.parameter_names)} != {self.dimension}."
            )

        self.mean_lr = config.mean_lr
        self.sigma_lr = float(config.sigma_lr)
        self.sigma_decay = float(config.sigma_decay)
        self.min_sigma = float(config.min_sigma)
        self.max_sigma = float(config.max_sigma)
        self.break_symmetry = config.break_symmetry
        self.convergence_eps = float(config.convergence_eps)
        self.convergence_patience = int(config.convergence_patience)

        if self.dimension < 0:
            raise ValueError("dimension must be non-negative")

        self.fixed_mode = self.dimension == 0

        if self.mean_lr <= 0.0:
            raise ValueError("mean_lr must be positive")

        if self.sigma_lr < 0.0:
            raise ValueError("sigma_lr must be non-negative")

        if not 0.0 < self.sigma_decay <= 1.0:
            raise ValueError(
                "sigma_decay must be in (0, 1]. Use 1.0 to disable sigma adaptation."
            )

        if self.min_sigma <= 0.0:
            raise ValueError("min_sigma must be positive")

        if self.max_sigma < self.min_sigma:
            raise ValueError("max_sigma must be >= min_sigma")

        if self.convergence_eps < 0.0:
            raise ValueError(
                "convergence_eps must be non-negative. Use 0 to disable the "
                + "convergence stop."
            )

        if self.convergence_patience < 1:
            raise ValueError("convergence_patience must be at least 1")

        # --- Runtime state ---
        if config.initial_mean is not None:
            initial_mean = np.asarray(config.initial_mean, dtype=np.float32)

            if initial_mean.shape != (self.dimension,):
                raise ValueError(
                    "initial_mean must have shape "
                    + f"({self.dimension},), got {initial_mean.shape}"
                )

            if not np.all(np.isfinite(initial_mean)):
                raise ValueError("initial_mean must contain finite values")

            if np.any(initial_mean < 0.0) or np.any(initial_mean > 1.0):
                raise ValueError("initial_mean values must be in [0, 1]")

            self.mean = initial_mean.copy()
        else:
            self.mean = np.full(shape=self.dimension, fill_value=0.5, dtype=np.float32)

        self.sigma = float(np.clip(config.sigma, self.min_sigma, self.max_sigma))

        # Random number generator.
        self.rng = np.random.default_rng(config.base_seed)

        # History tracking.
        self.fitness_baseline: float | None = None
        self.best_fitness = -float("inf")
        self.best_candidate = self.mean.copy()
        self.best_mechanism_idx: int | None = None
        self.population_history: list[tuple[np.ndarray, np.ndarray]] = []
        self.stalled_generations = 0

        # This is explicitly the average fitness of the sampled population,
        # not the fitness of the distribution mean.
        self.previous_population_mean_fitness: float | None = None

        # Typed metric logger for one generation at a time (see ESSchema).
        self.logger = MetricLogger.from_schema(ESSchema)

    @property
    def batch_capacity(self) -> int:
        """Population size: number of candidates evaluated per generation.

        Returns
        -------
        int
            Number of candidates; equals the inner optimizer's capacity once
            ``BilevelConfig`` has built the run.

        Raises
        ------
        RuntimeError
            If the size has not been set yet (``BilevelOptimizer`` sets it from
            the inner optimizer's capacity).
        """

        return super().batch_capacity

    @batch_capacity.setter
    def batch_capacity(self, value: int) -> None:
        """Set the population size and select the matching regime.

        ``dimension == 0`` accepts any positive value (fixed mode); ``1``
        switches to sequential (1+1)-ES; otherwise the value must be even
        unless ``break_symmetry`` is set, because the population is built
        from mirrored noise pairs.

        Parameters
        ----------
        value : int
            Number of candidates per generation.

        Raises
        ------
        ValueError
            If ``value`` is not positive, or is odd while ``break_symmetry`` is
            off and the dimension is positive (other than ``1``).
        """

        if value <= 0:
            raise ValueError("population_size must be positive")

        if self.fixed_mode:
            self._batch_capacity = value
            logger.info(
                "[ES] Fixed-mechanism batch mode enabled | batch_capacity=%d", value
            )
            return

        if value == 1:
            self._batch_capacity = 1
            logger.info(
                "[ES] Single-candidate mode enabled. Using sequential (1+1)-ES."
            )
            return

        if not self.break_symmetry and value % 2 != 0:
            raise ValueError(f"Antithetic ES requires an even batch size, got {value}.")

        self._batch_capacity = value

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""

        values = np.asarray(values, dtype=np.float64)
        output = np.empty_like(values, dtype=np.float64)
        positive = values >= 0.0
        negative = ~positive
        output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[negative])
        output[negative] = exp_values / (1.0 + exp_values)

        return output

    @staticmethod
    def _logit(values: np.ndarray) -> np.ndarray:
        """Convert values in [0, 1] to finite logit coordinates."""

        eps_bound = 1e-6
        clipped = np.clip(
            np.asarray(values, dtype=np.float64), eps_bound, 1.0 - eps_bound
        )

        return np.log(clipped / (1.0 - clipped))

    def _sample_population(self) -> np.ndarray:
        """Sample a population using antithetic logit-space noise.

        Returns:
            Population with shape
            ``(batch_capacity, dimension)`` and values in ``(0, 1)``.
        """

        capacity = self.batch_capacity

        if self.fixed_mode:
            return np.empty((capacity, 0), dtype=np.float32)
        if capacity == 1 and self.fitness_baseline is None:
            return self.mean[None, :].copy()

        half_pop = capacity // 2
        remaining = capacity - (2 * half_pop)
        noise_half = self.rng.standard_normal(
            (half_pop, self.dimension), dtype=np.float32
        )

        if half_pop > 0:
            noise_matrix = np.vstack([noise_half, -noise_half])
        else:
            noise_matrix = np.empty((0, self.dimension), dtype=np.float32)

        if remaining > 0:
            extra_noise = self.rng.standard_normal(
                (remaining, self.dimension), dtype=np.float32
            )
            noise_matrix = np.vstack([noise_matrix, extra_noise])

        # When requested, replace one mirrored sample with an independent
        # sample so the population is no longer strictly antithetic.
        if self.break_symmetry and capacity % 2 == 0 and half_pop > 0:
            noise_matrix[-1] = self.rng.standard_normal(
                self.dimension, dtype=np.float32
            )

        mean_logit = self._logit(self.mean)
        population_logit = mean_logit[None, :] + self.sigma * noise_matrix
        population = self._sigmoid(population_logit)

        return population.astype(np.float32)

    def _update_sigma(self, generation_mean_fitness: float) -> str:
        """Adapt sigma from population-level performance changes.

        The old implementation only allowed sigma to decrease, and
        ``sigma_lr`` was unused. This version applies a symmetric,
        multiplicative update:

        * improved population mean -> contract sigma;
        * deteriorated population mean -> expand sigma;
        * change within tolerance -> keep sigma unchanged.

        ``sigma_lr`` controls the strength of the update. For example,
        with ``sigma_decay=0.99``:

        * ``sigma_lr=1.0`` applies the full factor 0.99;
        * ``sigma_lr=0.5`` applies sqrt(0.99);
        * ``sigma_lr=0.0`` disables adaptation.

        Returns:
            One of ``"initialized"``, ``"contracted"``, ``"expanded"``,
            or ``"held"``.
        """

        previous = self.previous_population_mean_fitness

        if previous is None:
            self.previous_population_mean_fitness = generation_mean_fitness

            return "initialized"

        tolerance = SIGMA_PERFORMANCE_REL_TOL * max(
            1.0, abs(previous), abs(generation_mean_fitness)
        )
        improvement = generation_mean_fitness - previous

        # Convert the full decay factor into a learning-rate-controlled
        # multiplicative step. This is symmetric in log space.
        adaptation_factor = self.sigma_decay**self.sigma_lr
        old_sigma = self.sigma

        if improvement > tolerance:
            # Better population-level performance: exploit more.
            proposed_sigma = self.sigma * adaptation_factor
            action = "contracted"
        elif improvement < -tolerance:
            # Worse population-level performance: restore exploration.
            proposed_sigma = (
                self.sigma / adaptation_factor
                if adaptation_factor > 0.0
                else self.max_sigma
            )
            action = "expanded"
        else:
            proposed_sigma = self.sigma
            action = "held"

        self.sigma = float(np.clip(proposed_sigma, self.min_sigma, self.max_sigma))
        self.previous_population_mean_fitness = generation_mean_fitness

        logger.info(
            "[ES] SIGMA UPDATE | "
            + "action=%s | previous_population_mean_fitness=%.6f | "
            + "current_population_mean_fitness=%.6f | "
            + "improvement=%+.6f | tolerance=%.6f | "
            + "sigma=%.6f->%.6f | sigma_lr=%.6f | "
            + "sigma_decay=%.6f",
            action,
            previous,
            generation_mean_fitness,
            improvement,
            tolerance,
            old_sigma,
            self.sigma,
            self.sigma_lr,
            self.sigma_decay,
        )

        return action

    def _update_single_candidate(self, candidate: np.ndarray, fitness: float) -> None:
        """Sequential (1+1)-ES update.

        The first candidate becomes the remembered parent.
        Later candidates replace the parent only when they improve fitness.
        """

        candidate = np.asarray(candidate, dtype=np.float32).reshape(self.dimension)
        fitness = float(fitness)

        if not np.isfinite(fitness):
            raise ValueError("Single-candidate fitness must be finite")

        # First evaluation: remember the initial ES mean and its fitness.
        if self.fitness_baseline is None:
            self.mean = candidate.copy()
            self.fitness_baseline = fitness
            self.previous_population_mean_fitness = fitness
            self.best_fitness = fitness
            self.best_candidate = candidate.copy()
            self.best_mechanism_idx = 0

            logger.info(
                "[ES] SINGLE INITIALIZED | parent=%s | parent_fitness=%.6f",
                candidate.tolist(),
                fitness,
            )

            return

        parent_fitness = float(self.fitness_baseline)
        improvement = fitness - parent_fitness
        accepted = fitness > parent_fitness
        old_mean = self.mean.copy()
        old_sigma = float(self.sigma)

        if accepted:
            # The offspring becomes the new remembered parent.
            self.mean = candidate.copy()
            self.fitness_baseline = fitness

        # Track the globally best evaluated candidate.
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_candidate = candidate.copy()
            self.best_mechanism_idx = 0

        # Sigma adaptation only affects the new one-candidate mode.
        # With sigma_lr=0, sigma remains exactly fixed.
        adaptation_factor = self.sigma_decay**self.sigma_lr

        if self.sigma_lr > 0.0:
            if accepted:
                proposed_sigma = self.sigma / adaptation_factor
            else:
                proposed_sigma = self.sigma * adaptation_factor

            self.sigma = float(np.clip(proposed_sigma, self.min_sigma, self.max_sigma))

        self.previous_population_mean_fitness = self.fitness_baseline

        logger.info(
            "[ES] SINGLE UPDATE | "
            + "accepted=%s | "
            + "parent_fitness=%.6f | "
            + "candidate_fitness=%.6f | "
            + "improvement=%+.6f | "
            + "mean=%s->%s | "
            + "sigma=%.6f->%.6f",
            accepted,
            parent_fitness,
            fitness,
            improvement,
            old_mean.tolist(),
            self.mean.tolist(),
            old_sigma,
            self.sigma,
        )

    def _update_parameters(
        self, population: np.ndarray, fitness_scores: list[float] | np.ndarray
    ) -> None:
        population = np.asarray(population, dtype=np.float32)
        fitness_scores_array = np.asarray(fitness_scores, dtype=np.float32).reshape(-1)

        if population.ndim != 2:
            raise ValueError("population must be a 2D array")

        if population.shape != (fitness_scores_array.size, self.dimension):
            raise ValueError(
                "population shape and fitness count do not match: "
                + f"{population.shape} versus "
                + f"{fitness_scores_array.size} fitness values"
            )

        if fitness_scores_array.size == 0:
            raise ValueError("fitness_scores must not be empty")

        if not np.all(np.isfinite(fitness_scores_array)):
            raise ValueError("fitness_scores must all be finite")

        if self.fixed_mode:
            expected_size = population.shape[0]

            if fitness_scores_array.size != expected_size:
                raise ValueError(
                    "Fixed-mechanism mode expected "
                    + f"{expected_size} fitness values, got "
                    + f"{fitness_scores_array.size}."
                )

            generation_mean_fitness = float(np.mean(fitness_scores_array))
            best_idx = int(np.argmax(fitness_scores_array))
            best_fitness = float(fitness_scores_array[best_idx])
            self.fitness_baseline = generation_mean_fitness
            self.previous_population_mean_fitness = generation_mean_fitness

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_candidate = np.empty(0, dtype=np.float32)
                self.best_mechanism_idx = best_idx

            logger.info(
                "[ES] FIXED MECHANISM BATCH | "
                + "fitness=%s | mean=%.6f | "
                + "std=%.6f | min=%.6f | max=%.6f",
                fitness_scores_array.tolist(),
                generation_mean_fitness,
                float(np.std(fitness_scores_array)),
                float(np.min(fitness_scores_array)),
                float(np.max(fitness_scores_array)),
            )

            return

        generation_mean_fitness = float(np.mean(fitness_scores_array))
        normalized_fitness = fitness_scores_array.copy()

        if fitness_scores_array.size == 1:
            self._update_single_candidate(
                candidate=population[0], fitness=float(fitness_scores_array[0])
            )

            return
        else:
            fitness_mean = float(np.mean(normalized_fitness))
            fitness_std = float(np.std(normalized_fitness))

            if fitness_std <= EPS:
                # A flat population contains no directional information.
                normalized_fitness = np.zeros_like(normalized_fitness)
            else:
                normalized_fitness = (normalized_fitness - fitness_mean) / (
                    fitness_std + EPS
                )

        mean_logit = self._logit(self.mean)
        population_logit = self._logit(population)

        # Reconstruct the standardized perturbations used to generate
        # the candidates.
        eps_est = (population_logit - mean_logit[None, :]) / (self.sigma + EPS)
        population_size = len(normalized_fitness)
        half = population_size // 2
        strict_antithetic = (
            not self.break_symmetry and population_size % 2 == 0 and half > 0
        )

        if strict_antithetic:
            fitness_positive = normalized_fitness[:half]
            fitness_negative = normalized_fitness[half : 2 * half]
            epsilon_positive = eps_est[:half]
            gradient = np.mean(
                (fitness_positive - fitness_negative)[:, None] * epsilon_positive,
                axis=0,
            ) / (2.0 * self.sigma + EPS)
        else:
            gradient = np.mean(normalized_fitness[:, None] * eps_est, axis=0) / (
                self.sigma + EPS
            )

        gradient = np.asarray(gradient, dtype=np.float64)
        grad_norm = float(np.linalg.norm(gradient))

        if grad_norm > 5.0:
            gradient *= 5.0 / (grad_norm + EPS)

        logger.info(
            "[ES] PARAMETER GRADIENTS | %s",
            {
                name: float(gradient[index])
                for index, name in enumerate(self.parameter_names)
            },
        )

        # Mean update.
        new_mean_logit = mean_logit + self.mean_lr * gradient
        self.mean = self._sigmoid(new_mean_logit).astype(np.float32)

        # Sigma update. This now uses sigma_lr and can expand after a
        # deterioration instead of monotonically shrinking.
        self._update_sigma(generation_mean_fitness)

        # Track the best raw candidate fitness.
        best_idx = int(np.argmax(fitness_scores_array))
        best_fitness = float(fitness_scores_array[best_idx])

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_candidate = population[best_idx].copy()
            self.best_mechanism_idx = best_idx

    def _to_logger_payload(
        self,
        *,
        generation: int,
        inner: MetricSchema,
        population: np.ndarray,
        fitness: np.ndarray,
        mean: np.ndarray,
        sigma: float,
    ) -> ESSchema:
        """Convert one completed ES generation to its metric schema."""

        # In fixed mode the dimension is 0: the parameter names, the mean and
        # the candidates are empty, so the payload only carries the fitness
        # values of the evaluated batch.
        parameter_names = self.parameter_names
        logged_population = population
        logged_mean = mean
        logged_best = self.best_candidate

        best_idx = int(np.argmax(fitness))

        return ESSchema(
            iter=generation,
            generation=generation,
            sigma=sigma,
            population_size=len(fitness),
            fitness_mean=float(fitness.mean()),
            fitness_best=float(fitness[best_idx]),
            best_mechanism_idx=best_idx,
            best_fitness_global=float(self.best_fitness),
            by_mechanism={
                str(mechanism_idx): ESCandidateSchema(
                    fitness=float(fitness[mechanism_idx]),
                    by_parameter={
                        parameter_name: ESParameterSchema(
                            value=float(logged_population[mechanism_idx, parameter_idx])
                        )
                        for parameter_idx, parameter_name in enumerate(parameter_names)
                    },
                )
                for mechanism_idx in range(len(fitness))
            },
            search_mean={
                parameter_name: ESParameterSchema(
                    value=float(logged_mean[parameter_idx])
                )
                for parameter_idx, parameter_name in enumerate(parameter_names)
            },
            global_best={
                parameter_name: ESParameterSchema(
                    value=float(logged_best[parameter_idx])
                )
                for parameter_idx, parameter_name in enumerate(parameter_names)
            },
            generation_best={
                parameter_name: ESParameterSchema(
                    value=float(logged_population[best_idx, parameter_idx])
                )
                for parameter_idx, parameter_name in enumerate(parameter_names)
            },
            inner=inner,
        )

    def _update_convergence(self, previous_mean: np.ndarray) -> bool:
        """Update the stall streak after a generation and test the stop rule.

        The displacement is the Euclidean norm of ``self.mean - previous_mean``
        in the normalized ``[0, 1]^dimension`` space. A displacement below
        ``convergence_eps`` extends the streak, any other resets it, and the
        search has converged when the streak reaches ``convergence_patience``.
        This is a heuristic stopping rule, not taken from a paper. It is
        inactive in fixed mode, where the mean is empty, and when
        ``convergence_eps`` is ``0``.

        Parameters
        ----------
        previous_mean : numpy.ndarray
            Search mean before the generation's update, shape ``(dimension,)``,
            values in ``[0, 1]``.

        Returns
        -------
        bool
            ``True`` when the streak has reached ``convergence_patience``.
        """

        if self.fixed_mode:
            return False

        displacement = float(
            np.linalg.norm(
                np.asarray(self.mean, dtype=np.float64)
                - np.asarray(previous_mean, dtype=np.float64)
            )
        )

        if displacement < self.convergence_eps:
            self.stalled_generations += 1
        else:
            self.stalled_generations = 0

        logger.info(
            "[ES] CONVERGENCE CHECK | displacement=%.3e | eps=%.3e | stalled=%d/%d",
            displacement,
            self.convergence_eps,
            self.stalled_generations,
            self.convergence_patience,
        )

        return self.stalled_generations >= self.convergence_patience

    def train(self) -> dict[str, Any]:
        """Run ``self.episodes`` generations and return a summary.

        Each generation samples the population, calls ``self.env.reset()``
        then ``self.env.step(actions)`` until the episode ends, and reads the
        fitness array (shape ``(batch_capacity,)``). It appends the
        ``(population, fitness)`` pair to ``population_history``, applies the
        mean and sigma update, pushes the generation's ``ESSchema`` payload to
        the logger (the ``metrics`` entry of the last ``info`` becomes its
        ``inner`` field) and reports it. ``actions`` is a list with one
        dictionary per candidate, keyed by mechanism id.

        The run stops before ``self.episodes`` generations when the search
        mean has converged (``convergence_eps`` and ``convergence_patience``,
        see the class docstring); the generation that completes the streak is
        the last one run. Otherwise all ``self.episodes`` generations run.

        Returns
        -------
        dict
            ``episodes`` (generations actually run), ``converged`` (``True``
            when the convergence rule stopped the run), ``best_fitness``
            (best value seen so far),
            ``best_mechanism`` (the normalized candidate that reached it,
            shape ``(dimension,)``, values in ``[0, 1]``) and
            ``population_history``.

        Raises
        ------
        RuntimeError
            If no environment is attached, the population size is not set, a
            fitness is non-finite, the number of fitness values does not
            match the population size, or the episode is truncated or ends
            without a fitness.
        TypeError
            If ``self.episodes`` is ``None``.
        ValueError
            If the environment declares ``horizon=None``: such a regulator never
            terminates its episode, so a generation would never end.
        """

        if self.env is None:
            raise RuntimeError("ESOptimizer requires a RegulatorEnv")

        # RegulatorEnv(horizon=None) means "never terminate", which the
        # generation loop below cannot handle.
        if getattr(self.env, "horizon", 1) is None:
            raise ValueError(
                "ESOptimizer requires a regulator with a finite horizon; "
                + "got horizon=None"
            )

        converged = False
        generations_run = 0
        self.stalled_generations = 0

        for generation in range(self.episodes):
            logger.info(
                "[ES] Generation started | gen=%d | sigma=%.5f | mean_norm=%.4f",
                generation,
                self.sigma,
                float(np.linalg.norm(self.mean)),
            )

            pre_update_mean = self.mean.copy()
            pre_update_sigma = float(self.sigma)
            population = self._sample_population()

            self.env.reset()

            terminated = False
            truncated = False

            fitness: np.ndarray | None = None
            info: dict[str, Any] = {}

            actions = [
                unflatten(self.action_space, candidate) for candidate in population
            ]

            while not terminated and not truncated:
                _, fitness, terminated, truncated, info = self.env.step(actions)
                fitness = np.asarray(fitness, dtype=np.float32).reshape(-1)

                if fitness.size == 0:
                    logger.warning("[ES] No fitness returned; skipping update")

                if not np.all(np.isfinite(fitness)):
                    invalid_indices = np.flatnonzero(~np.isfinite(fitness)).tolist()

                    raise RuntimeError(
                        f"Non-finite fitness detected at indices {invalid_indices}"
                    )

                if fitness.size != population.shape[0]:
                    raise RuntimeError(
                        f"The environment returned {fitness.size} fitness values for "
                        + f"{population.shape[0]} candidates"
                    )

            if truncated and not terminated:
                raise RuntimeError(
                    "Regulator episode truncated before terminal ES fitness produced"
                )

            if fitness is None:
                raise RuntimeError("Regulator episode terminated without fitness")

            self.population_history.append((population.copy(), fitness.copy()))

            fitness_variance = float(fitness.var())

            logger.info(
                "[ES] BEFORE UPDATE | gen=%d | mean=%s | sigma=%.5f | population=%s"
                + " | fitness=%s",
                generation,
                pre_update_mean.tolist(),
                pre_update_sigma,
                population.tolist(),
                fitness.tolist(),
            )
            self._update_parameters(population, fitness)
            logger.info(
                "[ES] AFTER UPDATE | gen=%d | mean=%s | sigma=%.5f | best=%.5f",
                generation,
                self.mean.tolist(),
                self.sigma,
                self.best_fitness,
            )

            metrics = self._to_logger_payload(
                inner=info.get("metrics") if isinstance(info, dict) else None,
                population=population,
                fitness=fitness,
                mean=pre_update_mean,
                sigma=pre_update_sigma,
                generation=generation,
            )

            self.logger.push_data(metrics)
            self.report_metrics()
            logger.info(
                "[ES] gen=%d | best=%.4f | mean=%.4f+/-%.4f | var=%.4f | sigma=%.4f",
                generation,
                self.best_fitness,
                float(fitness.mean()),
                float(fitness.std()),
                fitness_variance,
                self.sigma,
            )

            generations_run += 1
            converged = self._update_convergence(pre_update_mean)

            if converged:
                logger.info(
                    "[ES] Converged | gen=%d | stalled=%d | eps=%.3e | sigma=%.6f",
                    generation,
                    self.stalled_generations,
                    self.convergence_eps,
                    self.sigma,
                )
                break

        return {
            "episodes": generations_run,
            "converged": converged,
            "best_fitness": self.best_fitness,
            "best_mechanism": self.best_candidate,
            "population_history": self.population_history,
        }
