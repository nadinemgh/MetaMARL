"""Context records exchanged between the fishery levels through the World actor.

``FitnessContext`` is what the regulator environment publishes for each
mechanism candidate once the inner rollouts have been aggregated: the scalar
objective the ES maximizes plus the summary statistics it was computed from.
It is a ``ContextSchema``, so it travels inside the ``metrics`` field of a
``MechanismContext`` appended to the ``World``.
"""

from typing import SupportsFloat

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """Fitness of one mechanism candidate, the sole scalar feedback of the bilevel loop.

    Built by ``FisheryRegulatorEnv.reward`` from the inner optimizer's
    rollouts and consumed by the ES outer loop. ``objective_score`` is
    ``harvest_score + sustainability_weight * mean_fish``; the other fields
    are the statistics reported next to it. The statistics are computed on the
    last steps of every evaluation episode (the tail window of
    ``FisheryRegulatorEnv``), then averaged over the episodes and seeds of the
    candidate.

    Attributes
    ----------
    objective_score : float
        Scalar the ES maximizes (dimensionless), equal to
        ``harvest_score + sustainability_weight * mean_fish``.
    mean_reward : float
        Mean per-step reward of the fishers over the tail window (reward
        units). Reported only; it does not enter the objective.
    collapse_rate : float
        Fraction in ``[0, 1]`` of the tail steps whose normalized biomass is
        below the sustainability threshold. Reported only.
    sustainability_penalty : float
        Mean relative shortfall of the normalized biomass below the
        threshold, in ``[0, 1]``. Reported only.
    total_fines : float
        Value the regulator environment passes as the fines paid (reward
        units). Reported only.
    mean_fish : float
        Mean normalized biomass, in the fraction of the carrying capacity
        ``K``. It enters the objective, weighted by the sustainability
        weight.
    min_fish : float
        Minimum normalized biomass over the tail steps (fraction of ``K``).
    mean_realized_harvest : float
        Mean of the logged realized harvest (biomass units per step).
    harvest_score : float
        Mean of the logged realized harvest divided by the maximum sustainable
        yield (dimensionless). It enters the objective.

    When to use: as the ``metrics`` payload of the ``done`` context that a
    regulator environment publishes for a candidate; build it with
    :meth:`from_metrics`, which computes the objective, rather than by hand.

    Examples
    --------
    >>> context = FitnessContext.from_metrics(
    ...     mean_reward=0.4,
    ...     collapse_rate=0.0,
    ...     sustainability_penalty=0.0,
    ...     sustainability_weight=2.0,
    ...     mean_fish=0.5,
    ...     harvest_score=1.0,
    ... )
    >>> context.objective_score
    2.0
    """

    objective_score: float
    mean_reward: float
    collapse_rate: float
    sustainability_penalty: float
    total_fines: float
    mean_fish: float
    min_fish: float
    mean_realized_harvest: float
    harvest_score: float

    @classmethod
    def from_metrics(
        cls,
        *,
        mean_reward: SupportsFloat,
        collapse_rate: SupportsFloat,
        sustainability_penalty: SupportsFloat,
        sustainability_weight: SupportsFloat,
        total_fines: SupportsFloat = 0.0,
        mean_fish: SupportsFloat = 0.0,
        min_fish: SupportsFloat = 0.0,
        mean_realized_harvest: SupportsFloat = 0.0,
        harvest_score: SupportsFloat = 0.0,
    ) -> "FitnessContext":
        """Build the context and compute ``objective_score`` from the statistics.

        The objective is ``harvest_score + sustainability_weight * mean_fish``;
        ``mean_reward``, ``collapse_rate`` and ``sustainability_penalty`` are
        stored for reporting but do not enter the objective. The stored statistics
        are converted to ``float``.

        Parameters
        ----------
        mean_reward : SupportsFloat
            Mean per-step reward of the fishers (reward units).
        collapse_rate : SupportsFloat
            Fraction in ``[0, 1]`` of tail steps below the sustainability
            threshold.
        sustainability_penalty : SupportsFloat
            Mean relative shortfall below the threshold, in ``[0, 1]``.
        sustainability_weight : SupportsFloat
            Weight of ``mean_fish`` in the objective (dimensionless). It is
            used for the objective only and is not stored on the context.
        total_fines : SupportsFloat, optional
            Fines paid, in reward units (default 0.0).
        mean_fish : SupportsFloat, optional
            Mean normalized biomass, fraction of ``K`` (default 0.0).
        min_fish : SupportsFloat, optional
            Minimum normalized biomass, fraction of ``K`` (default 0.0).
        mean_realized_harvest : SupportsFloat, optional
            Mean realized harvest, biomass units per step (default 0.0).
        harvest_score : SupportsFloat, optional
            Mean realized harvest over the maximum sustainable yield,
            dimensionless (default 0.0).

        Returns
        -------
        FitnessContext
            The populated context.
        """

        # An alternative objective that was considered is a weighted
        # log-utility: the sum of (1 - alpha) times the log of the mean reward and
        # alpha times the log of the mean fish level, where alpha is the
        # sustainability weight constrained to [0, 1] and both logged terms are
        # floored at 1e-8 to avoid log(0).
        objective = harvest_score + sustainability_weight * mean_fish

        return cls(
            objective_score=objective,
            mean_reward=float(mean_reward),
            collapse_rate=float(collapse_rate),
            sustainability_penalty=float(sustainability_penalty),
            total_fines=float(total_fines),
            mean_fish=float(mean_fish),
            min_fish=float(min_fish),
            mean_realized_harvest=float(mean_realized_harvest),
            harvest_score=float(harvest_score),
        )
