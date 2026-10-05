"""Context record the fresh-water regulator environment publishes per candidate.

``FitnessContext`` is what ``WaterRegulatorEnv`` appends to the World for each
mechanism candidate once the inner rollouts have been aggregated. It is a
``ContextSchema`` and travels inside the ``metrics`` field of a
``MechanismContext``.
"""

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """Fitness of one water-policy candidate.

    Attributes
    ----------
    objective_score : float
        Scalar the ES maximizes: ``economic_weight * economic_score +
        sustainability_weight * sustainability_score``.
    economic_score : float
        Mean over episodes and seeds of the mean per-step reward of the farms
        (crop satisfaction minus policy penalty, reward units).
    level_deviation : float
        Mean over episodes and seeds of the relative change of the lake's filled
        level that the farms' withdrawals cause with respect to a world without
        withdrawals, ``sum|L - L0| / sum|L0|`` (dimensionless, zero for no
        change).
    sustainability_score : float
        ``1 / (1 + level_deviation)``, in ``(0, 1]``; 1 means the withdrawals
        leave the lake level untouched.

    When to use: as the ``metrics`` payload of the ``done`` context that
    ``WaterRegulatorEnv`` publishes for a candidate; build it with
    :meth:`from_scores`.

    Examples
    --------
    >>> context = FitnessContext.from_scores(
    ...     economic_score=0.6,
    ...     level_deviation=0.25,
    ...     economic_weight=1.0,
    ...     sustainability_weight=2.0,
    ... )
    >>> round(context.sustainability_score, 4), round(context.objective_score, 4)
    (0.8, 2.2)
    """

    objective_score: float
    economic_score: float
    level_deviation: float
    sustainability_score: float

    @classmethod
    def from_scores(
        cls,
        *,
        economic_score: float,
        level_deviation: float,
        economic_weight: float = 1.0,
        sustainability_weight: float = 1.0,
    ) -> "FitnessContext":
        """Combine the two scores into the objective of the candidate.

        Parameters
        ----------
        economic_score : float
            Mean per-step reward of the farms (reward units).
        level_deviation : float
            Relative change of the lake level, at least 0 (dimensionless).
        economic_weight, sustainability_weight : float, optional
            Weights of the two terms of the objective (default 1.0 each).

        Returns
        -------
        FitnessContext
            The populated context.
        """
        sustainability_score = 1.0 / (1.0 + float(level_deviation))
        objective = (
            economic_weight * float(economic_score)
            + sustainability_weight * sustainability_score
        )
        return cls(
            objective_score=float(objective),
            economic_score=float(economic_score),
            level_deviation=float(level_deviation),
            sustainability_score=sustainability_score,
        )
