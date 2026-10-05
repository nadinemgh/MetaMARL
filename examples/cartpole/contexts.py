"""Context record the cart-pole regulator environment publishes per candidate.

``FitnessContext`` is what ``CartpoleRegulatorEnv`` appends to the World for
each mechanism candidate once the inner rollouts have been aggregated. It is a
``ContextSchema`` and travels inside the ``metrics`` field of a
``MechanismContext``.
"""

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """Fitness of one mechanism candidate of the cart-pole experiment.

    Attributes
    ----------
    objective_score : float
        Scalar the ES maximizes: the mean per-step reward of the balancing
        agent over the episodes and seeds of the candidate (reward units, at
        most ``1.0`` since the reward is ``1`` per step).
    mean_reward : float
        Same value as ``objective_score``, kept as a separate field for
        reporting symmetry with the fishery example.

    When to use: as the ``metrics`` payload of the ``done`` context that
    ``CartpoleRegulatorEnv`` publishes for a candidate; build it with
    :meth:`from_mean_reward`.

    Examples
    --------
    >>> FitnessContext.from_mean_reward(0.75).objective_score
    0.75
    """

    objective_score: float
    mean_reward: float

    @classmethod
    def from_mean_reward(cls, mean_reward: float) -> "FitnessContext":
        """Build the context whose objective is the mean reward.

        Parameters
        ----------
        mean_reward : float
            Mean per-step reward of the candidate (reward units).

        Returns
        -------
        FitnessContext
            The populated context.
        """

        return cls(objective_score=float(mean_reward), mean_reward=float(mean_reward))
