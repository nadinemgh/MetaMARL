"""Social observation: each targeted agent sees its peers' last actions.

For targeted agents ``1..N`` whose targeted mechanism has a ``d``-dimensional
action, the observation of agent ``i`` at step ``t`` receives the actions its
peers delivered at step ``t - 1``:

    o_i[k : k + (N - 1) * d] = [a_{j, t-1} for j != i]

with ``k = obs_offset`` and the peers ordered by identifier. The targeted
agents must leave those entries of their observation at zero, because the
contribution is summed with the observation each agent builds.

This is observation shaping only. The counterfactual social-influence reward
of Jaques et al. (2019), ``r_i + beta * sum_j D_KL[...]``, is not implemented;
``influence_weight`` is reserved for it and has no effect. The rule has no
searched parameter, so the action space is empty.

References
----------
Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P., Strouse, D.,
Leibo, J. Z., & de Freitas, N. (2019). Social influence as intrinsic motivation
for multi-agent deep reinforcement learning. Proceedings of the 36th
International Conference on Machine Learning, PMLR 97:3040-3049.
arXiv:1810.08647.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from gymnasium import Space

from core.annotations import override
from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig, empty_action_space


class SocialInfluenceMechanism(Mechanism):
    """Expose the peers' last delivered actions in each targeted observation.

    Parameters
    ----------
    obs_offset : int
        Index ``k`` of the first observation entry reserved for the peers'
        actions, non-negative. Entries ``k`` to ``k + (N - 1) * d - 1`` must
        exist in the targeted agents' observation and be left at zero by them.
    influence_weight : float
        Reserved for the reward bonus of Jaques et al. (2019), non-negative;
        it has no effect.
    acts_on : tuple[AgentID, MechanismID]
        Agent type and mechanism whose delivered action is exposed, for example
        ``("fisherman", "harvest")``. The agents whose identifier starts with
        ``"<agent>:"`` and that hold an action for that mechanism are both the
        recipients and the peers. To expose several mechanisms, declare one
        instance per mechanism with disjoint entries.
    **kwargs
        Forwarded to :class:`core.mechanism.base.Mechanism`.

    Raises
    ------
    ValueError
        If ``obs_offset`` or ``influence_weight`` is negative.

    Notes
    -----
    When to use: the regulator makes behaviour public instead of constraining
    or pricing it. The exposed value is the delivered action, after the
    regulator's other mechanisms and the agent's own decoding, not the raw
    policy output. Nothing is exposed at reset, where no action exists yet.

    Examples
    --------
    >>> import dataclasses
    >>> import numpy as np
    >>> from core.mechanism.base import MDPState
    >>> mechanism = SocialInfluence(
    ...     id="social", acts_on=("fisherman", "harvest"), obs_offset=1
    ... ).build("regulator")
    >>> mdp = MDPState(
    ...     actions={
    ...         "fisherman:0": {"harvest": 0.25},
    ...         "fisherman:1": {"harvest": 0.75},
    ...     },
    ...     obs={"fisherman:0": np.zeros(2), "fisherman:1": np.zeros(2)},
    ... )
    >>> delta = mechanism.observe(dataclasses.replace(mdp, t=1))
    >>> delta.obs["fisherman:0"][0].tolist()
    [0.0, 0.75]
    """

    def __init__(
        self, *, obs_offset: int, influence_weight: float = 0.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.obs_offset = obs_offset
        self.influence_weight = influence_weight

        if self.obs_offset < 0:
            raise ValueError("obs_offset must be non-negative.")
        if self.influence_weight < 0.0:
            raise ValueError("influence_weight must be non-negative.")

    @override(Mechanism)
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Contribute nothing to the transition; see :meth:`observe`."""
        return MDPState()

    @override(Mechanism)
    def observe(self, mdp: MDPState) -> MDPState:
        """Return the peers' actions of step ``mdp.t - 1`` as an obs residual.

        Parameters
        ----------
        mdp : MDPState
            Shared state after the transition; the delivered actions are read
            at ``mdp.t - 1`` and the size of each observation from the one of
            that step.

        Returns
        -------
        MDPState
            Residual holding only ``obs``: per targeted agent a ``float32``
            vector of the size of its observation, zero outside the reserved
            entries. Empty at reset and when an agent has no peer.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing or the reserved entries do not fit in an
            agent's observation.
        """
        if self.acts_on is None:
            raise ValueError("SocialInfluenceMechanism requires `acts_on`.")

        if mdp.t == 0:
            return MDPState()

        target_agent, target_mechanism = self.acts_on

        # Sorted so that the order of the peers does not depend on the order
        # in which the policies' actions reached the environment.
        delivered = {
            aid: np.asarray(a[target_mechanism][mdp.t - 1], dtype=np.float32).reshape(
                -1
            )
            for aid, a in sorted(
                mdp.actions.data.items(), key=lambda item: str(item[0])
            )
            if str(aid).startswith(f"{target_agent}:") and target_mechanism in a
        }

        dobs = {}

        for aid in delivered:
            peers = [action for other, action in delivered.items() if other != aid]

            if not peers:
                continue

            signal = np.concatenate(peers)
            size = np.asarray(mdp.obs[aid][mdp.t - 1]).reshape(-1).size
            end = self.obs_offset + signal.size

            if end > size:
                raise ValueError(
                    "SocialInfluenceMechanism needs observation entries "
                    + f"{self.obs_offset} to {end - 1} of {aid!r}, "
                    + f"whose observation has {size} entries."
                )

            contribution = np.zeros(size, dtype=np.float32)
            contribution[self.obs_offset : end] = signal
            dobs[aid] = contribution

        return MDPState(obs=dobs)


@dataclass(frozen=True, kw_only=True)
class SocialInfluence(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = SocialInfluenceMechanism
    action_space: Space = field(default_factory=empty_action_space)
    obs_offset: int
    influence_weight: float = 0.0
