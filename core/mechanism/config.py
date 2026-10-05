"""Configuration objects that build the mechanisms of an agent.

A mechanism is described by a frozen dataclass that holds its parameters and
knows which :class:`~core.mechanism.base.Mechanism` subclass to instantiate.
The agent configuration (:class:`~core.agents.base.AgentConfig`) stores these
descriptions and builds one mechanism per entry for the agent that owns it. The
module also holds :func:`empty_action_space`, the action space of a mechanism
that has no parameter for the regulator to search.
"""

from dataclasses import dataclass, fields
from typing import ClassVar, Optional

import numpy as np
from gymnasium import Space, spaces

from core.mechanism.base import Mechanism
from core.types import AgentID, MechanismID


def empty_action_space() -> spaces.Box:
    """Action space of a mechanism that has no searched parameter.

    A fixed rule still needs an entry in the regulator's action dictionary to
    be applied, so it declares a ``Box`` of shape ``(0,)``: the optimizer gives
    it an empty array and searches no dimension for it.

    Returns
    -------
    gymnasium.spaces.Box
        Space of shape ``(0,)``, bounds ``[0, 1]``, dtype ``float32``.

    Notes
    -----
    When to use: as the default ``action_space`` of a mechanism whose
    parameters are all fixed in its configuration, such as
    :class:`~core.mechanism.algorithms.penalty.ThresholdPenalty`.

    Examples
    --------
    >>> space = empty_action_space()
    >>> space.shape
    (0,)
    >>> space.sample().size
    0
    """
    return spaces.Box(low=0.0, high=1.0, shape=(0,), dtype=np.float32)


@dataclass(frozen=True)
class MechanismConfig:
    """Base description of a mechanism, from which it is built for an agent.

    A configuration is immutable and carries everything the mechanism
    constructor receives. Subclasses add their own parameters as fields, as
    keyword-only dataclass fields, and set ``mechanism_cls`` to the mechanism
    they build; :meth:`build` passes every field of the configuration to that
    class as a keyword argument. The base class itself points at the abstract
    :class:`~core.mechanism.base.Mechanism`, so it cannot be built.

    Attributes
    ----------
    action_space : gymnasium.Space
        Space of the regulator's action for this mechanism, a ``Box`` for the
        mechanisms of this package. The optimizer searches its dimensions.
    id : str or None
        Identifier of the mechanism, the key of its action in the regulator's
        action dictionary. Defaults to ``None``.
    acts_on : tuple[AgentID, MechanismID] or None
        Agent type and mechanism the mechanism acts on, for example
        ``("fisherman", "harvest")``. Defaults to ``None``; the mechanisms of
        this package raise a ``ValueError`` when they are used without it.
    obs_map : dict[str, str] or None
        Names under which the mechanism reads the environment, for example
        ``{"resource_level": "fish"}`` maps the resource level to the ``"fish"``
        state entry. Defaults to ``None``.
    default : numpy.ndarray or None
        Default action of the mechanism. Defaults to ``None``.
    mechanism_cls : type[Mechanism]
        Class variable, not a field: the mechanism class that :meth:`build`
        instantiates.

    Notes
    -----
    When to use: subclass it to make a mechanism available to an agent
    configuration; use the existing subclasses (``Quota``, ``Subsidy``,
    ``ThresholdPenalty``, ``SocialInfluence``) in an experiment.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.algorithms.quota import Quota
    >>> config = Quota(
    ...     id="quota",
    ...     action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    ...     acts_on=("fisherman", "harvest"),
    ...     obs_map={"resource_level": "fish"},
    ... )
    >>> mechanism = config.build("regulator")
    >>> type(mechanism).__name__, mechanism.aid, mechanism.id
    ('QuotaMechanism', 'regulator', 'quota')
    """

    action_space: Space
    id: MechanismID | None = None
    acts_on: tuple[AgentID, MechanismID] | None = None
    obs_map: Optional[dict[str, str]] = None
    default: np.ndarray | None = None

    mechanism_cls: ClassVar[type[Mechanism]] = Mechanism

    def build(self, aid: AgentID) -> Mechanism:
        """Instantiate ``mechanism_cls`` for the agent ``aid``.

        Parameters
        ----------
        aid : AgentID
            Identifier of the agent that owns the mechanism, typically the
            regulator.

        Returns
        -------
        Mechanism
            Instance of ``mechanism_cls`` built with ``aid`` and every field of
            this configuration as keyword arguments.
        """
        params = {f.name: getattr(self, f.name) for f in fields(self)}
        return self.mechanism_cls(aid=aid, **params)
