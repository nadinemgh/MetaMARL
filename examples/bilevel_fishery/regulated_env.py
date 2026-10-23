"""Single-stock fishery benchmark: fishers, harvest and restoration, stock dynamics.

``N`` fishers share one fish stock ``B`` (biomass). The stock follows the
standard discrete-time surplus-production model (Schaefer, 1954; Hilborn &
Walters, 1992), here with the Pella-Tomlinson production function
(Pella & Tomlinson, 1969) in the parametrization with an intrinsic growth rate
``r`` and a shape parameter ``p``:

    A(t) = max(B(t) + g(B(t)) + R(t) + noise(t), 0)

    B(t + 1) = clip(A(t) - C(t), 0, K)

    g(B) = (r / p) * B * (1 - (B / K)^p)

where ``K`` is the carrying capacity, ``R(t)`` the total restoration of the
fishers, ``A(t)`` the biomass available at the step, ``C(t) = min(H(t), A(t))``
the total catch actually delivered (``H(t)`` is the total catch requested) and
``noise(t) = sigma * N(0, 1) * B(t)`` Gaussian process noise proportional to
the stock. ``p = 1`` gives the Schaefer logistic model ``r * B * (1 - B / K)``.
The maximum sustainable yield reference points follow from ``g'(B) = 0``:
``B_msy = K * (1 / (p + 1))^(1 / p)``, ``MSY = r * K / (p + 1)^((p + 1) / p)``
and ``F_msy = MSY / B_msy``.

Each fisher is a ``Fisherman`` agent holding two mechanisms, ``Fishing`` and
``Restore``. Their raw policy outputs ``z`` are unbounded and are mapped to a
fraction ``f = sigmoid(z / 4)`` in ``(0, 1)``:

- the ``harvest`` action is the fraction of the fisher's maximal catch
  ``m * F_msy * B / N`` (``m`` the unregulated fishing-mortality multiplier)
  that is removed from the stock;
- the ``restore`` action is the fraction of the maximal restoration
  ``restoration_power * K / N`` that is added to the stock; ``restoration_power``
  is a parameter of ``RestoreConfig`` and the restoration also costs the fisher
  ``restoration_cost`` times the square of that fraction in reward.

The environment applies no quota and no fine. The leaders' mechanisms of
``core.mechanism.algorithms`` (quota, subsidy, penalty, social influence) act
on the fishers' actions and rewards through the ``MultiAgentEnv`` step, which
runs the leaders before the fishers. A fisher's reward is its harvest fraction
after the leaders' mechanisms have acted on it (a quota lowers it), minus its
restoration cost, plus whatever those mechanisms add to the reward, minus a
collapse penalty that the transition charges to every fisher. The harvest
fraction is read before the transition, so it ignores the pro-rata rationing
described below: when the biomass available cannot cover every request, a
fisher is still paid the fraction the quota let it request.

Within one step all the fishers act at the same time on the stock ``B(t)``
that the step starts with. A fisher's ``Fishing`` mechanism computes the catch
it requests from ``B(t)`` and writes it to ``state["harvest"]``, taking it out
of ``state["fish"]`` in place, and its ``Restore`` mechanism writes the biomass
it restores to ``state["restore"]`` and puts it into ``state["fish"]``. The
mechanisms of all the fishers read the same ``B(t)`` and their residuals are
summed afterwards, so the order of the fishers does not matter. The
environment also keeps ``B(t)`` in ``state["fish_start"]``, which no mechanism
writes to, and the ``pella_tomlinson`` transition reads it from there, rather
than rebuilding it from the sums, which in floating point would not return
exactly the stock the fishers read. The transition evaluates the production and
the noise on that stock and caps the total request at the biomass
available. When the total request exceeds it, every fisher receives the same
fraction of its request (a pro-rata share); the delivered total is ``C(t)``.
The catch is capped by the stock plus the growth of the step, restoration
included, whereas version 2 capped it by the stock alone. This is a modelling
choice of the reworked equations (commits 82fd437 and 553679a) that Nadine
should confirm: it lets a fisher catch biomass that is restored or produced
during the very step in which it is caught. Every step pushes the series of
``FisheryMetricSchema`` (stock, growth, requested and delivered catches,
reference points) into the environment's metric logger, when it was built with
a schema.

The observation of each fisher is a difference written into slot 0 (the stock
over ``K`` at the current step) and slot 2 (the harvest of the previous step
over ``K``), measured against what the current step already holds. The other
slots belong to the leaders' mechanisms, for example social influence, which
fill them through their own reset hooks. This module does not fix the number of
slots.

References
----------
Schaefer, M. B. (1954). Some aspects of the dynamics of populations important
to the management of the commercial marine fisheries. Bulletin of the
Inter-American Tropical Tuna Commission, 1(2), 27-56.
https://www.iattc.org/BulletinsENG.htm

Hilborn, R., & Walters, C. J. (1992). Quantitative Fisheries Stock Assessment:
Choice, Dynamics and Uncertainty. Chapman and Hall, New York.

Pella, J. J., & Tomlinson, P. K. (1969). A generalized stock production
model. Bulletin of the Inter-American Tropical Tuna Commission, 13(3),
421-458. https://www.iattc.org/BulletinsENG.htm
"""

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from gymnasium.core import ActType

from core.agents.base import Agent, AgentConfig
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.mechanism.utils import ACTION_TEMPERATURE
from core.utils import sigmoid

logger = logging.getLogger(__name__)
EPS = 1e-8


class Fisherman(Agent):
    # TODO (nadine) replace usage with inidividual harvest observation
    """Fisher agent: sees the normalized stock and is paid its harvest fraction.

    The agent holds the ``Fishing`` and ``Restore`` mechanisms (built from
    ``FishermanConfig``) and overrides the two hooks that define the benchmark:
    ``observation`` and ``reward``. Construction and the application of the
    mechanisms are inherited from :class:`core.agents.base.Agent`.

    When to use: build it through ``FishermanConfig``, which the environment
    does for you; instantiate it directly only to test the observation or the
    reward in isolation.

    The observation has nine slots, a layout inherited from the legacy fishery
    benchmark that the code marks as temporary. The fisher writes the stock
    fraction into slot 0 and the harvest fraction of the previous step into
    slot 2. The quota mechanism writes the allowed fraction (slot 1), the fixed
    quota (slot 3) and the parameters of its fine (slots 4 to 7), and the
    incentive mechanism writes the restoration subsidy (slot 8). The state
    keys the observation reads are ``fish``, ``harvest`` and ``restore``.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState
    >>> box = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
    >>> fisher = Fisherman(
    ...     id="fisherman:0",
    ...     policy_id="fisher_policy",
    ...     mechanisms={},
    ...     observation_space=box,
    ... )
    >>> mdp = MDPState(
    ...     state={"fish": [800.0, 640.0], "harvest": [100.0, 0.0]},
    ...     params={"K": 1000.0},
    ...     t=1,
    ... )
    >>> fisher.observation(mdp).obs["fisherman:0"][0].tolist()
    [0.6399999856948853, 0.0, 0.10000000149011612, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """

    def observation(self, mdp: MDPState) -> MDPState:
        """Build this fisher's observation at the current step.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the ``fish`` and ``harvest`` trajectories (biomass
            units), the carrying capacity ``K`` in ``params`` and the
            observations already written for the current step.

        Returns
        -------
        MDPState
            An MDP whose ``obs`` holds, for this agent, a ``float32`` residual
            with the shape of the observation space. Slot 0 receives
            ``fish[t] / K`` and slot 2 receives ``harvest[t - 1] / K`` (zero at
            the first step), each written as the difference from the value the
            current step already holds, so that adding the residual sets the
            slot. The other slots are left at zero: they belong to the leaders'
            mechanisms, for example social influence. The size of the
            observation is set by the observation space, not here.
        """

        K = max(mdp.params["K"], EPS)
        fish_norm = mdp.state["fish"][mdp.t] / K
        harvest_norm = mdp.state["harvest"][mdp.t - 1] / K if mdp.t > 0 else 0.0

        # Observation residual
        # Observations are per-step flows: the slot of the current step starts
        # at zero, so the difference is taken against what this step already
        # holds, not against the previous step.
        history = mdp.obs.data.get(self.id, [])
        current = (
            np.asarray(history[mdp.t], dtype=np.float32)
            if len(history) > mdp.t
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        do = np.zeros_like(current)
        do[[0, 2]] = [fish_norm, harvest_norm] - current[[0, 2]]

        return MDPState(obs={self.id: do})

    def reward(self, mdp: MDPState) -> MDPState:
        """Pay the fisher its harvest fraction of the current step.

        The reward is the ``harvest`` action stored in ``mdp.actions`` at
        ``mdp.t``. After the leaders' mechanisms and the ``Fishing`` mechanism
        have acted, that value is the fraction in ``(0, 1)`` the fisher
        requests once a quota has lowered it, so the reward is dimensionless.
        The pro-rata rationing of ``pella_tomlinson`` happens later, in the
        transition, and does not lower the reward
        (``tests/integration/test_fishery_regulator_composition.py`` checks the
        reward against these formulas).

        Parameters
        ----------
        mdp : MDPState
            Shared state whose ``actions`` hold this agent's ``harvest`` entry.

        Returns
        -------
        MDPState
            An MDP whose ``rewards`` hold this agent's reward.
        """

        harvest = mdp.actions[self.id]["harvest"][mdp.t]

        return MDPState(rewards={self.id: float(np.asarray(harvest).item())})


class FishermanConfig(AgentConfig):
    """Configuration that builds ``Fisherman`` agents.

    A frozen dataclass inherited from :class:`core.agents.base.AgentConfig`;
    only ``agent_cls`` changes.

    Attributes
    ----------
    policy_id : str
        Identifier of the RL policy that controls the fishers.
    mechanisms : MechanismConfig or tuple[MechanismConfig, ...]
        Mechanisms each fisher holds, normally ``FishingConfig`` (id
        ``"harvest"``) and ``RestoreConfig`` (id ``"restore"``).
    id : str or None
        Agent type; instances are named ``"<id>:<index>"``. Defaults to the
        class name of the agent.
    count : int
        Number of fishers built from this configuration (at least 1; default 1).
    shared_policy : bool
        Whether the fishers share one policy (default ``True``).
    observation_space : gymnasium.Space or None
        Observation space of a fisher, a ``Box`` of shape ``(9,)`` in the
        fishery. The fisher fills slots 0 and 2; the other slots belong to the
        leaders' mechanisms (see the temporary nine-slot layout in
        :class:`Fisherman`).

    When to use: in the ``agents`` of the inner optimizer configuration, once
    for the whole society of fishers.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> unbounded = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    >>> config = FishermanConfig(
    ...     id="fisherman",
    ...     policy_id="fisher_policy",
    ...     mechanisms=(
    ...         FishingConfig(action_space=unbounded, id="harvest"),
    ...         RestoreConfig(action_space=unbounded, id="restore"),
    ...     ),
    ... )
    >>> sorted(config.build().mechanisms)
    ['harvest', 'restore']
    """

    agent_cls: ClassVar[type[Agent]] = Fisherman


class Fishing(Mechanism):
    """Harvest mechanism of a fisher: requests a share of its maximal catch.

    The mechanism turns the raw policy output ``z`` into a harvest fraction
    ``sigmoid(z / 4)`` and requests ``fraction * m * F_msy * B / N`` biomass,
    where ``B`` is the stock at the current step, ``N`` the number of agents in
    ``mdp.aids``, ``F_msy`` the harvest rate at maximum sustainable yield (per
    step) and ``m`` the ``unregulated_f_multiplier`` (dimensionless; a value
    above 1 lets the fishers harvest above the sustainable rate). The request
    is written to ``state["harvest"]`` and taken out of ``state["fish"]`` in
    place. Every fisher computes its request from the same ``B`` and the
    residuals are summed afterwards, so the order of the fishers does not
    matter; the ``pella_tomlinson`` transition caps the total request at the
    biomass available and delivers the catches.

    When to use: as the ``harvest`` mechanism of ``FishermanConfig``, through
    ``FishingConfig``; the ``Quota`` mechanism of the regulator targets it with
    ``acts_on=("fisherman", "harvest")``.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState
    >>> box = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    >>> fishing = FishingConfig(action_space=box, id="harvest").build("fisherman:0")
    >>> fishing.decode(None, np.zeros(1, dtype=np.float32))
    0.5
    >>> mdp = MDPState(
    ...     state={"fish": 800.0},
    ...     params={"unregulated_f_multiplier": 2.0, "F_msy": 0.15},
    ...     aids={"fisherman:0", "fisherman:1"},
    ... )
    >>> fishing.apply(mdp, 0.5).state["harvest"]
    [60.0]
    """

    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Map the raw policy output to a harvest fraction.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Raw policy output ``z``, unbounded, of shape ``(1,)`` or anything
            ``numpy`` can flatten; only its first element is read.

        Returns
        -------
        float
            The fraction ``sigmoid(z / 4)`` in ``(0, 1)``.
        """

        z = np.asarray(action, dtype=np.float32).reshape(-1)

        return sigmoid(float(z[0]) / ACTION_TEMPERATURE)

    def apply(self, mdp: MDPState, harvest_frac: ActType) -> MDPState:
        """Record the catch this fisher requests from the current stock.

        Parameters
        ----------
        mdp : MDPState
            Shared state: the ``fish`` stock at ``mdp.t`` (biomass units), the
            agents in ``aids`` and, in ``params``, ``unregulated_f_multiplier``
            and ``F_msy``.
        harvest_frac : ActType
            Harvest fraction in ``[0, 1]`` returned by :meth:`decode`, possibly
            lowered by a regulator mechanism.

        Returns
        -------
        MDPState
            A residual whose ``state["harvest"]`` is the catch this fisher
            takes (biomass units) and whose ``state["fish"]`` is the same
            amount with a negative sign, so that the shared state removes it
            from the stock.
        """

        fish = mdp.state["fish"][mdp.t]
        n_fishers = len(mdp.aids)
        max_harvest_multiplier = mdp.params[
            "unregulated_f_multiplier"
        ]  # allow unsustainable
        catch_capacity = max_harvest_multiplier * mdp.params["F_msy"] * fish / n_fishers
        harvest = harvest_frac * catch_capacity

        return MDPState(state={"fish": -harvest, "harvest": harvest})


class Restore(Mechanism):
    """Restoration mechanism of a fisher: restores biomass to the stock.

    The mechanism turns the raw policy output ``z`` into an effort
    ``sigmoid(z / 4)`` in ``(0, 1)`` and restores ``effort * power * K / N``
    biomass, where ``power`` is ``restoration_power`` (dimensionless), ``K`` the
    carrying capacity and ``N`` the number of agents in ``mdp.aids``. The
    biomass is written to ``state["restore"]`` and put into ``state["fish"]``;
    the transition counts it in the growth of the step. With the default power
    of 0.05 and ``N`` fishers at full effort the stock gains ``0.05 * K`` per
    step. The effort also costs the fisher ``cost * effort^2`` in reward, where
    ``cost`` is ``restoration_cost``. Both parameters are set by
    ``RestoreConfig``; the ecology no longer carries a restoration parameter.
    The regulator's ``Subsidy`` mechanism can reward the effort on top of that,
    and targets this mechanism with ``acts_on=("fisherman", "restore")``.

    When to use: as the ``restore`` mechanism of ``FishermanConfig``, through
    ``RestoreConfig``.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState
    >>> box = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    >>> restore = RestoreConfig(action_space=box, id="restore").build("fisherman:0")
    >>> mdp = MDPState(
    ...     params={"K": 1000.0},
    ...     aids={"fisherman:0", "fisherman:1"},
    ... )
    >>> restore.apply(mdp, 1.0).state["restore"]
    [25.0]
    """

    def __init__(self, *, restoration_cost: float, restoration_power: float, **kwargs):
        super().__init__(**kwargs)

        self.restoration_cost = restoration_cost
        self.restoration_power = restoration_power

    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Map the raw policy output to a restoration effort.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Raw policy output ``z``, unbounded; only its first element is read.

        Returns
        -------
        float
            The effort ``sigmoid(z / 4)`` in ``(0, 1)``.
        """

        z = np.asarray(action, dtype=np.float32).reshape(-1)

        return sigmoid(float(z[0]) / ACTION_TEMPERATURE)

    def apply(self, mdp: MDPState, restoration_frac: ActType) -> MDPState:
        """Record the biomass this fisher restores at the current step.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the agents in ``aids`` and, in ``params``, the
            carrying capacity ``K`` (biomass units).
        restoration_frac : ActType
            Restoration effort in ``[0, 1]`` returned by :meth:`decode`.

        Returns
        -------
        MDPState
            A residual whose ``state["restore"]`` is the restored biomass
            (biomass units) and whose ``state["fish"]`` adds the same amount
            to the stock; ``rewards`` holds the quadratic restoration cost of
            the owning fisher.
        """

        n_fishers = len(mdp.aids)
        restoration = (
            self.restoration_power * mdp.params["K"] * restoration_frac / n_fishers
        )
        cost = self.restoration_cost * restoration_frac**2

        return MDPState(
            state={"fish": restoration, "restore": restoration},
            rewards={self.aid: -cost},
        )


class FishingConfig(MechanismConfig):
    """Configuration that builds the ``Fishing`` mechanism.

    A frozen dataclass inherited from
    :class:`core.mechanism.config.MechanismConfig`; only ``mechanism_cls``
    changes.

    Attributes
    ----------
    action_space : gymnasium.Space
        Space of the raw policy output, an unbounded ``Box`` of shape ``(1,)``
        and dtype ``float32`` in the fishery.
    id : str or None
        Mechanism identifier; the regulator targets ``"harvest"`` in
        ``acts_on``.
    acts_on : tuple[str, str] or None
        Unused by a fisher's own mechanism (default ``None``).
    obs_map : dict[str, str] or None
        Unused here (default ``None``).
    default : numpy.ndarray or None
        Unused here (default ``None``).

    When to use: in the ``mechanisms`` of ``FishermanConfig``, with
    ``id="harvest"``.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> box = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    >>> config = FishingConfig(action_space=box, id="harvest")
    >>> type(config.build("fisherman:0")).__name__
    'Fishing'
    """

    mechanism_cls: ClassVar[type[Mechanism]] = Fishing


@dataclass(frozen=True, kw_only=True)
class RestoreConfig(MechanismConfig):
    """Configuration that builds the ``Restore`` mechanism.

    A frozen dataclass inherited from
    :class:`core.mechanism.config.MechanismConfig`; only ``mechanism_cls``
    changes.

    Attributes
    ----------
    action_space : gymnasium.Space
        Space of the raw policy output, an unbounded ``Box`` of shape ``(1,)``
        and dtype ``float32`` in the fishery.
    id : str or None
        Mechanism identifier; the regulator targets ``"restore"`` in
        ``acts_on``.
    acts_on : tuple[str, str] or None
        Unused by a fisher's own mechanism (default ``None``).
    obs_map : dict[str, str] or None
        Unused here (default ``None``).
    default : numpy.ndarray or None
        Unused here (default ``None``).
    restoration_cost : float
        Reward cost of a full restoration effort; the cost of an effort ``f`` is
        ``restoration_cost * f^2`` (reward units, default 0.25).
    restoration_power : float
        Fraction of ``K`` that full effort of all the fishers adds to the stock
        per step (dimensionless, default 0.05).

    When to use: in the ``mechanisms`` of ``FishermanConfig``, with
    ``id="restore"``.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> box = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    >>> config = RestoreConfig(action_space=box, id="restore")
    >>> type(config.build("fisherman:0")).__name__
    'Restore'
    """

    restoration_cost: float = 0.25
    restoration_power: float = 0.05
    mechanism_cls: ClassVar[type[Mechanism]] = Restore


class FisheryRegulatedEnv(MultiAgentEnv):
    """Single-stock fishery that the inner level of the bilevel loop trains on.

    The environment owns the stock, its production function and the logging;
    the fishers' harvest and restoration are applied by their mechanisms and
    the regulation by the leaders' mechanisms, all through the step of
    :class:`core.envs.marl_regulated.MultiAgentEnv` (see the module docstring
    for the order of a step). It declares a reset hook, ``reset_fishery``, and a
    transition hook, ``pella_tomlinson``, that the base class discovers.

    Parameters
    ----------
    ecology_cfg : dict
        Ecology of the stock, read with these keys:

        - ``r`` : intrinsic growth rate per step (default 0.3).
        - ``K`` : carrying capacity in biomass units (falls back to
          ``max_fish``, then 1000.0; floored at ``1e-8``).
        - ``p`` : shape parameter of the production function (default 1.0,
          the Schaefer logistic; floored at ``1e-8``).
        - ``fish_init`` : initial stock in biomass units (falls back to
          ``B0``, then ``K``).
        - ``initial_stock_log_sigma`` : standard deviation of the log of the
          initial stock (default 0.05); 0 starts every episode at
          ``fish_init``.
        - ``sigma`` : standard deviation of the process noise as a fraction of
          the stock per step (default 0.05).
        - ``unregulated_f_multiplier`` : multiplier ``m`` of ``F_msy`` that
          bounds a fisher's catch (default 2.0).

        The restoration is not set here: ``restoration_power`` and
        ``restoration_cost`` belong to ``RestoreConfig``, and a
        ``restoration_effectiveness`` key is no longer read.
    **kwargs : Any
        Forwarded to :class:`core.envs.marl_regulated.MultiAgentEnv`: ``world``,
        ``mechanism_id``, ``horizon``, ``agents_cfg_dict``,
        ``leaders_cfg_dict``, ``seed``, ``schema`` (``FisheryMetricSchema``
        in the fishery; the environment logs through it and cannot reset
        without one), ``reporter_cfg`` and the other options of the base class.

    Attributes
    ----------
    K : float
        Carrying capacity (biomass units).
    fish_init : float
        Mean initial stock (biomass units).
    initial_stock_log_sigma : float
        Log-scale spread of the initial stock.
    ecology : dict[str, float]
        Parameters handed to the mechanisms as ``mdp.params``: ``r`` (per
        step), ``p``, ``K`` (biomass), ``B_msy`` (biomass), ``MSY`` (biomass
        per step), ``F_msy`` (per step), ``sigma``,
        and ``unregulated_f_multiplier``. The restoration parameters are set
        by ``RestoreConfig``, not by the ecology.

    When to use: as the ``env`` of the inner (society) optimizer of a fishery
    experiment, with ``FisheryMetricSchema`` as its schema. The carrying
    capacity ``K`` lives only here: the regulator environment reads none.

    Examples
    --------
    The reference points of the Schaefer model (``p = 1``) are
    ``B_msy = K / 2``, ``MSY = r * K / 4`` and ``F_msy = r / 2``:

    >>> from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
    >>> env = FisheryRegulatedEnv(
    ...     world=None,
    ...     mechanism_id=0,
    ...     agents_cfg_dict={},
    ...     leaders_cfg_dict={},
    ...     schema=FisheryMetricSchema,
    ...     ecology_cfg={"r": 0.3, "K": 1000.0, "p": 1.0, "sigma": 0.0},
    ... )
    >>> [round(env.ecology[name], 6) for name in ("B_msy", "MSY", "F_msy")]
    [500.0, 75.0, 0.15]
    """

    def __init__(self, *, ecology_cfg: dict, **kwargs: Any):
        super().__init__(**kwargs)

        r = ecology_cfg.get("r", 0.3)
        self.K = max(ecology_cfg.get("K", ecology_cfg.get("max_fish", 1000.0)), EPS)
        p = max(ecology_cfg.get("p", 1.0), EPS)
        B_msy = max(self.K * (1.0 / (p + 1.0)) ** (1.0 / p), EPS)
        MSY = r * self.K / (p + 1.0) ** ((p + 1.0) / p)
        self.fish_init = ecology_cfg.get("fish_init", ecology_cfg.get("B0", self.K))
        self.initial_stock_log_sigma = ecology_cfg.get("initial_stock_log_sigma", 0.05)
        self.ecology = {
            "r": r,
            "p": p,
            "K": self.K,
            "B_msy": B_msy,
            "MSY": MSY,
            "F_msy": MSY / B_msy,
            "sigma": ecology_cfg.get("sigma", 0.05),
            "unregulated_f_multiplier": ecology_cfg.get(
                "unregulated_f_multiplier", 2.0
            ),
        }

    # TODO (nadine) reset should not take mdp and init params should not be stateful
    @reset
    def reset_fishery(self, mdp: MDPState) -> MDPState:
        """Draw the initial stock and expose the ecology to the mechanisms.

        The stock starts at ``fish_init`` when ``initial_stock_log_sigma`` is 0.
        Otherwise it is drawn from a lognormal distribution with median
        ``fish_init`` and log-scale spread ``initial_stock_log_sigma``, using
        the environment's seeded generator, and clipped to ``[1e-8, K]``. The
        harvest recorded for the step starts at 0. The stock is stored twice,
        in ``state["fish"]``, which the fishers' mechanisms modify in place
        during the step, and in ``state["fish_start"]``, which they leave
        untouched so that the transition can read the stock the step started
        with.

        Parameters
        ----------
        mdp : MDPState
            State being reset; only used to satisfy the hook signature.

        Returns
        -------
        MDPState
            A residual with ``state["fish"]`` and ``state["fish_start"]``
            (biomass units, both the initial stock), ``state["harvest"]``
            (biomass units per step) and ``params`` set to
            the ``ecology`` dictionary itself, not a copy.

        Examples
        --------
        >>> from core.mechanism.base import MDPState
        >>> from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
        >>> env = FisheryRegulatedEnv(
        ...     world=None,
        ...     mechanism_id=0,
        ...     agents_cfg_dict={},
        ...     leaders_cfg_dict={},
        ...     schema=FisheryMetricSchema,
        ...     ecology_cfg={"K": 1000.0, "fish_init": 800.0, "sigma": 0.0,
        ...                  "initial_stock_log_sigma": 0.0},
        ... )
        >>> residual = env.reset_fishery(MDPState())
        >>> residual.state["fish"], residual.state["fish_start"]
        ([800.0], [800.0])
        >>> residual.state["harvest"]
        [0.0]
        """

        if self.initial_stock_log_sigma == 0.0:
            fish_init = self.fish_init
        else:
            # TODO (nadine) change to scipy truncnorm rather than clip to avoid
            #   flat signal
            fish_init = np.clip(
                self.rng.lognormal(
                    mean=np.log(max(self.fish_init, EPS)),
                    # sigma around sampling from lognormal distribution
                    sigma=self.initial_stock_log_sigma,
                ),
                EPS,
                self.K,
            )

        return MDPState(
            state={"fish": fish_init, "fish_start": fish_init, "harvest": 0.0},
            params=self.ecology,
        )

    def step(self, mdp: MDPState) -> MDPState:
        """Play one step and log the step's reward in ``reward_series``.

        The step is that of :class:`core.envs.marl_regulated.MultiAgentEnv`.
        Afterwards the mean over the fishers of the reward they received at
        this step is pushed to ``reward_series``, the series that the regulator
        environment reads to average the reward over the last steps of an
        episode.

        Parameters
        ----------
        mdp : MDPState
            State returned by ``reset`` or by the previous ``step``, holding
            the fishers' actions of the current step.

        Returns
        -------
        MDPState
            The state after the transition, as returned by the base class.
        """

        mdp = super().step(mdp)

        if self.logger is not None:
            step_reward = float(
                np.mean([mdp.rewards[aid][mdp.t - 1] for aid in self.followers])
            )
            self.logger.push(key=("reward_series",), value=step_reward)

        return mdp

    @transition
    def pella_tomlinson(self, mdp: MDPState) -> MDPState:
        """Advance the stock one step with the surplus-production equation.

        The fishers' mechanisms have already acted at step ``t = mdp.t``: each
        removed its request from ``state["fish"]`` and added it to
        ``state["harvest"]``, and each restoration added its biomass to
        ``state["fish"]`` and to ``state["restore"]``. The transition reads
        the stock at the beginning of the step, ``B``, from
        ``state["fish_start"]``, which the mechanisms never modify, instead of
        undoing their changes with ``fish - restore + harvest``: that sum
        differs from the stock the fishers read by rounding errors of the
        order of ``1e-12``. It computes the production ``g = (r / p) * B * (1 - (B /
        K)^p)`` and the process noise ``sigma * N(0, 1) * B``, both on ``B``.
        The growth is the production plus the noise plus the restoration, and
        the biomass available is ``max(B + growth, 0)``. The catch is the
        requested harvest, limited to the biomass available, and the next stock
        is the available biomass minus the catch, clipped to ``[0, K]``. The
        fishers are not rationed one by one: the request of the whole society
        is capped at once. The cap is the stock plus the growth of the step,
        restoration and noise included, whereas version 2 capped the catch by
        the stock alone; this is a modelling choice of the reworked equations
        for Nadine to confirm. Every follower also receives a collapse penalty, a
        smooth step from ``0.1`` towards ``0`` as the normalized next stock
        rises above ``0.2``.

        The method pushes ``B_msy``, ``MSY``, ``F_msy``, ``fish_stock`` (the
        stock ``B``), ``fish_stock_next``, the four ``fish_norm_next_*``
        fields and ``fish_norm_next_series``, ``growth`` (production,
        restoration and noise), ``growth_noise``, ``H_attempted`` (the total
        request), ``H_realized`` (the catch) and ``H_realized_series``, and
        ``total_usage_norm`` (the catch over ``K``) into the environment's
        metric logger, when it has one. For each follower it also pushes
        ``by_agent[aid].requested_harvest`` (the fraction it played times its
        catch capacity on ``B``) and ``by_agent[aid].delivered_harvest`` (that
        request, or its pro-rata share of ``H_realized`` when the cap binds).
        Without a schema it only advances the stock.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the ``fish``, ``fish_start``, ``harvest`` and
            ``restore`` trajectories (biomass units), the first, third and
            fourth written by the fishers' mechanisms, and the ``ecology``
            parameters in ``params``.

        Returns
        -------
        MDPState
            The state advanced by one time index, with the next stock appended
            to ``state["fish"]`` and ``state["fish_start"]`` (biomass units);
            the harvest and the restoration restart from zero at the new index.
            The collapse penalty is added to the rewards of the followers.

        Examples
        --------
        Two fishers request 100 together from a stock of 800 and nobody
        restores. With a production of ``0.3 * 800 * (1 - 0.8) = 48`` the
        biomass available is 848, the catch is the 100 requested and the next
        stock is 748:

        >>> import numpy as np
        >>> from core.mechanism.base import MDPState
        >>> from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
        >>> env = FisheryRegulatedEnv(
        ...     world=None,
        ...     mechanism_id=0,
        ...     agents_cfg_dict={},
        ...     leaders_cfg_dict={},
        ...     schema=FisheryMetricSchema,
        ...     ecology_cfg={"r": 0.3, "K": 1000.0, "p": 1.0, "fish_init": 800.0,
        ...                  "sigma": 0.0, "initial_stock_log_sigma": 0.0},
        ... )
        >>> env.rng = np.random.default_rng(env.seed)  # MultiAgentEnv.reset does it
        >>> mdp = MDPState().add(env.reset_fishery(MDPState()))
        >>> state = {"fish": -100.0, "harvest": 100.0, "restore": 0.0}
        >>> requests = MDPState(state=state)
        >>> advanced = env.pella_tomlinson(mdp.add(requests))
        >>> [round(float(value), 3) for value in advanced.state["fish"]]
        [700.0, 748.0]
        >>> advanced.state["harvest"]
        [100.0, 0.0]
        """

        # intervention already happened
        r = mdp.params["r"]
        p = mdp.params["p"]
        H = float(mdp.state["harvest"][mdp.t])
        R = float(mdp.state["restore"][mdp.t])
        # The stock at the start of the step, which the fishers' mechanisms
        # read before they changed state["fish"] in place.
        B = float(mdp.state["fish_start"][mdp.t])

        # Forward Euler discretization (dt = 1):
        # Biological growth is evaluated at the biomass B_t at the beginning
        # of the timestep, before harvesting and restoration.
        # These interventions affect B_{t+1}, but their influence on
        # biological growth appears from the following timestep onward.
        noise = mdp.params["sigma"] * self.rng.normal() * B
        biological_growth = (r / p) * B * (1.0 - (B / self.K) ** p)
        growth = biological_growth + noise + R
        available = max(B + growth, 0.0)
        H_realized = min(H, available)
        fish_next = np.float64(np.clip(available - H_realized, 0.0, self.K))

        # TODO (nadine): Refactor collapse penalty into a state-dependent
        # incentive mechanism evaluated after the environment transition.
        fish_norm_next = float(fish_next / self.K)
        collapse_penalty = 0.1 / (
            1.0 + np.exp(np.clip((fish_norm_next - 0.20) / 0.03, -60.0, 60.0))
        )
        mdp = mdp.add(
            MDPState(rewards={aid: -float(collapse_penalty) for aid in self.followers})
        )

        if self.logger is not None:
            self.logger.push(key=("B_msy",), value=mdp.params["B_msy"])
            self.logger.push(key=("MSY",), value=mdp.params["MSY"])
            self.logger.push(key=("F_msy",), value=mdp.params["F_msy"])
            self.logger.push(key=("fish_stock",), value=B)
            self.logger.push(key=("fish_stock_next",), value=fish_next)
            self.logger.push(
                key=("fish_norm_next_mean",), value=fish_next / max(self.K, EPS)
            )
            self.logger.push(
                key=("fish_norm_next_min",), value=fish_next / max(self.K, EPS)
            )
            self.logger.push(
                key=("fish_norm_next_max",), value=fish_next / max(self.K, EPS)
            )
            self.logger.push(
                key=("fish_norm_next_last",), value=fish_next / max(self.K, EPS)
            )
            self.logger.push(
                key=("fish_norm_next_series",), value=fish_next / max(self.K, EPS)
            )
            self.logger.push(key=("growth",), value=growth)
            self.logger.push(key=("growth_noise",), value=noise)
            self.logger.push(key=("H_attempted",), value=H)
            self.logger.push(key=("H_realized",), value=H_realized)
            self.logger.push(key=("H_realized_series",), value=H_realized)
            self.logger.push(
                key=("total_usage_norm",), value=H_realized / max(EPS, self.K)
            )

            # The shared state only holds the sum of the requests, so each
            # fisher's own request is rebuilt from the fraction it played,
            # which already carries the leaders' regulation, with the same
            # capacity ``Fishing.apply`` used on the stock B the step started with.
            catch_capacity = (
                mdp.params["unregulated_f_multiplier"]
                * mdp.params["F_msy"]
                * B
                / max(len(mdp.aids), 1)
            )
            # Pro-rata rationing: when the cap binds every fisher receives the
            # same fraction of its request, otherwise the request is delivered.
            delivered_share = H_realized / H if H_realized < H else 1.0

            for aid in self.followers:
                fractions = mdp.actions.data.get(aid, {}).get("harvest", [])

                if len(fractions) <= mdp.t:
                    continue

                request = float(np.asarray(fractions[mdp.t]).item()) * catch_capacity

                self.logger.push(
                    key=("by_agent", aid, "requested_harvest"), value=request
                )
                self.logger.push(
                    key=("by_agent", aid, "delivered_harvest"),
                    value=request * delivered_share,
                )

        # TODO (nadine) add observation
        return mdp.advance(
            state={
                "fish": fish_next,
                "fish_start": fish_next,
                "harvest": 0.0,
                "restore": 0.0,
            }
        )
