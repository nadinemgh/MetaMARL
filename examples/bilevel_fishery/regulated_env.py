"""Single-stock fishery benchmark: fishers, harvest and restoration, stock dynamics.

``N`` fishers share one fish stock ``B`` (biomass). The stock has a
Pella-Tomlinson-type production function (Pella & Tomlinson, 1969) in the
parametrization with an intrinsic growth rate ``r`` and a shape parameter
``p``:

    g(B) = (r / p) * B * (1 - (B / K)^p)

where ``K`` is the carrying capacity; ``p = 1`` gives the Schaefer logistic
model ``r * B * (1 - B / K)``. The maximum sustainable yield reference points
follow from ``g'(B) = 0``: ``B_msy = K * (1 / (p + 1))^(1 / p)``,
``MSY = r * K / (p + 1)^((p + 1) / p)`` and ``F_msy = MSY / B_msy``. The
environment adds Gaussian process noise proportional to the stock,
``sigma * N(0, 1) * B``, to the production.

Each fisher is a ``Fisherman`` agent holding two mechanisms, ``Fishing`` and
``Restore``. Their raw policy outputs ``z`` are unbounded and are mapped to a
fraction ``f = sigmoid(z / 4)`` in ``(0, 1)``:

- the ``harvest`` action is the fraction of the fisher's maximal catch
  ``m * F_msy * B / N`` (``m`` the unregulated fishing-mortality multiplier)
  that is removed from the stock;
- the ``restore`` action is the fraction of the maximal restoration
  ``restoration_effectiveness * K / N`` that is added to the stock.

The environment itself applies no regulation, no fine and no restoration cost.
The leaders' mechanisms of ``core.mechanism.algorithms`` (quota, subsidy,
penalty, social influence) act on the fishers' actions and rewards through the
``MultiAgentEnv`` step, which runs the leaders before the fishers. A fisher's
reward is its delivered harvest fraction, plus whatever the leaders' mechanisms
add.

Within one step the fishers act one after the other: each reads the stock that
the previous fisher left, removes its harvest and adds its restoration. Then
the ``pella_tomlinson`` transition recovers the step's harvest as the
difference between the stock one time index earlier and the current stock,
computes the production and the noise from the stock of that earlier index,
and caps the harvest at the available biomass. Every step pushes the series of
``FisheryMetricSchema`` (stock, growth, realized harvest, reference points)
into the environment's metric logger. The observation of each fisher is the
vector ``[stock / K, 0, usage / K, 0, 0]``, where the usage is the logged
harvest of the previous step; the entries left at zero are filled by the
leaders' mechanisms, for example social influence.

References
----------
Pella, J. J., & Tomlinson, P. K. (1969). A generalized stock production
model. Bulletin of the Inter-American Tropical Tuna Commission, 13(3),
421-458. https://www.iattc.org/BulletinsENG.htm
"""

import logging
from typing import Any, ClassVar

import numpy as np
from gymnasium.core import ActType

from core.agents.base import Agent, AgentConfig
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.utils import sigmoid

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

EPS = 1e-8
# These four indices are not used by any code and do not match the observation that
# ``Fisherman.observation`` builds: that vector has five entries, the normalized
# fish stock at index 0, the normalized usage at index 2 and zeros elsewhere
# (the leaders' mechanisms fill indices 3 and 4, for example the social
# influence on the harvest and on the restoration).
FISH_NORM = 0
USAGE_NORM = 1
HARVEST = 2
RESTORATION = 3


class Fisherman(Agent):
    """Fisher agent: sees the normalized stock and is paid its harvest fraction.

    The agent holds the ``Fishing`` and ``Restore`` mechanisms (built from
    ``FishermanConfig``) and overrides the two hooks that define the benchmark:
    ``observation`` and ``reward``. Construction and the application of the
    mechanisms are inherited from :class:`core.agents.base.Agent`.

    When to use: build it through ``FishermanConfig``, which the environment
    does for you; instantiate it directly only to test the observation or the
    reward in isolation.

    Examples
    --------
    >>> from core.mechanism.base import MDPState
    >>> fisher = Fisherman(id="fisherman:0", policy_id="fisher_policy", mechanisms={})
    >>> mdp = MDPState(state={"fish": 800.0, "usage": 0.0}, params={"K": 1000.0})
    >>> fisher.observation(mdp).obs["fisherman:0"][0].tolist()
    [0.800000011920929, 0.0, 0.0, 0.0, 0.0]
    """

    def observation(self, mdp: MDPState) -> MDPState:
        """Build this fisher's observation at the current step.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the ``fish`` and ``usage`` trajectories (biomass
            units) and the carrying capacity ``K`` in ``params``.

        Returns
        -------
        MDPState
            An MDP whose ``obs`` holds, for this agent, the ``float32`` vector
            ``[fish / K, 0, usage / K, 0, 0]`` of shape ``(5,)``, with the
            stock and the usage read at ``mdp.t``.
        """

        fish_norm = mdp.state["fish"][mdp.t] / max(mdp.params["K"], EPS)
        usage_norm = mdp.state["usage"][mdp.t] / max(mdp.params["K"], EPS)
        return MDPState(
            obs={
                self.id: np.asarray(
                    [fish_norm, 0.0, usage_norm, 0.0, 0.0], dtype=np.float32
                )
            }
        )

    def reward(self, mdp: MDPState) -> MDPState:
        """Pay the fisher its harvest fraction of the current step.

        The reward is the ``harvest`` action stored in ``mdp.actions`` at
        ``mdp.t``. After the ``Fishing`` mechanism has decoded the action, that
        value is the fraction in ``(0, 1)`` the fisher harvests, so the reward
        is dimensionless and grows with the catch.

        Parameters
        ----------
        mdp : MDPState
            Shared state whose ``actions`` hold this agent's ``harvest`` entry.

        Returns
        -------
        MDPState
            An MDP whose ``rewards`` hold this agent's reward.
        """

        return MDPState(rewards={self.id: mdp.actions[self.id]["harvest"][mdp.t]})


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
        Observation space of a fisher, a ``Box`` of shape ``(5,)`` in the
        fishery.

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
    """Harvest mechanism of a fisher: removes a share of its maximal catch.

    The mechanism turns the raw policy output ``z`` into a harvest fraction
    ``sigmoid(z / 4)`` and removes ``fraction * m * F_msy * B / N`` biomass from
    the stock, where ``B`` is the stock at the current step, ``N`` the number of
    agents in ``mdp.aids``, ``F_msy`` the harvest rate at maximum sustainable
    yield (per step) and ``m`` the ``unregulated_f_multiplier`` (dimensionless;
    a value above 1 lets the fishers harvest above the sustainable rate). The
    stock is read from the state each fisher receives, so the fishers harvest
    one after the other, each from what the previous one left.

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
    >>> fishing.apply(mdp, 0.5).state["fish"]
    [-60.0]
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
        ActType
            The fraction ``sigmoid(z / 4)`` in ``(0, 1)``, as a ``float``.
        """

        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return sigmoid(float(z[0]) / temperature)

    def apply(self, mdp: MDPState, harvest_frac: ActType) -> MDPState:
        """Remove the harvest from the stock, as a negative residual.

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
            A residual whose ``state["fish"]`` is minus the harvest (biomass
            units), to be added to the stock by the caller.
        """

        # Must apply DLETA
        fish = mdp.state["fish"][mdp.t]
        n_fishers = len(mdp.aids)
        max_harvest_multiplier = mdp.params[
            "unregulated_f_multiplier"
        ]  # allow unsustainable
        catch_capacity = max_harvest_multiplier * mdp.params["F_msy"] * fish / n_fishers
        harvest = harvest_frac * catch_capacity
        return MDPState(state={"fish": -harvest})


class Restore(Mechanism):
    """Restoration mechanism of a fisher: adds biomass to the stock.

    The mechanism turns the raw policy output ``z`` into an effort
    ``sigmoid(z / 4)`` in ``(0, 1)`` and adds ``effort * e * K / N`` biomass to
    the stock, where ``e`` is ``restoration_effectiveness`` (dimensionless),
    ``K`` the carrying capacity and ``N`` the number of agents in ``mdp.aids``.
    With the default effectiveness of 0.05 and ``N`` fishers at full effort the
    stock gains ``0.05 * K`` per step. The effort has no cost in the
    environment: a cost and a reward for it come from the regulator's
    ``Subsidy`` mechanism, which targets this one with
    ``acts_on=("fisherman", "restore")``.

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
    ...     params={"restoration_effectiveness": 0.01, "K": 1000.0},
    ...     aids={"fisherman:0", "fisherman:1"},
    ... )
    >>> restore.apply(mdp, 1.0).state["fish"]
    [5.0]
    """

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
        ActType
            The effort ``sigmoid(z / 4)`` in ``(0, 1)``, as a ``float``.
        """

        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return sigmoid(float(z[0]) / temperature)

    def apply(self, mdp: MDPState, restoration_frac: ActType) -> MDPState:
        """Add the restored biomass to the stock, as a positive residual.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the agents in ``aids`` and, in ``params``,
            ``restoration_effectiveness`` and ``K`` (biomass units).
        restoration_frac : ActType
            Restoration effort in ``[0, 1]`` returned by :meth:`decode`.

        Returns
        -------
        MDPState
            A residual whose ``state["fish"]`` is the restored biomass
            (biomass units), to be added to the stock by the caller.
        """

        # Must apply DLETA
        n_fishers = len(mdp.aids)
        restoration_power = mdp.params["restoration_effectiveness"]
        restoration = restoration_power * mdp.params["K"] * restoration_frac / n_fishers
        return MDPState(state={"fish": restoration})


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
        - ``restoration_effectiveness`` : fraction of ``K`` that full effort
          of all fishers restores per step (default 0.05).
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
        ``unregulated_f_multiplier`` and ``restoration_effectiveness``.

    When to use: as the ``env`` of the inner (society) optimizer of a fishery
    experiment, with ``FisheryMetricSchema`` as its schema and an ``ecology_cfg``
    whose ``K`` matches the regulator environment's.

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
            "restoration_effectiveness": float(
                ecology_cfg.get("restoration_effectiveness", 0.05)
            ),
        }

    @reset
    def reset_fishery(self, mdp: MDPState) -> MDPState:
        """Draw the initial stock and expose the ecology to the mechanisms.

        The stock starts at ``fish_init`` when ``initial_stock_log_sigma`` is 0.
        Otherwise it is drawn from a lognormal distribution with median
        ``fish_init`` and log-scale spread ``initial_stock_log_sigma``, using
        the environment's seeded generator, and clipped to ``[1e-8, K]``. The
        usage (the logged harvest of the previous step) starts at 0.

        Parameters
        ----------
        mdp : MDPState
            State being reset; only used to satisfy the hook signature.

        Returns
        -------
        MDPState
            A residual with ``state["fish"]`` (biomass units),
            ``state["usage"]`` (biomass units per step) and ``params`` set to
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
        >>> residual.state["fish"], residual.state["usage"]
        ([800.0], [0.0])
        """

        # Must add delta
        if self.initial_stock_log_sigma == 0.0:
            fish_init = self.fish_init
        else:
            fish_init = np.clip(
                self.rng.lognormal(
                    mean=np.log(max(self.fish_init, EPS)),
                    # Standard deviation of the lognormal initial-stock distribution.
                    sigma=self.initial_stock_log_sigma,
                ),
                EPS,
                self.K,
            )
        return MDPState(state={"fish": fish_init, "usage": 0.0}, params=self.ecology)

    @transition
    def pella_tomlinson(self, mdp: MDPState) -> dict[str, float]:
        """Advance the stock one step and log the step's dynamics.

        The fishers have already acted on ``mdp.state["fish"][t]``. At ``t = 0``
        the stock ``B`` is that value and the harvest ``H`` is 0. From ``t = 1``
        on, ``B`` is the stock of the previous index and ``H`` is the difference
        between it and the current stock. The transition then computes the
        production ``(r / p) * B * (1 - (B / K)^p)`` plus the process noise
        ``sigma * N(0, 1) * B``, forms the available biomass ``max(B + growth,
        0)``, takes the realized harvest ``min(H, available)`` and leaves
        ``available - realized`` as the next stock. Because ``H`` is a stock
        difference, it also contains the growth of the previous step and the
        restoration, and it is negative when the restoration exceeds the
        removal.

        The method pushes ``B_msy``, ``MSY``, ``F_msy``, ``fish_stock``,
        ``fish_stock_next``, the four ``fish_norm_next_*`` fields, ``growth``,
        ``growth_noise``, ``H_realized`` and ``total_usage_norm`` into the
        environment's metric logger.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the ``fish`` trajectory (biomass units) and the
            ``ecology`` parameters in ``params``.

        Returns
        -------
        MDPState
            The state advanced by one time index, with the next stock appended
            to ``state["fish"]`` (biomass units) and the realized harvest to
            ``state["usage"]`` (biomass units per step).

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
        ...     ecology_cfg={"r": 0.3, "K": 1000.0, "p": 1.0, "fish_init": 800.0,
        ...                  "sigma": 0.0, "initial_stock_log_sigma": 0.0},
        ... )
        >>> mdp = MDPState().add(env.reset_fishery(MDPState()))
        >>> advanced = env.pella_tomlinson(mdp)
        >>> [round(value, 3) for value in advanced.state["fish"]]
        [800.0, 848.0]
        """

        # intervention already happened
        r = mdp.params["r"]
        p = mdp.params["p"]

        if mdp.t == 0:
            B = mdp.state["fish"][mdp.t]
            H = 0.0
        else:
            B = mdp.state["fish"][mdp.t - 1]
            H = B - mdp.state["fish"][mdp.t]

        noise = mdp.params["sigma"] * self.rng.normal() * B

        biological_growth = (r / p) * B * (1.0 - (B / self.K) ** p)
        growth = biological_growth + noise
        available = max(B + growth, 0.0)

        H_realized = min(H, available)
        fish_next = available - H_realized

        self.logger.push(key=("B_msy",), value=mdp.params["B_msy"])
        self.logger.push(key=("MSY",), value=mdp.params["MSY"])
        self.logger.push(key=("F_msy",), value=mdp.params["F_msy"])
        self.logger.push(key=("fish_stock",), value=mdp.state["fish"][mdp.t])
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
        self.logger.push(key=("growth",), value=growth)
        self.logger.push(key=("growth_noise",), value=noise)
        self.logger.push(key=("H_realized",), value=H_realized)
        self.logger.push(key=("total_usage_norm",), value=H_realized / max(EPS, self.K))

        return mdp.advance(state={"fish": fish_next, "usage": H_realized})
