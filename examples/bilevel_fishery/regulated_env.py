"""Single-stock fishery benchmark regulated by a ``FisheryMechanism`` vector.

``N`` fishers share a stock ``B_t`` (biomass) with Pella-Tomlinson growth

    B_{t+1} = B_t + (r / p) * B_t * (1 - (B_t / K)^p) + noise + restoration - H_t

where ``K`` is the carrying capacity, ``r`` the intrinsic growth rate, ``p`` the
shape parameter (``p = 1`` gives the Schaefer logistic model), ``noise`` is
multiplicative Gaussian process noise and ``H_t`` the realized total harvest.
Each agent's action has two unbounded components squashed through a sigmoid:

- ``action[0]``: harvest fraction of its maximal request
  ``full_required_harvest = m * F_msy * B_t / N`` (``m`` the unregulated
  fishing-mortality multiplier);
- ``action[1]``: restoration effort, converted to biomass through
  ``restoration_effectiveness``, charged quadratically through
  ``restoration_effort_cost`` and rewarded linearly through the mechanism's
  ``restoration_subsidy``.

The regulation is applied inside the environment itself from the six fields
of the current ``FisheryMechanism``: the quota (``fixed_quota``,
``max_demand_frac``) caps the delivered harvest smoothly, the fine
(``fine_amount``) and the risk penalty (``risk_penalty_scale``,
``risk_penalty_power``) are subtracted from the reward, and a collapse
penalty kicks in when the next-step biomass falls below
``collapse_stock_frac``. Every step pushes the ``FisheryMetricSchema``
series (stock, growth, harvests, reference points, quota allowance) and the
per-agent harvest requests into the env's metric logger.

References
----------
Pella, J. J., & Tomlinson, P. K. (1969). A generalized stock production
model. Inter-American Tropical Tuna Commission Bulletin, 13(3), 416-497.
"""

import logging
from typing import ClassVar

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
# Observation layout: the vector holds the normalized fish stock, the normalized
# usage, the harvest and the restoration, in that order.
FISH_NORM = 0
USAGE_NORM = 1
HARVEST = 2
RESTORATION = 3


class Fisherman(Agent):
    def observation(self, mdp: MDPState) -> MDPState:
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
        return MDPState(rewards={self.id: mdp.actions[self.id]["harvest"][mdp.t]})


class FishermanConfig(AgentConfig):
    agent_cls: ClassVar[type[Agent]] = Fisherman


class Fishing(Mechanism):
    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return sigmoid(float(z[0]) / temperature)

    def apply(self, mdp: MDPState, harvest_frac: ActType) -> MDPState:
        """Must apply DLETA"""
        fish = mdp.state["fish"][mdp.t]
        n_fishers = len(mdp.aids)
        max_harvest_multiplier = mdp.params[
            "unregulated_f_multiplier"
        ]  # allow unsustainable
        catch_capacity = max_harvest_multiplier * mdp.params["F_msy"] * fish / n_fishers
        harvest = harvest_frac * catch_capacity
        return MDPState(state={"fish": -harvest})


class Restore(Mechanism):
    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return sigmoid(float(z[0]) / temperature)

    def apply(self, mdp: MDPState, restoration_frac: ActType) -> MDPState:
        """Must apply DLETA"""
        n_fishers = len(mdp.aids)
        restoration_power = mdp.params["restoration_effectiveness"]
        restoration = restoration_power * mdp.params["K"] * restoration_frac / n_fishers
        return MDPState(state={"fish": restoration})


class FishingConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = Fishing


class RestoreConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = Restore


class FisheryRegulatedEnv(MultiAgentEnv):
    def __init__(self, *, ecology_cfg: dict, **kwargs):
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
        """Must add delta"""
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
