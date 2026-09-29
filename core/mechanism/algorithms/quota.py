from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism, MDPState
from core.mechanism.config import MechanismConfig
from core.types import MultiAgentDict
from core.utils import (
    sigmoid,
    smooth_positive_zero_at_origin,
)


EPS = 1e-8

# TODO what if two mechanisms interfere by requiring context from each other ? 
# for example a penalty based on how much quota is violated ?
class QuotaMechanism(Mechanism):
    # Fixed algorithmic parameters.
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03

    def __init__(
            self,
            *,
            quota_transition_width: Optional[float] = 0.03,
            usage_transition_width: Optional[float] = 0.005,
            violation_transition_width: Optional[float] = 0.03,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.quota_transition_width = quota_transition_width
        self.usage_transition_width = usage_transition_width
        self.violation_transition_width = violation_transition_width

        if self.quota_transition_width <= 0:
            raise ValueError("quota_transition_width must be > 0.")
        if self.usage_transition_width <= 0:
            raise ValueError("usage_transition_width must be > 0.")
        if self.violation_transition_width <= 0:
            raise ValueError("violation_transition_width must be > 0.")


    # TODO action needs to know about #1 current resource level and #2 full required
    # TODO we assume prior normalizaiton of action space 
    # TODO we assume prior selection of action component.
    # TODO maybe this should be mapped to constraint rather action projection
    @override(Mechanism)
    def apply(self, mdp_state: MDPState, **kwargs) -> MDPState:
        # TODO pass the state to action projection ?
        resource_level = mdp_state.obs[self.obs_map["resource_level"]]

        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - self.u) / width)
        upper = sigmoid((1.0 - self.u) / width)
        current = sigmoid((resource_level - self.u) / width)

        allowed_frac = (current - lower) / max(upper - lower, EPS)
        mdp_state.state["allowed_frac"] = allowed_frac

        requested = {}
        delivered = {}
        delta = {}

        for agent_id, action in mdp_state.actions.items():
            # TODO (nadine) temp cloning only identifiable with parsing
            if not str(agent_id).startswith(f"{self.acts_on[0]}:"):
                continue
            action = np.asarray(action, dtype=np.float32)
            requested_action = action.copy()
            regulated_action = action.copy()
            requested_frac = float(action[self.acts_on[1]])
            regulated_action[self.acts_on[1]] = (
                requested_frac - smooth_positive_zero_at_origin(
                    requested_frac - allowed_frac,
                    self.usage_transition_width,
                )
            )
            requested[agent_id] = requested_action
            delivered[agent_id] = regulated_action 
            delta[agent_id] = regulated_action - requested_action

        mdp_state.state["requested_action_dict"] = requested
        mdp_state.state["delivered_action_dict"] = delivered
        mdp_state.state["action_delta_dict"] = delta

        # add allowed fraction into obs
        mdp_state.obs = {aid: np.array([mdp_state.state["allowed_frac"]], dtype=np.float32)
            for aid, _ in mdp_state.obs.items()
        }
        return delta

@dataclass(frozen=True, kw_only=True)
class Quota(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = QuotaMechanism
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03