from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from core.annotations import override
from core.mechanism.base import ActType, Mechanism, MDPState
from core.mechanism.config import MechanismConfig
from core.types import AgentID, MultiAgentDict
from core.utils import (
    sigmoid,
    smooth_positive_zero_at_origin,
)


EPS = 1e-8

# TODO what if two mechanisms interfere by requiring context from each other ? 
# for example a penalty based on how much quota is violated ?
class QuotaMechanism(Mechanism):
    """Apply a quota as a residual on targeted agents' actions."""
    # Fixed algorithmic parameters.
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03

    def __init__(
        self,
        *,
        quota_transition_width: float = 0.03,
        usage_transition_width: float = 0.005,
        violation_transition_width: float = 0.03,
        **kwargs,
    ):
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

    @override(Mechanism)
    def decode(self, action: ActType) -> float:
        """Decode the regulator action into a quota parameter in [0, 1]."""
        value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        return float(np.clip(value, 0.0, 1.0))

    @override(Mechanism)
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        if self.acts_on is None:
            raise ValueError("QuotaMechanism requires `acts_on`.")

        if self.obs_map is None or "resource_level" not in self.obs_map:
            raise ValueError("QuotaMechanism requires obs_map['resource_level'].")

        target_agent, target_mechanism = self.acts_on
        resource_level=mdp.state[self.obs_map["resource_level"]][mdp.t] / max(mdp.params["K"], EPS)
        
        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - action) / width)
        upper = sigmoid((1.0 - action) / width)
        current = sigmoid((resource_level - action) / width)
        allowed_frac = (current - lower) / max(upper - lower, EPS)

        da = {}

        for aid, a in (mdp.actions or {}).items():
            if not str(aid).startswith(f"{target_agent}:"):
                continue
    
            if target_mechanism not in a:
                continue

            requested = np.asarray(a[target_mechanism], dtype=np.float32)
            requested_frac = float(requested.reshape(-1)[0])
            excess = smooth_positive_zero_at_origin(
                requested_frac - allowed_frac,
                self.usage_transition_width,
            )
            delivered_frac = requested_frac - excess
            delivered = requested.copy().reshape(-1)
            delivered[0] = delivered_frac
            delivered = delivered.reshape(requested.shape)
            da[aid] = {target_mechanism: delivered - requested}
            obs = {aid: np.asarray([0.0, 0.0, 0.0, 0.0, allowed_frac, action]) for aid in self.acts_on}

        return mdp.add(MDPState(actions=da, state={"allowed_frac": allowed_frac}, obs=obs))

@dataclass(frozen=True, kw_only=True)
class Quota(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = QuotaMechanism
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03