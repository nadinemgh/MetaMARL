from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Optional

from gymnasium import Space
import numpy as np

from core.mechanism.base import Mechanism
from core.types import MechanismID, AgentID

@dataclass(frozen=True)
class MechanismConfig:
    action_space: Space
    id: MechanismID | None = None
    acts_on: tuple[AgentID, MechanismID] | None = None
    obs_map: Optional[dict[str, str]] = None
    default: np.ndarray | None = None

    mechanism_cls: ClassVar[type[Mechanism]] = Mechanism

    def build(self, aid: AgentID) -> Mechanism:
        params = {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }
        return self.mechanism_cls(aid, **params)