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
    """
    return spaces.Box(low=0.0, high=1.0, shape=(0,), dtype=np.float32)


@dataclass(frozen=True)
class MechanismConfig:
    action_space: Space
    id: MechanismID | None = None
    acts_on: tuple[AgentID, MechanismID] | None = None
    obs_map: Optional[dict[str, str]] = None
    default: np.ndarray | None = None

    mechanism_cls: ClassVar[type[Mechanism]] = Mechanism

    def build(self, aid: AgentID) -> Mechanism:
        params = {f.name: getattr(self, f.name) for f in fields(self)}
        return self.mechanism_cls(aid=aid, **params)
