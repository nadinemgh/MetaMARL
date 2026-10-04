"""Type aliases shared across the framework.

The aliases name the identifiers that flow between the ``World``, the
optimizers and the environments (``ContextID``, ``OptimizerID``) and the
container types exchanged with RLlib (``EnvType``, ``EnvConfigDict``). They
carry no runtime behaviour.
"""

from typing import Any, Hashable, TypeAlias, Union

from gymnasium import Env
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv

ContextID: TypeAlias = str
"""
Unique identifier for a context object.

Used to distinguish different context schemas stored in a World
(e.g., 'quota_violation', 'market_price', 'stock_level').

ContextIDs are semantic, not structural.
"""

OptimizerID: TypeAlias = str
"""
Unique identifier for an optimizer instance.

Used to:
- Namespace contexts inside a World
- Associate contexts with producing optimizers
- Support parent/child or upstream/downstream optimizer graphs

OptimizerIDs are expected to be stable for the lifetime of an experiment.
"""


EnvType: TypeAlias = Union[BaseEnv, MultiAgentEnv, Env]
"""
Represents a BaseEnv, MultiAgentEnv, ExternalEnv, ExternalMultiAgentEnv,
VectorEnv, gym.Env, or ActorHandle.
"""


EnvConfigDict: TypeAlias = dict
"""
Represents the env_config sub-dict of the algo config that is passed to
the env constructor.
"""

AgentID = Hashable
"""Represents a generic identifier for an agent (e.g., "agent1")."""

MultiAgentDict = dict[AgentID, Any]
"""A dict keyed by agent ids, e.g. {"agent-1": value}."""


PolicyID: TypeAlias = str
"""
Unique identifier for a policy.

Used to associate agents with the policy that controls their behavior and to
distinguish policies in multi-policy or multi-agent optimization settings.
"""


EpisodeID: TypeAlias = str
"""
Unique identifier for an environment episode.

Used to associate trajectories, metrics, contexts, and other runtime data with
the episode in which they were generated.
"""


MechanismID: TypeAlias = str
"""
Unique identifier for a action instance.
Note a mechanism is a specialized form of action

Used to distinguish mechanisms within composite or multi-mechanism settings
and to associate mechanism-specific parameters, contexts, or metrics with
their producing mechanism.
"""


SeedID: TypeAlias = str
"""
Unique identifier for an experimental seed.

Used to distinguish independent seeded runs and to associate evaluations,
trajectories, metrics, or other experiment data with the seed that generated
them.
"""
