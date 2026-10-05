"""Context schemas exchanged through the shared ``World`` actor.

A ``Context`` is the unit of communication between the two levels of the
bilevel loop. The outer (regulator) optimizer publishes one
``MechanismContext`` per candidate and training seed, the inner RL
environments fetch the candidate that matches their index and seed, and the
regulator appends a ``done`` ``MechanismContext`` carrying the aggregated
fitness of each candidate. ``EnvStepContext`` is the schema for one
environment transition; the ``World`` can store such records, but no
environment of the current fishery pipeline publishes them. ``MechanismStatus``
records where a published mechanism stands in the publish, train, evaluate
cycle. This module holds only the data definitions; the registry that stores
them is ``core.world.base.World``.

Examples
--------
>>> payload = MechanismContext(
...     index=1,
...     env_id=None,
...     seed=7,
...     status=MechanismStatus.published,
...     mechanism={"quota": 0.3},
...     metrics=None,
... )
>>> ctx = Context(id=None, opt_id="opt", step=0, env="RegulatorEnv", payload=payload)
>>> ctx.payload.status.value, ctx.payload.mechanism
('published', {'quota': 0.3})
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, SupportsFloat

from gymnasium.core import ActType, ObsType
from pydantic import BaseModel
from ray.rllib.utils.typing import MultiAgentDict

from core.types import ContextID, MechanismID, OptimizerID


class MechanismStatus(Enum):
    """Lifecycle state of a mechanism candidate stored in the ``World``.

    The nominal path is ``published -> train -> eval -> done``:

    - ``published``: the regulator appended the candidate to the World (one
      entry per ``(candidate index, seed)``) and nobody has claimed it yet.
    - ``train``: a training environment fetched it through
      ``World.get_mechanism_by_id(mode=train)``. Only ``published`` entries can
      move here, so each ``(index, seed)`` entry is handed out exactly once.
    - ``eval``: an evaluation environment fetched it. Both ``train`` and
      ``eval`` entries qualify, so several evaluation seeds can share one
      trained mechanism. The regulator flushes ``eval`` entries between inner
      runs (``World.flush(status=eval)``).
    - ``done``: written by regulators when they publish the aggregated fitness
      of a candidate (``mechanism=None``, ``metrics`` filled in).

    Two values sit outside that path. ``assigned`` is set by the legacy
    ``World.get_mechanism`` / ``try_get_mechanism`` accessors, which hand out
    any published mechanism without matching an index or a seed. ``init`` is
    a placeholder for a context created before publication; the core code
    never assigns it.

    The same enum also stamps ``EnvStepContext.status`` with the mode of the
    producing environment (``train`` or ``eval``), and
    ``MultiAgentEnv.mode`` is built with ``MechanismStatus(mode)`` from the
    strings ``"train"`` and ``"eval"``.

    When to use: to read or set ``MechanismContext.status``, to pick the fetch
    mode of ``World.get_mechanism_by_id`` (``train`` or ``eval``), or to select
    which contexts ``World.flush`` drops. Compare members with ``is`` or ``==``
    rather than comparing their string values.

    Examples
    --------
    >>> MechanismStatus("eval") is MechanismStatus.eval
    True
    >>> [status.value for status in MechanismStatus]
    ['init', 'published', 'assigned', 'train', 'eval', 'done']
    """

    init = "init"
    published = "published"
    assigned = "assigned"
    train = "train"
    eval = "eval"
    done = "done"


class ContextSchema(BaseModel):
    """Base class of every payload that can be stored in a ``Context``.

    A subclass is a pydantic model, so its fields are validated and
    serialised when it is built. ``model_config`` allows arbitrary field types,
    which lets payloads carry objects that pydantic cannot describe, such as
    gymnasium spaces or NumPy arrays. The ``World`` treats payload types as
    schemas: its singleton check compares the exact type of two payloads.

    Attributes
    ----------
    model_config : dict
        Pydantic configuration, ``{"arbitrary_types_allowed": True}``.

    When to use: subclass it for each new kind of record exchanged between the
    regulator and the environments, for example the fitness record of a
    candidate (``FitnessContext`` in the fishery example).

    Examples
    --------
    >>> class FitnessNote(ContextSchema):
    ...     objective: float
    >>> FitnessNote(objective=0.75).objective
    0.75
    """

    model_config = {"arbitrary_types_allowed": True}


class MechanismContext(ContextSchema):
    """One mechanism candidate, as published by the regulator.

    The regulator publishes one instance in status ``published`` for each pair
    of candidate index and training seed; an inner environment claims it
    through ``World.get_mechanism_by_id`` and the status moves to ``train`` and
    then ``eval``. After scoring, regulators publish a further instance in
    status ``done`` with ``mechanism=None`` and the fitness in ``metrics``. All
    fields except ``eval_seed`` and ``mechanism`` are required: ``seed``,
    ``env_id`` and ``metrics`` have no default and must be passed explicitly,
    even as ``None``.

    Attributes
    ----------
    index : int
        Position of the candidate in the regulator's current batch,
        dimensionless and zero-based. Training environments are built with a
        matching ``mechanism_id`` and fetch their candidate by this index.
    env_id : str or None
        Identifier of the environment that produced the context. Publication
        from the regulator leaves it ``None``. ``World.update_context`` rejects
        a ``None`` value; ``append_context``, the path used by
        ``RegulatorEnv.step`` and by the regulators of the examples, does not.
    seed : int or None
        Policy (training) seed this copy of the candidate is meant for. The
        regulator publishes one copy per training seed so that each seeded
        policy trains against its own instance; ``done`` contexts use ``None``.
    eval_seed : int or None
        Environment seed intended for evaluating the candidate. Default
        ``None``; the current regulators never set it.
    status : MechanismStatus
        Lifecycle state, see ``MechanismStatus``.
    mechanism : dict[MechanismID, ActType] or None
        The candidate itself: a mapping from mechanism identifier (for example
        ``"quota"``) to the action of that mechanism, as decoded by the outer
        optimizer. Default ``None``; it is ``None`` on ``done`` contexts.
    metrics : ContextSchema or None
        Aggregated fitness payload, for example ``FitnessContext`` in the
        fishery example; filled only on ``done`` contexts and ``None``
        otherwise. A required field.

    When to use: build one to publish a candidate or its aggregated fitness
    through ``World.append_context``, or read one returned by
    ``World.get_mechanism_by_id`` to learn which mechanism an environment must
    apply.

    Examples
    --------
    >>> candidate = MechanismContext(
    ...     index=0,
    ...     env_id=None,
    ...     seed=101,
    ...     status=MechanismStatus.published,
    ...     mechanism={"quota": 0.4},
    ...     metrics=None,
    ... )
    >>> candidate.status.value, candidate.eval_seed
    ('published', None)
    """

    index: int
    env_id: Optional[str]
    seed: Optional[int]
    eval_seed: Optional[int] = None
    status: MechanismStatus
    mechanism: Optional[dict[MechanismID, ActType]] = None
    metrics: Optional[ContextSchema]


class EnvStepContext(ContextSchema):
    """Snapshot of one environment transition.

    An environment that records its transitions in the ``World`` appends one of
    these per ``reset`` (with ``reward=0.0`` and ``action=None``) and per
    ``step``. The ``World`` can store such records through ``append_context``.
    No environment of the current fishery pipeline publishes them: the
    regulator derives fitness from the inner optimizer's metrics instead. All
    fields are required.

    Attributes
    ----------
    env_id : int or None
        Index of the sub-environment inside its vectorised env runner, as set
        on the environment by the ``tag_episode_with_env_idx`` callback when
        the first episode is created; ``None`` before that.
    seed : int or None
        Seed of the environment dynamics (``MultiAgentEnv.seed``). During
        training it equals ``policy_seed``; during evaluation it is one of the
        configured evaluation seeds.
    policy_seed : int or None
        Seed identifying which trained policy acts in this environment.
        Together with ``mechanism`` it selects the RLModule named
        ``<policy>_m<mechanism>_s<policy_seed>``.
    status : MechanismStatus
        Mode of the producing environment: ``train`` or ``eval``.
    mechanism : int or None
        Index of the mechanism candidate the environment runs
        (``MultiAgentEnv.mechanism_id``), not the mechanism object itself.
    observation : ObsType or MultiAgentDict
        Observation returned to the agent(s) after the transition.
    observation_map : list of str or None
        Optional names for the entries of the observation vector. No code in
        ``core`` fills it.
    reward : SupportsFloat or MultiAgentDict or list of float
        Reward of the transition (``0.0`` on the reset record), in reward units.
    action : ActType or MultiAgentDict
        Action that produced the transition (``None`` on the reset record).
    info : dict or MultiAgentDict or None
        The ``info`` dictionary returned by the environment.

    When to use: when an environment should leave a per-step trace in the
    ``World`` that a regulator can read back, for example to score a candidate
    from raw transitions rather than from aggregated metrics.

    Examples
    --------
    >>> step = EnvStepContext(
    ...     env_id=0,
    ...     seed=101,
    ...     policy_seed=101,
    ...     status=MechanismStatus.train,
    ...     mechanism=0,
    ...     observation={"fisher_0": [0.8]},
    ...     observation_map=None,
    ...     reward={"fisher_0": 0.0},
    ...     action=None,
    ...     info=None,
    ... )
    >>> step.status.value, step.reward
    ('train', {'fisher_0': 0.0})
    """

    env_id: Optional[int]
    seed: Optional[int]
    policy_seed: Optional[int]
    status: MechanismStatus
    mechanism: Optional[int]
    observation: ObsType | MultiAgentDict
    observation_map: Optional[list[str]]
    reward: SupportsFloat | MultiAgentDict | list[float]
    action: ActType | MultiAgentDict
    info: dict | MultiAgentDict | None


@dataclass
class Context:
    """Runtime envelope around a payload stored in the ``World``.

    The ``World`` indexes contexts by ``id`` and by ``opt_id``; the payload
    carries the content. The envelope is a plain dataclass, so unlike the
    payloads it performs no validation.

    Attributes
    ----------
    id : ContextID or None
        Unique identifier within the World. ``None`` until the context is
        registered; ``World.append_context`` always overwrites it with a new
        UUID string.
    opt_id : OptimizerID or None
        Identifier of the optimizer that owns the context. ``None`` means that
        no optimizer owns it: the World then stores the context without an
        optimizer entry.
    step : int
        Step counter of the producing environment, dimensionless; ``0`` marks
        the first transition of an episode.
    env : str
        Name of the producing environment; the regulator uses its class name.
    payload : ContextSchema
        The content, for example a ``MechanismContext``.

    When to use: wrap a payload in a ``Context`` before handing it to
    ``World.append_context``; the World returns the same kind of object from
    its accessors.

    Examples
    --------
    >>> note = ContextSchema()
    >>> ctx = Context(id=None, opt_id="opt", step=3, env="RegulatorEnv", payload=note)
    >>> ctx.id is None, ctx.step
    (True, 3)
    """

    id: ContextID | None
    opt_id: Optional[OptimizerID]
    step: int
    env: str
    payload: ContextSchema
