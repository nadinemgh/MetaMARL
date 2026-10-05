"""Environment-level metric schemas.

``AgentEnvStepSchema`` holds what is logged per agent at each step;
``EpisodeRolloutSchema`` holds episode-level statistics plus the ``by_agent``
dynamic node. Benchmarks subclass both to add their own fields (see
``examples/bilevel_fishery/metric_schema.py``). The schemas only declare
fields and how each one is reduced; the environments push the values to a
``MetricLogger`` built from them, and the reporting stack reads the reduced
result.
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

AgentID: TypeAlias = str


class AgentEnvStepSchema(MetricSchema):
    """Metrics logged for one agent during the steps of an environment.

    Each field is optional and ``None`` until a value is pushed. Floating-point
    fields are reduced with the mean over the pushed steps, ``terminated`` and
    ``truncated`` keep the last value. No core environment pushes these fields
    itself: they are the shared vocabulary that a benchmark fills, usually
    through a subclass that adds its own per-agent fields. The schema is the
    value type of the ``by_agent`` node of ``EpisodeRolloutSchema``.

    Attributes
    ----------
    iter : int or None
        Iteration counter inherited from ``MetricSchema`` (last value kept).
    base_action : float or None
        Counterpart of ``action`` named with the ``base_`` prefix (mean).
    action : float or None
        Scalar summary of the agent's action at a step (mean).
    observation : float or None
        Scalar summary of the agent's observation (mean).
    base_reward : float or None
        Counterpart of ``reward`` named with the ``base_`` prefix (mean).
    reward : float or None
        Reward of the agent at a step, in the benchmark's reward unit (mean).
    terminated, truncated : bool or None
        Whether the agent's episode ended by termination or by the time limit
        (last value).
    logp : float or None
        Log-probability of the chosen action under the policy (mean).
    value_pred : float or None
        Value estimate of the policy at the step, in reward units (mean).
    value_target : float or None
        Target the value estimate is trained toward, in reward units (mean).
    advantage : float or None
        Advantage estimate of the step, in reward units (mean).
    td_error : float or None
        Temporal-difference error, in reward units (mean).
    q_pred : float or None
        Action-value estimate, in reward units (mean).

    When to use: as the base class of a per-agent metric schema, to log how
    each agent behaved at every step of an episode and to query it through
    ``by_agent`` in the reporting queries.

    Examples
    --------
    >>> step = AgentEnvStepSchema(action=0.5, terminated=False)
    >>> step.action, step.reward
    (0.5, None)
    """

    # statistics collected at env-step level
    base_action: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    action: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    observation: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    base_reward: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    reward: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    terminated: Optional[bool] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    truncated: Optional[bool] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )

    # calculated stats
    logp: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_pred: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_target: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    advantage: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    td_error: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    q_pred: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )


class EpisodeRolloutSchema(
    MetricSchema
):  # attention this is aggregate by env not by env step
    """Episode-level metrics of one environment, plus the per-agent breakdown.

    The schema is aggregated by environment, not by environment step: the
    environment pushes one value per step and the logger reduces them at the end
    of the episode, according to the reduction listed for each field. The
    ``MultiAgentEnv`` pushes ``mechanism_id``, ``seed`` and ``policy_seed`` at
    reset, ``iter`` at every step, and the same mean follower reward of the step
    to the five ``reward_*`` fields, so their reductions give the total, the
    mean, the extremes and the last value of that per-step series. The Ray
    adaptor also builds an instance from RLlib's results (mean, minimum and
    maximum return, episode lengths and episode counts). The fields that
    nothing pushes stay ``None``.

    Attributes
    ----------
    iter : int or None
        Iteration counter inherited from ``MetricSchema`` (last value kept).
    env_id : int or None
        Index of the sub-environment inside its runner (last).
    mechanism_id : int or None
        Index of the mechanism candidate the environment trains against (last).
    seed : int or None
        Seed of the environment dynamics (last).
    policy_seed : int or None
        Seed of the trained policy (last).
    reward_total : float or None
        Sum of the per-step rewards over the episode, in the benchmark's
        reward unit (sum).
    reward_mean : float or None
        Mean per-step reward (mean).
    reward_min, reward_max : float or None
        Smallest and largest per-step reward (min and max).
    reward_terminal : float or None
        Reward of the last step (last).
    value_terminal, value_penultimate : float or None
        Value estimates at the last and second-to-last step, in reward units.
        These fields declare no reduction, so they follow the default of
        ``MetricSchema`` (mean).
    episode_len_mean, episode_len_max, episode_len_min : float or None
        Mean, largest and smallest episode length, in steps. All three are
        reduced with the mean.
    num_episodes : int or None
        Number of episodes in the reporting window (last).
    num_episodes_lifetime : int or None
        Number of episodes since the start of the run (last).
    by_agent : dict[AgentID, AgentEnvStepSchema]
        Per-agent metrics, a dynamic node filled at runtime, empty by default.
        Benchmarks override it with their own agent schema.

    When to use: as the schema of an environment's logger, or as the base class
    of a benchmark schema that adds its own episode-level fields such as stock
    levels, so that the reporting queries can select the reward series and the
    per-agent breakdown of an episode.

    Examples
    --------
    >>> from core.metrics.logger import MetricLogger
    >>> logger = MetricLogger.from_schema(EpisodeRolloutSchema)
    >>> for reward in (1.0, 2.0, 4.0):
    ...     logger.push(key=("reward_total",), value=reward)
    ...     logger.push(key=("reward_max",), value=reward)
    >>> logger.push(key=("by_agent", "f0", "action"), value=0.5)
    >>> reduced = logger.reduce()
    >>> reduced.reward_total, reduced.reward_max, reduced.by_agent["f0"].action
    (7.0, 4.0, 0.5)
    """

    env_id: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    mechanism_id: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    seed: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    policy_seed: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )

    # Reward (R) statistics
    reward_total: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SUM}
    )
    reward_mean: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    reward_min: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MIN}
    )
    reward_max: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MAX}
    )
    reward_terminal: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )

    # Value (V) statistics
    value_terminal: Optional[float] = None
    value_penultimate: Optional[float] = None
    episode_len_mean: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    episode_len_max: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    episode_len_min: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    num_episodes: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    num_episodes_lifetime: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )

    by_agent: dict[AgentID, AgentEnvStepSchema] = Field(default_factory=dict)
