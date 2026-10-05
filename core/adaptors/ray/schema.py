"""Typed view of an RLlib training/evaluation ``ResultDict``.

``RaySchema`` has an optional ``train`` and ``eval`` branch, each with the
rollout statistics (``aggregate`` plus the ``by_mechanism -> by_seed ->
by_episode`` grouping of the environment episode schemas), the per-policy
learner statistics and performance timers. Instances are produced by the
builders in ``core.adaptors.ray.utils`` and logged by ``RayOptimizer`` through
a ``MetricLogger`` built from ``RaySchema``. Every class derives from
``MetricSchema``, so each field carries the reduction (``MEAN``, ``LAST``...)
that the logger applies when it summarises an iteration or a whole run.
"""

from typing import Optional

from pydantic import Field

from core.envs.schema import EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema
from core.types import EpisodeID, MechanismID, PolicyID, SeedID


class PolicyLearnerSchema(MetricSchema):
    """Learner statistics of one RLModule (policy) for one training iteration.

    The fields mirror the statistics RLlib's learner reports for one module
    (see ``core.adaptors.ray.utils.build_learner`` for the source key of each
    one) plus a few derived quantities. Every field is optional and
    defaults to ``None``, meaning "not reported". Unless stated otherwise the
    values are dimensionless floats. ``json_schema_extra["reduce"]`` gives
    the reduction the logger applies across the samples of an iteration; every
    field uses ``MEAN``, and ``total_loss`` declares no reduction, which the
    logger treats as ``MEAN`` as well.

    Attributes
    ----------
    batch_size : float or None
        Mean number of samples per update, as RLlib reports it.
    total_loss, policy_loss, value_loss : float or None
        The optimized losses (reward-squared units for ``value_loss``).
    residual_variance : float or None
        Unexplained fraction of the value target, in [0, 1] when reported.
        ``build_learner`` never fills it.
    sample_staleness : float or None
        Age of the samples in learner updates: the sum of the lag indicators
        RLlib exposes (see ``build_learner``).
    policy_entropy : float or None
        Entropy of the action distribution, in nats.
    policy_entropy_coeff : float or None
        Coefficient weighting the entropy bonus in the loss.
    policy_relative_entropy : float or None
        Entropy divided by its coefficient.
    entropy_pressure : float or None
        Entropy multiplied by its coefficient, the size of the entropy term.
    policy_kl : float or None
        KL divergence between the behaviour and the current policy, in nats.
    policy_kl_coeff : float or None
        Coefficient weighting the KL penalty in the loss.
    value_mean, value_target : float or None
        Mean predicted and target returns, in reward units.
    gradient_norm : float or None
        Global norm of the gradient of the update.
    gradient_noise : float or None
        Noise statistic of the gradient estimate, when RLlib reports one.

    When to use: as the leaf of the learner tree to inspect whether one
    policy trains sensibly (loss, entropy, KL and gradient size), for example
    when a regulator candidate gives a flat or exploding learning curve.

    Examples
    --------
    >>> stats = PolicyLearnerSchema(policy_loss=0.25, policy_entropy=1.2)
    >>> stats.policy_loss, stats.value_loss
    (0.25, None)
    """

    batch_size: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # Value, Q, advantage debugging
    total_loss: Optional[float] = Field(
        default=None, json_schema_extra={"source": "total_loss"}
    )
    residual_variance: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    sample_staleness: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # Policy (π) debugging
    policy_loss: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_entropy: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_entropy_coeff: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_relative_entropy: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    entropy_pressure: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_kl: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_kl_coeff: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # Value (V) debgging
    value_loss: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_mean: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_target: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # Reward metrics. N.B. episode == trajectory

    # Gradient debugging
    gradient_norm: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gradient_noise: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )


class PerformanceSchema(MetricSchema):
    """Throughput and timing counters of one RLlib iteration.

    Every field is optional and defaults to ``None`` (not reported). The
    counters are reduced with ``LAST`` (the most recent value of the run is
    kept) and the throughput and timers with ``MEAN``.

    Attributes
    ----------
    env_steps_this_iter : float or None
        Environment steps sampled in the iteration (steps).
    env_steps_lifetime : float or None
        Environment steps sampled since the algorithm was built (steps).
    agent_steps_this_iter_sum, agent_steps_lifetime_sum : float or None
        Agent steps sampled in the iteration and since the algorithm was
        built, summed over the agents (steps).
    env_steps_throughput : float or None
        Environment steps sampled per second (steps / s).
    training_iteration_s : float or None
        Wall-clock duration of the whole iteration (seconds).
    training_step_s : float or None
        Wall-clock duration of one training step (seconds).
    sample_s : float or None
        Wall-clock duration of sampling (seconds).
    learner_update_s : float or None
        Wall-clock duration of the learner update (seconds).
    weights_seq_no : float or None
        RLlib's counter of weight broadcasts to the env runners.

    When to use: to check where the wall-clock time of an inner iteration
    goes (sampling against learning) and how many environment steps the
    regulator's fitness evaluations consume.

    Examples
    --------
    >>> perf = PerformanceSchema(env_steps_this_iter=1000.0, sample_s=2.0)
    >>> perf.env_steps_this_iter / perf.sample_s
    500.0
    """

    env_steps_this_iter: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    env_steps_lifetime: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    agent_steps_this_iter_sum: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    agent_steps_lifetime_sum: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    env_steps_throughput: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    training_iteration_s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    training_step_s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    sample_s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    learner_update_s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    weights_seq_no: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )


class SeedRolloutSchema(MetricSchema):
    """Episodes collected under one training seed, keyed by episode ID.

    Attributes
    ----------
    by_episode : dict of str to EpisodeRolloutSchema
        Episode statistics keyed by
        ``env=<idx>|m=<mechanism>|ps=<policy_seed>|ss=<env_seed>|n=<k>``, the
        tagged episode ID without its ``|raw=`` suffix followed by the position
        of the episode among those its sub-environment ended in the iteration
        (see ``build_rollout``). Empty by default.

    When to use: as the innermost level of ``RolloutSchema.by_mechanism``;
    read it to compare the episodes that one policy seed produced.

    Examples
    --------
    >>> episode = EpisodeRolloutSchema(mechanism_id=0, seed=1, reward_mean=2.0)
    >>> rollout = SeedRolloutSchema(by_episode={"env=0|m=0|ps=1|ss=1|n=0": episode})
    >>> rollout.by_episode["env=0|m=0|ps=1|ss=1|n=0"].reward_mean
    2.0
    """

    by_episode: dict[EpisodeID, EpisodeRolloutSchema] = Field(default_factory=dict)


class MechanismRolloutSchema(MetricSchema):
    """Rollouts of one mechanism candidate, keyed by training seed.

    Attributes
    ----------
    by_seed : dict of str to SeedRolloutSchema
        Rollouts keyed by the seed of the episode, as a string. Empty by
        default.

    When to use: as the middle level of ``RolloutSchema.by_mechanism``; the
    regulator reads one of these per candidate mechanism to compute its
    fitness.

    Examples
    --------
    >>> MechanismRolloutSchema().by_seed
    {}
    """

    by_seed: dict[SeedID, SeedRolloutSchema] = Field(default_factory=dict)


class RolloutSchema(MetricSchema):
    """Rollout statistics of one iteration.

    Attributes
    ----------
    aggregate : EpisodeRolloutSchema
        Episode schema summarised over every episode of the iteration, as
        RLlib reports it (reward and episode length statistics, episode
        counts). Required.
    by_mechanism : dict of str to MechanismRolloutSchema
        Per-mechanism, per-seed, per-episode breakdown that the regulator uses
        to compute one fitness per candidate. Keyed by mechanism ID as a
        string; empty by default.

    When to use: as the ``rollout`` branch of ``TrainSchema`` and
    ``EvalSchema``; ``aggregate`` is for monitoring, ``by_mechanism`` for the
    regulator's fitness.

    Examples
    --------
    >>> rollout = RolloutSchema(aggregate=EpisodeRolloutSchema(reward_mean=1.5))
    >>> rollout.aggregate.reward_mean, rollout.by_mechanism
    (1.5, {})
    """

    aggregate: EpisodeRolloutSchema
    by_mechanism: dict[MechanismID, MechanismRolloutSchema] = Field(
        default_factory=dict
    )


class SeedLearnerSchema(MetricSchema):
    """Learner statistics of the policies trained under one seed.

    Attributes
    ----------
    by_policy : dict of str to PolicyLearnerSchema
        Statistics keyed by the base policy ID (the part of the learner ID
        before ``_m<mechanism>_s<seed>``). Empty by default.

    When to use: as the innermost level of ``LearnerSchema.by_mechanism``;
    read it to compare the policies of different agent types that were
    trained together for one mechanism and seed.

    Examples
    --------
    >>> stats = SeedLearnerSchema(by_policy={"fisher": PolicyLearnerSchema()})
    >>> list(stats.by_policy)
    ['fisher']
    """

    by_policy: dict[PolicyID, PolicyLearnerSchema] = Field(default_factory=dict)


class MechanismLearnerSchema(MetricSchema):
    """Learner statistics of one mechanism candidate, keyed by training seed.

    The seed is the training (policy initialization) seed, not the seed of an
    evaluation environment.

    Attributes
    ----------
    by_seed : dict of str to SeedLearnerSchema
        Statistics keyed by the policy seed, as a string. Empty by default.

    When to use: as the middle level of ``LearnerSchema.by_mechanism``.

    Examples
    --------
    >>> MechanismLearnerSchema().by_seed
    {}
    """

    by_seed: dict[SeedID, SeedLearnerSchema] = Field(default_factory=dict)


class LearnerSchema(MetricSchema):
    """Learner statistics of one iteration for every trained RLModule.

    The tree mirrors the layout of the RLModules, named
    ``<policy>_m<mechanism>_s<seed>``: ``by_mechanism -> by_seed ->
    by_policy``. ``core.adaptors.ray.utils.build_learner`` fills it from the
    ``learners`` block of an RLlib result.

    Attributes
    ----------
    by_mechanism : dict of str to MechanismLearnerSchema
        Statistics keyed by mechanism ID as a string. Empty by default.

    When to use: as the ``learner`` branch of ``TrainSchema``, to follow how
    each seeded policy of each mechanism candidate learns.

    Examples
    --------
    >>> policy = PolicyLearnerSchema(policy_loss=0.5)
    >>> seed = SeedLearnerSchema(by_policy={"fisher": policy})
    >>> mechanism = MechanismLearnerSchema(by_seed={"101": seed})
    >>> learner = LearnerSchema(by_mechanism={"0": mechanism})
    >>> learner.by_mechanism["0"].by_seed["101"].by_policy["fisher"].policy_loss
    0.5
    """

    by_mechanism: dict[MechanismID, MechanismLearnerSchema] = Field(
        default_factory=dict
    )


class TrainSchema(MetricSchema):
    """Training branch of a result: rollouts, learner statistics and timers.

    Attributes
    ----------
    rollout : RolloutSchema
        Episodes sampled during the training iteration.
    learner : LearnerSchema
        Per-module learner statistics.
    performance : PerformanceSchema
        Throughput and timers.

    When to use: as the ``train`` branch of ``RaySchema``; built by
    ``RayOptimizer`` once per inner training iteration.

    Examples
    --------
    >>> train = TrainSchema(
    ...     rollout=RolloutSchema(aggregate=EpisodeRolloutSchema()),
    ...     learner=LearnerSchema(),
    ...     performance=PerformanceSchema(),
    ... )
    >>> train.learner.by_mechanism
    {}
    """

    rollout: RolloutSchema
    learner: LearnerSchema
    performance: PerformanceSchema


class EvalSchema(MetricSchema):
    """Evaluation branch of a result: rollouts and timers, no learner statistics.

    Attributes
    ----------
    rollout : RolloutSchema
        Episodes sampled by the evaluation env runners.
    performance : PerformanceSchema
        Throughput and timers of the evaluation pass.

    When to use: as the ``eval`` branch of ``RaySchema``; it holds the episodes
    from which the regulator reads the outcome of an inner training run.

    Examples
    --------
    >>> evaluation = EvalSchema(
    ...     rollout=RolloutSchema(aggregate=EpisodeRolloutSchema(reward_mean=3.0)),
    ...     performance=PerformanceSchema(),
    ... )
    >>> evaluation.rollout.aggregate.reward_mean
    3.0
    """

    rollout: RolloutSchema
    performance: PerformanceSchema


class RaySchema(MetricSchema):
    """Typed root of an RLlib ``ResultDict``.

    The ``train`` and ``eval`` branches are both optional. ``RayOptimizer``
    builds one instance per training iteration (with both branches, the
    ``eval`` one only when the result nests an evaluation block) and one per
    evaluation pass (``eval`` only), and pushes it into a ``MetricLogger``
    created from this class. Both branches are reduced with ``LAST`` at the
    subtree level, so the summary of a run is its last iteration.

    Attributes
    ----------
    train : TrainSchema or None
        Training branch; ``None`` when nothing was logged.
    eval : EvalSchema or None
        Evaluation branch; ``None`` when nothing was logged.

    When to use: pass it as the ``schema`` of ``RayOptimizerConfig.reporting``
    so the inner optimizer logs typed metrics that the regulator environment
    can aggregate into a fitness per mechanism.

    Examples
    --------
    >>> RaySchema().train is None
    True
    >>> evaluation = EvalSchema(
    ...     rollout=RolloutSchema(aggregate=EpisodeRolloutSchema(reward_mean=1.0)),
    ...     performance=PerformanceSchema(),
    ... )
    >>> RaySchema(eval=evaluation).eval.rollout.aggregate.reward_mean
    1.0
    """

    train: Optional[TrainSchema] = Field(
        default=None, json_schema_extra={"subtree_reduce": ReduceProtocol.LAST}
    )
    eval: Optional[EvalSchema] = Field(
        default=None, json_schema_extra={"subtree_reduce": ReduceProtocol.LAST}
    )
