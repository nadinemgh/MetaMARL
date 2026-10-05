"""Metric schema of one Evolution Strategies generation.

This module holds the pydantic ``MetricSchema`` classes through which
``ESOptimizer`` logs and reports one generation: ``ESParameterSchema`` for one
normalized parameter, ``ESCandidateSchema`` for one member of the population
and ``ESSchema`` for the whole generation. Series fields (``sigma``,
``fitness_mean``, ...) grow by one value per generation; ``by_mechanism`` holds
each candidate's fitness and parameter values, ``search_mean``/``global_best``/
``generation_best`` are keyed by parameter name, and ``inner`` is the reduced
metric schema of the inner optimizer, specialized at runtime (``RaySchema`` for
RLlib). The generation index is stored twice: in the ``iter`` field inherited
from ``MetricSchema`` and in the ``generation`` field of ``ESSchema``.
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

MechanismID: TypeAlias = str
ParameterName: TypeAlias = str


class ESParameterSchema(MetricSchema):
    """One normalized mechanism parameter tracked as a series over generations.

    Attributes
    ----------
    value : float or None
        Value of the parameter in ``[0, 1]`` (the ES search space, before the
        mechanism decodes it to its own units). Reduced as a series: each push
        appends one value. ``None`` until set.

    When to use: as the leaf of ``ESCandidateSchema`` and of the
    ``search_mean``, ``global_best`` and ``generation_best`` mappings of
    ``ESSchema``; there is rarely a reason to build one by hand.

    Examples
    --------
    >>> ESParameterSchema(value=0.4).value
    0.4
    """

    value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )


class ESCandidateSchema(MetricSchema):
    """One candidate of the population: its fitness series and its parameters.

    Attributes
    ----------
    fitness : float or None
        Scalar returned by the regulator environment for the candidate
        (objective units defined by the example). Reduced as a series.
    by_parameter : dict[str, ESParameterSchema]
        Normalized parameter values of the candidate, keyed by parameter name
        (``"quota"``, or ``"quota[0]"`` for a mechanism with several values).
        Empty by default; filled at runtime.

    When to use: as an entry of ``ESSchema.by_mechanism``, which keys the
    candidates of a generation by their index in the population.

    Examples
    --------
    >>> candidate = ESCandidateSchema(
    ...     fitness=1.5, by_parameter={"quota": ESParameterSchema(value=0.25)}
    ... )
    >>> candidate.by_parameter["quota"].value
    0.25
    """

    fitness: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )

    # Parameter names are runtime-defined by MechanismSpace.
    by_parameter: dict[str, ESParameterSchema] = Field(default_factory=dict)


class ESSchema(MetricSchema):
    """Metrics of one ES generation (see module docstring).

    ``ESOptimizer`` builds one instance per generation and pushes it to its
    ``MetricLogger``; the reporter then peeks the accumulated series. The
    generation index is stored in the inherited ``iter`` field and in the
    ``generation`` field.

    Attributes
    ----------
    generation : int or None
        Index of the generation (zero-based, no unit); the same value as
        ``iter``. Reduced as a series: each push appends one value.
    sigma : float or None
        Standard deviation of the search distribution in logit space, as it was
        when the population was sampled (dimensionless).
    population_size : int or None
        Number of candidates evaluated in the generation.
    fitness_mean, fitness_best : float or None
        Mean and maximum fitness of the generation's population (objective
        units defined by the example).
    best_mechanism_idx : int or None
        Index, in the population, of the generation's best candidate.
    best_fitness_global : float or None
        Best fitness seen since the start of the run.
    by_mechanism : dict[str, ESCandidateSchema]
        Fitness and parameters of each candidate, keyed by its population index
        as a string.
    search_mean : dict[str, ESParameterSchema]
        Mean of the search distribution at the start of the generation, keyed
        by parameter name, values in ``[0, 1]``.
    global_best : dict[str, ESParameterSchema]
        Best candidate seen so far, keyed by parameter name.
    generation_best : dict[str, ESParameterSchema]
        Best candidate of this generation, keyed by parameter name.
    inner : MetricSchema or None
        Reduced metrics of the inner optimizer for this generation.

    When to use: as the metric schema of the outer optimizer when declaring its
    reporting queries (``ESConfig().reporting(queries, schema=ESSchema)``).

    Examples
    --------
    >>> schema = ESSchema(
    ...     iter=0,
    ...     sigma=0.15,
    ...     population_size=1,
    ...     fitness_mean=1.0,
    ...     fitness_best=1.0,
    ...     best_mechanism_idx=0,
    ...     best_fitness_global=1.0,
    ...     by_mechanism={
    ...         "0": ESCandidateSchema(
    ...             fitness=1.0, by_parameter={"quota": ESParameterSchema(value=0.5)}
    ...         )
    ...     },
    ... )
    >>> schema.by_mechanism["0"].by_parameter["quota"].value
    0.5
    """

    generation: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    sigma: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    population_size: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    fitness_mean: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    fitness_best: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    best_mechanism_idx: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    best_fitness_global: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    by_mechanism: dict[MechanismID, ESCandidateSchema] = Field(default_factory=dict)
    search_mean: dict[MechanismID, ESParameterSchema] = Field(default_factory=dict)
    global_best: dict[MechanismID, ESParameterSchema] = Field(default_factory=dict)
    generation_best: dict[ParameterName, ESParameterSchema] = Field(
        default_factory=dict
    )
    inner: Optional[MetricSchema] = None
