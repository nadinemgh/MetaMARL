"""Enumerations of the reporting layer.

Two small enumerations whose string values are part of the configuration
format: the reporting backend and the time axis of a report. The reporters of
:mod:`core.reporting` do not read either of them.
"""

from enum import Enum


class ReporterType(Enum):
    """Reporting backend named in a configuration.

    ``wandb`` streams figures to Weights & Biases; ``local`` writes to disk
    (CSV or TensorBoard event files). The value is the string a configuration
    file would hold.

    When to use: to validate or branch on the backend name read from a
    configuration.

    Examples
    --------
    >>> ReporterType("wandb") is ReporterType.wandb
    True
    >>> [member.value for member in ReporterType]
    ['wandb', 'local']
    """

    wandb: str = "wandb"
    local: str = "local"


class Resolution(Enum):
    """Time axis of a report.

    ``env`` counts environment steps, ``inner`` counts inner-loop training
    iterations and ``outer`` counts outer-loop generations. The value is the
    name of that axis as a string.

    When to use: to name the x axis of a report by its unit (steps,
    iterations or generations).

    Examples
    --------
    >>> Resolution.outer.value
    'generation'
    >>> Resolution("train_iters") is Resolution.inner
    True
    """

    env: str = "env_steps"
    inner: str = "train_iters"
    outer: str = "generation"
