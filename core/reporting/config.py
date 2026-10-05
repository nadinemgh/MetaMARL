"""Reporter configurations: serializable factories for reporter instances.

A ``ReporterConfig`` travels through the optimizer configs and is copied into
every environment; ``build(label=...)`` creates the backend-specific reporter
for one owner (an optimizer or an environment instance). This module holds the
abstract base class; the concrete configurations live next to their reporters
in :mod:`core.reporting.csv`, :mod:`core.reporting.tensor_board` and
:mod:`core.reporting.wandb`.
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Self, Union

from core.reporting.base import Reporter


class ReporterConfig(ABC):
    """Serializable description of a reporter, built into one instance per owner.

    A configuration holds what is known when an experiment is declared (the
    project name and the backend options) and is cheap to copy; the reporter
    itself, which may open files or a network connection, is only created by
    :meth:`build`. ``world`` and ``outer_iters`` are filled in later by the
    optimizer that owns the config, before :meth:`build` is called;
    :meth:`copy` gives each environment its own instance. The class is
    abstract: a backend subclasses it and implements :meth:`build`.

    Parameters
    ----------
    project : str
        Name of the project or run group of the backend.

    Attributes
    ----------
    project_name : str
        The ``project`` argument.
    world : str or None
        Name of the world reported on; ``None`` until the optimizer sets it.
    outer_iters : int or None
        Number of outer-loop iterations of the run (a count, no unit); ``None``
        until the optimizer sets it.

    When to use: subclass it to add a backend; otherwise use one of the
    concrete configurations (CSV, TensorBoard, Weights & Biases) in the
    experiment configuration.

    Examples
    --------
    A copy is independent of the original:

    >>> from core.reporting.csv import CSVConfig
    >>> config = CSVConfig(project="fishery")
    >>> config.world is None
    True
    >>> config.world = "lake"
    >>> clone = config.copy()
    >>> clone.world = "other"
    >>> config.world
    'lake'
    """

    def __init__(self, project: str):
        self.project_name: str = project
        self._world_name: Union[str | None] = None
        self._outer_iters: Union[int | None] = None

    @property
    def world(self) -> Union[str, None]:
        """Name of the world reported on (``None`` until the optimizer sets it)."""

        return self._world_name

    @world.setter
    def world(self, world: str) -> None:
        """Set the world name used in run names and output directories."""

        self._world_name = world

    @property
    def outer_iters(self) -> Union[int, None]:
        """Number of outer-loop iterations of the run (``None`` until set)."""

        return self._outer_iters

    @outer_iters.setter
    def outer_iters(self, outer_iters: int) -> None:
        """Set the number of outer-loop iterations forwarded to the backend."""

        self._outer_iters = outer_iters

    def copy(self) -> Self:
        """Return a deep copy of this reporter configuration."""

        return copy.deepcopy(self)

    @abstractmethod
    def build(self, *, label: Optional[str] = None) -> Reporter:
        """Create the backend-specific reporter described by this config.

        Parameters
        ----------
        label : str or None, default None
            The owner of the reporter (an optimizer class name or an
            environment id), used by the backend to keep the outputs of two
            owners apart. The optimizers and environments always pass it.

        Returns
        -------
        Reporter
            A new reporter, not yet shared with any other owner.
        """

        ...
