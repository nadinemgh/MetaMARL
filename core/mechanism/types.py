"""Time-indexed trees that store what a mechanism and an environment exchange.

A :class:`Trajectory` is a tree of dictionaries whose leaves are lists with one
entry per timestep. The state, the actions, the rewards and the observations of
an :class:`~core.mechanism.base.MDPState` are all trajectories, and the
residuals returned by the mechanisms are trajectories too, which is how they are
composed with the shared state. :class:`FlowTrajectory` differs only in how an
unwritten timestep is filled: stocks keep their previous value, flows restart
from zero.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Trajectory(Generic[T]):
    """Tree of per-timestep histories, treated as a stock.

    Every leaf of the tree is a list holding one value per timestep. A value
    that is not a ``dict`` becomes a one-entry list when the trajectory is built
    (a ``list`` is copied and taken as a history, so a NumPy array is a single
    timestep, not a history). The operations never modify the trajectory they
    are called on: :meth:`add`, :meth:`append` and :meth:`update` return a new
    one. A trajectory records stocks, such as the fish stock or the actions in
    force, so a timestep that nothing has written keeps the previous value; see
    :class:`FlowTrajectory` for quantities that do not persist.

    Attributes
    ----------
    data : dict
        The tree, with ``list`` leaves, one entry per timestep. Keys are agent
        identifiers, mechanism identifiers or state names. Defaults to an empty
        dictionary.

    Notes
    -----
    When to use: to hold anything that evolves along an episode and that
    mechanisms must be able to adjust at the current step, such as the state of
    the environment or the actions of the agents. Reach for
    :class:`FlowTrajectory` for rewards and observations.

    Examples
    --------
    >>> stock = Trajectory({"fish": 10.0})
    >>> stock = stock.append({"fish": 8.0})
    >>> stock["fish"]
    [10.0, 8.0]
    >>> stock.length
    2
    >>> stock.add(1, [Trajectory({"fish": -2.0})])["fish"]
    [10.0, 6.0]
    """

    data: dict[Any, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = self._normalize(self.data)

    @classmethod
    def _normalize(cls, node: Any) -> Any:
        """Turn every non-dict leaf into a trajectory list."""

        if isinstance(node, dict):
            return {key: cls._normalize(value) for key, value in node.items()}

        if isinstance(node, list):
            return node.copy()

        return [node]

    @classmethod
    def _copy_tree(cls, node: Any) -> Any:
        if isinstance(node, dict):
            return {key: cls._copy_tree(value) for key, value in node.items()}

        return node.copy()

    @classmethod
    def _length(cls, node: Any) -> int:
        if isinstance(node, dict):
            return max((cls._length(value) for value in node.values()), default=0)

        return len(node)

    @classmethod
    def _reaches(cls, node: Any, t: int) -> bool:
        """Whether some leaf holds at least ``t`` entries.

        Equivalent to ``cls._length(node) >= t`` for ``t > 0``, but it stops at
        the first leaf long enough, where the length visits every leaf.
        """
        if isinstance(node, dict):
            return any(cls._reaches(value, t) for value in node.values())

        return len(node) >= t

    @classmethod
    def _fill_value(cls, history: list[Any]) -> Any:
        """Value of a timestep that has not been written yet.

        A trajectory records stocks (state, actions in force), so an unwritten
        timestep keeps the previous value; see ``FlowTrajectory`` for flows.
        """

        return history[-1] if history else 0

    @classmethod
    def _carry_forward(cls, node: Any) -> None:
        """Open the next timestep on every existing leaf with its fill value."""

        if isinstance(node, dict):
            for value in node.values():
                cls._carry_forward(value)

            return

        if node:
            node.append(cls._fill_value(node))

    @classmethod
    def _set_at_t(
        cls, destination: dict[Any, Any], source: dict[Any, Any], t: int
    ) -> None:
        for key, value in source.items():
            if isinstance(value, dict):
                if key not in destination:
                    destination[key] = {}

                if not isinstance(destination[key], dict):
                    raise TypeError(
                        f"Tree mismatch at key {key!r}: "
                        + "destination is a leaf but source is a branch."
                    )

                cls._set_at_t(destination[key], value, t)
                continue

            value = value[-1]

            if key not in destination:
                destination[key] = [0] * t + [value]

                continue

            if isinstance(destination[key], dict):
                raise TypeError(
                    f"Tree mismatch at key {key!r}: "
                    + "destination is a branch but source is a leaf."
                )

            history = destination[key]

            while len(history) < t:
                history.append(cls._fill_value(history))

            if len(history) == t:
                history.append(value)
            else:
                history[t] = value

    @classmethod
    def _add_at_t(
        cls,
        destination: dict[Any, Any],
        source: dict[Any, Any],
        t: int,
        owned: set[int],
    ) -> None:
        """Add the latest values from source into destination at timestep t.

        ``destination`` belongs to the result being built, but the branches and
        lists below it may still be shared with the trajectory it was copied
        from. ``owned`` holds the ids of the nodes this result already owns; any
        other node is copied before it is written to.
        """

        for key, value in source.items():
            if isinstance(value, dict):
                if key not in destination:
                    destination[key] = {}
                    owned.add(id(destination[key]))

                if not isinstance(destination[key], dict):
                    raise TypeError(
                        f"Tree mismatch at key {key!r}: "
                        + "destination is a leaf but source is a branch."
                    )

                if id(destination[key]) not in owned:
                    destination[key] = dict(destination[key])
                    owned.add(id(destination[key]))

                cls._add_at_t(destination[key], value, t, owned)
                continue

            latest = value[-1]

            if key not in destination:
                destination[key] = [0] * t + [latest]
                owned.add(id(destination[key]))

                continue

            if isinstance(destination[key], dict):
                raise TypeError(
                    f"Tree mismatch at key {key!r}: "
                    + "destination is a branch but source is a leaf."
                )

            if id(destination[key]) not in owned:
                destination[key] = destination[key].copy()
                owned.add(id(destination[key]))

            history = destination[key]

            while len(history) <= t:
                history.append(cls._fill_value(history))

            # Avoid += here because leaves can be numpy arrays.
            # += could mutate an array shared with the original trajectory.
            history[t] = history[t] + latest

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.data

    def copy(self) -> "Trajectory[T]":
        """Return an independent copy of the tree.

        The lists are copied, so appending to the copy leaves the original
        alone. The values inside the lists, such as NumPy arrays, are shared.
        """
        return self._from_tree(self._copy_tree(self.data))

    @classmethod
    def _from_tree(cls, tree: dict[Any, Any]) -> "Trajectory[T]":
        """Wrap a tree that is already normalised, without walking it again."""
        trajectory = cls.__new__(cls)
        trajectory.data = tree
        return trajectory

    def write(self, path: tuple[Any, ...], t: int, value: Any) -> None:
        """Overwrite step ``t`` of the leaf at ``path``, in place.

        Unlike the other operations, this one modifies the trajectory it is
        called on. It copies the branches along ``path`` and the leaf before
        writing, because :meth:`add` shares the lists and branches it does not
        write to with the trajectory it was called on, and a write into a
        shared list would reach that trajectory too.

        Parameters
        ----------
        path : tuple
            Keys from the root to the leaf, for example ``(agent_id,
            mechanism_id)``.
        t : int
            Index of the step to overwrite, which must already exist.
        value : Any
            New value of the step.

        Raises
        ------
        KeyError
            If a key of ``path`` does not exist.
        IndexError
            If the leaf holds no step ``t``.

        Notes
        -----
        When to use: to record a value at the current step of a trajectory
        that the caller holds, as :meth:`core.mechanism.base.Mechanism.__call__`
        does with the decoded action.

        Examples
        --------
        >>> before = Trajectory({"f0": {"harvest": 0.5}})
        >>> after = before.add(0, [])
        >>> after.write(("f0", "harvest"), 0, 0.25)
        >>> after["f0"]["harvest"], before["f0"]["harvest"]
        ([0.25], [0.5])
        """
        node = self.data

        for key in path[:-1]:
            node[key] = dict(node[key])
            node = node[key]

        history = node[path[-1]].copy()
        history[t] = value
        node[path[-1]] = history

    @property
    def length(self) -> int:
        """Number of timesteps recorded, that is the length of the longest leaf.

        An empty trajectory has length ``0``.
        """
        return self._length(self.data)

    def add(self, t: int, deltas: list["Trajectory[T]"]) -> "Trajectory[T]":
        """Return a copy with the latest value of each delta added at step ``t``.

        For each leaf of each delta, the last entry is summed into the entry of
        the same path at step ``t``. A path that does not exist yet is created,
        with zeros before ``t``. A leaf that lags behind ``t`` is first filled
        up to ``t`` with its fill value, then summed. The sum is not done in
        place, so arrays shared with this trajectory are not modified. The
        lists and branches that no delta writes to are shared with this
        trajectory rather than copied; no operation of this class except
        :meth:`write` modifies a list in place, and :meth:`write` copies what it
        modifies, so the sharing is never visible through them.

        As in :meth:`update`, a step cannot be skipped: ``t`` is at most the
        current :attr:`length`, which is the step ``add`` opens when it is
        equal. A leaf may lag behind the rest of the tree, a whole tree may not
        lag behind ``t``. The check applies only when a delta holds a value to
        add, so adding nothing is always allowed.

        Parameters
        ----------
        t : int
            Timestep at which the deltas are added, between ``0`` and
            :attr:`length` included.
        deltas : list of Trajectory
            Residual trajectories; only the last entry of each leaf is used.

        Returns
        -------
        Trajectory
            New trajectory of the same type as this one.

        Raises
        ------
        ValueError
            If a delta holds a value and ``t`` is negative, or larger than the
            current length (a step cannot be skipped).
        TypeError
            If a path is a leaf in one tree and a branch in the other.
        """
        if any(delta.length for delta in deltas):
            if t < 0:
                raise ValueError("t must be non-negative")

            if t > 0 and not self._reaches(self.data, t):
                raise ValueError(
                    f"Cannot skip from trajectory length {self.length} "
                    + f"to timestep {t}"
                )

        # The result shares every list and branch that no delta writes to with
        # this trajectory, and copies the others before writing. A full copy
        # would cost the size of the whole tree at every add, and an
        # environment composes one residual per agent per step, which made an
        # episode quadratic in the number of agents.
        tree = dict(self.data)
        owned = {id(tree)}

        for delta in deltas:
            self._add_at_t(tree, delta.data, t, owned)

        return self._from_tree(tree)

    def append(self, values: dict[Any, Any]) -> "Trajectory[T]":
        """Return a copy with one more timestep holding ``values``.

        Equivalent to :meth:`update` at step :attr:`length`. Leaves that
        ``values`` does not mention receive their fill value for the new step.
        """
        return self.update(self.length, values)

    def update(self, t: int, values: dict[Any, Any]) -> "Trajectory[T]":
        """Return a copy with ``values`` written at step ``t``.

        Writing at an existing step overwrites it. Writing at step
        :attr:`length` opens a new step, on which every existing leaf first
        receives its fill value. A leaf that lags behind ``t`` is filled up to
        ``t`` before it is written, and a new path is created with zeros before
        ``t``. As in :meth:`add`, only the last entry of each leaf of ``values``
        is used.

        Parameters
        ----------
        t : int
            Timestep to write, between ``0`` and :attr:`length` included.
        values : dict
            Tree of values, normalised like the ``data`` of a trajectory.

        Returns
        -------
        Trajectory
            New trajectory of the same type as this one.

        Raises
        ------
        ValueError
            If ``t`` is negative, or larger than the current length (a step
            cannot be skipped).
        TypeError
            If a path is a leaf in one tree and a branch in the other.
        """
        result = self.copy()

        if t < 0:
            raise ValueError("t must be non-negative")

        if t == result.length:
            self._carry_forward(result.data)
        elif t > result.length:
            raise ValueError(
                f"Cannot skip from trajectory length {result.length} to timestep {t}"
            )

        incoming = type(self)(values)

        self._set_at_t(result.data, incoming.data, t)

        return result


class FlowTrajectory(Trajectory[T]):
    """Trajectory of per-step values, such as rewards and observations.

    A reward is earned at one timestep and does not persist to the next, so an
    unwritten timestep starts from ``0`` instead of the previous value. Deltas
    added at the same timestep are still summed, which is how a penalty or a
    subsidy composes with an agent's utility.

    Observations follow the same rule: every agent rebuilds its whole
    observation at each step, so adding it on top of the previous one would
    hand the policy a running sum. Contributions written at the same timestep
    are summed, which lets a mechanism fill entries of the observation vector
    that the agent leaves at zero.

    Attributes
    ----------
    data : dict
        The tree, with ``list`` leaves, one entry per timestep. Defaults to an
        empty dictionary.

    Notes
    -----
    When to use: for the rewards and the observations of an episode, and for
    the residuals that a mechanism adds to them.

    Examples
    --------
    >>> reward = FlowTrajectory({"fisherman:0": 1.0}).append({})
    >>> reward["fisherman:0"]
    [1.0, 0]
    >>> reward.add(1, [FlowTrajectory({"fisherman:0": -0.25})])["fisherman:0"]
    [1.0, -0.25]
    """

    @classmethod
    def _fill_value(cls, history: list[Any]) -> Any:
        return 0
