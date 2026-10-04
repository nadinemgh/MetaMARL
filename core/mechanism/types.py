from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Trajectory(Generic[T]):
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
        cls, destination: dict[Any, Any], source: dict[Any, Any], t: int
    ) -> None:
        """Add the latest values from source into destination at timestep t."""

        for key, value in source.items():
            if isinstance(value, dict):
                if key not in destination:
                    destination[key] = {}

                if not isinstance(destination[key], dict):
                    raise TypeError(
                        f"Tree mismatch at key {key!r}: "
                        + "destination is a leaf but source is a branch."
                    )

                cls._add_at_t(destination[key], value, t)
                continue

            latest = value[-1]

            if key not in destination:
                destination[key] = [0] * t + [latest]

                continue

            if isinstance(destination[key], dict):
                raise TypeError(
                    f"Tree mismatch at key {key!r}: "
                    + "destination is a branch but source is a leaf."
                )

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
        return type(self)(self._copy_tree(self.data))

    @property
    def length(self) -> int:
        return self._length(self.data)

    def add(self, t: int, deltas: list["Trajectory[T]"]) -> "Trajectory[T]":
        result = self.copy()

        for delta in deltas:
            self._add_at_t(result.data, delta.data, t)

        return result

    def append(self, values: dict[Any, Any]) -> "Trajectory[T]":
        return self.update(self.length, values)

    def update(self, t: int, values: dict[Any, Any]) -> "Trajectory[T]":
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
    """Trajectory of per-step flows, such as rewards.

    A reward is earned at one timestep and does not persist to the next, so an
    unwritten timestep starts from ``0`` instead of the previous value. Deltas
    added at the same timestep are still summed, which is how a penalty or a
    subsidy composes with an agent's utility.
    """

    @classmethod
    def _fill_value(cls, history: list[Any]) -> Any:
        return 0
