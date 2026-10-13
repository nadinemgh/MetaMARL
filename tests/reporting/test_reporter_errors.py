"""``Reporter`` path and query resolution: error messages name the offending values."""

import pytest

from core.reporting.base import Reporter
from core.reporting.query import Query


class RecordingReporter(Reporter):
    """Concrete ``Reporter`` whose backend does nothing."""

    def _report(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.unit
def test_nested_series_error_names_the_path():
    with pytest.raises(ValueError, match=r"reduction is required: \('a',\)") as exc:
        RecordingReporter()._resolve_path(path=("a",), metrics={"a": [[1.0], [2.0]]})

    assert "{path}" not in str(exc.value)


@pytest.mark.unit
def test_length_mismatch_error_names_the_x_series():
    query = Query(title="t", x=("x",), y=("y",))
    metrics = {"x": [0, 1, 2], "y": [1.0, 2.0]}

    with pytest.raises(ValueError, match=r"x=\('x',\) \(3\)") as exc:
        RecordingReporter()._resolve_query(metrics, query)

    assert "{query.x}" not in str(exc.value)
