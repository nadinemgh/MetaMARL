"""Small numeric, string and composition helpers shared across the package.

Three families live here. Tolerant scalar conversions are used by the metric
and reporting layers (``to_float``, ``finite``, ``safe_ratio``,
``flatten_numeric``, ``sanitize_key``, ``is_mapping``) together with the unique
identifier helper ``generate_uuid`` used by the ``World`` and the optimizer
configurations. Smooth, differentiable stand-ins for ``max``, ``min`` and
clipping are used when shaping mechanism rewards and agent utilities
(``sigmoid``, ``smooth_positive``, ``smooth_min``, ``smooth_cap_01``,
``smooth_positive_zero_at_origin``). Composition helpers (``intersect``,
``logical_or_dict``) combine the partial descriptions that several
mechanisms contribute to one ``MDPState``.
"""

import re
import uuid
from collections.abc import Container, Mapping
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

# Generic utils

#: Smallest transition width accepted by the smooth helpers; narrower widths
#: are clamped to it so that a division by the width cannot blow up.
EPS = 1e-8


def generate_uuid(registry: Container[Any]) -> str:
    """Generate a UUID4 string that is not already present in ``registry``.

    Draws ``uuid.uuid4()`` values until one is absent from ``registry``, so
    the result is unique within it. The registry is only read, never updated.

    Parameters
    ----------
    registry : Container[Any]
        Existing identifiers to avoid: anything that supports ``in``, such as
        a set, the keys of a dict or the dict itself.

    Returns
    -------
    str
        A fresh ``uuid.uuid4()`` string absent from ``registry``.

    When to use: to name a new context or optimizer in a registry that must not
    contain duplicates, as the ``World`` and the ``build_optimizer`` methods do.

    Examples
    --------
    >>> uid = generate_uuid({"a", "b"})
    >>> uid not in {"a", "b"}
    True
    """

    while True:
        id = str(uuid.uuid4())

        if id not in registry:
            return id


def safe_ratio(num: Any, den: Any) -> Optional[float]:
    """Divide two scalars, returning ``None`` instead of raising.

    Each operand goes through ``finite``, so strings that are not numbers,
    ``None``, NaN and infinities all make the ratio undefined.

    Parameters
    ----------
    num, den : Any
        Numerator and denominator; any value ``finite`` can convert.

    Returns
    -------
    float or None
        ``num / den``, or ``None`` if either operand is missing, non-finite,
        or ``den`` is zero.

    When to use: when turning two optional metrics into a rate (for example
    steps per second) where a missing or zero denominator should yield a
    missing value rather than an exception or an infinity.

    Examples
    --------
    >>> safe_ratio(1.0, 4.0)
    0.25
    >>> safe_ratio(1.0, 0.0) is None
    True
    """

    fnum = finite(num)
    fden = finite(den)

    if fnum is None or fden is None or fden == 0.0:
        return None

    return float(fnum) / float(fden)


def to_float(x: Any) -> Optional[float]:
    """Best-effort conversion of a scalar (incl. NumPy scalars) to ``float``.

    Never raises: any failure of the conversion is reported as ``None``.

    Parameters
    ----------
    x : Any
        Value to convert. ``numpy.generic`` instances go through ``.item()``.

    Returns
    -------
    float or None
        The converted value, or ``None`` if ``x`` is ``None`` or the
        conversion raises. NaN and inf are returned as-is; use ``finite`` to
        reject them.

    When to use: to read a metric whose type is not under your control (a
    NumPy scalar, a number stored as text, or a missing value) before logging or
    comparing it.

    Examples
    --------
    >>> to_float(np.float32(2.5)), to_float("abc")
    (2.5, None)
    """

    if x is None:
        return None

    try:
        if isinstance(x, np.generic):
            return float(x.item())

        return float(x)
    except Exception:
        return None


def finite(x: Any) -> Optional[float]:
    """Convert to ``float`` and reject NaN/inf.

    Parameters
    ----------
    x : Any
        Value passed to ``to_float``.

    Returns
    -------
    float or None
        The finite float, or ``None`` if conversion failed or the value is
        NaN or infinite.

    When to use: when a missing, NaN or infinite value must be treated the same
    way, for example before averaging per-episode statistics or writing them to
    a reporter.

    Examples
    --------
    >>> finite("2.5")
    2.5
    >>> finite(float("nan")) is None, finite(float("inf")) is None
    (True, True)
    """

    fx = to_float(x)

    if fx is None or not np.isfinite(fx):
        return None

    return fx


def sanitize_key(s: str) -> str:
    """Make a string safe to use as a metric key (e.g. for W&B).

    Runs of characters outside ``[0-9a-zA-Z_-]`` are collapsed into a single
    underscore. The input is coerced with ``str`` first.

    Parameters
    ----------
    s : str
        Raw key, for example a metric path such as ``"info/stock level (t)"``.

    Returns
    -------
    str
        The key with every run of unsupported characters replaced by one
        underscore. The result is not shortened and can end with an underscore.

    When to use: before sending a metric name to a backend that restricts the
    characters of its keys; the CSV, TensorBoard and W&B reporters use it.

    Examples
    --------
    >>> sanitize_key("info/stock level (t)")
    'info_stock_level_t_'
    """

    # keep alnum, underscore, dash; replace everything else with underscore
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(s))


def is_mapping(x: Any) -> bool:
    """Return ``True`` if ``x`` is a ``collections.abc.Mapping``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` for a ``dict`` or any other registered mapping type.

    When to use: to branch between a nested mapping and a leaf value when
    walking a metric tree.

    Examples
    --------
    >>> is_mapping({"a": 1}), is_mapping([("a", 1)])
    (True, False)
    """

    return isinstance(x, Mapping)


def flatten_numeric(value: Any) -> list[float]:
    """Flatten a scalar, sequence or array into a flat list of floats.

    Parameters
    ----------
    value : Any
        Anything ``numpy.asarray`` accepts: a scalar, a nested list, or an
        array of any shape.

    Returns
    -------
    list of float
        Row-major flattening of ``value``; a scalar yields a one-element list.

    Raises
    ------
    TypeError
        If ``value`` is ``None`` (or another object NumPy stores as a
        zero-dimensional ``object`` array that ``float`` rejects).
    ValueError
        If an element cannot be converted to ``float``, or if the input is a
        ragged nested sequence.

    When to use: to turn a logged value of unknown shape (a scalar, a list of
    per-agent values or an array) into a flat list of floats for reporting.

    Examples
    --------
    >>> flatten_numeric([[1, 2], [3, 4]])
    [1.0, 2.0, 3.0, 4.0]
    """

    arr = np.asarray(value)

    if arr.ndim == 0:
        return [float(arr)]

    return [float(x) for x in arr.reshape(-1)]


ACTION_TEMPERATURE = 4.0
"""Temperature of the map ``sigmoid(z / ACTION_TEMPERATURE)`` from a raw policy
output ``z`` to a fraction in ``(0, 1)``.

The fishery's ``Fishing`` and ``Restore`` mechanisms decode their action with
it. ``QuotaMechanism.apply`` and ``SubsidyMechanism.apply`` act before the
targeted agents, read the raw action and decode or re-encode it themselves, so
they must use the same value; one constant keeps them consistent. The value is
a heuristic scale, not taken from the literature.
"""


def sigmoid(x: float) -> float:
    """Numerically stable logistic function ``1 / (1 + exp(-x))``.

    The two-branch form avoids overflow in ``exp`` for large ``|x|``: for
    ``x >= 0`` it evaluates ``1 / (1 + exp(-x))`` and otherwise
    ``exp(x) / (1 + exp(x))``. The output increases monotonically from ``0`` to
    ``1`` and equals ``0.5`` at the origin.

    Parameters
    ----------
    x : float
        Argument of the logistic function, dimensionless.

    Returns
    -------
    float
        Value in ``[0, 1]``; it saturates to exactly ``0.0`` or ``1.0`` for
        very large ``|x|``.

    When to use: to turn a signed margin into a smooth gate or probability, for
    example in mechanism and agent reward shaping.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(-1000.0)
    0.0
    """
    if x >= 0.0:
        z = np.exp(-x)

        return float(1.0 / (1.0 + z))

    z = np.exp(x)

    return float(z / (1.0 + z))


def smooth_positive(x: float, width: float) -> float:
    """Smooth approximation of ``max(x, 0)`` (softplus with a width scale).

    Computes ``width * log(1 + exp(x / width))``. The result is positive in
    exact arithmetic and equals ``width * ln 2`` at ``x = 0``; in floating
    point it underflows to ``0.0`` once ``x / width`` falls below about -745.
    Use ``smooth_positive_zero_at_origin`` when an exact zero at the origin is
    required.

    Parameters
    ----------
    x : float
        Input value.
    width : float
        Transition scale, in the same units as ``x``. Clamped to at least
        ``EPS``; smaller widths approach the hard hinge.

    Returns
    -------
    float
        Softplus value, in the units of ``x``; ``>= 0`` and close to
        ``max(x, 0)`` when ``|x|`` is large compared with ``width``.

    When to use: for a differentiable hinge, such as the part of a harvest
    above a quota, when a hard ``max(x, 0)`` would give a zero gradient.

    Examples
    --------
    >>> round(smooth_positive(0.0, 1.0), 4)
    0.6931
    >>> round(smooth_positive(10.0, 1.0), 4)
    10.0
    """

    width = max(float(width), EPS)

    return float(width * np.logaddexp(0.0, float(x) / width))


def smooth_min(a: float, b: float, width: float) -> float:
    """Smooth transition between ``a`` and ``b`` approximating ``min(a, b)``.

    Returns a sigmoid-weighted blend ``(1 - w) * a + w * b`` where
    ``w = sigmoid((a - b) / width)``, so the output leans towards ``b`` when
    ``a > b`` and towards ``a`` otherwise.

    Parameters
    ----------
    a, b : float
        Values to blend.
    width : float
        Transition scale, in the same units as ``a`` and ``b``. Clamped to at
        least ``EPS``.

    Returns
    -------
    float
        Blended value, between ``min(a, b)`` and ``max(a, b)``, in the units of
        ``a`` and ``b``.

    When to use: to cap one quantity by another (a harvest by the available
    stock, for instance) without a kink, so that gradients flow to both.

    Examples
    --------
    >>> round(smooth_min(1.0, 3.0, 0.1), 4)
    1.0
    >>> smooth_min(2.0, 2.0, 0.1)
    2.0
    """

    a = float(a)
    b = float(b)
    width = max(float(width), EPS)
    weight_on_b = sigmoid((a - b) / width)

    return float((1.0 - weight_on_b) * a + weight_on_b * b)


def smooth_cap_01(x: float) -> float:
    """Smooth saturation of a non-negative value into ``[0, 1)``.

    Computes ``1 - exp(-x)``: zero at the origin, slope one there, and
    asymptotically one. Negative inputs give negative outputs; the caller is
    expected to pass ``x >= 0``.

    Parameters
    ----------
    x : float
        Non-negative quantity, dimensionless.

    Returns
    -------
    float
        Value in ``[0, 1)`` for ``x >= 0``; negative for ``x < 0``.

    When to use: to map an unbounded non-negative quantity (a violation
    magnitude, for example) into a bounded fraction while staying smooth.

    Examples
    --------
    >>> smooth_cap_01(0.0)
    0.0
    >>> round(smooth_cap_01(1.0), 4)
    0.6321
    """

    x = float(x)

    return float(1.0 - np.exp(-x))


def smooth_positive_zero_at_origin(x: float, width: float) -> float:
    """Smooth approximation of ``max(x, 0)`` that is exactly zero at ``x=0``.

    Shifts ``smooth_positive`` down by ``width * ln 2`` so that the origin maps
    to zero, then clamps at zero from below (the shifted softplus is negative
    for ``x < 0``). For large positive ``x`` the result approaches
    ``x - width * ln 2``, so it stays slightly below ``max(x, 0)``.

    Parameters
    ----------
    x : float
        Input value.
    width : float
        Transition scale, in the same units as ``x``. Clamped to at least
        ``EPS``.

    Returns
    -------
    float
        Non-negative value in the units of ``x``, ``0.0`` for ``x <= 0``.

    When to use: for a penalty or subsidy that must vanish exactly when the
    constraint is not violated yet stay differentiable around the threshold.

    Examples
    --------
    >>> smooth_positive_zero_at_origin(0.0, 0.5)
    0.0
    >>> round(smooth_positive_zero_at_origin(2.0, 0.5), 4)
    1.6625
    """

    width = max(float(width), EPS)
    value = width * (np.logaddexp(0.0, float(x) / width) - np.log(2.0))

    return float(max(0.0, value))


def intersect(
    x: spaces.Space | None, dxs: list[spaces.Space | None]
) -> spaces.Space | None:
    """Narrow a gymnasium space by a list of constraining spaces.

    ``None`` always means "this component contributes nothing": it is dropped
    from ``dxs``, and a missing base takes the place of the first remaining
    delta. Two ``Box`` spaces are intersected by taking the element-wise
    maximum of the lower bounds and the minimum of the upper bounds, keeping the
    dtype of the base; two ``Dict`` spaces are intersected key by key
    (recursively), and a key present in only some of the inputs is kept,
    narrowed by those that define it. The inputs are never modified.

    Parameters
    ----------
    x : gymnasium.spaces.Space or None
        Base space, a ``Box`` or a ``Dict`` of such spaces.
    dxs : list of gymnasium.spaces.Space or None
        Spaces that further constrain ``x``; ``None`` entries are ignored. They
        must be of the same kind as ``x`` (``Box`` with ``Box``, ``Dict`` with
        ``Dict``).

    Returns
    -------
    gymnasium.spaces.Space or None
        ``x`` itself when there is nothing to intersect with, the single delta
        when ``x`` is ``None`` and one delta remains, otherwise a new ``Box`` or
        ``Dict``. ``None`` if ``x`` is ``None`` and no delta remains.

    Raises
    ------
    TypeError
        If the spaces are of different kinds or of an unsupported kind (only
        ``Box`` and ``Dict`` are handled).
    ValueError
        If the intersection of two boxes is empty (a lower bound above an upper
        bound); equal bounds are allowed and give a degenerate box.

    When to use: when several mechanisms each restrict the action, state or
    observation space of the same environment and the result must satisfy all
    of them; ``MDPState.add`` composes its spaces with it.

    Examples
    --------
    >>> base = spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)
    >>> tighter = spaces.Box(low=2.0, high=12.0, shape=(1,), dtype=np.float32)
    >>> narrowed = intersect(base, [tighter, None])
    >>> narrowed.low, narrowed.high
    (array([2.], dtype=float32), array([10.], dtype=float32))
    """

    dxs = [dx for dx in dxs if dx is not None]

    if not dxs:
        return x

    if x is None:
        if len(dxs) == 1:
            return dxs[0]

        x = dxs[0]
        dxs = dxs[1:]

    if isinstance(x, spaces.Dict):
        result = dict(x.spaces)
        keys = set(result)

        for dx in dxs:
            if not isinstance(dx, spaces.Dict):
                raise TypeError(f"Cannot intersect Dict with {type(dx).__name__}")

            keys.update(dx.spaces)

        composed = {}

        for key in keys:
            base = result.get(key)
            child_dxs = [dx.spaces[key] for dx in dxs if key in dx.spaces]
            composed[key] = intersect(base, child_dxs)

        return spaces.Dict(composed)

    if isinstance(x, spaces.Box):
        low = x.low.copy()
        high = x.high.copy()

        for dx in dxs:
            if not isinstance(dx, spaces.Box):
                raise TypeError(f"Cannot intersect Box with {type(dx).__name__}")

            np.maximum(low, dx.low, out=low)
            np.minimum(high, dx.high, out=high)

        if np.any(low > high):
            raise ValueError("Mechanism composition produced an empty space.")

        return spaces.Box(low=low, high=high, dtype=x.dtype)

    raise TypeError(f"Cannot intersect spaces of type {type(x).__name__}")


def logical_or_dict(
    x: dict[Any, Any] | None, dxs: list[dict[Any, Any] | None]
) -> dict[Any, bool]:
    """Combine dictionaries of flags with a logical or, key by key.

    Missing entries (``None`` or empty dicts, in ``x`` or in ``dxs``) count as
    "no flags". The result has the union of the keys, and every value is coerced
    to ``bool`` (a key absent from one dict counts as ``False`` there). The
    inputs are not modified.

    Parameters
    ----------
    x : dict or None
        Base flags, for example ``{agent_id: terminated}``.
    dxs : list of dict or None
        Further flag dictionaries; ``None`` entries are ignored.

    Returns
    -------
    dict
        A new dict mapping each key to ``True`` if any input flags it as true,
        else ``False``.

    When to use: to merge termination and truncation flags contributed by
    several mechanisms, so that an agent ends as soon as any of them says so;
    ``MDPState.add`` uses it for ``terminateds`` and ``truncateds``.

    Examples
    --------
    >>> logical_or_dict({"a": False, "b": True}, [{"a": True}, None]) == {
    ...     "a": True, "b": True
    ... }
    True
    """

    x = x or {}

    if not any(dxs):
        # The common case inside ``MDPState.add``: no residual sets a flag.
        # The environment composes one residual per agent per step over flags
        # for every agent, so this path stays in C: a plain copy when every
        # value is already a bool, as after any earlier composition.
        if set(map(type, x.values())) <= {bool}:
            return dict(x)
        return dict(zip(x, map(bool, x.values())))

    result = dict(x)
    keys = set(result)

    for dx in dxs:
        if dx:
            keys.update(dx)

    for key in keys:
        result[key] = bool(x.get(key, False)) or any(
            dx.get(key, False) for dx in dxs if dx is not None
        )

    return result
