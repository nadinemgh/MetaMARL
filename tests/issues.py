"""Stable IDs and CI skip reasons for tracked limitations."""

KNOWN_ISSUES = {
    "OBS-001": (
        "OBS-001: Dynamic observation sizes and leader "
        + "contributions are not consistently composed."
    ),
    "MET-001": (
        "MET-001: Fishery transition requires a metric logger "
        + "(schema=None unsupported)."
    ),
    "MET-002": (
        "MET-002: Regulator reduction does not handle missing/NaN episode metrics."
    ),
    "APPO-001": (
        "APPO-001: Inner CSV queries fail under APPO; possible "
        + "threading/synchronization issue (unconfirmed)."
    ),
}
