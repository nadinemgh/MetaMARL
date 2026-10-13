ACTION_TEMPERATURE = 4.0
"""Temperature of the map ``sigmoid(z / ACTION_TEMPERATURE)`` from a raw policy
output ``z`` to a fraction in ``(0, 1)``.

The fishery's ``Fishing`` and ``Restore`` mechanisms decode their action with
it. ``QuotaMechanism.apply`` and ``SubsidyMechanism.apply`` act before the
targeted agents, read the raw action and decode or re-encode it themselves, so
they must use the same value; one constant keeps them consistent. The value is
a heuristic scale, not taken from the literature.
"""