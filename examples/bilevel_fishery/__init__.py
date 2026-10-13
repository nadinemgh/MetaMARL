"""Bilevel fishery example: an ES regulator over a quota, APPO fishers inside.

The package holds the single-stock fishery benchmark and the experiment that
runs it as a bilevel optimization. The inner level is a society of fishers
trained with APPO in ``regulated_env`` (the stock dynamics, the harvest and
restoration mechanisms, the metrics each step logs through ``metric_schema``).
The outer level is an evolution-strategies regulator whose environment,
``regulator_env``, scores each candidate mechanism and publishes the result
through the fitness record of ``contexts``. The reporting queries live in
``queries``, ``config.yaml`` declares the experiment, and ``debug`` is the
runnable entry point that builds the same experiment in Python. The regulatory
mechanisms themselves (quota, subsidy, penalty, social influence) belong to
``core.mechanism.algorithms``, not to this package. Nothing is re-exported
from here.
"""
