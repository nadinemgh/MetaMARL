"""Fresh-water irrigation example: farms, a reservoir and an ES regulator.

Many corn farms request irrigation water from one reservoir. The outer level
searches the eight rules of a water policy (a quota that tightens as the lake
falls, fines for requests above it, a penalty on large requests when the river
depends on the release; two of the eight rules, the under-irrigation penalty
scale and the farm area, are searched and observed but read by no dynamics); the
inner level trains the farms with reinforcement learning. The lake is a built-in
surrogate by default and the external Raven hydrological model on request.

Run it with ``uv run python -m examples.fresh_water.debug --help``.
"""
