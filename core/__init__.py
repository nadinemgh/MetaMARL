"""Core library of the bilevel fishery framework.

The package is organised by level of the bilevel loop. ``core.world`` holds
the shared blackboard both levels talk to, ``core.mechanism`` defines the
regulatory mechanisms the outer level optimizes, ``core.agents`` the agents
whose behaviour the mechanisms shape, ``core.envs`` contains the environments
(the outer ``RegulatorEnv`` and the inner mechanism-regulated multi-agent
benchmarks), ``core.optimizers`` the outer evolution-strategies optimizer and
the inner learners, ``core.metrics`` and ``core.reporting`` cover logging and
reporting, ``core.config`` loads experiments from YAML and starts them from the
command line, and ``core.adaptors`` binds everything to Ray RLlib. The
top-level modules ``types``, ``utils``, ``annotations`` and ``callbacks`` are
shared helpers. This ``__init__`` re-exports nothing; import from the
submodules directly.
"""
