"""Cart-pole example: an ES regulator over an inert dial, one RL agent inside.

The package wraps Gymnasium's ``CartPole-v1`` as a bilevel benchmark and checks
a whole bilevel run on a task with a known solution. The inner level trains a
balancing agent with PPO or APPO in ``regulated_env`` (the Gymnasium dynamics,
the push mechanism, the inert dial mechanism and the metrics each step logs
through ``metric_schema``). The outer level is an evolution-strategies
regulator whose environment, ``regulator_env``, scores each candidate by the
agent's mean step reward and publishes it through the fitness record of
``contexts``. The reporting queries live in ``queries``, ``debug`` builds and
runs the experiment, and ``main_ppo`` and ``main_appo`` are its two entry
points. Nothing is re-exported from here.
"""
