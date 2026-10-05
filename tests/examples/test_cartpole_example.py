"""Cart-pole example: dynamics, reward, composition with the regulator, end-to-end run.

The inner environment wraps Gymnasium's ``CartPole-v1`` and does not
re-implement it, so the dynamics tests pin the wrapping, not the physics: the
push reaches the right Gymnasium action, the reward is the Gymnasium reward, the
flags of Gymnasium reach RLlib and the state the agent sees is the state
Gymnasium returned. One expected state is computed by hand from the equations of
motion of ``CartPole-v1`` (Euler integration, ``tau = 0.02``, gravity 9.8, cart
mass 1.0, pole mass 0.1, half pole length 0.5, force magnitude 10) for the
upright pole at rest pushed to the right:

    temp = 10 / 1.1                                  = 9.090909
    theta_acc = -temp / (0.5 * (4/3 - 0.1 / 1.1))    = -14.634146
    x_acc = temp - 0.05 * theta_acc / 1.1            =  9.756098

so after one step ``x_dot = 0.02 * x_acc = 0.195122`` and
``theta_dot = 0.02 * theta_acc = -0.292683``, with ``x`` and ``theta`` still 0.

The composition tests step the environment behind the real RLlib adapter with a
scripted stand-in for the ``World`` actor, as
``tests/integration/test_fishery_regulator_composition.py`` does, so they run
without Ray. The end-to-end tests launch ``examples.cartpole.main_*`` as child
processes with a time limit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.agents.base import AgentConfig
from core.envs import marl_regulated
from core.mechanism.base import MDPState
from core.reporting.csv import CSVConfig
from core.world.context import MechanismContext, MechanismStatus
from examples.cartpole.metric_schema import CartpoleMetricSchema
from examples.cartpole.regulated_env import (
    DIAL_SPACE,
    PUSH_SPACE,
    CartpoleAgentConfig,
    CartpoleRegulatedEnv,
    DialConfig,
    Push,
    PushConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT = "agent_0:0"
REGULATOR = "cartpole_regulator"
FLOAT32_ATOL = 1e-5


class ScriptedWorld:
    """Stand-in for the ``World`` actor: hands out the scripted candidates once."""

    def __init__(self, answers: list[MechanismContext | None]) -> None:
        self.answers = list(answers)
        self.get_mechanism_by_id = SimpleNamespace(remote=self._fetch)

    def _fetch(self, **kwargs: Any) -> MechanismContext | None:
        return self.answers.pop(0) if self.answers else None


def candidate(dial: float) -> MechanismContext:
    """Candidate the regulator publishes: a dial value."""
    return MechanismContext(
        index=0,
        env_id=None,
        seed=0,
        status=MechanismStatus.train,
        mechanism={"dial": np.asarray([dial], dtype=np.float32)},
        metrics=None,
    )


@pytest.fixture
def build_env(tmp_path, monkeypatch):
    """Factory of a cart-pole environment, with or without a regulator."""
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref, *a, **k: ref)

    def build(
        *,
        candidates: list[MechanismContext | None] | None = None,
        with_regulator: bool = False,
        horizon: int = 50,
        seed: int = 0,
    ) -> CartpoleRegulatedEnv:
        follower = CartpoleAgentConfig(
            id=AGENT,
            policy_id="cartpole_policy",
            mechanisms=(PushConfig(id="push", action_space=PUSH_SPACE),),
            observation_space=spaces.Box(-np.inf, np.inf, (4,), np.float32),
        )
        leaders = {}
        if with_regulator:
            leaders[REGULATOR] = AgentConfig(
                id=REGULATOR,
                policy_id="dial_policy",
                mechanisms=(DialConfig(id="dial", action_space=DIAL_SPACE),),
            )
        return CartpoleRegulatedEnv(
            world=ScriptedWorld(candidates or []),
            opt_id="inner",
            env_name="cartpole_test",
            horizon=horizon,
            agents_cfg_dict={AGENT: follower},
            leaders_cfg_dict=leaders,
            mechanism_id=0,
            seed=seed,
            policy_seed=seed,
            mode="train",
            reporter_cfg=CSVConfig(project="cartpole_test", output_dir=str(tmp_path)),
            queries=(),
            schema=CartpoleMetricSchema,
        )

    return build


def start(env: CartpoleRegulatedEnv) -> MDPState:
    """Reset the way the adapter does, with a fresh state naming the agent."""
    return env.reset(MDPState(aids={AGENT}))


def play(env: CartpoleRegulatedEnv, mdp: MDPState, push: Any) -> MDPState:
    """Write the agent's action into the state and step the environment once."""
    mdp.update(actions={AGENT: {"push": push}})
    return env.step(mdp)


# --- Push mechanism -------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0), (1, 1), (np.int64(1), 1), (np.asarray([0]), 0), (np.asarray([[1]]), 1)],
)
def test_push_decode_returns_a_plain_int(raw, expected):
    push = PushConfig(id="push", action_space=PUSH_SPACE).build(AGENT)
    decoded = push.decode(None, raw)
    assert decoded == expected
    assert type(decoded) is int


@pytest.mark.unit
def test_push_decode_accepts_its_own_output():
    push: Push = PushConfig(id="push", action_space=PUSH_SPACE).build(AGENT)
    once = push.decode(None, np.int64(1))
    assert push.decode(None, once) == once


@pytest.mark.unit
def test_push_and_dial_apply_no_residual():
    push = PushConfig(id="push", action_space=PUSH_SPACE).build(AGENT)
    dial = DialConfig(id="dial", action_space=DIAL_SPACE).build(REGULATOR)
    assert push.apply(MDPState(), 1).state.data == {}
    assert dial.apply(MDPState(), np.asarray([0.3], np.float32)).state.data == {}


# --- Environment dynamics and reward --------------------------------------------


@pytest.mark.unit
def test_environment_needs_exactly_one_agent(build_env, tmp_path):
    follower = CartpoleAgentConfig(
        id="a",
        policy_id="p",
        mechanisms=(PushConfig(id="push", action_space=PUSH_SPACE),),
    )
    with pytest.raises(ValueError, match="exactly one agent"):
        CartpoleRegulatedEnv(
            world=None,
            mechanism_id=0,
            agents_cfg_dict={"a": follower, "b": follower},
            schema=CartpoleMetricSchema,
        )


@pytest.mark.unit
def test_reset_returns_the_gymnasium_state_as_a_float32_observation(build_env):
    env = build_env()
    mdp = start(env)
    observation = mdp.obs[AGENT][0]
    assert observation.shape == (4,)
    assert observation.dtype == np.float32
    # CartPole-v1 draws each coordinate of the initial state in [-0.05, 0.05].
    assert np.all(np.abs(observation) <= 0.05)
    assert mdp.state["terminated"] == [0.0]
    assert mdp.state["truncated"] == [0.0]


@pytest.mark.unit
def test_first_reset_is_seeded_and_the_stream_continues_afterwards(build_env):
    first = start(build_env(seed=7)).obs[AGENT][0]
    second = start(build_env(seed=7)).obs[AGENT][0]
    other = start(build_env(seed=8)).obs[AGENT][0]
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)

    env = build_env(seed=7)
    start(env)
    assert not np.array_equal(start(env).obs[AGENT][0], first)


@pytest.mark.unit
def test_step_pushing_right_from_rest_matches_the_hand_computed_state(build_env):
    env = build_env()
    mdp = start(env)
    env.gym_env.unwrapped.state = np.zeros(4)

    mdp = play(env, mdp, 1)

    np.testing.assert_allclose(
        mdp.obs[AGENT][mdp.t], [0.0, 0.195122, 0.0, -0.292683], atol=FLOAT32_ATOL
    )
    assert mdp.rewards[AGENT][mdp.t - 1] == 1.0
    assert mdp.terminateds[AGENT] is False
    assert mdp.truncateds["__all__"] is False


@pytest.mark.unit
def test_step_pushing_left_mirrors_pushing_right(build_env):
    env = build_env()
    mdp = start(env)
    env.gym_env.unwrapped.state = np.zeros(4)

    mdp = play(env, mdp, np.int64(0))

    np.testing.assert_allclose(
        mdp.obs[AGENT][mdp.t], [0.0, -0.195122, 0.0, 0.292683], atol=FLOAT32_ATOL
    )


@pytest.mark.unit
def test_reward_is_one_per_step_including_the_terminating_step(build_env):
    env = build_env()
    mdp = start(env)
    # A pole 0.2094 rad off vertical (12 degrees is 0.20944) falls on the next step.
    env.gym_env.unwrapped.state = np.asarray([0.0, 0.0, 0.2094, 1.0])

    mdp = play(env, mdp, 1)

    assert mdp.rewards[AGENT][mdp.t - 1] == 1.0
    assert mdp.terminateds[AGENT] is True
    assert mdp.terminateds["__all__"] is True
    assert mdp.truncateds["__all__"] is False


@pytest.mark.unit
def test_cart_leaving_the_track_terminates_the_episode(build_env):
    env = build_env()
    mdp = start(env)
    # |x| > 2.4 terminates; x moves by tau * x_dot = 0.02 * 3 = 0.06 to 2.43.
    env.gym_env.unwrapped.state = np.asarray([2.37, 3.0, 0.0, 0.0])

    mdp = play(env, mdp, 1)

    assert mdp.terminateds["__all__"] is True


@pytest.mark.unit
def test_gymnasium_time_limit_truncates_without_terminating(build_env):
    env = build_env(horizon=1000)
    mdp = start(env)
    env.gym_env.unwrapped.state = np.zeros(4)
    # CartPole-v1 truncates when its 500th step has been played.
    env.gym_env._elapsed_steps = 499

    mdp = play(env, mdp, 1)

    assert mdp.truncateds[AGENT] is True
    assert mdp.truncateds["__all__"] is True
    assert mdp.terminateds["__all__"] is False


@pytest.mark.unit
def test_horizon_truncates_every_agent(build_env):
    env = build_env(horizon=2)
    mdp = start(env)
    env.gym_env.unwrapped.state = np.zeros(4)

    mdp = play(env, mdp, 1)
    assert mdp.truncateds["__all__"] is False
    mdp = play(env, mdp, 0)
    assert mdp.truncateds["__all__"] is True
    assert mdp.terminateds["__all__"] is False


@pytest.mark.unit
def test_logger_reduces_the_state_series_of_an_episode(build_env):
    env = build_env()
    mdp = start(env)
    env.gym_env.unwrapped.state = np.zeros(4)
    mdp = play(env, mdp, 1)
    mdp = play(env, mdp, 1)

    states = [np.asarray(mdp.state["cartpole"][t]) for t in (1, 2)]
    reduced = env.logger.reduce()

    assert reduced.cart_position == pytest.approx(
        np.mean([s[0] for s in states]), abs=FLOAT32_ATOL
    )
    assert reduced.pole_angle_abs_max == pytest.approx(
        max(abs(s[2]) for s in states), abs=FLOAT32_ATOL
    )
    assert reduced.reward_total == 2.0
    assert reduced.reward_mean == 1.0


# --- Composition with the regulator, behind the RLlib adapter --------------------


def run_episode(adapter: RLlibMultiAgentEnvAdapter, pushes: list[int]) -> list:
    """Play the scripted pushes; return the observation, reward and flags per step."""
    adapter.reset()
    trace = []
    for push in pushes:
        obs, rewards, terminateds, truncateds, _ = adapter.step(
            {AGENT: {"push": np.int64(push)}}
        )
        trace.append((obs[AGENT].copy(), rewards[AGENT], terminateds, truncateds))
        if terminateds["__all__"] or truncateds["__all__"]:
            break
    return trace


@pytest.mark.unit
def test_adapter_exposes_the_push_space_of_the_agent(build_env):
    adapter = RLlibMultiAgentEnvAdapter(build_env())
    assert adapter.agents == [AGENT]
    assert adapter.action_spaces[AGENT]["push"] == PUSH_SPACE
    assert adapter.observation_spaces[AGENT].shape == (4,)


@pytest.mark.unit
def test_the_inert_dial_leaves_dynamics_and_reward_untouched(build_env):
    pushes = [1, 0, 1, 1, 0, 0, 1, 0]
    runs = {}
    for label, kwargs in {
        "none": {},
        "low": {"with_regulator": True, "candidates": [candidate(0.05)]},
        "high": {"with_regulator": True, "candidates": [candidate(0.95)]},
    }.items():
        runs[label] = run_episode(
            RLlibMultiAgentEnvAdapter(build_env(**kwargs)), pushes
        )

    for label in ("low", "high"):
        assert len(runs[label]) == len(runs["none"])
        for (obs_a, r_a, *_), (obs_b, r_b, *_) in zip(runs[label], runs["none"]):
            np.testing.assert_array_equal(obs_a, obs_b)
            assert r_a == r_b == 1.0


@pytest.mark.unit
def test_candidate_reaches_the_leader_and_is_kept_across_steps(build_env):
    env = build_env(with_regulator=True, candidates=[candidate(0.25)])
    adapter = RLlibMultiAgentEnvAdapter(env)
    adapter.reset()
    adapter.step({AGENT: {"push": np.int64(1)}})
    adapter.step({AGENT: {"push": np.int64(0)}})

    dial = adapter._mdp.actions[REGULATOR]["dial"]
    values = [float(np.asarray(v).reshape(-1)[0]) for v in dial]
    assert len(values) >= 2
    assert values == [0.25] * len(values)
    assert env.published_mechanism_assigned


@pytest.mark.unit
def test_episode_ends_when_the_pole_falls_and_pays_one_per_step(build_env):
    env = build_env(with_regulator=True, candidates=[candidate(0.5)], horizon=500)
    trace = run_episode(RLlibMultiAgentEnvAdapter(env), [1] * 60)

    # Pushing right every step topples the pole within a few tens of steps.
    assert 5 < len(trace) < 60
    *running, last = trace
    assert all(not step[2]["__all__"] for step in running)
    assert last[2]["__all__"] is True
    assert last[3]["__all__"] is False
    assert [step[1] for step in trace] == [1.0] * len(trace)
    assert env.logger.reduce().reward_total == float(len(trace))


@pytest.mark.unit
def test_episodes_after_a_reset_are_independent_in_the_logger(build_env):
    env = build_env(horizon=3)
    adapter = RLlibMultiAgentEnvAdapter(env)
    run_episode(adapter, [1, 0, 1])
    first = env.logger.reduce().reward_total
    run_episode(adapter, [0, 1, 0])
    assert (first, env.logger.reduce().reward_total) == (3.0, 3.0)


# --- Regulator environment -------------------------------------------------------


@pytest.mark.unit
def test_regulator_reward_averages_seeds_and_scores_missing_mechanisms(monkeypatch):
    import ray

    from examples.cartpole.regulator_env import CartpoleRegulatorEnv

    published = []
    world = SimpleNamespace(
        append_context=SimpleNamespace(remote=lambda ctx: published.append(ctx))
    )
    monkeypatch.setattr(ray, "get", lambda ref, *a, **k: ref)
    env = CartpoleRegulatorEnv(
        world=world, optimizer=None, horizon=1, agents_cfgs={}, seeds=[0, 1]
    )

    def episode(*means):
        return {str(i): SimpleNamespace(reward_mean=[m]) for i, m in enumerate(means)}

    by_seed = {
        "0": SimpleNamespace(by_episode=episode(1.0, 0.5)),
        "1": SimpleNamespace(by_episode=episode(0.25)),
    }
    rollout = SimpleNamespace(by_mechanism={"2": SimpleNamespace(by_seed=by_seed)})
    metrics = SimpleNamespace(eval=SimpleNamespace(rollout=rollout))

    # Seed 0 averages to 0.75 and seed 1 to 0.25, so mechanism 2 scores 0.5;
    # mechanisms 0 and 1 have no rollout.
    assert env.reward(metrics) == [-np.inf, -np.inf, 0.5]
    (context,) = published
    assert context.payload.status is MechanismStatus.done
    assert context.payload.index == 2
    assert context.payload.metrics.objective_score == 0.5


@pytest.mark.unit
def test_regulator_reward_refuses_an_unknown_split_and_empty_metrics():
    from examples.cartpole.regulator_env import CartpoleRegulatorEnv

    with pytest.raises(ValueError):
        CartpoleRegulatorEnv(
            world=None,
            optimizer=None,
            horizon=1,
            agents_cfgs={},
            seeds=[0],
            aggregation_status="published",
        )
    env = CartpoleRegulatorEnv(
        world=None, optimizer=None, horizon=1, agents_cfgs={}, seeds=[0]
    )
    empty = SimpleNamespace(
        eval=SimpleNamespace(rollout=SimpleNamespace(by_mechanism={}))
    )
    with pytest.raises(ValueError, match="no mechanism"):
        env.reward(empty)


# --- End to end, as child processes ----------------------------------------------

CHILD_TIMEOUT_S = 240


def run_child(module: str, args: tuple[str, ...], workdir: Path) -> dict:
    """Run ``python -m module args`` in ``workdir`` with a time limit."""
    env = dict(os.environ, WANDB_MODE="offline", PYTHONPATH=str(REPO_ROOT))
    # A fixed hash seed would make the child's world identifiers repeat across runs.
    env.pop("PYTHONHASHSEED", None)
    try:
        done = subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{module} exceeded {CHILD_TIMEOUT_S} s; stdout: {exc.stdout!r}"[-800:]
        )
    return {
        "returncode": done.returncode,
        "output": done.stdout + "\n" + done.stderr,
        "workdir": workdir,
    }


SMOKE_ARGS = ("--outer-iters", "2", "--horizon", "20", "--reporter", "csv")


@pytest.fixture(scope="module")
def appo_run(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("cartpole_appo")
    return run_child(
        "examples.cartpole.debug",
        ("--algo", "appo", "--train-iters", "2", *SMOKE_ARGS),
        workdir,
    )


@pytest.mark.integration
def test_debug_script_runs_to_completion_with_appo(appo_run):
    assert appo_run["returncode"] == 0, appo_run["output"][-2000:]
    assert "Run finished | iters=2" in appo_run["output"]


@pytest.mark.unit
@pytest.mark.parametrize("algo", ["ppo", "appo"])
def test_entry_points_fix_the_algorithm_and_forward_the_options(monkeypatch, algo):
    import importlib

    entry = importlib.import_module(f"examples.cartpole.main_{algo}")
    seen = []
    monkeypatch.setattr(entry, "run", seen.append)
    entry.main(["--outer-iters", "3"])
    assert seen == [["--algo", algo, "--outer-iters", "3"]]


@pytest.mark.unit
def test_importing_the_scripts_runs_nothing_and_defaults_are_the_full_runs():
    from examples.cartpole import debug

    args = debug.parse_args(["--algo", "ppo"])
    assert (args.outer_iters, args.train_iters, args.horizon) == (100, 100, 1000)
    assert debug.parse_args(["--algo", "appo"]).train_iters == 200


@pytest.mark.integration
def test_appo_run_writes_csv_files_and_every_query_renders(appo_run):
    csv_files = sorted((appo_run["workdir"] / "results").rglob("*.csv"))
    assert len(csv_files) == 5, [path.name for path in csv_files]
    assert "could not render query" not in appo_run["output"]


@pytest.mark.integration
def test_appo_run_reports_the_fitness_of_a_balancing_agent(appo_run):
    fitness_files = list((appo_run["workdir"] / "results").rglob("Fitness_over*.csv"))
    assert fitness_files
    values = [
        float(line.split(",")[3])
        for line in fitness_files[0].read_text().splitlines()[1:]
        if "Candidates" in line
    ]
    # Reward is 1 per step, so a mean step reward cannot exceed 1.
    assert values
    assert all(0.0 < value <= 1.0 for value in values)


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "PPO finishes many episodes per training step and "
        + "core.callbacks.log_and_report_episode_metrics logs each as an 'item' under "
        + "the same key; RLlib's ItemStats.merge accepts one incoming value."
    ),
)
def test_debug_script_runs_to_completion_with_ppo(tmp_path):
    result = run_child(
        "examples.cartpole.debug",
        ("--algo", "ppo", "--train-iters", "1", *SMOKE_ARGS),
        tmp_path,
    )
    assert result["returncode"] == 0, result["output"][-1500:]
