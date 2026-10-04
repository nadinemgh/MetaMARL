"""Unit tests for ``_evaluate_with_fixed_duration_once`` in ``core.callbacks``.

The function is the strict single-round evaluation loop. A fake
``EnvRunnerGroup`` runs the remote function synchronously on fake workers, so
the exact-once guards (missing results, incomplete units, stale iterations) can
be triggered deterministically and no Ray runtime is needed. The first half of
the file covers the new API stack, the second half the old API stack branch,
which the project does not use but which is kept as in RLlib.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS, NUM_EPISODES

from core.callbacks import _evaluate_with_fixed_duration_once

# --------------------------------------------------------------------------- #
# _evaluate_with_fixed_duration_once
# --------------------------------------------------------------------------- #


class FakeStat:
    """A metrics ``Stats`` leaf exposing ``peek()``."""

    def __init__(self, value):
        self.value = value

    def peek(self):
        return self.value


class FakeEpisode:
    def __init__(self, env_steps, agent_steps):
        self._env_steps = env_steps
        self._agent_steps = agent_steps

    def env_steps(self):
        return self._env_steps

    def agent_steps(self):
        return self._agent_steps


class FakeWorker:
    """Evaluation env runner producing ``num_episodes`` episodes on ``sample``.

    ``episodes_returned`` overrides the count actually delivered (to simulate
    an incomplete round) and ``report_num_episodes=False`` omits the
    ``NUM_EPISODES`` key from the metrics.
    """

    def __init__(
        self,
        worker_index,
        *,
        env_steps_per_episode=10,
        agent_steps_per_episode=20,
        episodes_returned=None,
        report_num_episodes=True,
    ):
        self.worker_index = worker_index
        self.env_steps_per_episode = env_steps_per_episode
        self.agent_steps_per_episode = agent_steps_per_episode
        self.episodes_returned = episodes_returned
        self.report_num_episodes = report_num_episodes
        self.sample_calls: list[dict] = []
        self._last_count = 0

    def sample(self, *, num_timesteps, num_episodes, force_reset):
        self.sample_calls.append(
            {
                "num_timesteps": num_timesteps,
                "num_episodes": num_episodes,
                "force_reset": force_reset,
            }
        )
        if self.episodes_returned is not None:
            count = self.episodes_returned
        elif num_episodes is not None:
            count = num_episodes
        else:
            count = max(1, num_timesteps // self.env_steps_per_episode)
        self._last_count = count
        return [
            FakeEpisode(self.env_steps_per_episode, self.agent_steps_per_episode)
            for _ in range(count)
        ]

    def get_metrics(self):
        if not self.report_num_episodes:
            return {}
        return {NUM_EPISODES: FakeStat(self._last_count)}


class FakeGroup:
    """``EnvRunnerGroup`` stand-in running the remote function in-process."""

    def __init__(self, workers, *, drop_results=0):
        self.workers = list(workers)
        self.drop_results = drop_results
        self.foreach_calls: list[dict] = []

    def num_healthy_remote_workers(self):
        return len(self.workers)

    def healthy_worker_ids(self):
        return [w.worker_index for w in self.workers]

    def foreach_env_runner(self, *, func, kwargs, local_env_runner, timeout_seconds):
        self.foreach_calls.append(
            {
                "kwargs": kwargs,
                "local_env_runner": local_env_runner,
                "timeout_seconds": timeout_seconds,
            }
        )
        results = [func(w, **kwargs) for w in self.workers]
        if self.drop_results:
            results = results[: len(results) - self.drop_results]
        return results


class FakeMetrics:
    """``MetricsLogger`` stand-in recording ``aggregate`` and answering ``peek``."""

    def __init__(self, num_episodes=None, eval_results=None):
        self.aggregate_calls: list[dict] = []
        self.num_episodes = num_episodes
        self.eval_results = eval_results if eval_results is not None else {"ok": True}

    def aggregate(self, stats, *, key):
        self.aggregate_calls.append({"stats": stats, "key": key})

    def peek(self, key, *, default=None, latest_merged_only=False):
        assert latest_merged_only is True
        if key == (EVALUATION_RESULTS, ENV_RUNNER_RESULTS, NUM_EPISODES):
            if self.num_episodes is None:
                # Derive from what was aggregated, like a real logger would.
                total = 0
                for call in self.aggregate_calls:
                    for met in call["stats"]:
                        if NUM_EPISODES in met:
                            total += met[NUM_EPISODES].peek()
                return total if self.aggregate_calls else default
            return self.num_episodes
        if key == EVALUATION_RESULTS:
            return self.eval_results
        return default


def make_algo(
    group,
    *,
    unit="episodes",
    duration=4,
    count_steps_by="env_steps",
    iteration=3,
    force_reset=True,
    timeout=12.5,
    metrics=None,
):
    config = SimpleNamespace(
        evaluation_duration_unit=unit,
        evaluation_duration=duration,
        evaluation_num_env_runners=len(group.workers),
        evaluation_force_reset_envs_before_iteration=force_reset,
        evaluation_sample_timeout_s=timeout,
        enable_env_runner_and_connector_v2=True,
        count_steps_by=count_steps_by,
    )
    return SimpleNamespace(
        config=config,
        evaluation_config=SimpleNamespace(),
        eval_env_runner_group=group,
        iteration=iteration,
        metrics=metrics if metrics is not None else FakeMetrics(),
    )


@pytest.mark.unit
def test_episodes_are_split_evenly_across_runners_in_one_round():
    workers = [FakeWorker(1), FakeWorker(2)]
    group = FakeGroup(workers)
    algo = make_algo(group, unit="episodes", duration=4)

    eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
        algo, group
    )

    # Exactly one round, two episodes per runner, force reset on round 0.
    assert len(group.foreach_calls) == 1
    call = group.foreach_calls[0]
    assert call["kwargs"]["num"] == [None, 2, 2]
    assert call["kwargs"]["round"] == 0
    assert call["kwargs"]["iter"] == 3
    assert call["local_env_runner"] is False
    assert call["timeout_seconds"] == 12.5
    for worker in workers:
        assert worker.sample_calls == [
            {"num_timesteps": None, "num_episodes": 2, "force_reset": True}
        ]

    assert env_steps == 4 * 10
    assert agent_steps == 4 * 20
    assert eval_results == {"ok": True}
    assert len(algo.metrics.aggregate_calls) == 1
    assert algo.metrics.aggregate_calls[0]["key"] == (
        EVALUATION_RESULTS,
        ENV_RUNNER_RESULTS,
    )
    assert len(algo.metrics.aggregate_calls[0]["stats"]) == 2


@pytest.mark.unit
def test_uneven_episode_split_gives_remainder_to_first_runners():
    group = FakeGroup([FakeWorker(1), FakeWorker(2), FakeWorker(3)])
    algo = make_algo(group, unit="episodes", duration=5)

    _evaluate_with_fixed_duration_once(algo, group)

    assert group.foreach_calls[0]["kwargs"]["num"] == [None, 2, 2, 1]


@pytest.mark.unit
def test_timesteps_unit_counts_env_steps():
    workers = [
        FakeWorker(1, env_steps_per_episode=10),
        FakeWorker(2, env_steps_per_episode=10),
    ]
    group = FakeGroup(workers)
    algo = make_algo(group, unit="timesteps", duration=40, count_steps_by="env_steps")

    _, env_steps, agent_steps = _evaluate_with_fixed_duration_once(algo, group)

    assert env_steps == 40
    assert agent_steps == 80
    for worker in workers:
        assert worker.sample_calls[0]["num_timesteps"] == 20
        assert worker.sample_calls[0]["num_episodes"] is None


@pytest.mark.unit
def test_timesteps_unit_counts_agent_steps_when_configured():
    workers = [
        FakeWorker(1, env_steps_per_episode=10, agent_steps_per_episode=20),
        FakeWorker(2, env_steps_per_episode=10, agent_steps_per_episode=20),
    ]
    group = FakeGroup(workers)
    # 40 requested timesteps -> 20 per worker -> 2 episodes -> 40 agent steps
    # each -> 80 agent steps in total, which differs from the request.
    algo = make_algo(group, unit="timesteps", duration=40, count_steps_by="agent_steps")

    with pytest.raises(RuntimeError, match="completed=80"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_force_reset_flag_is_forwarded():
    worker = FakeWorker(1)
    group = FakeGroup([worker])
    algo = make_algo(group, duration=1, force_reset=False)

    _evaluate_with_fixed_duration_once(algo, group)

    assert worker.sample_calls[0]["force_reset"] is False


@pytest.mark.unit
def test_missing_runner_result_raises_instead_of_retrying():
    group = FakeGroup([FakeWorker(1), FakeWorker(2)], drop_results=1)
    algo = make_algo(group, duration=4)

    with pytest.raises(RuntimeError, match="expected=2, received=1"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_incomplete_round_raises():
    group = FakeGroup([FakeWorker(1), FakeWorker(2, episodes_returned=1)])
    algo = make_algo(group, duration=4)

    with pytest.raises(RuntimeError, match="requested=4, completed=3"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_metrics_without_num_episodes_count_as_zero_units():
    group = FakeGroup([FakeWorker(1, report_num_episodes=False)])
    algo = make_algo(group, duration=2)

    with pytest.raises(RuntimeError, match="completed=0"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_stale_iteration_results_are_discarded():
    group = FakeGroup([FakeWorker(1)])
    algo = make_algo(group, duration=2, iteration=5)

    # The remote function echoes ``algo.iteration`` read before sampling; a
    # worker that answers for an older iteration is ignored, so the round is
    # incomplete.
    original = group.foreach_env_runner

    def foreach_with_stale_iter(**kwargs):
        results = original(**kwargs)
        return [(e, a, m, it - 1) for e, a, m, it in results]

    group.foreach_env_runner = foreach_with_stale_iter

    with pytest.raises(RuntimeError, match="completed=0"):
        _evaluate_with_fixed_duration_once(algo, group)


@pytest.mark.unit
def test_no_healthy_workers_warns_and_returns_empty(caplog):
    group = FakeGroup([])
    metrics = FakeMetrics(num_episodes=0, eval_results={})
    algo = make_algo(group, duration=4, metrics=metrics)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
            algo, group
        )

    assert (eval_results, env_steps, agent_steps) == ({}, 0, 0)
    assert group.foreach_calls == []
    assert "all workers crashing" in caplog.text
    assert "empty set of episode summary" in caplog.text


@pytest.mark.unit
def test_zero_duration_skips_sampling(caplog):
    group = FakeGroup([FakeWorker(1)])
    metrics = FakeMetrics(num_episodes=0)
    algo = make_algo(group, duration=0, metrics=metrics)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    assert env_steps == 0
    assert group.foreach_calls == []
    assert "all workers crashing" not in caplog.text
    assert "empty set of episode summary" in caplog.text


# --------------------------------------------------------------------------- #
# Old API stack branch
# --------------------------------------------------------------------------- #


class FakeBatch:
    """Sample batch of the old API stack, reporting its step counts."""

    def __init__(self, env_steps, agent_steps):
        self._env_steps = env_steps
        self._agent_steps = agent_steps

    def env_steps(self):
        return self._env_steps

    def agent_steps(self):
        return self._agent_steps


class OldStackWorker:
    """Rollout worker returning one fixed batch per ``sample`` call."""

    def __init__(self, worker_id, *, env_steps=10, agent_steps=20):
        self.worker_id = worker_id
        self.env_steps = env_steps
        self.agent_steps = agent_steps
        self.sample_count = 0

    def sample(self):
        self.sample_count += 1

        return FakeBatch(self.env_steps, self.agent_steps)

    def get_metrics(self):
        return [f"metrics-{self.worker_id}"]


class OldStackGroup:
    """Group answering the asynchronous fetch of the old API stack.

    ``healthy_counts`` lists the number of healthy workers reported after each
    fetch, so a test can make workers "crash" between rounds. ``stale_rounds``
    makes the first fetches answer for an older iteration.
    """

    def __init__(self, workers, *, healthy_counts=None, stale_rounds=0, empty_rounds=0):
        self.workers = {w.worker_id: w for w in workers}
        self.healthy_counts = list(healthy_counts or [])
        self.stale_rounds = stale_rounds
        self.empty_rounds = empty_rounds
        self.fetch_calls: list[dict] = []

    def healthy_worker_ids(self):
        return list(self.workers)

    def num_healthy_remote_workers(self):
        if self.healthy_counts:
            return self.healthy_counts.pop(0)

        return len(self.workers)

    def foreach_env_runner_async_fetch_ready(self, *, func, remote_worker_ids, tag):
        self.fetch_calls.append({"ids": list(remote_worker_ids), "tag": tag})

        if self.empty_rounds > 0:
            self.empty_rounds -= 1

            return []

        results = [func(self.workers[i]) for i in remote_worker_ids]

        if self.stale_rounds > 0:
            self.stale_rounds -= 1
            results = [(b, m, it - 1) for b, m, it in results]

        return results


def make_old_stack_algo(
    group,
    *,
    unit="episodes",
    duration=2,
    count_steps_by="env_steps",
    iteration=7,
    timeout=10.0,
    reward_estimators=None,
    keep_custom=False,
):
    config = SimpleNamespace(
        evaluation_duration_unit=unit,
        evaluation_duration=duration,
        evaluation_num_env_runners=len(group.workers),
        evaluation_force_reset_envs_before_iteration=True,
        evaluation_sample_timeout_s=timeout,
        enable_env_runner_and_connector_v2=False,
        count_steps_by=count_steps_by,
    )
    evaluation_config = SimpleNamespace(
        rollout_fragment_length=5,
        num_envs_per_env_runner=2,
        keep_per_episode_custom_metrics=keep_custom,
    )

    return SimpleNamespace(
        config=config,
        evaluation_config=evaluation_config,
        eval_env_runner_group=group,
        iteration=iteration,
        reward_estimators=reward_estimators or [],
    )


@pytest.fixture
def summarize_calls(monkeypatch):
    """Replace RLlib's ``summarize_episodes`` by a recorder reporting 3 episodes."""
    calls: list[dict] = []

    def fake_summarize(episodes, new_episodes, keep_custom_metrics):
        calls.append(
            {
                "episodes": list(episodes),
                "new_episodes": list(new_episodes),
                "keep_custom_metrics": keep_custom_metrics,
            }
        )

        return {NUM_EPISODES: 3, "episode_reward_mean": 1.5}

    monkeypatch.setattr("core.callbacks.summarize_episodes", fake_summarize)

    return calls


@pytest.mark.unit
def test_old_stack_counts_one_unit_per_returned_batch(summarize_calls):
    workers = [OldStackWorker(1), OldStackWorker(2)]
    group = OldStackGroup(workers)
    algo = make_old_stack_algo(group, unit="episodes", duration=2, keep_custom=True)

    eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
        algo, group
    )

    assert group.fetch_calls == [
        {"ids": [1, 2], "tag": "env_runner_sample_and_get_metrics"}
    ]
    assert (env_steps, agent_steps) == (20, 40)
    assert eval_results == {
        ENV_RUNNER_RESULTS: {NUM_EPISODES: 3, "episode_reward_mean": 1.5}
    }
    assert summarize_calls == [
        {
            "episodes": ["metrics-1", "metrics-2"],
            "new_episodes": ["metrics-1", "metrics-2"],
            "keep_custom_metrics": True,
        }
    ]


@pytest.mark.unit
def test_old_stack_runs_a_second_round_on_fewer_workers(summarize_calls):
    workers = [OldStackWorker(1), OldStackWorker(2)]
    group = OldStackGroup(workers)
    algo = make_old_stack_algo(group, unit="episodes", duration=3)

    _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    # Round 0 uses both workers, round 1 needs a single episode.
    assert [call["ids"] for call in group.fetch_calls] == [[1, 2], [1]]
    assert env_steps == 30
    assert [w.sample_count for w in workers] == [2, 1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "count_steps_by, duration, expected_ids, expected_steps",
    [
        # One worker asked for 10 units: 10 env steps meet an env-step budget.
        ("env_steps", 10, [1], (10, 20)),
        # The same worker brings 20 agent steps, so a 20-step budget selects a
        # second worker (index 1 * 10 units < 20) and ends with 40 agent steps.
        ("agent_steps", 20, [1, 2], (20, 40)),
    ],
)
def test_old_stack_timesteps_unit_selects_workers_by_fragment_size(
    summarize_calls, count_steps_by, duration, expected_ids, expected_steps
):
    workers = [OldStackWorker(1), OldStackWorker(2)]
    group = OldStackGroup(workers)
    algo = make_old_stack_algo(
        group, unit="timesteps", duration=duration, count_steps_by=count_steps_by
    )

    _, env_steps, agent_steps = _evaluate_with_fixed_duration_once(algo, group)

    # Each worker is credited ``rollout_fragment_length * num_envs_per_env_runner``
    # = 5 * 2 = 10 units when workers are selected for a round.
    assert [call["ids"] for call in group.fetch_calls] == [expected_ids]
    assert (env_steps, agent_steps) == expected_steps


@pytest.mark.unit
def test_old_stack_discards_steps_and_metrics_of_an_older_iteration(summarize_calls):
    group = OldStackGroup([OldStackWorker(1)], stale_rounds=1)
    algo = make_old_stack_algo(group, unit="episodes", duration=1)

    _, env_steps, agent_steps = _evaluate_with_fixed_duration_once(algo, group)

    assert (env_steps, agent_steps) == (0, 0)
    assert summarize_calls[0]["episodes"] == []


@pytest.mark.unit
def test_old_stack_collects_batches_for_reward_estimators(summarize_calls):
    group = OldStackGroup([OldStackWorker(1)])
    algo = make_old_stack_algo(
        group, duration=1, reward_estimators=["importance_sampling"]
    )

    _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    assert env_steps == 10
    assert len(group.fetch_calls) == 1


@pytest.mark.unit
def test_old_stack_gives_up_after_the_sample_timeout(
    monkeypatch, caplog, summarize_calls
):
    clock = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr("core.callbacks.time.time", lambda: next(clock))
    group = OldStackGroup([OldStackWorker(1)], empty_rounds=5)
    algo = make_old_stack_algo(group, duration=1, timeout=10.0)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        eval_results, env_steps, agent_steps = _evaluate_with_fixed_duration_once(
            algo, group
        )

    # The first empty fetch happens 100 s after the start: the loop stops.
    assert len(group.fetch_calls) == 1
    assert (env_steps, agent_steps) == (0, 0)
    assert summarize_calls[0]["episodes"] == []
    assert eval_results[ENV_RUNNER_RESULTS][NUM_EPISODES] == 3
    assert "empty set of episode summary" not in caplog.text


@pytest.mark.unit
def test_old_stack_keeps_waiting_while_the_timeout_is_not_reached(
    monkeypatch, summarize_calls
):
    clock = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr("core.callbacks.time.time", lambda: next(clock))
    group = OldStackGroup([OldStackWorker(1)], empty_rounds=1)
    algo = make_old_stack_algo(group, duration=1, timeout=10.0)

    _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    assert len(group.fetch_calls) == 2
    assert env_steps == 10


@pytest.mark.unit
def test_old_stack_warns_when_every_worker_crashes(caplog, summarize_calls):
    group = OldStackGroup([OldStackWorker(1)], healthy_counts=[1, 0])
    algo = make_old_stack_algo(group, duration=2)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        _, env_steps, _ = _evaluate_with_fixed_duration_once(algo, group)

    # One batch arrived before the worker died; the duration was never met.
    assert env_steps == 10
    assert len(group.fetch_calls) == 1
    assert "all workers crashing" in caplog.text


@pytest.mark.unit
def test_old_stack_warns_about_an_empty_episode_summary(caplog, monkeypatch):
    monkeypatch.setattr(
        "core.callbacks.summarize_episodes", lambda *args, **kwargs: {NUM_EPISODES: 0}
    )
    group = OldStackGroup([OldStackWorker(1)])
    algo = make_old_stack_algo(group, duration=1)

    with caplog.at_level(logging.WARNING, logger="core.callbacks"):
        _evaluate_with_fixed_duration_once(algo, group)

    assert "empty set of episode summary" in caplog.text
    assert "1 episodes" in caplog.text
