"""``wait_for_learner_thread``: evaluation waits for APPO's background updates.

The tests drive RLlib's real ``_LearnerThread`` and ``CircularBuffer``; only
the gradient step is replaced by a slow stand-in, so that the race between
``Algorithm.train`` returning and the last update finishing is wide enough to
observe.
"""

import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest
from ray.rllib.algorithms.appo.torch.appo_torch_learner import APPOTorchLearner
from ray.rllib.algorithms.appo.utils import CircularBuffer
from ray.rllib.algorithms.impala.impala_learner import _LearnerThread
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from core.adaptors.ray import learner_drain
from core.adaptors.ray.learner_drain import wait_for_consumer, wait_for_learner_thread

#: Duration of one stand-in update, in seconds: long enough that a caller
#: returning without waiting would observe it unfinished.
UPDATE_S = 0.3


class Batch:
    """Stand-in for a train batch; ``CircularBuffer`` only asks its size."""

    def env_steps(self):
        return 1


@pytest.fixture
def learner_thread():
    """Start a real learner thread on a fresh buffer with a slow update."""

    applied = []
    release = threading.Event()
    release.set()

    def update(self, training_data, timesteps, _no_metrics_reduce):
        release.wait()
        time.sleep(UPDATE_S)
        applied.append(training_data.batch)

    buffer = CircularBuffer(num_batches=4, iterations_per_batch=1)
    learner = SimpleNamespace(
        metrics=MetricsLogger(), _num_updates=0, _num_updates_lock=threading.Lock()
    )
    thread = _LearnerThread(update_method=update, in_queue=buffer, learner=learner)
    thread.start()

    yield SimpleNamespace(
        thread=thread, buffer=buffer, applied=applied, release=release
    )

    # The thread only checks ``stopped`` once it holds a batch.
    thread.stopped = True
    release.set()
    buffer.add(Batch())
    thread.join(timeout=5)


@pytest.mark.unit
def test_wait_returns_after_the_last_update(learner_thread):
    batches = [Batch(), Batch()]
    for batch in batches:
        learner_thread.buffer.add(batch)

    # The buffer empties when the second batch is taken, a full update before
    # that batch is applied: an emptiness check alone would return too early.
    while len(learner_thread.buffer) > 0:
        time.sleep(0.001)
    assert len(learner_thread.applied) < 2

    wait_for_consumer(learner_thread.thread, learner_thread.buffer, timeout_s=10)

    # The buffer samples at random, so only the set of applied batches is fixed.
    assert sorted(map(id, learner_thread.applied)) == sorted(map(id, batches))


@pytest.mark.unit
def test_wait_returns_promptly_when_the_thread_is_idle(learner_thread):
    start = time.monotonic()
    wait_for_consumer(learner_thread.thread, learner_thread.buffer, timeout_s=10)
    assert time.monotonic() - start < UPDATE_S

    # A second wait on the already observed buffer behaves the same.
    learner_thread.buffer.add(Batch())
    wait_for_consumer(learner_thread.thread, learner_thread.buffer, timeout_s=10)
    assert len(learner_thread.applied) == 1


@pytest.mark.unit
def test_wait_times_out_on_a_stuck_update(learner_thread, monkeypatch):
    monkeypatch.setattr(learner_drain, "_POLL_INTERVAL_S", 0.05)
    learner_thread.release.clear()
    learner_thread.buffer.add(Batch())

    with pytest.raises(TimeoutError):
        wait_for_consumer(learner_thread.thread, learner_thread.buffer, timeout_s=0.2)


@pytest.mark.unit
def test_wait_fails_when_the_thread_is_dead(monkeypatch):
    monkeypatch.setattr(learner_drain, "_POLL_INTERVAL_S", 0.05)
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    buffer = CircularBuffer(num_batches=4, iterations_per_batch=1)

    with pytest.raises(RuntimeError, match="died"):
        wait_for_consumer(thread, buffer, timeout_s=10)


@pytest.mark.unit
def test_impala_deque_is_rejected():
    with pytest.raises(NotImplementedError, match="CircularBuffer"):
        wait_for_consumer(threading.current_thread(), deque(), timeout_s=10)


@pytest.mark.unit
def test_synchronous_learners_need_no_wait():
    # Anything outside the IMPALA family (PPO included) updates in place.
    wait_for_learner_thread(SimpleNamespace())


def bare_appo_learner(num_gpus_per_learner):
    """An APPO learner object without ``build``: no thread, no buffer."""

    learner = object.__new__(APPOTorchLearner)
    learner.config = SimpleNamespace(num_gpus_per_learner=num_gpus_per_learner)
    return learner


@pytest.mark.unit
def test_missing_private_attributes_fail_loudly():
    with pytest.raises(RuntimeError, match="private names"):
        wait_for_learner_thread(bare_appo_learner(num_gpus_per_learner=0))


@pytest.mark.unit
def test_gpu_loader_path_is_rejected():
    with pytest.raises(NotImplementedError, match="GPU"):
        wait_for_learner_thread(bare_appo_learner(num_gpus_per_learner=1))
