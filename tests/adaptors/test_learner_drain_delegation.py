"""``wait_for_learner_thread``: hand-over of the learner thread and its buffer.

The existing ``test_learner_drain.py`` covers the rejection paths and
``wait_for_consumer`` against a real learner thread. This file covers the
remaining branch of ``wait_for_learner_thread``: an IMPALA/APPO learner that
does expose its private thread and buffer must hand both, with the timeout, to
``wait_for_consumer``. ``wait_for_consumer`` is replaced by a recorder here, so
no thread runs.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from ray.rllib.algorithms.appo.torch.appo_torch_learner import APPOTorchLearner

from core.adaptors.ray import learner_drain
from core.adaptors.ray.learner_drain import (
    LEARNER_DRAIN_TIMEOUT_S,
    wait_for_learner_thread,
)


def appo_learner_with_thread(thread, buffer, num_gpus_per_learner=0):
    """An APPO learner object without ``build``, given a thread and a buffer."""

    learner = object.__new__(APPOTorchLearner)
    learner.config = SimpleNamespace(num_gpus_per_learner=num_gpus_per_learner)
    learner._learner_thread = thread
    learner._learner_thread_in_queue = buffer
    return learner


@pytest.fixture
def recorded_waits(monkeypatch):
    calls = []
    monkeypatch.setattr(
        learner_drain, "wait_for_consumer", lambda *args: calls.append(args)
    )
    return calls


@pytest.mark.unit
def test_an_appo_learner_hands_its_thread_and_buffer_to_the_wait(recorded_waits):
    thread = threading.current_thread()
    buffer = object()

    wait_for_learner_thread(appo_learner_with_thread(thread, buffer), timeout_s=7.5)

    assert recorded_waits == [(thread, buffer, 7.5)]


@pytest.mark.unit
def test_the_default_timeout_is_the_module_constant(recorded_waits):
    thread = threading.current_thread()
    buffer = object()

    wait_for_learner_thread(appo_learner_with_thread(thread, buffer))

    assert recorded_waits == [(thread, buffer, LEARNER_DRAIN_TIMEOUT_S)]


@pytest.mark.unit
def test_a_non_impala_learner_never_reaches_the_wait(recorded_waits):
    wait_for_learner_thread(SimpleNamespace(config=SimpleNamespace()))

    assert recorded_waits == []
