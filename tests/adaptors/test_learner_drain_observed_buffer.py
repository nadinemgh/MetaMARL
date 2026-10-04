"""``_ObservedCircularBuffer``: only the consumer thread's empty check counts.

The buffer records that its consumer found it empty by setting an event from
``__len__``. A ``len`` call from any other thread, or on a non-empty buffer,
proves nothing about the consumer and must leave the event unset.
"""

from __future__ import annotations

import threading

import pytest

from core.adaptors.ray.learner_drain import _ObservedCircularBuffer


class Batch:
    """Stand-in for a train batch; ``CircularBuffer`` only asks its size."""

    def env_steps(self):
        return 1


def observed_buffer() -> _ObservedCircularBuffer:
    buffer = _ObservedCircularBuffer(num_batches=4, iterations_per_batch=1)
    buffer._drain_consumer = None
    buffer._drain_idle = threading.Event()
    return buffer


def len_as_consumer(buffer: _ObservedCircularBuffer) -> int:
    """Call ``len(buffer)`` from a thread registered as the buffer's consumer."""

    results: list[int] = []
    thread = threading.Thread(target=lambda: results.append(len(buffer)), daemon=True)
    buffer._drain_consumer = thread
    thread.start()
    thread.join(timeout=5)
    return results[0]


@pytest.mark.unit
def test_empty_check_from_the_consumer_sets_the_event():
    buffer = observed_buffer()

    assert len_as_consumer(buffer) == 0
    assert buffer._drain_idle.is_set()


@pytest.mark.unit
def test_empty_check_from_another_thread_is_ignored():
    buffer = observed_buffer()
    buffer._drain_consumer = threading.Thread(target=lambda: None)

    assert len(buffer) == 0
    assert not buffer._drain_idle.is_set()


@pytest.mark.unit
def test_non_empty_check_from_the_consumer_is_ignored():
    buffer = observed_buffer()
    buffer.add(Batch())

    assert len_as_consumer(buffer) == 1
    assert not buffer._drain_idle.is_set()
