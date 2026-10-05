"""Wait for APPO's background learner thread before its weights are read.

APPO (like IMPALA) does not update the policy inside ``Algorithm.train``: the
learner's ``update`` only appends the batch to a ``CircularBuffer``, and a
daemon ``_LearnerThread`` takes it from there and applies the gradient step
in the background. ``Algorithm.train`` can therefore return while the last
update is still running, and ``Algorithm.evaluate`` immediately copies the
learner's weights to the evaluation env runners. Whether the evaluated policy
includes that last update then depends on thread timing, which makes the
evaluation (and the regulator's fitness built on it) differ between otherwise
identical runs.

:func:`wait_for_learner_thread` closes that race. The learner thread waits for
work by spinning on ``len(buffer) == 0`` inside ``CircularBuffer.sample``, and
it only reaches that loop once the previous update is complete. Observing the
thread itself find the buffer empty therefore proves that every queued batch
has been applied. The observation is made by switching the class of the live
buffer to :class:`_ObservedCircularBuffer`, whose ``__len__`` records that
event; the class has to be switched in place because the thread is already
blocked inside ``sample`` of that very object.

The function relies on private RLlib attributes (``_learner_thread``,
``_learner_thread_in_queue``). When they are missing it raises instead of
skipping the wait, so that an RLlib upgrade cannot silently bring the race
back.
"""

from __future__ import annotations

import logging
import threading
import time

from ray.rllib.algorithms.appo.utils import CircularBuffer
from ray.rllib.algorithms.impala.impala_learner import IMPALALearner
from ray.rllib.core.learner.learner import Learner

logger = logging.getLogger(__name__)

#: Upper bound on the wait, in seconds. One update takes tens of milliseconds
#: in the fishery configurations, so reaching it means the thread is stuck.
LEARNER_DRAIN_TIMEOUT_S = 60.0

#: Interval, in seconds, at which the waiting caller checks that the learner
#: thread is still alive and that the timeout has not expired.
_POLL_INTERVAL_S = 0.5


class _ObservedCircularBuffer(CircularBuffer):
    """``CircularBuffer`` that records when its consumer thread finds it empty.

    Only the consumer (the learner thread) sets the event: its sole call to
    ``len`` is the wait loop at the top of ``sample``, before it takes a
    batch, so an empty result there means it holds no batch and has finished
    the previous update. A ``len`` call from any other thread proves nothing
    about the consumer and is ignored.
    """

    _drain_consumer: threading.Thread
    _drain_idle: threading.Event

    def __len__(self) -> int:
        size = super().__len__()
        if size == 0 and threading.current_thread() is self._drain_consumer:
            self._drain_idle.set()
        return size


def wait_for_learner_thread(
    learner: Learner, timeout_s: float = LEARNER_DRAIN_TIMEOUT_S
) -> None:
    """Block until the learner's background thread has applied every batch.

    Learners that update synchronously (every learner outside the IMPALA
    family, PPO included) need no wait and return immediately.

    Parameters
    ----------
    learner : ray.rllib.core.learner.learner.Learner
        The learner whose weights are about to be read, typically reached
        through ``LearnerGroup.foreach_learner``.
    timeout_s : float, optional
        Maximum wait in seconds (default :data:`LEARNER_DRAIN_TIMEOUT_S`).

    Raises
    ------
    NotImplementedError
        If batches reach the learner thread through GPU loader threads, or
        through a ``deque`` (IMPALA) instead of APPO's ``CircularBuffer``.
        Neither path guarantees that the buffer stays empty while the caller
        waits, so the wait would not prove anything.
    RuntimeError
        If the private RLlib attributes this function relies on are missing,
        or if the learner thread has died.
    TimeoutError
        If the thread has not drained the buffer within ``timeout_s``.

    When to use: right before reading an APPO learner's weights for
    evaluation, once ``Algorithm.train`` has returned and no further batch
    will be queued. ``PolicyActor.evaluate`` does it for every evaluation pass.

    Examples
    --------
    Learners outside the IMPALA family update synchronously, so the call
    returns at once:

    >>> from types import SimpleNamespace
    >>> wait_for_learner_thread(SimpleNamespace())

    On a built APPO algorithm the function runs on each learner actor:

    >>> algo.learner_group.foreach_learner(wait_for_learner_thread)  # doctest: +SKIP

    References
    ----------
    .. [1] Espeholt, L., Soyer, H., Munos, R., et al. (2018). IMPALA:
       Scalable Distributed Deep-RL with Importance Weighted Actor-Learner
       Architectures. arXiv:1802.01561. The decoupled actor-learner
       architecture that RLlib's IMPALA and APPO learners follow.
    """

    if not isinstance(learner, IMPALALearner):
        return

    if learner.config.num_gpus_per_learner > 0:
        raise NotImplementedError(
            "Waiting for the learner thread is not implemented when batches "
            + "reach it through GPU loader threads (num_gpus_per_learner > 0)."
        )

    try:
        thread = learner._learner_thread
        buffer = learner._learner_thread_in_queue
    except AttributeError as err:
        raise RuntimeError(
            "RLlib's IMPALA/APPO learner no longer exposes the learner thread "
            + "or its input buffer under the expected private names; update "
            + f"{__name__} for this RLlib version."
        ) from err

    wait_for_consumer(thread, buffer, timeout_s)


def wait_for_consumer(
    thread: threading.Thread, buffer: object, timeout_s: float
) -> None:
    """Block until ``thread`` has found ``buffer`` empty, waiting for a batch.

    Parameters
    ----------
    thread : threading.Thread
        The learner thread that consumes ``buffer``.
    buffer : object
        The thread's input queue; must be a ``CircularBuffer``.
    timeout_s : float
        Maximum wait in seconds.

    Raises
    ------
    NotImplementedError
        If ``buffer`` is not a ``CircularBuffer``.
    RuntimeError
        If ``thread`` is not alive.
    TimeoutError
        If ``thread`` has not found the buffer empty within ``timeout_s``.

    When to use: when you hold the learner thread and its buffer directly, as
    in a test; ``wait_for_learner_thread`` finds both on a real learner. The
    first call switches the class of ``buffer`` to ``_ObservedCircularBuffer``
    in place and keeps it for later calls.

    Examples
    --------
    An idle consumer is blocked in ``sample`` on an empty buffer, so the wait
    returns as soon as the observation is in place:

    >>> import threading
    >>> buffer = CircularBuffer(num_batches=2, iterations_per_batch=1)
    >>> consumer = threading.Thread(target=buffer.sample, daemon=True)
    >>> consumer.start()
    >>> wait_for_consumer(consumer, buffer, timeout_s=10.0)
    >>> class Batch:
    ...     def env_steps(self):
    ...         return 1
    >>> _ = buffer.add(Batch())  # lets the consumer finish
    >>> consumer.join(timeout=10.0)
    >>> consumer.is_alive()
    False
    """

    if not isinstance(buffer, CircularBuffer):
        raise NotImplementedError(
            "Waiting for the learner thread is only implemented for APPO's "
            + f"CircularBuffer, not for {type(buffer).__name__}."
        )

    if not isinstance(buffer, _ObservedCircularBuffer):
        buffer._drain_consumer = thread
        buffer._drain_idle = threading.Event()
        buffer.__class__ = _ObservedCircularBuffer

    # Only an observation made from now on counts: no batch is added while
    # the caller waits, so the consumer finding the buffer empty after this
    # point means it has consumed and applied everything that was queued.
    buffer._drain_idle.clear()

    start = time.monotonic()
    while not buffer._drain_idle.wait(timeout=_POLL_INTERVAL_S):
        if not thread.is_alive():
            raise RuntimeError(
                "The learner thread died before applying every queued batch."
            )
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                "The learner thread did not apply the queued batches within "
                + f"{timeout_s:.0f} s."
            )

    logger.debug("Learner thread drained in %.3f s.", time.monotonic() - start)
