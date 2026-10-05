"""Entry point of the cart-pole experiment with a PPO inner level.

Run ``uv run python -m examples.cartpole.main_ppo`` for the full
configuration (100 generations of 100 PPO iterations on horizon 1000). It
accepts the options of :mod:`examples.cartpole.debug` except ``--algo``.

Known limitation of the framework, not of this example: PPO's training step
receives one metrics dictionary per finished episode (241 of them for a batch of
4000 steps of 20-step episodes, measured), while
``core.callbacks.log_and_report_episode_metrics`` logs every episode as an
``item`` under the same key (the identifier loses its unique ``raw`` suffix).
RLlib's ``ItemStats.merge`` accepts a single incoming value, so the first PPO
iteration raises an ``AssertionError``. The APPO entry point is not affected.
"""

import sys
from collections.abc import Sequence
from typing import Optional

from examples.cartpole.debug import main as run


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the cart-pole experiment with PPO.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Options of :func:`examples.cartpole.debug.parse_args` without
        ``--algo``; ``None`` reads ``sys.argv``.

    When to use: as the body of the ``main_ppo`` script. It starts Ray, so it
    has no doctest.
    """
    options = sys.argv[1:] if argv is None else list(argv)
    run(["--algo", "ppo", *options])


if __name__ == "__main__":
    main()
