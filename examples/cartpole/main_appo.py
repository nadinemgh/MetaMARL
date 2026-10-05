"""Entry point of the cart-pole experiment with an APPO inner level.

Run ``uv run python -m examples.cartpole.main_appo`` for the full
configuration (100 generations of 200 APPO iterations on horizon 1000). It
accepts the options of :mod:`examples.cartpole.debug` except ``--algo``.
"""

import sys
from collections.abc import Sequence
from typing import Optional

from examples.cartpole.debug import main as run


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the cart-pole experiment with APPO.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Options of :func:`examples.cartpole.debug.parse_args` without
        ``--algo``; ``None`` reads ``sys.argv``.

    When to use: as the body of the ``main_appo`` script. It starts Ray, so it
    has no doctest.
    """
    options = sys.argv[1:] if argv is None else list(argv)
    run(["--algo", "appo", *options])


if __name__ == "__main__":
    main()
