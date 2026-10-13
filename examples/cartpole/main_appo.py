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


# TODO the default mechanism config and fisherman, and observation spaces and action spaces part of config
# TODO where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
# TODO num_fisherman
# TODO wire up the evaluation cfg
# TODO seeding API
# TODO experimentation helpers
# TODO review ray configz
# TODO move this to the config !
# TODO dimension inferred from mechanism ?
# TODO implement early stop for plateau
# TODO fix this its using old api stack
# TODO use the new api stack and better custom model integration
# TODO review
# TODO review
# TODO
# TODO add this after run done
