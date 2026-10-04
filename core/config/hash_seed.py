"""Fix Python's string-hash seed so that runs with the same seeds are identical.

Python randomises the hash of ``str`` objects per process, so the iteration
order of a ``set`` of strings changes from one run to the next. RLlib's
env-to-module connector iterates ``MultiAgentEpisode.get_agents_that_stepped()``,
a ``set`` of agent ids, so the order of the agents in each inference batch, and
therefore which agent receives which draw of the (seeded) exploration noise,
depends on that hash seed. The seed is read once, when the interpreter starts,
so it cannot be changed from inside a running process: the entry points call
:func:`ensure_hash_seed` first, and it restarts the process with the variable
set when it is missing. Ray runs in ``local_mode`` here (see
``core.adaptors.ray.runtime``), so every actor lives in that same process.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

#: Hash seed used when ``PYTHONHASHSEED`` is not set. Any fixed value gives
#: reproducible runs; 0 is the conventional choice.
DEFAULT_HASH_SEED = "0"


def ensure_hash_seed(default: str = DEFAULT_HASH_SEED) -> None:
    """Restart the current process with ``PYTHONHASHSEED`` fixed, if it is unset.

    When the variable is already set, its value is respected and logged (a
    value of ``random`` keeps runs non-reproducible, which is logged as a
    warning). Otherwise the variable is exported and the process replaces
    itself with the same interpreter and the same command line
    (``sys.orig_argv``), so the call never returns in that case. The restarted
    process finds the variable set and continues normally.

    Parameters
    ----------
    default : str, optional
        Value exported when ``PYTHONHASHSEED`` is unset (default ``"0"``).

    When to use
    -----------
    As the first statement of a command-line entry point, before any work
    whose cost would be paid twice by the restart.

    Examples
    --------
    >>> import os
    >>> os.environ["PYTHONHASHSEED"] = "0"
    >>> ensure_hash_seed()  # already set: returns without restarting
    """

    value = os.environ.get("PYTHONHASHSEED")

    if value is not None:
        if value == "random":
            logger.warning(
                "PYTHONHASHSEED=random: agent order in RLlib batches, and "
                + "therefore results, will differ between runs."
            )
        else:
            logger.info("PYTHONHASHSEED=%s", value)
        return

    os.environ["PYTHONHASHSEED"] = default
    logger.info("PYTHONHASHSEED was unset; restarting with PYTHONHASHSEED=%s", default)
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
