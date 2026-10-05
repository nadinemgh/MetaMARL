"""Command-line entry point that builds and runs experiments from YAML files.

The module exposes the ``metamarl`` command with two sub-commands, ``run`` and
``check``, on top of the declarative loader in ``core.config.yaml``. ``check``
validates a configuration and constructs its ``experiment`` object without
running it; ``run`` additionally applies the file's ``run`` section. Run it as
``python -m core.config.cli run config.yaml``; started this way it first calls
``ensure_hash_seed``, which restarts the process with a fixed string-hash seed
so that runs with equal seeds are reproducible.
"""

from __future__ import annotations

import argparse
import logging
import sys

from core.config.hash_seed import ensure_hash_seed
from core.config.yaml import ConfigError, load_experiment, run_experiment


def main() -> int:
    """Parse ``sys.argv`` and execute the ``run`` or ``check`` sub-command.

    The first positional argument selects the command and the second is the
    path of the YAML configuration. ``check`` builds the ``experiment`` section
    through ``load_experiment`` and prints ``Config OK: <module>.<class>`` to
    standard output; ``run`` calls ``run_experiment``, which builds the
    experiment and then applies the ``run`` section. A ``ConfigError`` is
    logged at ERROR level together with the exception that caused it, so the
    failing YAML path and the root cause are both visible.

    Returns
    -------
    int
        Process exit status: ``0`` on success, ``2`` after a configuration
        error, and ``130`` after a ``KeyboardInterrupt``.

    Raises
    ------
    SystemExit
        Raised by ``argparse`` (exit status ``2``) when the command or the
        config path is missing or the command is unknown.

    When to use: as the body of a console script or of ``python -m
    core.config.cli``; call it directly from Python only when ``sys.argv``
    already holds the intended arguments.

    Examples
    --------
    >>> import sys, tempfile
    >>> from pathlib import Path
    >>> from unittest import mock
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / "config.yaml"
    ...     _ = path.write_text(
    ...         "experiment:\\n  _target_: fractions.Fraction\\n  _args_: [1, 2]\\n"
    ...     )
    ...     with mock.patch.object(sys, "argv", ["metamarl", "check", str(path)]):
    ...         status = main()
    Config OK: fractions.Fraction
    >>> status
    0
    """

    parser = argparse.ArgumentParser(
        prog="metamarl", description="Run MetaMARL experiments from YAML."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Build and run an experiment.")

    run_parser.add_argument("config")

    check_parser = subparsers.add_parser(
        "check", help="Validate and construct a config without running it."
    )

    check_parser.add_argument("config")

    args = parser.parse_args()

    try:
        if args.command == "check":
            experiment = load_experiment(args.config)

            print(
                f"Config OK: {type(experiment).__module__}.{type(experiment).__name__}"
            )

            return 0

        run_experiment(args.config)

        return 0
    except ConfigError as exc:
        # The ConfigError only names the failing YAML path; the traceback
        # carries the chained exception that explains why it failed.
        logging.error("Configuration error: %s", exc, exc_info=exc)

        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    # The library modules leave logging to the entry point, so configure it
    # here, before the hash-seed check, whose notice must be visible.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    # Before any work: the call restarts the process when the seed is unset.
    ensure_hash_seed()
    sys.exit(main())
