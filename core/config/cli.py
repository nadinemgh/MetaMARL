"""Command-line entry point that builds and runs experiments from YAML files.

The ``metamarl`` command offers ``run`` and ``check``. Hydra composes the YAML
and applies optional CLI overrides; ``core.config.yaml`` constructs the
experiment with the existing MetaMARL builder calls. ``entry`` fixes Python's
hash seed before execution for reproducible agent ordering.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from typing import NoReturn

from core.config.hash_seed import ensure_hash_seed
from core.config.yaml import ConfigError, load_experiment, load_yaml, run_experiment


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``run`` or ``check`` and return a process exit status.

    Extra ``KEY=VALUE`` arguments are Hydra overrides. ``--print-config``
    displays the fully composed configuration without building the experiment.
    """

    parser = argparse.ArgumentParser(
        prog="metamarl", description="Run MetaMARL experiments from YAML."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Build and run an experiment.")
    check_parser = subparsers.add_parser(
        "check", help="Construct a config without running it."
    )

    for command_parser in (run_parser, check_parser):
        command_parser.add_argument("config")
        command_parser.add_argument(
            "overrides",
            nargs="*",
            metavar="KEY=VALUE",
            help="Hydra overrides, e.g. params.learning_rate=0.0003",
        )
        command_parser.add_argument(
            "--print-config",
            action="store_true",
            help="Print the composed config without constructing an experiment.",
        )

    args = parser.parse_args(argv)

    try:
        if args.print_config:
            import yaml

            config = load_yaml(args.config, overrides=args.overrides)

            print(yaml.safe_dump(config, sort_keys=False), end="")

            return 0

        if args.command == "check":
            experiment = load_experiment(args.config, overrides=args.overrides)

            print(
                f"Config OK: {type(experiment).__module__}.{type(experiment).__name__}"
            )

            return 0

        try:
            run_experiment(args.config, overrides=args.overrides)
        finally:
            # Release Ray if this experiment imported it, even on failure.
            if "ray" in sys.modules:
                import ray

                ray.shutdown()

        return 0
    except ConfigError as exc:
        logging.error("Configuration error: %s", exc, exc_info=True)

        return 2
    except KeyboardInterrupt:
        return 130


def entry(argv: Sequence[str] | None = None) -> NoReturn:
    """Configure logging and hash seed, then exit with :func:`main`'s status.

    Accepting ``argv`` lets an experiment-specific ``run.py`` delegate to this
    generic entry point without implementing its own argument parsing.
    """

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    ensure_hash_seed()
    sys.exit(main(argv))


if __name__ == "__main__":
    entry()
