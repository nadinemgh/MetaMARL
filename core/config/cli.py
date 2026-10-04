from __future__ import annotations

import argparse
import logging
import sys

from core.config.hash_seed import ensure_hash_seed
from core.config.yaml import ConfigError, load_experiment, run_experiment


def main() -> int:
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
    # Same format as the library modules that configure logging at import, so
    # the hash-seed notice is visible before any of them is imported.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    # Before any work: the call restarts the process when the seed is unset.
    ensure_hash_seed()
    sys.exit(main())
