"""Allow ``python -m energy_intelligence``.

Delegates to the same ``main`` used by the ``energy-intel`` console script, so
there is a single implementation of the CLI.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
