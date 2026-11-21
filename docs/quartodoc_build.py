"""Run the quartodoc CLI with Google parser options for this repo's docstrings.

A `Returns:` or `Yields:` section here is one value whose text wraps over
several lines. Griffe's default reads each line as its own value, one table row
each, and quartodoc has no setting for it, so the options go into its defaults
table.
"""

import quartodoc.parsers
from quartodoc.__main__ import cli

GOOGLE = {"returns_multiple_items": False}


def configure() -> None:
    """Make quartodoc parse Google docstrings with `GOOGLE`."""

    quartodoc.parsers.DEFAULT_OPTIONS["google"] = GOOGLE


if __name__ == "__main__":
    configure()
    cli()
