"""The task suite: what turns single runs into rates.

The chaos harness breaks one call. This turns that into a number — 44 scenarios
scored against pinned criteria, run at each injection rate, written to a results
file that a published figure can be traced back to.

    python -m suite.run --dry-run       # composition, calling nothing
    python -m suite.run --chaos 0.0     # the baseline
    python -m suite.run --chaos 0.25 --seed 7

Two populations, never averaged into one number: 28 scenarios derived from
Project 01's golden rows, and 16 written to reach branches the golden set cannot
(out of scope, under-specified, pre-archive dates, the 2026-02-01 boundary, the
conditional headings). A rate that mixed real classification problems with
scenarios written to exercise the author's own branches would be a number about
nothing in particular.
"""

from suite.scenarios import Scenario, all_scenarios, load, summary
from suite.score import Score, score_run, summarise

__all__ = [
    "Scenario",
    "Score",
    "all_scenarios",
    "load",
    "score_run",
    "summarise",
    "summary",
]
