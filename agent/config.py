"""Bounds, model selection, and the `.env` loader. See docs/DESIGN.md §10.

The bounds below were set before the first chaos run, which is the only time
setting them means anything. An unbounded agent loop is both a bug and a bill,
and the bill arrives specifically during chaos testing, when every injected
failure invites a retry and every retry costs a call.

Nothing here imports a third-party package. The loop, the tool contract, the
chaos middleware and the scorer all run on a fresh clone with no `pip install`,
because a reliability harness whose own dependencies can fail is measuring the
wrong thing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ENV = REPO_ROOT / ".env"


def load_env(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Load `.env` into os.environ. Returns the names set, never the values.

    Stdlib, for the reason above. A real environment variable wins by default:
    someone who exports a key for one command means it for that command, and a
    stale `.env` silently overriding it is the failure mode that makes dotenv
    loaders untrustworthy.
    """
    path = DEFAULT_ENV if path is None else path
    if not path.is_file():
        return []

    applied: list[str] = []
    # A leading BOM is normal from a Windows editor and would otherwise become
    # part of the first key's name.
    for raw in path.read_text(encoding="utf-8").lstrip("﻿").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # `.env.example` ships every key present and empty, so a copied-but-
        # unfilled file would otherwise set ANTHROPIC_API_KEY="" — which reads
        # as "set" to anything testing membership rather than truth.
        if key and value and (override or not os.environ.get(key)):
            os.environ[key] = value
            applied.append(key)
    return applied


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard limits for one run. Exceeding any ends it in `budget_exhausted`.

    Every field is a count or a duration the loop checks *before* spending,
    not after, so the bound is a refusal rather than a post-mortem.
    """

    #: Longest legitimate path is 7 tools + 2 retries + the draft.
    max_iterations: int = 12
    max_tool_calls: int = 20
    #: Per tool, which is what catches a retry storm on one tool specifically.
    #: A global cap alone lets 20 calls to `lookup_schedule` look healthy.
    max_calls_per_tool: int = 4
    max_tokens: int = 60_000
    max_wall_clock_s: float = 120.0

    def relaxed_for_slow_chaos(self) -> "Budget":
        """Wall-clock raised to 600 s for the `slow` mode.

        The point of `slow` is that tools return correctly but late, so leaving
        the 120 s cap in place would convert every slow run into a timeout and
        measure the bound instead of the behaviour. Every other bound stays:
        a slow tool is no reason to allow more calls.
        """
        return replace(self, max_wall_clock_s=600.0)


#: Suite-level stop. Separate from the per-run budget because the failure it
#: prevents is different: a run that hangs costs one run, a suite that hangs
#: costs a night.
MAX_SUITE_WALL_CLOCK_S: float = 45 * 60

#: Refuse to start a suite whose estimated spend exceeds this, checked before
#: the first call rather than after the bill.
DEFAULT_MAX_RUN_USD: float = 5.00


def max_run_usd() -> float:
    raw = os.environ.get("MAX_RUN_USD", "").strip()
    if not raw:
        return DEFAULT_MAX_RUN_USD
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"MAX_RUN_USD={raw!r} is not a number; unset it or give a value "
            "like 5.00"
        ) from None


#: The model driving the loop and tool 7. Pinned exactly: model behaviour
#: changes under the same name, so every result file records the id actually
#: used.
#:
#: Open weights at frontier scale, served through NVIDIA's catalog so no GPU is
#: needed. Chosen because the agent's job is multi-step reasoning over tool
#: output, which is where the capability gap between 120B and 550B actually
#: shows up: the failure this project keeps hitting is not arithmetic but
#: judgement — reading a schedule entry and deciding whether it describes these
#: goods.
#:
#: **The trade this makes, stated rather than buried.** Project 01 deliberately
#: did *not* use 550B for its open-weight row, on the grounds that nobody is
#: self-hosting 550B. Making it the default here costs two things:
#:
#:   - the self-hostability demonstration. `nemotron-3-super-120b-a12b` has a
#:     published NIM container and runs on one node; this does not.
#:   - the clean bridge to Project 03, whose whole point is running the same
#:     model on a rented GPU and pricing it.
#:
#: Both are recoverable: the wire format is identical, so `--model
#: nvidia/nemotron-3-super-120b-a12b` reproduces every run against the
#: self-hostable model, and Project 03 should use that id. Any published
#: comparison should report both rows rather than only the stronger one.
#:
#: Hosted model ids get retired: Project 01's original pick returned "410 Gone:
#: reached its end of life" on its first live call. Expect to change this, and
#: verify a successor by calling it rather than by reading a docs page.
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

#: The self-hostable sibling. Same wire format, same reasoning switch — the
#: only difference is the id, which is what keeps the Project 03 bridge open.
SELF_HOSTABLE_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def agent_model() -> str:
    return os.environ.get("AGENT_MODEL", "").strip() or DEFAULT_MODEL


def gazette_dir() -> Path:
    """Where the pinned notifications live.

    Overridable so a sibling checkout of Project 01 can be read instead — the
    SHA-256 check in `agent.tools.gazette` reports it if the copies differ,
    which is the point of allowing the override at all.
    """
    override = os.environ.get("GAZETTE_PRIMARY_DIR", "").strip()
    return Path(override) if override else REPO_ROOT / "data" / "reference" / "primary"
