"""The seam onto Project 01, and the integrity check on the Gazette sources.

Project 01 — `gst-eval-harness`, pinned at `096aced` — owns the label space,
the scope screen, the Gazette lookup and the scorer. This project imports them
rather than reimplementing them, because two definitions of "which slabs exist"
is one definition too many and they will diverge on the day it matters.

**But a fresh clone has to work.** `make test` must pass before anyone runs a
git+https install, so this module imports upstream when it is there and falls
back to a vendored copy of the *constants only* when it is not. Two things keep
that from becoming quiet drift:

- `tests/test_upstream_parity.py` asserts the fallback equals upstream,
  character for character, whenever upstream is importable. CI installs it, so
  the assertion runs on every push.
- Nothing with real logic is duplicated. The lookup and the scorer have no
  fallback; without upstream they report `unavailable`, which the agent then
  has to handle — an ordinary structured tool error, on the ordinary path.

**Path rebinding.** Upstream resolves its Gazette PDFs relative to the process
working directory, which is correct for a repository that is always run from
its own root and wrong the moment another repository imports it. The module
globals are rebound here to absolute paths under this repo's
`data/reference/primary/`, so the lookup answers from the vendored, hash-pinned
copies regardless of where Python was started.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_DIR = Path(os.environ.get("GAZETTE_PRIMARY_DIR") or REPO_ROOT / "data" / "reference" / "primary")
MANIFEST = PRIMARY_DIR / "MANIFEST.json"

# --------------------------------------------------------------------------
# Label space — vendored fallback, asserted equal to upstream by a test
# --------------------------------------------------------------------------

_FALLBACK_VALID_SLABS: tuple[str, ...] = ("0", "0.25", "1.5", "3", "5", "18", "40")
_FALLBACK_ABOLISHED_SLABS = frozenset({"12", "28"})
_FALLBACK_SLAB_ABOLISHED_ON = {"12": "2025-09-22", "28": "2026-02-01"}
_FALLBACK_UNANSWERABLE = "UNANSWERABLE"
_FALLBACK_UNANSWERABLE_REASONS = frozenset(
    {
        "no-product-kind",
        "model-number-only",
        "rate-fact-absent",
        "packaging-indeterminate",
        "multi-good-no-dominant",
    }
)
_FALLBACK_OUT_OF_SCOPE_TERMS: tuple[str, ...] = (
    "beer",
    "wine",
    "whisky",
    "whiskey",
    "whiskies",
    "vodka",
    "rum",
    "liquor",
    "alcoholic",
)

try:  # pragma: no cover - exercised by CI with upstream installed
    from harness import schema as _upstream_schema

    UPSTREAM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _upstream_schema = None  # type: ignore[assignment]
    UPSTREAM_AVAILABLE = False


if UPSTREAM_AVAILABLE:
    VALID_SLABS = _upstream_schema.VALID_SLABS
    ABOLISHED_SLABS = _upstream_schema.ABOLISHED_SLABS
    SLAB_ABOLISHED_ON = _upstream_schema.SLAB_ABOLISHED_ON
    UNANSWERABLE = _upstream_schema.UNANSWERABLE
    UNANSWERABLE_REASONS = _upstream_schema.UNANSWERABLE_REASONS
    OUT_OF_SCOPE_TERMS = _upstream_schema.OUT_OF_SCOPE_TERMS
    out_of_scope_term = _upstream_schema.out_of_scope_term
else:
    import re as _re

    VALID_SLABS = _FALLBACK_VALID_SLABS
    ABOLISHED_SLABS = _FALLBACK_ABOLISHED_SLABS
    SLAB_ABOLISHED_ON = _FALLBACK_SLAB_ABOLISHED_ON
    UNANSWERABLE = _FALLBACK_UNANSWERABLE
    UNANSWERABLE_REASONS = _FALLBACK_UNANSWERABLE_REASONS
    OUT_OF_SCOPE_TERMS = _FALLBACK_OUT_OF_SCOPE_TERMS

    # The plural suffix is not decoration: catalogue categories are written
    # "Cigarettes", "Beers", and a bare word-boundary anchor misses every one.
    _OOS_RE = _re.compile(
        r"\b(?:" + "|".join(_re.escape(t) for t in OUT_OF_SCOPE_TERMS) + r")(?:e?s)?\b",
        _re.I,
    )

    def out_of_scope_term(text: str) -> str | None:  # type: ignore[misc]
        """Return the out-of-scope family this text names, or None."""
        m = _OOS_RE.search(text)
        return m.group(0).lower() if m else None


#: Slabs a *current* answer may take, plus the refusal sentinel.
ANSWERABLE_SLABS = frozenset(VALID_SLABS)

# --------------------------------------------------------------------------
# Notification dates. The branch the whole workflow turns on.
# --------------------------------------------------------------------------

#: 9/2025-CT(R) supersedes 1/2017 and abolishes the 12 % slab.
IN_FORCE_09_2025 = "2025-09-22"
#: 19/2025-CT(R) omits Schedule VII, abolishing 28 %.
IN_FORCE_19_2025 = "2026-02-01"

#: Anything dated before this is under a schedule this repository does not
#: archive. The correct behaviour is to decline, not to extrapolate backwards:
#: the pre-2025 table is exactly the one models recite from memory, and
#: answering from it would be the failure this project exists to measure.
ARCHIVE_STARTS = IN_FORCE_09_2025


def notification_in_force(on_date: str) -> str | None:
    """Which rate notification governed `on_date` (ISO yyyy-mm-dd).

    None means the date precedes the archive. Callers must treat that as a
    refusal, never as a licence to use the nearest schedule they have.
    """
    if on_date < ARCHIVE_STARTS:
        return None
    if on_date < IN_FORCE_19_2025:
        return "9/2025-CT(R)"
    return "9/2025-CT(R) as amended by 19/2025-CT(R)"


# --------------------------------------------------------------------------
# Source integrity
# --------------------------------------------------------------------------


class SourceIntegrityError(RuntimeError):
    """A pinned Gazette document is missing or is not the document we pinned."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_primary(*, strict: bool = True) -> dict[str, str]:
    """Check every pinned document against MANIFEST.json.

    Returns `{filename: "ok" | reason}`. A tool that answers from a document it
    cannot verify is worse than one that refuses, so `lookup_schedule` calls
    this and returns `source_mismatch` rather than a rate.

    Deliberately not cached. It runs once per process via `primary_status`,
    and re-reading three files to prove an answer's provenance is cheap next to
    publishing a rate from a document that quietly changed.
    """
    if not MANIFEST.exists():
        if strict:
            raise SourceIntegrityError(f"no manifest at {MANIFEST}")
        return {"MANIFEST.json": "missing"}

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    status: dict[str, str] = {}
    for doc in manifest.get("documents", []):
        name = doc["file"]
        path = PRIMARY_DIR / name
        if not path.exists():
            status[name] = "missing"
            continue
        actual = _sha256(path)
        expected = doc.get("sha256", "")
        status[name] = "ok" if actual == expected else f"sha256 mismatch: {actual}"
    return status


_primary_status: dict[str, str] | None = None


def primary_status() -> dict[str, str]:
    """`verify_primary`, computed once per process."""
    global _primary_status
    if _primary_status is None:
        _primary_status = verify_primary(strict=False)
    return _primary_status


def primary_ok() -> bool:
    status = primary_status()
    return bool(status) and all(v == "ok" for v in status.values())


# --------------------------------------------------------------------------
# The Gazette lookup, rebound onto this repository's vendored copies
# --------------------------------------------------------------------------

try:  # pragma: no cover - needs upstream + pypdf
    from harness.collect import schedule_lookup as _schedule_lookup

    # Upstream resolves these relative to the working directory. Rebind to the
    # vendored, hash-verified copies so the answer does not depend on where
    # Python was started. `_text` is cached on the path *string*, so rebinding
    # before first use is sufficient and does not need the cache cleared.
    _schedule_lookup.PRIMARY = PRIMARY_DIR
    _schedule_lookup.RATED = PRIMARY_DIR / "09-2025-CTR.pdf"
    _schedule_lookup.EXEMPT = PRIMARY_DIR / "10-2025-CTR.pdf"

    LOOKUP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _schedule_lookup = None  # type: ignore[assignment]
    LOOKUP_AVAILABLE = False


def lookup_heading(heading: str) -> Any:
    """Upstream `schedule_lookup.lookup`, or None when unavailable.

    Returns upstream's `Match`. Callers must not unwrap it blindly: `ambiguous`
    and `chapter_only` are answers, not failures, and collapsing them into a
    slab is the judgement this tool refuses to make.
    """
    if not LOOKUP_AVAILABLE:
        return None
    return _schedule_lookup.lookup(heading)


def amended_2026() -> dict[str, list[tuple[str, str, str]]]:
    """Notification 19/2025's relocations, from upstream's checked transcription."""
    if not LOOKUP_AVAILABLE:
        return {}
    return dict(_schedule_lookup.AMENDED_2026)


def schedule_slab() -> dict[str, str | None]:
    """Schedule -> combined GST rate. VII maps to None: it no longer exists."""
    if not LOOKUP_AVAILABLE:
        return {}
    return dict(_schedule_lookup.SCHEDULE_SLAB)
