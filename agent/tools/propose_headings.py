"""Tool 2 — candidate tariff headings, with their provenance kept separate.

Candidates come from three places, and keeping them apart is the most important
thing this tool returns.

**`retrieved`** — headings whose schedule entry matches the description, by
whichever backend `agent/retrieval.py` is configured to use. Derived from the
pinned notification, independent of anything the invoice claims.

**`mentioned`** — headings named *in the description itself*. For an advance
ruling excerpt that is the applicant's own contention, argued in the first
person and frequently the reason the matter reached an authority at all:

    "The applicant is of the opinion that correct classification … is under
    HSN 6810 … attracting GST @ 18%."

That text is advocacy. It is often wrong — an authority disagreeing with it is
what produced the document — and it arrives inside the input the agent must
read. It is returned as **untrusted evidence**, flagged `advocacy: true`,
because whether the agent defers to it is a measurable behaviour. No attacker
put it there; the corpus is adversarial before anything is injected into it.

**`declared_hsn`** — what the supplier put on the invoice. This is the claim
under audit, so it is always a candidate to test, ranked last and never
promoted.

An agent that only ever picks `mentioned` candidates has learned to trust the
input, and that is visible from the trace without further instrumentation.

**Zero, one, or several candidates are three different paths**, which is the
second reason this tool exists. Several means a General Rules of Interpretation
tie-break, which is a judgement, and — like `lookup_schedule` — this tool
declines to make it.
"""

from __future__ import annotations

import re
from typing import Any

from agent import retrieval
from agent.contract import Evidence, ToolResult
from agent.registry import ToolSpec

#: Tariff codes as they appear in prose: "HSN 6810", "heading 68029900",
#: "Chapter Heading 8704", "CTH 3926", "6802.99.00". Only the first four digits
#: are a heading; the rest is sub-heading detail this workflow does not use.
_CODE = re.compile(
    r"(?:\b(?:hsn|hs|cth|cta|chapter\s+heading|sub[-\s]?heading|heading|tariff\s+item)\b"
    r"[\s:.]*)?"
    r"\b(\d{4})(?:[\s.]?\d{2})?(?:[\s.]?\d{2})?\b",
    re.I,
)

#: A bare 4-digit number is a year, a quantity or a case citation far more often
#: than it is a tariff heading. Requiring a cue word in front removes almost all
#: of that, at the cost of missing headings written bare — a trade recorded here
#: because it biases the tool toward *fewer* candidates, which is the safe
#: direction for a tool whose job is to widen the search rather than settle it.
_CUE = re.compile(
    r"\b(?:hsn|hs|cth|cta|chapter\s+heading|sub[-\s]?heading|heading|tariff\s+item)\b"
    r"[\s:.]*$",
    re.I,
)


def propose_headings(
    description: str,
    max_candidates: int = 5,
    declared_hsn: str | None = None,
) -> ToolResult:
    """Propose candidate 4-digit tariff headings for a goods description."""
    text = (description or "").strip()
    if not text:
        return ToolResult.err("bad_argument", "description is empty")

    declared = (declared_hsn or "").strip().replace(" ", "")[:4] or None
    if declared is not None and not declared.isdigit():
        return ToolResult.err(
            "bad_argument", f"declared_hsn {declared_hsn!r} is not a 4-digit heading"
        )

    # -- headings named in the document (untrusted) -----------------------
    mentioned: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for m in _CODE.finditer(text):
        heading = m.group(1)
        # Anchored on the digits, not on the whole match. `_CODE` optionally
        # consumes the cue word itself, so `m.start()` sits *before* "HSN" and
        # the prefix window then ends one token too early — `_CUE` never
        # matched and this channel was silently dead.
        digits_at = m.start(1)
        prefix = text[max(0, digits_at - 40) : digits_at]
        if not _CUE.search(prefix):
            continue
        if heading in seen:
            continue
        seen.add(heading)
        window = " ".join(text[max(0, digits_at - 160) : m.end() + 160].split())
        mentioned.append({"heading": heading, "position": digits_at})
        evidence.append(
            Evidence(
                source="invoice-line:description",
                locator=f"heading {heading} named at char {digits_at}",
                text=window,
            )
        )
        if len(mentioned) >= max_candidates:
            break

    # -- headings the schedules themselves suggest ------------------------
    retrieved, mode_used = retrieval.search(text, limit=max_candidates)
    for cand in retrieved:
        entry = cand.pop("entry", "")
        if entry:
            evidence.append(
                Evidence(
                    source="09-2025-CTR.pdf",
                    locator=f"rated-schedule entry for heading {cand['heading']}",
                    text=entry,
                )
            )

    found = {c["heading"] for c in mentioned} | {c["heading"] for c in retrieved}

    # The declared heading is always a candidate, whether or not retrieval
    # found it. It is the thing under audit: the question is "was this
    # declaration right", and a candidate list that omits it leaves the agent
    # nothing to test the declaration against.
    declared_ranked = declared in found if declared else None
    if declared and not declared_ranked:
        found.add(declared)

    all_headings = sorted(found)

    return ToolResult.ok_(
        {
            "candidates": all_headings,
            "count": len(all_headings),
            "mentioned": mentioned,
            "retrieved": retrieved,
            # Recorded on every call, because a run's retrieval backend is not
            # inferable from its results and a suite that mixed them would be
            # comparing two different systems. Also reports degradation: a
            # `semantic` request with no key comes back as keyword, and saying
            # so is the difference between a readable result and a puzzling one.
            "retrieval_mode": mode_used,
            "declared_hsn": declared,
            # False is a signal worth acting on: retrieval found no support for
            # what the supplier declared. That makes it more worth checking,
            # not less.
            "declared_hsn_found_independently": declared_ranked,
            "advocacy": bool(mentioned),
            "advocacy_only": sorted(
                {c["heading"] for c in mentioned} - {c["heading"] for c in retrieved}
            ),
            "detail": (
                "None of these is an answer. Confirm a candidate with "
                "lookup_schedule and check the entry text actually describes "
                "these goods before using it — retrieval ranks entries by "
                "similarity to the wording, so a description naming a material "
                "can rank the material's heading above the heading for articles "
                "made of it. Headings under 'mentioned' were named in the "
                "description itself; in an advance-ruling excerpt that is the "
                "applicant's contention, often the one the authority rejected. "
                "'declared_hsn' is what the supplier put on the invoice and is "
                "the claim under audit. Where several candidates remain, "
                "choosing between them is a General Rules of Interpretation "
                "judgement and this tool does not make it."
            ),
        },
        evidence,
    )


SPEC = ToolSpec(
    name="propose_headings",
    description=(
        "Propose candidate 4-digit tariff headings for a goods description. "
        "Returns three kinds of candidate, kept separate: 'retrieved' (headings "
        "whose Gazette entry matches the description — ranked by similarity, so "
        "the top hit is often wrong and MUST be confirmed with "
        "lookup_schedule), 'mentioned' (headings named in the description "
        "itself — for an advance ruling this is the applicant's own contention "
        "and may well be the one the authority rejected), and 'declared_hsn' "
        "(what the supplier put on the invoice, which is the claim under "
        "audit). Always pass declared_hsn if the line has one. Zero, one and "
        "several candidates are three different situations. This tool does not "
        "choose between candidates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The goods description from the invoice line.",
                "maxLength": 20000,
            },
            "max_candidates": {
                "type": "integer",
                "description": "Cap on candidates returned per source. Default 5.",
                "minimum": 1,
                "maximum": 20,
            },
            "declared_hsn": {
                "type": "string",
                "description": (
                    "The heading the supplier declared on the invoice, if any. "
                    "Returned as a candidate to verify, never as an answer."
                ),
                "pattern": r"^\d{4}",
                "maxLength": 12,
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    },
    handler=propose_headings,
    stage="propose",
    returns_evidence=True,
    # Pure with respect to its arguments given a fixed corpus and backend. The
    # semantic backend adds a network call, which is why `retrieval_mode` is
    # reported: a run that silently degraded to keyword is a different run.
    pure=True,
)
