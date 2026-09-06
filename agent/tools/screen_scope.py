"""Tool 1 — is there anything here to classify at all?

Two of this tool's three verdicts are terminal. That is what makes it a real
step rather than a formality: an agent that cannot stop early cannot be right
about alcoholic liquor, for which no GST slab exists to predict.

**It decides only what can be decided by inspection, and says so.** The scope
screen is a regex over a closed list of families and is exact. Whether a
description is *too vague to classify* is a judgement — the difference between
"cotton shirt" and "shirt" is knowledge about the tariff, not a property of the
string — so this tool reports the signals and leaves the call to the agent.

That refusal is deliberate and it matches `lookup_schedule`, which declines to
resolve an ambiguous heading for the same reason. A tool that guesses is a tool
whose wrong answers are invisible, and the whole project is about making wrong
answers visible.
"""

from __future__ import annotations

import re

from agent.contract import ToolResult
from agent.gst import out_of_scope_term
from agent.registry import ToolSpec

#: A description that is essentially a part number: "XR-4400", "Item 55123-A",
#: "Model No. 7712/B". Real catalogue rows look like this, and no tariff
#: heading follows from one.
_MODEL_NUMBER_ONLY = re.compile(
    r"^\W*(?:item|model|part|sku|art(?:icle)?|cat(?:alogue)?)?\W*(?:no\.?|#|:)?\s*"
    r"[A-Za-z]{0,5}[-\s/]?\d{2,}[A-Za-z0-9\-/]*\W*$",
    re.I,
)

#: Tokens carrying no information about what a good *is*. Filtered before
#: counting meaningful words, so "Pack of 12 x 500 ml" does not read as a
#: six-word description of something.
_NOISE = re.compile(
    r"^(?:\d+(?:\.\d+)?|[a-z]{1,2}|pack|packs|packet|packets|box|boxes|bag|bags"
    r"|pcs?|pieces?|nos?|units?|set|sets|x|of|the|and|or|with|for|per|each"
    r"|ml|ltr?|litres?|liters?|kgs?|gms?|grams?|mg|mm|cm|mtr?|metres?|meters?)$",
    re.I,
)

#: Below this many meaningful words, a description is very unlikely to
#: determine a heading. A signal, not a verdict — see the module docstring.
THIN_DESCRIPTION_WORDS = 3


def _meaningful_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]+[A-Za-z\-']*", text) if not _NOISE.match(w)]


def screen_scope(description: str) -> ToolResult:
    """Screen a goods description for GST scope.

    Returns verdict `out_of_scope` (no slab exists — stop), `under_specified`
    (a part number with no goods in it — stop), or `in_scope` with signals the
    agent should weigh. `in_scope` is not a promise that the description
    determines a heading; it means nothing disqualifying was found.
    """
    text = (description or "").strip()
    if not text:
        return ToolResult.ok_(
            {
                "verdict": "under_specified",
                "reason": "no-product-kind",
                "detail": "description is empty",
                "signals": {"meaningful_words": 0},
            }
        )

    # Alcoholic liquor for human consumption is outside GST by constitutional
    # exclusion, taxed under state excise. There is no slab to predict, so this
    # is a categorical stop and not a difficulty judgement.
    if term := out_of_scope_term(text):
        return ToolResult.ok_(
            {
                "verdict": "out_of_scope",
                "reason": "constitutional-exclusion",
                "family": term,
                "detail": (
                    f"names {term!r}: alcoholic liquor for human consumption is "
                    "outside GST (state excise applies). No slab exists."
                ),
            }
        )

    words = _meaningful_words(text)

    if _MODEL_NUMBER_ONLY.match(text):
        return ToolResult.ok_(
            {
                "verdict": "under_specified",
                "reason": "model-number-only",
                "detail": "description is a part or model number with no goods named",
                "signals": {"meaningful_words": len(words)},
            }
        )

    return ToolResult.ok_(
        {
            "verdict": "in_scope",
            "reason": None,
            "detail": "nothing disqualifying found; this is not a finding that "
            "the description determines a heading",
            "signals": {
                "meaningful_words": len(words),
                # True when the description is thin enough that
                # `under_specified` is worth considering. The agent decides;
                # this tool will not, because whether three words are enough
                # depends on which three.
                "thin": len(words) < THIN_DESCRIPTION_WORDS,
                "chars": len(text),
                # Long ruling excerpts behave differently from catalogue rows —
                # they argue for an answer. Flagged so the agent knows it is
                # reading advocacy, and so the trace records that it was told.
                "long_form": len(text) > 600,
            },
        }
    )


SPEC = ToolSpec(
    name="screen_scope",
    description=(
        "Screen a goods description for GST scope before any lookup. Returns "
        "verdict 'out_of_scope' (alcoholic liquor — no GST slab exists, stop "
        "here), 'under_specified' (a part number with no goods named, stop "
        "here), or 'in_scope' with signals. 'in_scope' does NOT mean the "
        "description determines a heading; it means nothing disqualifying was "
        "found. Judging whether a description is too vague is your call, not "
        "this tool's."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The goods description exactly as it appears on the invoice line.",
                "maxLength": 20000,
            }
        },
        "required": ["description"],
        "additionalProperties": False,
    },
    handler=screen_scope,
    stage="screen",
    returns_evidence=False,
    pure=True,
)
